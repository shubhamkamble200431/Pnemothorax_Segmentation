#!/usr/bin/env python3
"""
=================================================================================
COMPLETE PNEUMOTHORAX SEGMENTATION SYSTEM (CORRECTED)
Part 1: Imports and Configuration
=================================================================================
"""
import math
import os
import glob
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not available. DICOM metadata will not be used.")

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.ndimage as ndi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration - edit these paths before running"""
    # ------------------ Dataset ------------------
    DATA_ROOT = r"/home/ml/Desktop/shubham/FYP_2.0/final_dataset"
    TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train", "png_images")
    TRAIN_MASKS_DIR = os.path.join(DATA_ROOT, "train", "png_masks")
    TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "test", "png_images")
    TEST_MASKS_DIR = os.path.join(DATA_ROOT, "test", "png_masks")

    # Optional: folder containing original DICOMs
    DICOM_DIR = None  # e.g. "./dicoms" or None

    # ------------------ Outputs ------------------
    OUTPUT_DIR = "./outputs"
    MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    PRED_DIR = os.path.join(RESULTS_DIR, "predictions")

    # ------------------ Model selection ------------------
    MODELS_TO_TRAIN = [1,4]  # PTXSeg-Net only

    # ------------------ Training params ------------------
    IMG_SIZE = 512
    IMG_CHANNELS = 1
    EPOCHS = 25
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    PATIENCE = 15
    MIN_DELTA = 0.001

    # ------------------ Data split ------------------
    TRAIN_SPLIT = 0.85
    VAL_SPLIT = 0.15
    RANDOM_SEED = 42

    # ------------------ Hardware ------------------
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # ------------------ Model architecture ------------------
    ENCODER_CHANNELS = [64, 128, 256, 512, 1024]

    # ------------------ Post-processing / TTA ------------------
    OPTIMAL_THRESHOLD = 0.5
    MIN_COMPONENT_SIZE = 100
    
    TTA_TRANSFORMS = [
        ('none', lambda x: x, lambda x: x),
        ('hflip', np.fliplr, np.fliplr),
        ('vflip', np.flipud, np.flipud),
        ('rot90', lambda x: np.rot90(x, k=1), lambda x: np.rot90(x, k=3)),
        ('rot270', lambda x: np.rot90(x, k=3), lambda x: np.rot90(x, k=1)),
    ]
    
    ENSEMBLE_GLOB = "*.pth"

    # Grid-search parameters
    THRESHOLDS_TO_SEARCH = np.linspace(0.2, 0.8, 13)
    COMPONENT_SIZES_TO_SEARCH = [0, 50, 100, 200, 500]

    # ------------------ Visualization ------------------
    PRED_OVERLAY_ALPHA = 0.5
    TOP_K_PRED_VIS = 5

    @classmethod
    def create_directories(cls):
        """Create necessary output directories"""
        for dir_path in [cls.MODELS_DIR, cls.PLOTS_DIR, cls.RESULTS_DIR, cls.PRED_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Output directories created at: {cls.OUTPUT_DIR}")

    @classmethod
    def get_model_config(cls, model_id: int) -> Dict:
        """Get configuration for specific model"""
        configs = {
            1: {'name': 'baseline_unet', 'attention': False, 'residual': False, 
                'deep_supervision': False, 'pretrain': None, 'lr': 0.0001, 
                'batch_size': 4, 'loss': 'dice', 'epochs': 25},
            2: {'name': 'unet_imagenet', 'attention': False, 'residual': False,
                'deep_supervision': False, 'pretrain': 'imagenet', 'lr': 0.00001,
                'batch_size': 8, 'loss': 'focal', 'epochs': 50},
            3: {'name': 'unet_autoencoder', 'attention': False, 'residual': False,
                'deep_supervision': False, 'pretrain': 'autoencoder', 'lr': 0.00001,
                'batch_size': 8, 'loss': 'focal', 'epochs': 50},
            4: {'name': 'attention_unet', 'attention': True, 'residual': False,
                'deep_supervision': False, 'pretrain': 'autoencoder', 'lr': 0.0001,
                'batch_size': 4, 'loss': 'dice', 'epochs': 25},
            5: {'name': 'attention_residual_unet', 'attention': True, 'residual': True,
                'deep_supervision': False, 'pretrain': 'autoencoder', 'lr': 0.00001,
                'batch_size': 8, 'loss': 'focal', 'epochs': 50},
            6: {'name': 'ptxseg_net', 'attention': True, 'residual': True,
                'deep_supervision': True, 'pretrain': 'autoencoder', 'lr': 0.00001,
                'batch_size': 8, 'loss': 'focal', 'epochs': 50},
            7: {'name': 'ptxseg_net_high_lr', 'attention': True, 'residual': True,
                'deep_supervision': True, 'pretrain': 'autoencoder', 'lr': 0.0001,
                'batch_size': 8, 'loss': 'focal', 'epochs': 25},
            8: {'name': 'ptxseg_net_large_batch', 'attention': True, 'residual': True,
                'deep_supervision': True, 'pretrain': 'autoencoder', 'lr': 0.00001,
                'batch_size': 16, 'loss': 'focal', 'epochs': 50},
            9: {'name': 'ptxseg_net_dice', 'attention': True, 'residual': True,
                'deep_supervision': True, 'pretrain': 'autoencoder', 'lr': 0.0001,
                'batch_size': 8, 'loss': 'dice', 'epochs': 25},
            10: {'name': 'ptxseg_net_combined', 'attention': True, 'residual': True,
                 'deep_supervision': True, 'pretrain': 'autoencoder', 'lr': 0.00001,
                 'batch_size': 8, 'loss': 'combined', 'epochs': 25},
            11: {'name': 'ptxseg_net_extended', 'attention': True, 'residual': True,
                 'deep_supervision': True, 'pretrain': 'autoencoder', 'lr': 0.00001,
                 'batch_size': 8, 'loss': 'focal', 'epochs': 100},
            12: {'name': 'ptxseg_net_deep', 'attention': True, 'residual': True,
                 'deep_supervision': True, 'pretrain': 'autoencoder', 'lr': 0.0001,
                 'batch_size': 8, 'loss': 'dice', 'epochs': 10, 'deeper': True}
        }
        return configs.get(model_id, configs[6])
    """
=================================================================================
COMPLETE PNEUMOTHORAX SEGMENTATION SYSTEM (CORRECTED)
Part 2: Utilities and Dataset
=================================================================================
"""

# =============================================================================
# UTILITIES
# =============================================================================

def list_by_stem(images_dir: str) -> Dict[str, str]:
    """Return mapping stem -> fullpath for png files in a folder."""
    if not os.path.exists(images_dir):
        return {}
    files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    return {Path(f).stem: f for f in files}


def pair_by_stem(img_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    """
    Create (image_path, mask_path) pairs for files named like:

        train_image_1_0.png  <->  train_mask_1_0.png
        test_image_1_1.png   <->  test_mask_1_1.png

    i.e. same name except 'image' vs 'mask'.
    """
    if not os.path.isdir(img_dir):
        raise RuntimeError(f"Image dir not found: {img_dir}")
    if not os.path.isdir(mask_dir):
        raise RuntimeError(f"Mask dir not found: {mask_dir}")

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    pairs: List[Tuple[str, str]] = []

    print(f"\nScanning images in: {img_dir}")
    print(f"Found {len(img_paths)} images")

    for img_path in img_paths:
        img_name = os.path.basename(img_path)

        # main rule: replace the first occurrence of "image" with "mask"
        if "image" in img_name:
            mask_name = img_name.replace("image", "mask", 1)
        else:
            # fallback: change the second token to 'mask'
            stem = Path(img_name).stem
            parts = stem.split("_")
            if len(parts) >= 2:
                parts[1] = "mask"
                mask_name = "_".join(parts) + ".png"
            else:
                print(f"[WARN] Could not infer mask name for {img_name}")
                continue

        mask_path = os.path.join(mask_dir, mask_name)

        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            print(f"[WARN] No mask for {img_name}, expected: {mask_path}")

    print(f"Paired {len(pairs)} image/mask files")
    return pairs



def extract_label_from_stem(stem: str) -> Optional[int]:
    """Extract final label token (0 or 1) from filename stem."""
    parts = stem.split('_')
    if parts and parts[-1] in ('0', '1'):
        return int(parts[-1])
    for p in reversed(parts):
        if p in ('0', '1'):
            return int(p)
    return None


def read_image_gray(path: str, size: int) -> np.ndarray:
    """Read and resize grayscale image, normalize to [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Unable to read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def read_mask(path: str, size: int) -> np.ndarray:
    """Read and resize binary mask."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Unable to read mask: {path}")
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8)
    return m


# =============================================================================
# DATASET
# =============================================================================

class PneumothoraxDataset(Dataset):
    """Dataset for pneumothorax segmentation."""
    
    def __init__(self, pairs: List[Tuple[str, str]], augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = read_image_gray(img_path, Config.IMG_SIZE)
        mask = read_mask(mask_path, Config.IMG_SIZE)

        if self.augment:
            img, mask = self.apply_augmentation(img, mask)

        img_t = torch.from_numpy(img).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, mask_t

    def apply_augmentation(self, img, mask):
        """Apply random augmentations."""
        # Horizontal flip
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        
        # Rotation
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        
        # Brightness adjustment
        if random.random() < 0.5:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-0.05, 0.05)
            img = np.clip(alpha * img + beta, 0, 1)
        
        return img, mask


def load_dataset_new_structure():
    """
    Load dataset from train/test structure.
    Returns train_pairs, val_pairs, test_pairs.
    """
    print(f"\n{'='*80}\nLOADING DATASET\n{'='*80}")
    
    # Check if directories exist
    for dir_path, name in [
        (Config.TRAIN_IMAGES_DIR, "Train images"),
        (Config.TRAIN_MASKS_DIR, "Train masks"),
        (Config.TEST_IMAGES_DIR, "Test images"),
        (Config.TEST_MASKS_DIR, "Test masks")
    ]:
        if not os.path.exists(dir_path):
            raise RuntimeError(f"{name} directory not found: {dir_path}")
    
    train_pairs_all = pair_by_stem(Config.TRAIN_IMAGES_DIR, Config.TRAIN_MASKS_DIR)
    test_pairs = pair_by_stem(Config.TEST_IMAGES_DIR, Config.TEST_MASKS_DIR)
    
    print(f"Found {len(train_pairs_all)} train pairs and {len(test_pairs)} test pairs")
    
    if len(train_pairs_all) == 0:
        raise RuntimeError("No training pairs found. Check your dataset structure.")
    
    # Split train into train/val
    random.seed(Config.RANDOM_SEED)
    random.shuffle(train_pairs_all)
    n_total = len(train_pairs_all)
    n_train = int(n_total * Config.TRAIN_SPLIT)
    train_pairs = train_pairs_all[:n_train]
    val_pairs = train_pairs_all[n_train:]
    
    print(f"Split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs
"""
=================================================================================
COMPLETE PNEUMOTHORAX SEGMENTATION SYSTEM (CORRECTED)
Part 3: Model Components and Architecture
=================================================================================
"""

# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class ConvBlock(nn.Module):
    """Basic convolutional block."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResidualBlock(nn.Module):
    """Residual convolutional block."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant features."""
    
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# =============================================================================
# PTXSegNet MODEL
# =============================================================================

class PTXSegNet(nn.Module):
    """Complete PTXSeg-Net architecture with attention and deep supervision."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.use_attention = config.get('attention', True)
        self.use_residual = config.get('residual', True)
        self.deep_supervision = config.get('deep_supervision', True)

        channels = Config.ENCODER_CHANNELS.copy()
        if config.get('deeper', False):
            channels = [64, 128, 256, 512, 1024, 2048]

        BlockType = ResidualBlock if self.use_residual else ConvBlock

        # Encoder
        self.enc1 = BlockType(Config.IMG_CHANNELS, channels[0])
        self.enc2 = BlockType(channels[0], channels[1])
        self.enc3 = BlockType(channels[1], channels[2])
        self.enc4 = BlockType(channels[2], channels[3])
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = BlockType(channels[3], channels[4])

        # Decoder
        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2)
        self.dec4 = BlockType(channels[4], channels[3])

        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3 = BlockType(channels[3], channels[2])

        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2 = BlockType(channels[2], channels[1])

        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.dec1 = BlockType(channels[1], channels[0])

        # Attention gates
        if self.use_attention:
            self.att4 = AttentionGate(F_g=channels[3], F_l=channels[3], 
                                     F_int=max(channels[3]//2, 16))
            self.att3 = AttentionGate(F_g=channels[2], F_l=channels[2], 
                                     F_int=max(channels[2]//2, 16))
            self.att2 = AttentionGate(F_g=channels[1], F_l=channels[1], 
                                     F_int=max(channels[1]//2, 16))
            self.att1 = AttentionGate(F_g=channels[0], F_l=channels[0], 
                                     F_int=max(channels[0]//2, 16))

        # Final output
        self.out = nn.Conv2d(channels[0], 1, 1)

        # Deep supervision outputs
        if self.deep_supervision:
            self.dsv4 = nn.Conv2d(channels[3], 1, 1)
            self.dsv3 = nn.Conv2d(channels[2], 1, 1)
            self.dsv2 = nn.Conv2d(channels[1], 1, 1)
            self.dsv1 = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # Decoder with attention
        d4 = self.up4(b)
        if self.use_attention:
            e4 = self.att4(d4, e4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        if self.use_attention:
            e3 = self.att3(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if self.use_attention:
            e2 = self.att2(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if self.use_attention:
            e1 = self.att1(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = torch.sigmoid(self.out(d1))

        if self.deep_supervision and self.training:
            dsv4 = torch.sigmoid(self.dsv4(d4))
            dsv3 = torch.sigmoid(self.dsv3(d3))
            dsv2 = torch.sigmoid(self.dsv2(d2))
            dsv1 = torch.sigmoid(self.dsv1(d1))

            dsv4 = F.interpolate(dsv4, size=out.shape[2:], mode='bilinear', 
                               align_corners=False)
            dsv3 = F.interpolate(dsv3, size=out.shape[2:], mode='bilinear', 
                               align_corners=False)
            dsv2 = F.interpolate(dsv2, size=out.shape[2:], mode='bilinear', 
                               align_corners=False)

            return out, [dsv4, dsv3, dsv2, dsv1]

        return out


# =============================================================================
# AUTOENCODER
# =============================================================================

class Autoencoder(nn.Module):
    """Autoencoder for pretraining."""
    
    def __init__(self):
        super().__init__()
        channels = Config.ENCODER_CHANNELS

        self.enc1 = ConvBlock(Config.IMG_CHANNELS, channels[0])
        self.enc2 = ConvBlock(channels[0], channels[1])
        self.enc3 = ConvBlock(channels[1], channels[2])
        self.enc4 = ConvBlock(channels[2], channels[3])
        self.bottleneck = ConvBlock(channels[3], channels[4])
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2)
        self.dec4 = ConvBlock(channels[3], channels[3])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3 = ConvBlock(channels[2], channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2 = ConvBlock(channels[1], channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.dec1 = ConvBlock(channels[0], channels[0])
        self.out = nn.Conv2d(channels[0], Config.IMG_CHANNELS, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(self.up4(b))
        d3 = self.dec3(self.up3(d4))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))

        return torch.sigmoid(self.out(d1))
    """
=================================================================================
COMPLETE PNEUMOTHORAX SEGMENTATION SYSTEM (CORRECTED)
Part 4: Loss Functions, Metrics, and Training
=================================================================================
"""

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Focal + Dice loss."""
    
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        return 0.5 * self.focal(inputs, targets) + 0.5 * self.dice(inputs, targets)


def get_loss_function(loss_type: str):
    """Get loss function by name."""
    losses = {
        'focal': FocalLoss(),
        'dice': DiceLoss(),
        'combined': CombinedLoss(),
        'bce': nn.BCELoss()
    }
    return losses.get(loss_type, FocalLoss())


# =============================================================================
# METRICS
# =============================================================================

class MetricsCalculator:
    """Calculate segmentation metrics."""
    
    @staticmethod
    def dice_coefficient(pred, target, threshold=0.5):
        pred = (pred > threshold).float()
        target = target.float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2. * intersection / (union + 1e-8)).item() if union > 0 else 1.0

    @staticmethod
    def jaccard_index(pred, target, threshold=0.5):
        pred = (pred > threshold).float()
        target = target.float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection / (union + 1e-8)).item() if union > 0 else 1.0

    @staticmethod
    def pixel_accuracy(pred, target, threshold=0.5):
        pred = (pred > threshold).float()
        target = target.float()
        correct = (pred == target).sum()
        return (correct / target.numel()).item()

    @staticmethod
    def sensitivity_specificity(pred, target, threshold=0.5):
        pred = (pred > threshold).float()
        target = target.float()
        TP = ((pred == 1) & (target == 1)).sum().float()
        FP = ((pred == 1) & (target == 0)).sum().float()
        TN = ((pred == 0) & (target == 0)).sum().float()
        FN = ((pred == 0) & (target == 1)).sum().float()
        sensitivity = TP / (TP + FN + 1e-8)
        specificity = TN / (TN + FP + 1e-8)
        return sensitivity.item(), specificity.item()


# =============================================================================
# VISUALIZER
# =============================================================================

class Visualizer:
    """Visualization utilities."""
    
    @staticmethod
    def plot_training_history(history: Dict, model_name: str):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice
        axes[0, 1].plot(history['train_dice'], label='Train Dice')
        axes[0, 1].plot(history['val_dice'], label='Val Dice')
        axes[0, 1].set_title('Dice Coefficient')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Jaccard
        if 'val_jaccard' in history:
            axes[1, 0].plot(history['val_jaccard'], label='Val Jaccard')
            axes[1, 0].set_title('Jaccard Index')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Jaccard')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Accuracy
        if 'val_accuracy' in history:
            axes[1, 1].plot(history['val_accuracy'], label='Val Accuracy')
            axes[1, 1].set_title('Pixel Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(Config.PLOTS_DIR, f'{model_name}_history.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Training history plot saved to: {save_path}")


# =============================================================================
# AUTOENCODER PRETRAINING
# =============================================================================

def pretrain_autoencoder(train_loader, val_loader, epochs=20):
    """Pretrain autoencoder for feature extraction."""
    print(f"\n{'='*80}\nPretraining Autoencoder\n{'='*80}")
    
    model = Autoencoder().to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images = images.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(Config.MODELS_DIR, 'autoencoder_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved autoencoder checkpoint")
    
    print(f"Autoencoder pretraining complete! Best val loss: {best_loss:.4f}")
    return model
"""
=================================================================================
COMPLETE PNEUMOTHORAX SEGMENTATION SYSTEM (CORRECTED)
Part 5: Training, TTA, Ensemble, and Post-processing
=================================================================================
"""

# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Model trainer with validation and early stopping."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.criterion = get_loss_function(config['loss'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config['lr'],
            weight_decay=Config.WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        self.history = {
            'train_loss': [], 'val_loss': [], 'train_dice': [],
            'val_dice': [], 'val_jaccard': [], 'val_accuracy': [],
            'val_sensitivity': [], 'val_specificity': []
        }
        self.best_dice = 0.0
        self.patience_counter = 0
        self.metrics_calc = MetricsCalculator()

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_dice = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, masks in pbar:
            images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
            self.optimizer.zero_grad()

            if self.config.get('deep_supervision', False):
                outputs, aux_outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                for i, aux_out in enumerate(aux_outputs):
                    loss += [0.8, 0.6, 0.4, 0.2][i] * self.criterion(aux_out, masks)
            else:
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            dice = self.metrics_calc.dice_coefficient(outputs, masks)
            total_dice += dice
            pbar.set_postfix({'loss': loss.item(), 'dice': dice})

        return total_loss / len(self.train_loader), total_dice / len(self.train_loader)

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = total_dice = total_jaccard = total_accuracy = 0
        total_sensitivity = total_specificity = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, masks in pbar:
                images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                dice = self.metrics_calc.dice_coefficient(outputs, masks)
                jaccard = self.metrics_calc.jaccard_index(outputs, masks)
                accuracy = self.metrics_calc.pixel_accuracy(outputs, masks)
                sens, spec = self.metrics_calc.sensitivity_specificity(outputs, masks)

                total_dice += dice
                total_jaccard += jaccard
                total_accuracy += accuracy
                total_sensitivity += sens
                total_specificity += spec
                pbar.set_postfix({'loss': loss.item(), 'dice': dice})

        n = len(self.val_loader)
        return {
            'loss': total_loss/n, 'dice': total_dice/n, 'jaccard': total_jaccard/n,
            'accuracy': total_accuracy/n, 'sensitivity': total_sensitivity/n,
            'specificity': total_specificity/n
        }

    def train(self, epochs):
        """Full training loop."""
        print(f"\n{'='*80}\nTraining {self.config['name']}\n{'='*80}\n")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}\n{'-'*80}")
            train_loss, train_dice = self.train_epoch()
            val_metrics = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['train_dice'].append(train_dice)
            for key in val_metrics:
                self.history[f'val_{key}'].append(val_metrics[key])

            print(f"\nTrain Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
            print(f"Val Jaccard: {val_metrics['jaccard']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val Sensitivity: {val_metrics['sensitivity']:.4f}, Val Specificity: {val_metrics['specificity']:.4f}")

            self.scheduler.step(val_metrics['dice'])

            if val_metrics['dice'] > self.best_dice + Config.MIN_DELTA:
                self.best_dice = val_metrics['dice']
                self.patience_counter = 0
                self.save_checkpoint('best')
                print(f"✓ New best model saved! Dice: {self.best_dice:.4f}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= Config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        print(f"\n{'='*80}\nTraining completed! Best Dice: {self.best_dice:.4f}\n{'='*80}\n")
        return self.history

    def save_checkpoint(self, name):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': self.best_dice,
            'history': self.history,
            'config': self.config
        }
        path = os.path.join(Config.MODELS_DIR, f"{self.config['name']}_{name}.pth")
        torch.save(checkpoint, path)


# =============================================================================
# TTA AND ENSEMBLE
# =============================================================================

def apply_tta_and_inverse(img_np: np.ndarray, model, device) -> np.ndarray:
    """Apply test-time augmentation."""
    model.eval()
    h, w = img_np.shape
    preds = []
    
    with torch.no_grad():
        for name, fwd, inv in Config.TTA_TRANSFORMS:
            aug = fwd(img_np)
            t = torch.from_numpy(aug).unsqueeze(0).unsqueeze(0).float().to(device)
            out = model(t)
            if isinstance(out, tuple):
                out = out[0]
            soft = out[0, 0].cpu().numpy()
            soft_inv = inv(soft)
            
            if soft_inv.shape != (h, w):
                soft_inv = cv2.resize(soft_inv, (w, h), interpolation=cv2.INTER_LINEAR)
            preds.append(soft_inv)
    
    avg = np.mean(preds, axis=0)
    return avg


def ensemble_predict(img_np: np.ndarray, checkpoint_paths: List[str], device) -> np.ndarray:
    """Ensemble predictions from multiple checkpoints with TTA."""
    all_softs = []
    
    for ckpt in checkpoint_paths:
        try:
            state = torch.load(ckpt, map_location=device)
            
            # Extract config
            if isinstance(state, dict) and 'config' in state:
                config = state['config']
            else:
                config = {'attention': True, 'residual': True, 'deep_supervision': False}
            
            # Create and load model
            net = PTXSegNet(config)
            sd = state.get('model_state_dict', state) if isinstance(state, dict) else state
            
            model_state = net.state_dict()
            pretrained = {k: v for k, v in sd.items() 
                         if k in model_state and model_state[k].shape == v.shape}
            model_state.update(pretrained)
            net.load_state_dict(model_state)
            
            net.to(device)
            net.eval()
            
            soft = apply_tta_and_inverse(img_np, net, device)
            all_softs.append(soft)
            
            del net
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Warning: failed to load {ckpt}: {e}. Skipping.")
            continue
    
    if not all_softs:
        raise RuntimeError("No predictions produced by ensemble.")
    
    ensemble_soft = np.mean(all_softs, axis=0)
    return ensemble_soft


# =============================================================================
# POST-PROCESSING
# =============================================================================

def postprocess_softmask(soft: np.ndarray, threshold: float, 
                        min_component_size: int) -> np.ndarray:
    """Post-process soft mask with hole filling and component removal."""
    bin_mask = (soft >= threshold).astype(np.uint8)

    # Fill holes
    bin_mask = ndi.binary_fill_holes(bin_mask).astype(np.uint8)

    # Remove small components
    labeled, num = ndi.label(bin_mask)
    sizes = ndi.sum(bin_mask, labeled, range(1, num+1))
    cleaned = np.zeros_like(bin_mask)
    for i, s in enumerate(sizes, start=1):
        if s >= min_component_size:
            cleaned[labeled == i] = 1

    # Morphological smoothing
    if cleaned.sum() > 0:
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    return cleaned.astype(np.uint8)


def grid_search_threshold_and_size(val_pairs: List[Tuple[str,str]], 
                                   checkpoint_paths: List[str], 
                                   device) -> Tuple[float,int]:
    """Grid search for optimal threshold and min component size."""
    print("Running grid search for threshold and min component size...")
    
    val_softs = []
    val_truths = []
    
    for img_p, mask_p in tqdm(val_pairs, desc="Ensembling val images"):
        img = read_image_gray(img_p, Config.IMG_SIZE)
        soft = ensemble_predict(img, checkpoint_paths, device)
        val_softs.append(soft)
        truth = read_mask(mask_p, Config.IMG_SIZE)
        val_truths.append(truth)

    best = (0.0, Config.OPTIMAL_THRESHOLD, Config.MIN_COMPONENT_SIZE)
    
    for b in Config.THRESHOLDS_TO_SEARCH:
        for c in Config.COMPONENT_SIZES_TO_SEARCH:
            dices = []
            for soft, truth in zip(val_softs, val_truths):
                binm = postprocess_softmask(soft, threshold=b, min_component_size=c)
                inter = (binm & truth).sum()
                union = binm.sum() + truth.sum()
                dice = (2.0 * inter / (union + 1e-8)) if union>0 else 1.0
                dices.append(dice)
            
            mean_d = float(np.mean(dices)) if dices else 0.0
            if mean_d > best[0]:
                best = (mean_d, float(b), int(c))
    
    print(f"Grid search done. Best Dice={best[0]:.4f} with threshold={best[1]} and min_comp_size={best[2]}")
    return best[1], best[2]
"""
=================================================================================
COMPLETE PNEUMOTHORAX SEGMENTATION SYSTEM (CORRECTED)
Part 6: Evaluation, Visualization, Severity Estimation, and Main
=================================================================================
"""

# =============================================================================
# SEVERITY ESTIMATION
# =============================================================================

def severity_estimation_from_mask(pred_mask: np.ndarray, img_path: str, 
                                 dicom_dir: Optional[str]=None) -> Dict:
    """Estimate pneumothorax severity from predicted mask."""
    info = {}
    Npx = int(pred_mask.sum())
    info['pneumothorax_pixels'] = Npx

    # Try to get pixel spacing from DICOM
    stem = Path(img_path).stem
    pixel_area_mm2 = None
    
    if dicom_dir and PYDICOM_AVAILABLE:
        try:
            dicom_path = os.path.join(dicom_dir, stem + ".dcm")
            if os.path.exists(dicom_path):
                ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                spacing = ds.get('PixelSpacing', None)
                if spacing:
                    sx, sy = float(spacing[0]), float(spacing[1])
                    pixel_area_mm2 = sx * sy
                    info['pixel_spacing_mm'] = (sx, sy)
        except Exception as e:
            info['dicom_note'] = f"DICOM read failed: {e}"

    if pixel_area_mm2:
        info['pneumothorax_area_mm2'] = float(Npx * pixel_area_mm2)
    else:
        info['pneumothorax_area_mm2'] = None

    # Estimate hemithorax area
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        info['severity_percent'] = None
        return info
    
    H, W = img.shape
    chest_mask = (img < 250).astype(np.uint8)
    x_coords = np.where(chest_mask.sum(axis=0) > 0)[0]
    y_coords = np.where(chest_mask.sum(axis=1) > 0)[0]
    
    if x_coords.size and y_coords.size:
        bw = x_coords[-1] - x_coords[0] + 1
        bh = y_coords[-1] - y_coords[0] + 1
        hemithorax_px = (bw * bh) / 2.0
    else:
        hemithorax_px = H * W / 2.0

    info['hemithorax_pixels_est'] = int(hemithorax_px)
    
    if pixel_area_mm2:
        info['hemithorax_area_mm2'] = float(hemithorax_px * pixel_area_mm2)

    if hemithorax_px > 0:
        severity_pct = 100.0 * Npx / hemithorax_px
        info['severity_percent'] = float(severity_pct)
        
        # Clinical classification
        if severity_pct >= 25.0:
            info['severity_label'] = "Large (>=25%)"
        elif severity_pct >= 20.0:
            info['severity_label'] = "Moderate (20-25%)"
        elif severity_pct >= 5.0:
            info['severity_label'] = "Small (5-20%)"
        else:
            info['severity_label'] = "Very small (<5%)"
    else:
        info['severity_percent'] = None
        info['severity_label'] = "Unknown"

    return info


# =============================================================================
# VISUALIZATION
# =============================================================================

def _blend_color_overlay(img_gray, color_rgb, mask_bool, alpha=0.5):
    """Blend color overlay on grayscale image."""
    H, W = img_gray.shape
    base = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32)
    overlay = base.copy()
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * np.array(color_rgb, dtype=np.float32)
    out = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return out


def show_topk_overlays_with_errors(image_paths: List[str],
                                   true_masks: List[np.ndarray],
                                   pred_masks: List[np.ndarray],
                                   scores: List[float],
                                   top_k: int = 5,
                                   save_dir: Optional[str] = None,
                                   figsize_per_image: Tuple[int,int]=(5,5),
                                   alpha_true: float = 0.4,
                                   alpha_pred: float = 0.4,
                                   alpha_error: float = 0.8):
    """Create visualization of top-k predictions with error overlay."""
    assert len(image_paths) == len(true_masks) == len(pred_masks) == len(scores)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Select cases with both prediction and ground truth
    selected = []
    for i, (img_p, gt, pred, sc) in enumerate(zip(image_paths, true_masks, pred_masks, scores)):
        if pred.sum() > 0 and gt.sum() > 0:
            selected.append((i, img_p, gt, pred, sc))
    
    if not selected:
        print("No cases where prediction and ground-truth both indicate pneumothorax.")
        return []

    selected = sorted(selected, key=lambda x: x[4], reverse=True)[:top_k]

    n = len(selected)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig_w = figsize_per_image[0] * cols
    fig_h = figsize_per_image[1] * rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else [axes]
    saved_paths = []

    for ax_idx, (idx, img_p, gt, pred, sc) in enumerate(selected):
        ax = axes[ax_idx]
        
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        H, W = img.shape
        
        if gt.shape != (H, W):
            gt_disp = cv2.resize(gt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            gt_disp = gt.astype(np.uint8)
        
        if pred.shape != (H, W):
            pred_disp = cv2.resize(pred.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            pred_disp = pred.astype(np.uint8)

        imgf = img.astype(np.float32) / 255.0
        missed = (gt_disp == 1) & (pred_disp == 0)

        # Blend true mask (blue)
        blended = _blend_color_overlay(imgf, color_rgb=(0.0, 0.0, 1.0), 
                                       mask_bool=(gt_disp==1), alpha=alpha_true).astype(np.float32)/255.0
        
        # Blend predicted mask (yellow)
        mask_pred_bool = (pred_disp==1)
        blended[mask_pred_bool] = (1 - alpha_pred) * blended[mask_pred_bool] + alpha_pred * np.array([1.0,1.0,0.0])
        
        # Blend missed pixels (red)
        mask_missed_bool = missed
        blended[mask_missed_bool] = (1 - alpha_error) * blended[mask_missed_bool] + alpha_error * np.array([1.0,0.0,0.0])

        out_img = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        ax.imshow(out_img)
        ax.axis('off')
        
        title = f"{Path(img_p).stem}\nDice={sc:.3f}  GT={int(gt_disp.sum())}px  Pred={int(pred_disp.sum())}px"
        ax.set_title(title, fontsize=9)

        if save_dir:
            out_file = os.path.join(save_dir, f"overlay_{Path(img_p).stem}.png")
            plt.imsave(out_file, out_img)
            saved_paths.append(out_file)

    for j in range(len(selected), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    if save_dir:
        grid_out = os.path.join(save_dir, f"overlay_grid_top{n}.png")
        fig.savefig(grid_out, dpi=150, bbox_inches='tight', pad_inches=0.1)
        saved_paths.append(grid_out)
        plt.close(fig)
        return saved_paths
    else:
        plt.show()
        return []


# =============================================================================
# EVALUATOR
# =============================================================================

class Evaluator:
    """Ensemble evaluator with TTA and post-processing."""
    
    def __init__(self, checkpoint_glob: str, test_pairs: List[Tuple[str,str]], 
                 val_pairs: List[Tuple[str,str]], config=None):
        self.checkpoint_paths = sorted(glob.glob(os.path.join(Config.MODELS_DIR, checkpoint_glob)))
        self.test_pairs = test_pairs
        self.val_pairs = val_pairs
        self.config = config
        self.metrics_calc = MetricsCalculator()

    def evaluate(self):
        """Run full evaluation with ensemble and TTA."""
        print(f"\n{'='*80}\nEnsemble evaluation (TTA + snapshots)\n{'='*80}\n")
        
        if not self.checkpoint_paths:
            raise RuntimeError("No checkpoints found for ensemble.")

        # Grid search on validation
        b_opt, c_opt = grid_search_threshold_and_size(self.val_pairs, self.checkpoint_paths, Config.DEVICE)

        # Run ensemble on test
        all_results = []
        predictions = []
        ground_truths = []
        dices = []

        for img_p, mask_p in tqdm(self.test_pairs, desc="Ensembling Test Images"):
            img = read_image_gray(img_p, Config.IMG_SIZE)
            truth = read_mask(mask_p, Config.IMG_SIZE)
            
            soft = ensemble_predict(img, self.checkpoint_paths, Config.DEVICE)
            binm = postprocess_softmask(soft, threshold=b_opt, min_component_size=c_opt)
            
            # Compute metrics
            inter = (binm & truth).sum()
            union = binm.sum() + truth.sum()
            dice = (2.0 * inter / (union + 1e-8)) if union>0 else 1.0
            jaccard = (inter / (union - inter + 1e-8)) if union>0 else 1.0
            accuracy = float((binm == truth).sum()) / truth.size
            
            TP = int(((binm == 1) & (truth == 1)).sum())
            TN = int(((binm == 0) & (truth == 0)).sum())
            FP = int(((binm == 1) & (truth == 0)).sum())
            FN = int(((binm == 0) & (truth == 1)).sum())
            sens = TP / (TP + FN + 1e-8)
            spec = TN / (TN + FP + 1e-8)

            severity = severity_estimation_from_mask(binm, img_p, dicom_dir=Config.DICOM_DIR)

            result = {
                'image': img_p,
                'mask': mask_p,
                'dice': float(dice),
                'jaccard': float(jaccard),
                'accuracy': float(accuracy),
                'sensitivity': float(sens),
                'specificity': float(spec),
                'threshold_used': float(b_opt),
                'min_component_size_used': int(c_opt),
                'severity': severity
            }
            all_results.append(result)
            predictions.append(binm)
            ground_truths.append(truth)
            dices.append(dice)

        metrics = {
            'dice_mean': float(np.mean(dices)),
            'dice_std': float(np.std(dices)),
            'count': len(dices)
        }
        print("\nEnsemble Test Results:", metrics)

        # Save results
        results_path = os.path.join(Config.RESULTS_DIR, "ensemble_test_results.json")
        with open(results_path, "w") as f:
            json.dump({'metrics': metrics, 'details': all_results}, f, indent=2)

        return metrics, all_results, predictions, ground_truths


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("PNEUMOTHORAX SEGMENTATION SYSTEM")
    print("="*80)

    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    random.seed(Config.RANDOM_SEED)

    Config.create_directories()
    print(f"Device: {Config.DEVICE}")
    print(f"Models to train: {Config.MODELS_TO_TRAIN}")

    # Load dataset
    train_pairs, val_pairs, test_pairs = load_dataset_new_structure()

    # Pretrain autoencoder if needed
    autoencoder_path = os.path.join(Config.MODELS_DIR, 'autoencoder_best.pth')
    if not os.path.exists(autoencoder_path):
        print("\nAutoencoder checkpoint not found. Pretraining...")
        train_loader_ae = DataLoader(
            PneumothoraxDataset(train_pairs, augment=True),
            batch_size=Config.BATCH_SIZE, shuffle=True,
            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
        )
        val_loader_ae = DataLoader(
            PneumothoraxDataset(val_pairs, augment=False),
            batch_size=Config.BATCH_SIZE, shuffle=False,
            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
        )
        pretrain_autoencoder(train_loader_ae, val_loader_ae)

    all_results = {}

    # Train models
    for model_id in Config.MODELS_TO_TRAIN:
        config = Config.get_model_config(model_id)
        
        train_loader = DataLoader(
            PneumothoraxDataset(train_pairs, augment=True),
            batch_size=config['batch_size'], shuffle=True,
            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
        )
        val_loader = DataLoader(
            PneumothoraxDataset(val_pairs, augment=False),
            batch_size=config['batch_size'], shuffle=False,
            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
        )

        model = PTXSegNet(config)

        # Load pretrained weights
        if config.get('pretrain') == 'autoencoder' and os.path.exists(autoencoder_path):
            auto_state = torch.load(autoencoder_path, map_location=Config.DEVICE)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in auto_state.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"✓ Loaded {len(pretrained_dict)} layers from autoencoder")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        trainer = Trainer(model, train_loader, val_loader, config)
        history = trainer.train(config['epochs'])
        
        # Plot history
        try:
            Visualizer.plot_training_history(history, config['name'])
        except Exception as e:
            print(f"Could not plot training history: {e}")

    # Ensemble evaluation
    evaluator = Evaluator(Config.ENSEMBLE_GLOB, test_pairs, val_pairs)
    metrics, details, predictions, ground_truths = evaluator.evaluate()
    all_results['ensemble'] = {'metrics': metrics, 'details_count': len(details)}

    # Visualize top predictions
    image_paths = [r['image'] for r in details]
    true_masks = [read_mask(r['mask'], Config.IMG_SIZE) for r in details]
    scores = [r.get('dice', 0.0) for r in details]

    saved_overlay_paths = show_topk_overlays_with_errors(
        image_paths, true_masks, predictions, scores,
        top_k=5, save_dir=Config.PRED_DIR,
        figsize_per_image=(5,5), alpha_true=0.4, alpha_pred=0.4, alpha_error=0.8
    )
    print("Saved overlay images:", saved_overlay_paths)

    # Save all results
    with open(os.path.join(Config.RESULTS_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Results saved to: {Config.RESULTS_DIR}")
    print(f"Predictions saved to: {Config.PRED_DIR}")
    print(f"Models saved to: {Config.MODELS_DIR}")


if __name__ == "__main__":
    main()