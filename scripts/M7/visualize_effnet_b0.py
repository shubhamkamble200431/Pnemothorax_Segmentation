# visualise_comparison.py
# 3-row comparison visualisation for best/medium/worst dice scores:
#   Row 1: GT overlay (blue)
#   Row 2: Predicted overlay (yellow)
#   Row 3: Error overlay (red)

import os
import glob
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import matplotlib.pyplot as plt


# ----------------- CONFIG -----------------

class Config:
    dataset_root = "/home/ml/Desktop/shubham/FYP_2.0/final_dataset"
    ckpt_path = "checkpoints/unet_effnet_b4.pth"  # trained model
    model_name = "unet_resnet34"                      # backbone choice
    img_size = 1024                                   # same as training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold = 0.5                                    # binary threshold
    batch_size = 8                                     # for computing dice scores
    output_dir = "b4_outputs"                     # output directory

cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)


# ----------------- DATASET -----------------

class PneumothoraxDataset(Dataset):
    """
    dataset_root/
      train/
        png_images/train_image_1_0.png
        png_masks/train_mask_1_0.png
      test/
        png_images/test_image_1_1.png
        png_masks/test_mask_1_1.png

    _x_1.png → pneumothorax present
    _x_0.png → no pneumothorax (mask all black)
    """

    def __init__(self, root_dir: str, split: str = "test", transforms=None):
        super().__init__()
        assert split in ["train", "test"]
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms

        img_dir = os.path.join(root_dir, split, "png_images")
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No PNG images found in {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def _get_mask_path(self, img_path: str) -> str:
        mask_path = img_path.replace(
            os.path.join(self.split, "png_images"),
            os.path.join(self.split, "png_masks")
        )
        mask_path = mask_path.replace("image_", "mask_")
        return mask_path

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self._get_mask_path(img_path)

        # -------- IMAGE (robust loading for model input) --------
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        # If 3-channel, convert to grayscale
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # If not uint8 (e.g. 16-bit), normalize to 0–255 uint8
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # -------- MASK --------
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        mask = (mask > 0).astype(np.float32)  # binary

        # Store original image for plotting (before normalization)
        img_for_plot = cv2.resize(image, (cfg.img_size, cfg.img_size))
        img_for_plot = cv2.cvtColor(img_for_plot, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

        # Normalize image to [0,1] for Albumentations
        image = image.astype(np.float32) / 255.0

        # Add channel dim for Albumentations: (H,W,1)
        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]    # tensor (1,H,W)
            mask = augmented["mask"]      # tensor or np

            # Ensure mask is (1,H,W) tensor
            if isinstance(mask, torch.Tensor):
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.ndim == 3 and mask.shape[0] != 1:
                    mask = mask.permute(2, 0, 1)
                mask = mask.float()
            else:
                mask = np.squeeze(mask)
                if mask.ndim == 2:
                    mask = mask[None, ...]
                elif mask.ndim == 3 and mask.shape[0] != 1:
                    mask = np.transpose(mask, (2, 0, 1))
                mask = torch.from_numpy(mask.astype(np.float32))
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()

        return image, mask, img_for_plot, img_path


def get_test_transforms():
    return A.Compose(
        [
            A.Resize(cfg.img_size, cfg.img_size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )


# ----------------- MODEL -----------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNetEncoderDecoder(nn.Module):
    def __init__(self, backbone_name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(
            backbone_name,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            pretrained=pretrained,
            in_chans=1,
        )
        enc_channels = self.encoder.feature_info.channels()

        self.center = ConvBlock(enc_channels[-1], 512)
        self.dec4 = UpBlock(512, enc_channels[-2], 256)
        self.dec3 = UpBlock(256, enc_channels[-3], 128)
        self.dec2 = UpBlock(128, enc_channels[-4], 64)
        self.dec1 = UpBlock(64, enc_channels[-5], 32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        center = self.center(x4)
        d4 = self.dec4(center, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        out = self.final_conv(d1)
        return out


def create_model(model_name: str, pretrained: bool = False) -> nn.Module:
    if model_name == "unet_resnet34":
        backbone = "resnet34"
    elif model_name == "unet_resnet50":
        backbone = "resnet50"
    elif model_name == "unet_effnet_b0":
        backbone = "tf_efficientnet_b0_ns"
    elif model_name == "unet_effnet_b4":
        backbone = "tf_efficientnet_b4_ns"
    elif model_name == "unet_effnet_b7":
        backbone = "tf_efficientnet_b7_ns"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return UNetEncoderDecoder(backbone_name=backbone, pretrained=pretrained)


# ----------------- METRICS -----------------

def dice_score_tensor(pred, target, smooth=1e-6):
    """Calculate dice score between prediction and target."""
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return (2. * inter + smooth) / (pred.sum() + target.sum() + smooth)


# ----------------- COMPUTE DICE PER IMAGE -----------------

def compute_dice_per_image(model: nn.Module, dataset: PneumothoraxDataset) -> Tuple[List[str], List[float], Dict[str, np.ndarray]]:
    """
    Compute dice score for each image in the dataset.
    Returns:
        file_list: list of image paths
        dice_list: list of dice scores
        preds_map: dict mapping image path to predicted mask
    """
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    dice_list = []
    file_list = []
    preds_map = {}
    
    model.eval()
    print("Computing dice scores for all images...")
    
    with torch.no_grad():
        for images, masks, imgs_for_plot, img_paths in tqdm(loader):
            images = images.to(cfg.device)
            masks = masks.to(cfg.device)
            
            logits = model(images)
            if logits.shape[2:] != masks.shape[2:]:
                logits = F.interpolate(
                    logits,
                    size=masks.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
            
            probs = torch.sigmoid(logits)
            preds = (probs > cfg.threshold).float()
            
            for i in range(preds.shape[0]):
                pred_np = preds[i, 0].cpu().numpy().astype(np.uint8)
                mask_np = masks[i, 0].cpu().numpy().astype(np.uint8)
                
                dice = dice_score_tensor(
                    torch.from_numpy(pred_np),
                    torch.from_numpy(mask_np)
                ).item()
                
                img_path = img_paths[i]
                dice_list.append(dice)
                file_list.append(img_path)
                preds_map[img_path] = pred_np
    
    return file_list, dice_list, preds_map


# ----------------- SELECT SAMPLES -----------------

def pick_samples_for_ranges(file_list: List[str], dice_list: List[float], dataset: PneumothoraxDataset) -> Tuple[int, int, int]:
    """
    Select best, median, and worst samples from positive mask images.
    Returns indices for best, median, worst.
    """
    # Find positive mask images
    positive_items = []
    for i, img_path in enumerate(file_list):
        mask_path = dataset._get_mask_path(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and np.any(mask > 0):
            positive_items.append((i, dice_list[i]))
    
    if len(positive_items) == 0:
        raise RuntimeError("No positive-mask images found in dataset.")
    
    # Sort by dice score (descending)
    positive_items = sorted(positive_items, key=lambda x: x[1], reverse=True)
    
    best_idx = positive_items[0][0]
    worst_idx = positive_items[-1][0]
    med_idx = positive_items[len(positive_items) // 2][0]
    
    return best_idx, med_idx, worst_idx


# ----------------- OVERLAY HELPERS -----------------

def overlay_image_with_mask(img: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float], alpha: float = 0.6) -> np.ndarray:
    """
    Overlay colored mask on image.
    img: HxWx3 numpy in 0..1
    mask: HxW numpy binary {0,1}
    color: RGB tuple in 0..1
    """
    img = img.copy()
    colored_mask = np.stack([mask * color[0], mask * color[1], mask * color[2]], axis=-1)
    out = img * (1 - alpha * mask[..., None]) + colored_mask * (alpha * mask[..., None])
    out = np.clip(out, 0, 1)
    return out


# ----------------- VISUALIZATION -----------------

def visualize_comparison(model: nn.Module, dataset: PneumothoraxDataset, file_list: List[str], 
                        dice_list: List[float], preds_map: Dict[str, np.ndarray]):
    """
    Create comparison plots for best/medium/worst dice scores.
    """
    # Pick samples
    best_idx, med_idx, worst_idx = pick_samples_for_ranges(file_list, dice_list, dataset)
    
    ranges = [
        (best_idx, "High (Best)"),
        (med_idx, "Medium (Median)"),
        (worst_idx, "Poor (Worst)")
    ]
    
    color_gt = (0.0, 0.0, 1.0)      # blue
    color_pred = (1.0, 1.0, 0.0)    # yellow
    color_error = (1.0, 0.0, 0.0)   # red
    
    for idx, range_title in ranges:
        img_path = file_list[idx]
        dice = dice_list[idx]
        
        # Get data
        _, mask_t, img_for_plot, _ = dataset[idx]
        mask_np = mask_t.squeeze(0).numpy().astype(np.uint8)
        pred_np = preds_map[img_path]
        
        # Compute error (XOR)
        error_np = ((pred_np.astype(np.uint8) ^ mask_np.astype(np.uint8)) > 0).astype(np.float32)
        
        # Create overlays
        gt_overlay = overlay_image_with_mask(img_for_plot, mask_np, color=color_gt, alpha=0.6)
        pred_overlay = overlay_image_with_mask(img_for_plot, pred_np, color=color_pred, alpha=0.6)
        error_overlay = overlay_image_with_mask(img_for_plot, error_np, color=color_error, alpha=0.7)
        
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(6, 16))
        
        axes[0].imshow(gt_overlay)
        axes[0].set_title(f"GT Overlay (Blue)\nDice Score: {dice:.4f}", fontsize=14)
        axes[0].axis("off")
        
        axes[1].imshow(pred_overlay)
        axes[1].set_title("Predicted Overlay (Yellow)", fontsize=14)
        axes[1].axis("off")
        
        axes[2].imshow(error_overlay)
        axes[2].set_title("Error Overlay (Red - XOR)", fontsize=14)
        axes[2].axis("off")
        
        fig.suptitle(f"Range: {range_title} - {cfg.model_name}", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save
        range_key = range_title.split()[0].lower()
        out_path = os.path.join(cfg.output_dir, f"comparison_{range_key}.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close(fig)


# ----------------- MAIN -----------------

def main():
    # Create dataset
    transforms = get_test_transforms()
    dataset = PneumothoraxDataset(cfg.dataset_root, split="test", transforms=transforms)
    print(f"Found {len(dataset)} test images")
    
    # Load model
    model = create_model(cfg.model_name, pretrained=False).to(cfg.device)
    
    if not os.path.isfile(cfg.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt_path}")
    
    state_dict = torch.load(cfg.ckpt_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    print(f"[INFO] Loaded weights from {cfg.ckpt_path}")
    
    # Compute dice scores for all images
    file_list, dice_list, preds_map = compute_dice_per_image(model, dataset)
    
    print(f"\nDice Score Statistics:")
    print(f"  Mean: {np.mean(dice_list):.4f}")
    print(f"  Std:  {np.std(dice_list):.4f}")
    print(f"  Min:  {np.min(dice_list):.4f}")
    print(f"  Max:  {np.max(dice_list):.4f}")
    
    # Create visualizations
    visualize_comparison(model, dataset, file_list, dice_list, preds_map)
    
    print(f"\nDone! Check overlay images in: {cfg.output_dir}")


if __name__ == "__main__":
    main()