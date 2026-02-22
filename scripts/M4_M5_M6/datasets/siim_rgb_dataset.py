import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class SIIMRGBDataset(Dataset):
    def __init__(self, root, split="train", img_size=256):
        self.img_size = img_size

        img_dir = os.path.join(root, "png_images")
        mask_dir = os.path.join(root, "png_masks")

        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

        assert len(self.img_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        img = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask