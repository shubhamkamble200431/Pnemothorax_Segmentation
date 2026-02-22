import os
import glob
from typing import List, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class SIIMDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 512):
        assert split in ["train", "test"]
        self.root = root
        self.split = split
        self.img_size = img_size

        img_dir = os.path.join(root, split, "png_images")
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))

        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def _mask_path(self, img_path: str) -> str:
        mask_path = img_path.replace(
            os.path.join(self.split, "png_images"),
            os.path.join(self.split, "png_masks")
        )
        mask_path = mask_path.replace("image_", "mask_")
        return mask_path

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self._mask_path(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size, self.img_size)).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask, img_path