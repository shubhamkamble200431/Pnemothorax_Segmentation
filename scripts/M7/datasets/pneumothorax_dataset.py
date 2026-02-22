import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PneumothoraxDataset(Dataset):

    def __init__(self, root_dir, split="train", transforms=None):

        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms

        img_dir = os.path.join(root_dir, split, "png_images")
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    def __len__(self):
        return len(self.img_paths)

    def _get_mask_path(self, img_path):

        mask_path = img_path.replace(
            os.path.join(self.split, "png_images"),
            os.path.join(self.split, "png_masks")
        )
        return mask_path.replace("image_", "mask_")

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        mask_path = self._get_mask_path(img_path)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask