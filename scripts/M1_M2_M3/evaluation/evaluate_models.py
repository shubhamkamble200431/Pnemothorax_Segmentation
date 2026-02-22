import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage as ndi
import timm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5
MIN_COMPONENT_AREA = 100
TTA_KEYS = ["none", "hflip"]


def dice(pred, gt, eps=1e-6):
    p = pred.astype(np.float32).flatten()
    g = gt.astype(np.float32).flatten()
    inter = (p * g).sum()
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else float((2 * inter + eps) / (denom + eps))


def postprocess(mask):
    m = (mask >= THRESHOLD).astype(np.uint8)
    m = ndi.binary_fill_holes(m).astype(np.uint8)

    if MIN_COMPONENT_AREA > 0:
        lab, n = ndi.label(m)
        sizes = ndi.sum(m, lab, range(1, n + 1))
        clean = np.zeros_like(m)
        for i, s in enumerate(sizes, 1):
            if s >= MIN_COMPONENT_AREA:
                clean[lab == i] = 1
        m = clean

    if m.sum() > 0:
        k = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    return m.astype(np.uint8)


class SIIMDataset(Dataset):
    def __init__(self, img_files: List[str], mask_files: List[str], size=512):
        self.img_files = img_files
        self.mask_files = mask_files
        self.size = size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, i):
        img = cv2.imread(self.img_files[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_files[i], cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.size, self.size)).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

        return torch.from_numpy(img).unsqueeze(0), mask, self.img_files[i]


def predict_tta(model, img):
    h, w = img.shape[1:]
    preds = []

    with torch.no_grad():
        for k in TTA_KEYS:
            x = img.clone()
            if k == "hflip":
                x = torch.flip(x, dims=[2])

            out = model(x.unsqueeze(0).to(DEVICE))
            if isinstance(out, tuple):
                out = out[0]

            p = torch.sigmoid(out)[0, 0].cpu().numpy()

            if k == "hflip":
                p = np.flip(p, axis=1)

            if p.shape != (h, w):
                p = cv2.resize(p, (w, h))

            preds.append(p)

    return np.mean(preds, axis=0)


def load_checkpoint(model: nn.Module, path: str):
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict):
        state = state.get("model_state_dict", state)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model


def evaluate(model, dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []

    for img, mask, path in loader:
        img = img[0]
        gt = mask[0].numpy()

        soft = predict_tta(model, img)
        pred = postprocess(soft)

        results.append({
            "image": path[0],
            "dice": dice(pred, gt)
        })

    return results