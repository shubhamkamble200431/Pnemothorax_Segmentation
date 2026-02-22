import numpy as np


def overlay(img, mask, color=(1, 1, 0), alpha=0.6):
    img = img.astype(np.float32)
    mask = mask.astype(np.float32)

    if img.max() > 1:
        img = img / 255.0

    colored = np.stack([mask * c for c in color], axis=-1)
    out = img * (1 - alpha * mask[..., None]) + colored * alpha
    return np.clip(out, 0, 1)


def error_map(pred, gt):
    return (pred ^ gt).astype(np.float32)