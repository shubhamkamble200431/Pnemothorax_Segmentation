import numpy as np
import torch


def dice_score(pred, target, eps=1e-6):
    p = pred.reshape(-1).float()
    t = target.reshape(-1).float()
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    if denom == 0:
        return 1.0
    return ((2 * inter + eps) / (denom + eps)).item()


def iou_score(pred, target, eps=1e-6):
    p = pred.reshape(-1).float()
    t = target.reshape(-1).float()
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    if union == 0:
        return 1.0
    return ((inter + eps) / (union + eps)).item()


def pixel_accuracy(pred, target):
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total