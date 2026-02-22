import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, eps=1e-6):
        p = pred.view(-1)
        t = target.view(-1)

        inter = (p * t).sum()
        denom = p.sum() + t.sum()

        dice = (2 * inter + eps) / (denom + eps)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)