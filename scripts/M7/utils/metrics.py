import torch


def dice_score(pred, target, eps=1e-6):

    pred = pred.view(-1)
    target = target.view(-1)

    inter = (pred * target).sum()

    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)