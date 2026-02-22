import numpy as np
import torch
import cv2


TTA_KEYS = ["none", "hflip"]


def apply_tta(img, key):
    if key == "hflip":
        return torch.flip(img, dims=[2])
    return img


def inverse_tta(pred, key):
    if key == "hflip":
        return np.flip(pred, axis=1)
    return pred


def predict_tta(model, img, device):
    h, w = img.shape[1:]
    preds = []

    with torch.no_grad():
        for k in TTA_KEYS:
            x = apply_tta(img, k).unsqueeze(0).to(device)
            out = model(x)

            if isinstance(out, tuple):
                out = out[0]

            prob = torch.sigmoid(out)[0, 0].cpu().numpy()
            prob = inverse_tta(prob, k)

            if prob.shape != (h, w):
                prob = cv2.resize(prob, (w, h))

            preds.append(prob)

    return np.mean(preds, axis=0)