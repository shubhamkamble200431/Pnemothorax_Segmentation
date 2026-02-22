import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import os


class ConvBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.b(x)


class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in, 2, 2)
        self.conv = ConvBlock(c_in + c_skip, c_out)
    def forward(self, x, s):
        x = self.up(x)
        dy, dx = s.size(2) - x.size(2), s.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([s, x], 1))


class UNetEncoderDecoder(nn.Module):
    def __init__(self, backbone="tf_efficientnet_b4_ns"):
        super().__init__()
        self.enc = timm.create_model(backbone, features_only=True,
                                     out_indices=(0,1,2,3,4),
                                     pretrained=False, in_chans=1)
        ch = self.enc.feature_info.channels()
        self.center = ConvBlock(ch[-1], 512)
        self.d4 = UpBlock(512, ch[-2], 256)
        self.d3 = UpBlock(256, ch[-3], 128)
        self.d2 = UpBlock(128, ch[-4], 64)
        self.d1 = UpBlock(64,  ch[-5], 32)
        self.out = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        x0,x1,x2,x3,x4 = self.enc(x)
        x = self.center(x4)
        x = self.d4(x, x3)
        x = self.d3(x, x2)
        x = self.d2(x, x1)
        x = self.d1(x, x0)
        return self.out(x)


def lung_mask(img):
    img_eq = cv2.equalizeHist(img)
    _, th = cv2.threshold(img_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (img_eq < th).astype(np.uint8)
    k = np.ones((15,15), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)


def severity(pt, lung):
    lp = lung.sum()
    return None if lp == 0 else (pt.sum() / lp) * 100.0


def predict(image_path, ckpt_path, size=1024, thr=0.5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    H, W = img.shape

    x = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
    x = torch.from_numpy(x)[None, None]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetEncoderDecoder().to(device)

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict):
        sd = sd.get("model_state_dict", sd.get("state_dict", sd))
    sd = {k.replace("module.",""): v for k,v in sd.items() if k.replace("module.","") in model.state_dict()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    with torch.no_grad():
        p = torch.sigmoid(model(x.to(device))).squeeze().cpu().numpy()

    mask = (p > thr).astype(np.uint8)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    lung = lung_mask(img)
    sev = severity(mask, lung)

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[...,2] = np.maximum(overlay[...,2], mask * 255)

    out = "ptx_overlay_result.png"
    cv2.imwrite(out, overlay)

    return sev, out


if __name__ == "__main__":
    img_path = input().strip()
    ckpt = "checkpoints/unet_effnet_b4_1024_focal.pth"
    if os.path.isfile(img_path) and os.path.isfile(ckpt):
        predict(img_path, ckpt)