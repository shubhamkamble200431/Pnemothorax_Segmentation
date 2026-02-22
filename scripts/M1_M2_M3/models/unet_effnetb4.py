import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4


class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c + skip_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], 1)
        return self.conv(x)


class EfficientNetB4_UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        enc = efficientnet_b4(weights="IMAGENET1K_V1" if pretrained else None).features

        self.e0 = enc[0]
        self.e1 = enc[1:3]
        self.e2 = enc[3:5]
        self.e3 = enc[5:7]
        self.e4 = enc[7:]

        self.u4 = UpBlock(1792, 160, 512)
        self.u3 = UpBlock(512, 112, 256)
        self.u2 = UpBlock(256, 56, 128)
        self.u1 = UpBlock(128, 32, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x0 = self.e0(x)
        x1 = self.e1(x0)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)

        d4 = self.u4(x4, x3)
        d3 = self.u3(d4, x2)
        d2 = self.u2(d3, x1)
        d1 = self.u1(d2, x0)

        return torch.sigmoid(self.out(d1))