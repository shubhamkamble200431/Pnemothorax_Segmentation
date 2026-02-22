import torch
import torch.nn as nn
from torchvision.models import resnet50


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


class ResNet50_UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        enc = resnet50(weights="IMAGENET1K_V1" if pretrained else None)

        self.input = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.e1 = enc.layer1
        self.e2 = enc.layer2
        self.e3 = enc.layer3
        self.e4 = enc.layer4

        self.u4 = UpBlock(2048, 1024, 512)
        self.u3 = UpBlock(512, 512, 256)
        self.u2 = UpBlock(256, 256, 128)
        self.u1 = UpBlock(128, 64, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x0 = self.input(x)
        x1 = self.e1(self.pool(x0))
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)

        d4 = self.u4(x4, x3)
        d3 = self.u3(d4, x2)
        d2 = self.u2(d3, x1)
        d1 = self.u1(d2, x0)

        return torch.sigmoid(self.out(d1))