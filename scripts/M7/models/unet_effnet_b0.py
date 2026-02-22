import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetEffNetB0(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.encoder = timm.create_model(
            "tf_efficientnet_b0_ns",
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            pretrained=pretrained,
            in_chans=1,
        )

        enc_ch = self.encoder.feature_info.channels()

        self.center = ConvBlock(enc_ch[-1], 512)
        self.dec4 = UpBlock(512, enc_ch[-2], 256)
        self.dec3 = UpBlock(256, enc_ch[-3], 128)
        self.dec2 = UpBlock(128, enc_ch[-4], 64)
        self.dec1 = UpBlock(64, enc_ch[-5], 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)

        center = self.center(x4)
        d4 = self.dec4(center, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        return self.final(d1)