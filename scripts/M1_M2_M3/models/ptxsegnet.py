import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
        )
        self.skip = nn.Identity() if in_c == out_c else nn.Sequential(
            nn.Conv2d(in_c, out_c, 1),
            nn.BatchNorm2d(out_c)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class AttentionGate(nn.Module):
    def __init__(self, Fg, Fl, Fint):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(Fg, Fint, 1), nn.BatchNorm2d(Fint))
        self.Wx = nn.Sequential(nn.Conv2d(Fl, Fint, 1), nn.BatchNorm2d(Fint))
        self.psi = nn.Sequential(nn.Conv2d(Fint, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.Wg(g) + self.Wx(x)))


class PTXSegNet(nn.Module):
    def __init__(self, in_ch=1, base=[64,128,256,512,1024], attention=True, residual=True, deep_supervision=True):
        super().__init__()
        Block = ResidualBlock if residual else ConvBlock
        c = base

        self.enc1 = Block(in_ch, c[0])
        self.enc2 = Block(c[0], c[1])
        self.enc3 = Block(c[1], c[2])
        self.enc4 = Block(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = Block(c[3], c[4])

        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, 2)
        self.dec4 = Block(c[4], c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.dec3 = Block(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.dec2 = Block(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.dec1 = Block(c[1], c[0])

        self.att = attention
        if attention:
            self.att4 = AttentionGate(c[3], c[3], c[3]//2)
            self.att3 = AttentionGate(c[2], c[2], c[2]//2)
            self.att2 = AttentionGate(c[1], c[1], c[1]//2)
            self.att1 = AttentionGate(c[0], c[0], c[0]//2)

        self.out = nn.Conv2d(c[0], 1, 1)
        self.ds = deep_supervision
        if deep_supervision:
            self.dsv4 = nn.Conv2d(c[3], 1, 1)
            self.dsv3 = nn.Conv2d(c[2], 1, 1)
            self.dsv2 = nn.Conv2d(c[1], 1, 1)
            self.dsv1 = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        e4 = self.att4(d4, e4) if self.att else e4
        d4 = self.dec4(torch.cat([d4, e4], 1))

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3) if self.att else e3
        d3 = self.dec3(torch.cat([d3, e3], 1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2) if self.att else e2
        d2 = self.dec2(torch.cat([d2, e2], 1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1) if self.att else e1
        d1 = self.dec1(torch.cat([d1, e1], 1))

        out = torch.sigmoid(self.out(d1))

        if self.ds and self.training:
            dsv = [
                F.interpolate(torch.sigmoid(self.dsv4(d4)), out.shape[2:]),
                F.interpolate(torch.sigmoid(self.dsv3(d3)), out.shape[2:]),
                F.interpolate(torch.sigmoid(self.dsv2(d2)), out.shape[2:]),
                torch.sigmoid(self.dsv1(d1))
            ]
            return out, dsv

        return out