import os, glob, random, json, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class Config:
    DATA_ROOT = "./dataset"
    TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train", "png_images")
    TRAIN_MASKS_DIR = os.path.join(DATA_ROOT, "train", "png_masks")
    TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "test", "png_images")
    TEST_MASKS_DIR = os.path.join(DATA_ROOT, "test", "png_masks")

    OUTPUT_DIR = "./outputs"
    MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    PRED_DIR = os.path.join(RESULTS_DIR, "predictions")

    MODELS_TO_TRAIN = [1]

    IMG_SIZE = 512
    IMG_CHANNELS = 1
    ENCODER_CHANNELS = [64,128,256,512,1024]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RANDOM_SEED = 42
    TRAIN_SPLIT = 0.85

    OPTIMAL_THRESHOLD = 0.5
    MIN_COMPONENT_SIZE = 100

    @classmethod
    def create_dirs(cls):
        for d in [cls.MODELS_DIR, cls.RESULTS_DIR, cls.PRED_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def get_model_config(cls, i):
        C = {
            1: dict(name="baseline_unet",attention=False,residual=False,deep_supervision=False,loss="focal",lr=1e-5,epochs=50,batch_size=8),
            2: dict(name="unet_imagenet",attention=False,residual=False,deep_supervision=False,loss="focal",lr=1e-5,epochs=50,batch_size=8),
            3: dict(name="unet_autoencoder",attention=False,residual=False,deep_supervision=False,loss="focal",lr=1e-5,epochs=50,batch_size=8),
            4: dict(name="attention_unet",attention=True,residual=False,deep_supervision=False,loss="focal",lr=1e-5,epochs=50,batch_size=8),
            5: dict(name="attention_residual_unet",attention=True,residual=True,deep_supervision=False,loss="focal",lr=1e-5,epochs=50,batch_size=8),
            6: dict(name="ptxseg_net",attention=True,residual=True,deep_supervision=True,loss="focal",lr=1e-5,epochs=50,batch_size=8),
            7: dict(name="ptxseg_net_high_lr",attention=True,residual=True,deep_supervision=True,loss="focal",lr=1e-4,epochs=15,batch_size=4),
            8: dict(name="ptxseg_net_large_batch",attention=True,residual=True,deep_supervision=True,loss="focal",lr=1e-5,epochs=50,batch_size=16),
            9: dict(name="ptxseg_net_dice",attention=True,residual=True,deep_supervision=True,loss="dice",lr=1e-4,epochs=25,batch_size=4),
            10: dict(name="ptxseg_net_combined",attention=True,residual=True,deep_supervision=True,loss="combined",lr=1e-5,epochs=15,batch_size=8),
            11: dict(name="ptxseg_net_extended",attention=True,residual=True,deep_supervision=True,loss="focal",lr=1e-5,epochs=100,batch_size=8),
            12: dict(name="ptxseg_net_deep",attention=True,residual=True,deep_supervision=True,loss="dice",lr=1e-4,epochs=10,batch_size=4,deeper=True)
        }
        return C[i]


def pair_by_stem(img_dir, mask_dir):
    imgs = sorted(glob.glob(os.path.join(img_dir,"*.png")))
    pairs=[]
    for p in imgs:
        m = p.replace("image","mask")
        if os.path.exists(m):
            pairs.append((p,m))
    return pairs


def read_img(p,s): 
    return cv2.resize(cv2.imread(p,0),(s,s)).astype(np.float32)/255.

def read_mask(p,s):
    return (cv2.resize(cv2.imread(p,0),(s,s),cv2.INTER_NEAREST)>127).astype(np.uint8)


class PneumoDataset(Dataset):
    def __init__(self,pairs): self.pairs=pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        ip,mp=self.pairs[i]
        x=read_img(ip,Config.IMG_SIZE)
        y=read_mask(mp,Config.IMG_SIZE)
        return torch.from_numpy(x)[None].float(),torch.from_numpy(y)[None].float()


class ConvBlock(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.b=nn.Sequential(
            nn.Conv2d(c1,c2,3,1,1),nn.BatchNorm2d(c2),nn.LeakyReLU(0.2,True),
            nn.Conv2d(c2,c2,3,1,1),nn.BatchNorm2d(c2),nn.LeakyReLU(0.2,True)
        )
    def forward(self,x): return self.b(x)


class ResidualBlock(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        self.c1=nn.Conv2d(c1,c2,3,1,1)
        self.c2=nn.Conv2d(c2,c2,3,1,1)
        self.b1=nn.BatchNorm2d(c2); self.b2=nn.BatchNorm2d(c2)
        self.r = nn.Conv2d(c1,c2,1) if c1!=c2 else nn.Identity()
    def forward(self,x):
        y=self.b1(self.c1(x))
        y=self.b2(self.c2(y))
        return F.leaky_relu(y+self.r(x),0.2)


class AttentionGate(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.g=nn.Conv2d(c,c//2,1)
        self.x=nn.Conv2d(c,c//2,1)
        self.p=nn.Sequential(nn.Conv2d(c//2,1,1),nn.Sigmoid())
    def forward(self,g,x):
        a=self.p(F.relu(self.g(g)+self.x(x)))
        return x*a


class PTXSegNet(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        ch=Config.ENCODER_CHANNELS.copy()
        if cfg.get("deeper"): ch.append(2048)
        B=ResidualBlock if cfg["residual"] else ConvBlock

        self.e1=B(1,ch[0]); self.e2=B(ch[0],ch[1])
        self.e3=B(ch[1],ch[2]); self.e4=B(ch[2],ch[3])
        self.pool=nn.MaxPool2d(2)
        self.b=B(ch[3],ch[4])

        self.u4=nn.ConvTranspose2d(ch[4],ch[3],2,2); self.d4=B(ch[4],ch[3])
        self.u3=nn.ConvTranspose2d(ch[3],ch[2],2,2); self.d3=B(ch[3],ch[2])
        self.u2=nn.ConvTranspose2d(ch[2],ch[1],2,2); self.d2=B(ch[2],ch[1])
        self.u1=nn.ConvTranspose2d(ch[1],ch[0],2,2); self.d1=B(ch[1],ch[0])

        self.att = cfg["attention"]
        if self.att:
            self.a4,self.a3,self.a2,self.a1 = AttentionGate(ch[3]),AttentionGate(ch[2]),AttentionGate(ch[1]),AttentionGate(ch[0])

        self.out = nn.Conv2d(ch[0],1,1)
        self.ds = cfg["deep_supervision"]

    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1))
        e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
        b=self.b(self.pool(e4))

        d4=self.u4(b);  e4=self.a4(d4,e4) if self.att else e4
        d4=self.d4(torch.cat([d4,e4],1))

        d3=self.u3(d4); e3=self.a3(d3,e3) if self.att else e3
        d3=self.d3(torch.cat([d3,e3],1))

        d2=self.u2(d3); e2=self.a2(d2,e2) if self.att else e2
        d2=self.d2(torch.cat([d2,e2],1))

        d1=self.u1(d2); e1=self.a1(d1,e1) if self.att else e1
        d1=self.d1(torch.cat([d1,e1],1))

        out=torch.sigmoid(self.out(d1))
        return out


class DiceLoss(nn.Module):
    def forward(self,p,t):
        p=p.view(-1); t=t.view(-1)
        inter=(p*t).sum()
        return 1-(2*inter+1)/(p.sum()+t.sum()+1)


def train_model(cfg,train_pairs,val_pairs):
    net=PTXSegNet(cfg).to(Config.DEVICE)
    opt=torch.optim.Adam(net.parameters(),lr=cfg["lr"])
    loss_fn=DiceLoss()

    tl=DataLoader(PneumoDataset(train_pairs),batch_size=cfg["batch_size"],shuffle=True)
    vl=DataLoader(PneumoDataset(val_pairs),batch_size=cfg["batch_size"])

    best=0
    for _ in range(cfg["epochs"]):
        net.train()
        for x,y in tl:
            x,y=x.to(Config.DEVICE),y.to(Config.DEVICE)
            p=net(x); l=loss_fn(p,y)
            opt.zero_grad(); l.backward(); opt.step()

        net.eval(); dice=0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(Config.DEVICE),y.to(Config.DEVICE)
                p=(net(x)>0.5).float()
                dice+=(2*(p*y).sum()/(p.sum()+y.sum()+1)).item()
        dice/=len(vl)

        if dice>best:
            best=dice
            torch.save(net.state_dict(),os.path.join(Config.MODELS_DIR,cfg["name"]+"_best.pth"))


def main():
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)
    Config.create_dirs()

    tr=pair_by_stem(Config.TRAIN_IMAGES_DIR,Config.TRAIN_MASKS_DIR)
    te=pair_by_stem(Config.TEST_IMAGES_DIR,Config.TEST_MASKS_DIR)

    random.shuffle(tr)
    n=int(len(tr)*Config.TRAIN_SPLIT)
    train_pairs,val_pairs=tr[:n],tr[n:]

    for i in Config.MODELS_TO_TRAIN:
        cfg=Config.get_model_config(i)
        train_model(cfg,train_pairs,val_pairs)


if __name__=="__main__":
    main()