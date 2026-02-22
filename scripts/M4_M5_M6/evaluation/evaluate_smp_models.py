import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets.siim_rgb_dataset import SIIMRGBDataset
from models.smp_models import MODEL_REGISTRY
from utils.metrics import dice_score, iou_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader):
    model.eval()

    dices, ious = [], []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = (torch.sigmoid(model(imgs)) > 0.5).float()

            for i in range(len(preds)):
                dices.append(dice_score(preds[i], masks[i]))
                ious.append(iou_score(preds[i], masks[i]))

    return np.mean(dices), np.mean(ious)


def main():
    dataset = SIIMRGBDataset("dataset")
    loader = DataLoader(dataset, batch_size=8)

    for name, builder in MODEL_REGISTRY.items():
        model = builder().to(DEVICE)
        model.load_state_dict(torch.load(f"checkpoints/{name}_best.pth"))

        dice, iou = evaluate(model, loader)
        print(name, dice, iou)


if __name__ == "__main__":
    main()