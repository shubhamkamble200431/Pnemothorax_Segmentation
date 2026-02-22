import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from datasets.siim_rgb_dataset import SIIMRGBDataset
from models.smp_models import MODEL_REGISTRY
from utils.metrics import dice_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader, epochs, lr, name):
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_dice = 0

    for epoch in range(epochs):
        model.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        dices = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = (torch.sigmoid(model(imgs)) > 0.5).float()

                for i in range(len(preds)):
                    dices.append(dice_score(preds[i], masks[i]))

        mean_dice = sum(dices) / len(dices)

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), f"checkpoints/{name}_best.pth")

    return best_dice


def main():

    dataset = SIIMRGBDataset("dataset")
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    train_loader = DataLoader(dataset, batch_size=8, sampler=train_idx)
    val_loader = DataLoader(dataset, batch_size=8, sampler=val_idx)

    for name, builder in MODEL_REGISTRY.items():
        model = builder()
        train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, name=name)


if __name__ == "__main__":
    main()