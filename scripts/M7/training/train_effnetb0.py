import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet_effnet_b0 import UNetEffNetB0
from datasets.pneumothorax_dataset import PneumothoraxDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():

    train_ds = PneumothoraxDataset("dataset", "train")
    val_ds = PneumothoraxDataset("dataset", "test")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    model = UNetEffNetB0().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = 1e9

    for epoch in range(25):

        model.train()
        total_loss = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss {total_loss/len(train_loader):.4f}")

        torch.save(model.state_dict(), "checkpoints/unet_effnet_b0.pth")


if __name__ == "__main__":
    train()