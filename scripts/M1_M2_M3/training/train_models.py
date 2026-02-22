import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datasets.siim_dataset import SIIMDataset
from models import PTXSegNet, ResNet50_UNet, EfficientNetB4_UNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one(model, train_loader, val_loader, epochs, lr, name):

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_dice = 0

    for epoch in range(epochs):

        # ---- TRAIN ----
        model.train()

        for imgs, masks, _ in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)

            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---- VALIDATION ----
        model.eval()
        dices = []

        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                preds = (model(imgs) > 0.5).float()

                inter = (preds * masks).sum()
                union = preds.sum() + masks.sum()

                dice = (2 * inter + 1e-6) / (union + 1e-6)
                dices.append(dice.item())

        mean_dice = sum(dices) / len(dices)

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), f"checkpoints/{name}_best.pth")

    return best_dice


def main():

    dataset = SIIMDataset("dataset", split="train", img_size=512)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4)

    models = {
        "ptxsegnet": PTXSegNet(),
        "resnet50_unet": ResNet50_UNet(),
        "effnetb4_unet": EfficientNetB4_UNet()
    }

    for name, model in models.items():
        train_one(model, train_loader, val_loader, epochs=25, lr=1e-4, name=name)


if __name__ == "__main__":
    main()