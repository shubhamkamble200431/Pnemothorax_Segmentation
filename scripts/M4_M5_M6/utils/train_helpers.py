import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from utils.metrics import dice_score


def train_model(model, train_loader, val_loader, epochs, lr, device, name):
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_dice = 0
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(epochs):

        # ---- TRAIN ----
        model.train()
        running_loss = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0
        dice_scores = []
        y_true, y_pred = [], []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item()

                preds = (torch.sigmoid(outputs) > 0.5).float()

                for i in range(len(preds)):
                    dice_scores.append(dice_score(preds[i], masks[i]))

                    gt_cls = int(masks[i].sum() > 0)
                    pr_cls = int(preds[i].sum() > 0)
                    y_true.append(gt_cls)
                    y_pred.append(pr_cls)

        val_loss /= len(val_loader)
        avg_dice = np.mean(dice_scores)
        acc = accuracy_score(y_true, y_pred)

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(avg_dice)
        history["val_acc"].append(acc)

        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            torch.save(model.state_dict(), f"checkpoints/{name}_best.pth")

    total_time = time.time() - start_time
    return history, best_val_dice, acc, total_time