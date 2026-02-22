import torch
from torch.utils.data import DataLoader
from models.unet_effnet_b0 import UNetEffNetB0
from datasets.pneumothorax_dataset import PneumothoraxDataset
from utils.metrics import dice_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate():

    dataset = PneumothoraxDataset("dataset", "test")
    loader = DataLoader(dataset, batch_size=4)

    model = UNetEffNetB0(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/unet_effnet_b0.pth"))
    model.eval()

    dices = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            probs = torch.sigmoid(model(imgs))
            preds = (probs > 0.5).float()

            for i in range(len(preds)):
                dices.append(dice_score(preds[i], masks[i]).item())

    print("Mean Dice:", sum(dices) / len(dices))


if __name__ == "__main__":
    evaluate()