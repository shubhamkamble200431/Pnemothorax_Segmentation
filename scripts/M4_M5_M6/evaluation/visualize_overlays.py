import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets.siim_rgb_dataset import SIIMRGBDataset
from models.smp_models import MODEL_REGISTRY


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def overlay(img, mask, color=(1, 1, 0), alpha=0.5):
    colored = np.stack([mask * c for c in color], axis=-1)
    return img * (1 - alpha * mask[..., None]) + colored * alpha


def main():

    dataset = SIIMRGBDataset("dataset")
    loader = DataLoader(dataset, batch_size=1)

    imgs, masks = next(iter(loader))
    img = imgs[0].permute(1, 2, 0).numpy()
    gt = masks[0, 0].numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("Image")

    plt.subplot(1, 4, 2)
    plt.imshow(gt, cmap="gray")
    plt.title("GT")

    col = 3

    for name, builder in MODEL_REGISTRY.items():
        model = builder().to(DEVICE)
        model.load_state_dict(torch.load(f"checkpoints/{name}_best.pth"))

        pred = (torch.sigmoid(model(imgs.to(DEVICE)))[0, 0] > 0.5).cpu().numpy()

        plt.subplot(1, 4, col)
        plt.imshow(overlay(img, pred))
        plt.title(name)
        col += 1

    plt.show()


if __name__ == "__main__":
    main()