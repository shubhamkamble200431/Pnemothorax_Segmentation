import numpy as np
import matplotlib.pyplot as plt


def overlay(img, mask, color, alpha=0.6):
    m = mask.astype(np.float32)
    colored = np.stack([m * c for c in color], axis=-1)
    return np.clip(img * (1 - alpha * m[..., None]) + colored * alpha, 0, 1)


def visualize(image, gt, pred, title):
    err = (gt ^ pred).astype(np.float32)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(overlay(image, gt, (0, 0.5, 1)))
    ax[0].set_title("Ground Truth")

    ax[1].imshow(overlay(image, pred, (1, 1, 0)))
    ax[1].set_title("Prediction")

    ax[2].imshow(overlay(image, err, (1, 0, 0)))
    ax[2].set_title("Error")

    for a in ax:
        a.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()