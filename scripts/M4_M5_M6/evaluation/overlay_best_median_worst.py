import numpy as np
import matplotlib.pyplot as plt


def overlay(img, mask, color, alpha=0.6):
    colored = np.stack([mask * c for c in color], axis=-1)
    return img * (1 - alpha * mask[..., None]) + colored * alpha


def visualize_range_samples(file_list, dice_list, dataset_map, preds_map, title):
    positive_items = [(i, d) for i, d in enumerate(dice_list)
                      if dataset_map[file_list[i]][0].sum() > 0]

    positive_items = sorted(positive_items, key=lambda x: x[1], reverse=True)

    best = positive_items[0][0]
    worst = positive_items[-1][0]
    med = positive_items[len(positive_items) // 2][0]

    indices = [best, med, worst]

    for idx in indices:
        fp = file_list[idx]
        mask, img = dataset_map[fp]
        pred = preds_map[fp]
        err = (pred ^ mask).astype(np.float32)

        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.imshow(overlay(img, mask, (0, 0, 1)))
        plt.title("GT")

        plt.subplot(1, 3, 2)
        plt.imshow(overlay(img, pred, (1, 1, 0)))
        plt.title("Prediction")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay(img, err, (1, 0, 0)))
        plt.title("Error")

        plt.suptitle(f"{title} — Dice {dice_list[idx]:.4f}")
        plt.show()