import numpy as np
import matplotlib.pyplot as plt


def overlay_mask(image, mask, color=(1.0, 1.0, 0.0), alpha=0.5):
    """
    Overlay segmentation mask on RGB image.

    image: HxWx3 float [0,1]
    mask:  HxW binary {0,1}
    color: RGB tuple in [0,1]
    """

    image = image.astype(np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    mask = mask.astype(np.float32)

    colored = np.stack([
        mask * color[0],
        mask * color[1],
        mask * color[2]
    ], axis=-1)

    output = image * (1 - alpha * mask[..., None]) + colored * alpha
    return np.clip(output, 0, 1)


def error_map(pred, gt):
    """
    XOR error map between prediction and ground truth.
    """
    return (pred.astype(np.uint8) ^ gt.astype(np.uint8)).astype(np.float32)


def show_sample(image, gt_mask, pred_mask, title=None):
    """
    Display image, GT, prediction, and error.
    """

    err = error_map(pred_mask, gt_mask)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    ax[0].imshow(image)
    ax[0].set_title("Image")

    ax[1].imshow(gt_mask, cmap="gray")
    ax[1].set_title("Ground Truth")

    ax[2].imshow(pred_mask, cmap="gray")
    ax[2].set_title("Prediction")

    ax[3].imshow(err, cmap="hot")
    ax[3].set_title("Error")

    for a in ax:
        a.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def compare_models(image, gt_mask, pred_dict):
    """
    Compare predictions from multiple models.

    pred_dict: {model_name: mask}
    """

    n = len(pred_dict) + 2
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("GT")
    axes[1].axis("off")

    i = 2
    for name, pred in pred_dict.items():
        axes[i].imshow(overlay_mask(image, pred))
        axes[i].set_title(name)
        axes[i].axis("off")
        i += 1

    plt.tight_layout()
    plt.show()