import numpy as np
import matplotlib.pyplot as plt


def overlay(img, mask, color, alpha=0.6):
    colored = np.stack([mask*c for c in color], axis=-1)
    return img*(1-alpha*mask[...,None]) + colored*alpha


def visualize(img, gt, pred):

    error = gt ^ pred

    plt.figure(figsize=(6,12))

    plt.subplot(3,1,1)
    plt.imshow(overlay(img, gt, (0,0,1)))
    plt.title("GT")

    plt.subplot(3,1,2)
    plt.imshow(overlay(img, pred, (1,1,0)))
    plt.title("Prediction")

    plt.subplot(3,1,3)
    plt.imshow(overlay(img, error, (1,0,0)))
    plt.title("Error")

    plt.show()