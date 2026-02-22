import csv
import numpy as np


def save_summary_csv(summaries, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Mean Dice", "Mean IoU"])

        for name, s in summaries.items():
            writer.writerow([name, s["dice"], s["iou"]])


def compute_summary(dice_list, iou_list):
    return {
        "dice": float(np.mean(dice_list)),
        "iou": float(np.mean(iou_list))
    }