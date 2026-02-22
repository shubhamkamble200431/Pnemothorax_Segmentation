# Pneumothorax Segmentation on SIIM-ACR Dataset

This repository provides a research-grade implementation of deep learning models for automatic pneumothorax segmentation on chest X-rays using the SIIM-ACR dataset.

## Models Implemented

### 1. PTXSegNet (Proposed Model)

Attention-Residual U-Net with Deep Supervision.

**Key features:**

* Residual encoder-decoder blocks
* Attention gates on skip connections
* Deep supervision outputs
* Designed for medical segmentation tasks

---

### 2. ResNet50-UNet

Encoder-decoder architecture using ResNet-50 backbone.

**Key features:**

* ImageNet pretrained encoder
* Strong hierarchical feature extraction
* Standard U-Net decoder

---

### 3. EfficientNet-B4 UNet

High-capacity UNet using EfficientNet-B4 encoder.

**Key features:**

* Compound scaling backbone
* High resolution feature maps
* Suitable for fine-grained segmentation

---

## Dataset

SIIM-ACR Pneumothorax Segmentation Dataset:

https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

Expected directory structure:

```
dataset_root/
 ├── train/
 │    ├── png_images/
 │    └── png_masks/
 └── test/
      ├── png_images/
      └── png_masks/
```

---

## Installation

```
pip install -r requirements.txt
```

---

## Inference / Evaluation

Run unified evaluation:

```
python evaluation/evaluate_models.py
```

---

## Visualization

Generate comparison overlays:

```
python evaluation/visualize_comparison.py
```

---

## Metrics Reported

* Dice coefficient
* IoU (Jaccard index)
* Pixel precision / recall / F1
* Case-level classification metrics

---

## Results (Example)

| Model                | Dice | IoU |
| -------------------- | ---- | --- |
| PTXSegNet            | —    | —   |
| ResNet50-UNet        | —    | —   |
| EfficientNet-B4 UNet | —    | —   |

---

## Citation

If you use this code, please cite:

```
@misc{ptxsegnet2025,
  title={Pneumothorax Segmentation using Deep Neural Networks},
  author={Your Name},
  year={2025}
}
```

---

## License

MIT License
