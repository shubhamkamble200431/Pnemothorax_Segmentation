# Pneumothorax Segmentation using Segmentation Models PyTorch (SMP)

This repository implements three deep learning models for pneumothorax segmentation on chest X-rays using the SIIM-ACR dataset.

## Models Implemented

### U-Net (ResNet34 Encoder)

Standard encoder–decoder segmentation architecture.

### U-Net++

Nested U-Net architecture with dense skip connections.

### MAnet (Attention UNet variant)

Multi-scale attention network used as an nnU-Net-style baseline.

---

## Dataset

SIIM-ACR Pneumothorax Segmentation Dataset:

https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

Expected structure:

```
dataset/
 ├── png_images/
 └── png_masks/
```

---

## Installation

```
pip install -r requirements.txt
```

---

## Training

```
python training/train_smp_models.py
```

---

## Evaluation

```
python evaluation/evaluate_smp_models.py
```

---

## Visualization

```
python evaluation/visualize_overlays.py
```

---

## Metrics

* Dice coefficient
* IoU
* Pixel accuracy
* Case-level classification

---

## License

MIT License
