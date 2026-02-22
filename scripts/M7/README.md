# Pneumothorax Segmentation using U-Net + EfficientNet-B0

This repository implements a deep learning model for pneumothorax segmentation on chest X-ray images using a U-Net architecture with an EfficientNet-B0 encoder.

## Model Implemented

### U-Net (EfficientNet-B0 Encoder)

Encoder–decoder segmentation network where EfficientNet-B0 serves as the backbone for feature extraction, providing strong performance with relatively low computational cost.

---

## Dataset

SIIM-ACR Pneumothorax Segmentation Dataset:

https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

### Expected Dataset Structure

```
dataset/
 ├── train/
 │   ├── png_images/
 │   └── png_masks/
 └── test/
     ├── png_images/
     └── png_masks/
```

Image and mask filenames must correspond:

```
train_image_x.png  ↔  train_mask_x.png
test_image_x.png   ↔  test_mask_x.png
```

Binary masks should represent pneumothorax regions (white = foreground).

---

## Installation

Install required packages:

```
pip install -r requirements.txt
```

---

## Training

Train the EfficientNet-B0 U-Net model:

```
python training/train_effnetb0.py
```

---

## Evaluation

Evaluate the trained model on the test set:

```
python evaluation/evaluate_effnetb0.py
```

---

## Visualization

Generate qualitative comparison plots (best / median / worst cases):

```
python visualization/visualize_comparison.py
```

This will produce overlays showing:

- Ground Truth mask (blue)
- Predicted mask (yellow)
- Error regions (red)

---

## Inference (Single Image)

Run prediction on a single image:

```
python inference/predict_single.py
```

---

## Metrics Reported

- Dice coefficient (DSC)
- Intersection over Union (IoU)
- Pixel accuracy
- Precision and Recall
- Case-level classification performance

---

## Outputs

The system generates:

- Model checkpoints
- Evaluation summaries (CSV)
- Prediction masks
- Overlay visualizations
- Performance plots

---

## License

MIT License
