import numpy as np
import cv2
import scipy.ndimage as ndi


def postprocess_mask(soft, threshold=0.5, min_size=100):
    mask = (soft >= threshold).astype(np.uint8)
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    if min_size > 0:
        labeled, num = ndi.label(mask)
        sizes = ndi.sum(mask, labeled, range(1, num + 1))
        clean = np.zeros_like(mask)
        for i, s in enumerate(sizes, 1):
            if s >= min_size:
                clean[labeled == i] = 1
        mask = clean

    if mask.sum() > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask.astype(np.uint8)