"""Pixel-level mitigations: ROI masking (optionally dilated) and Telea inpainting."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .utils import MEAN_RGB


def apply_roi_mask(img: Image.Image, roi: np.ndarray, margin_px: int = 0, fill=MEAN_RGB) -> Image.Image:
    """Keep ROI pixels (optionally dilated by margin_px), replace the rest by `fill`."""
    visible = roi.astype(np.uint8)
    if margin_px > 0:
        k = 2 * margin_px + 1
        visible = cv2.dilate(visible, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    arr = np.asarray(img).copy()
    arr[visible == 0] = np.asarray(fill, dtype=np.uint8)
    return Image.fromarray(arr)


# Backwards-compatible name used by the pilot.
apply_oracle_mask = apply_roi_mask


def apply_inpaint(img: Image.Image, artifact_mask: np.ndarray, radius: int = 3) -> Image.Image:
    if artifact_mask is None or artifact_mask.sum() == 0:
        return img
    bgr = cv2.cvtColor(np.asarray(img).copy(), cv2.COLOR_RGB2BGR)
    repaired = cv2.inpaint(bgr, artifact_mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB))


def apply_method(img: Image.Image, method: str, roi: np.ndarray, artifact_mask: np.ndarray | None) -> Image.Image:
    if method == "erm":
        return img
    if method == "mask":
        return apply_roi_mask(img, roi, 0)
    if method.startswith("dilate"):
        return apply_roi_mask(img, roi, int(method.replace("dilate", "")))
    if method == "inpaint":
        return apply_inpaint(img, artifact_mask)
    if method == "mask_inpaint":
        return apply_roi_mask(apply_inpaint(img, artifact_mask), roi, 0)
    raise ValueError(method)
