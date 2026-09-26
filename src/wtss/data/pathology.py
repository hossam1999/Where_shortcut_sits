"""TCGA lung (LUAD vs LUSC) whole-slide cohort with GrandQC artifact masks.

A slide is represented by the mean frozen embedding of its non-empty 224-px tiles at ~4 µm/px.
Views (pixel methods, applied on the level image before tiling):
  erm       every tile containing tissue or an artifact (pen on glass included)
  mask      ROI = tissue region (GrandQC tissue + artifacts inside it, hole-filled); outside -> white glass;
            tiles without ROI are dropped
  inpaint   artifact pixels of the trap's artifact class Telea-inpainted (oracle)
  insert    generic artifact library drawn on tiles (Universal I2E); mask_insert = insert then mask
Because the eraser and the head are linear, tile-level erasure equals slide-level erasure of the mean.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .. import paths

Image.MAX_IMAGE_PIXELS = None
PREP = paths.DATA / "path" / "prepared"
LEVELS = paths.DATA / "path" / "levels"
GQC = paths.DATA / "path" / "grandqc"
CLASS = {"fold": 2, "dark_foreign": 3, "pen": 4, "bubble_edge": 5, "out_of_focus": 6}
TILE = 224


def mask_path(slide: str):
    hits = list(GQC.glob(f"*/mask_qc/{slide}.svs_mask.png"))
    return hits[0] if hits else None


def load_slide(slide: str):
    """(rgb level image, GrandQC class map resized to it, ROI mask)."""
    rgb = np.asarray(Image.open(LEVELS / f"{slide}.jpg").convert("RGB"))
    H, W = rgb.shape[:2]
    cls = np.asarray(Image.open(mask_path(slide)).resize((W, H), Image.NEAREST))
    tissue = (cls == 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(tissue.astype(np.uint8), cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi = np.zeros((H, W), np.uint8)
    cv2.drawContours(roi, cnts, -1, 1, thickness=-1)
    return rgb, cls, roi.astype(bool)


def tiles(rgb: np.ndarray, keep: np.ndarray, min_frac: float = 0.1):
    """Non-overlapping TILE x TILE tiles whose `keep` fraction >= min_frac; returns list of (y, x)."""
    H, W = keep.shape
    out = []
    for y in range(0, H - TILE + 1, TILE):
        for x in range(0, W - TILE + 1, TILE):
            if keep[y:y + TILE, x:x + TILE].mean() >= min_frac:
                out.append((y, x))
    return out
