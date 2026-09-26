"""Chest radiography (NIH ChestX-ray14) cohorts for WP1.

Two cohorts, both pneumothorax (PTX) vs no-PTX, patient-grouped splits:

1. ``nih_ptx`` — synthetic-tube cohort (controlled overlap with the lung ROI, as for the ruler on
   ISIC). Positives are NEATX-annotated PTX images *without* a chest drain (so no real device is
   correlated with the label); negatives are 'No Finding' images, 3 per positive patient-balanced.
2. ``nih_drain`` — real chest-drain trap. A = chest drain. PTX-positive drain labels from NEATX
   (Jiménez-Sánchez et al. 2023/2025, two non-expert annotators, zenodo 14944064); drain labels
   for PTX-negative images from a drain detector trained on NEATX (scripts/data/prepare_cxr.py).

Lung ROI: torchxrayvision PSPNet (chestx_det), union of left and right lung at p > 0.5.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import paths
from .isic2018 import Cohort

NIH_PNG = paths.CXR_NIH / "png" / "images"
PREP = paths.DATA / "cxr" / "prepared"
TUBE_GEOMETRY = {518: (100, 12), 224: (43, 6)}  # ~0.45% of the image, like the ISIC ruler


def nih_table() -> pd.DataFrame:
    d = pd.read_csv(paths.CXR_NIH / "data" / "Data_Entry_2017_v2020.csv")
    d = d.rename(columns={"Image Index": "image_id", "Patient ID": "patient", "Finding Labels": "labels",
                          "View Position": "view"})
    d["ptx"] = d.labels.str.contains("Pneumothorax").astype(int)
    d["no_finding"] = (d.labels == "No Finding").astype(int)
    n = pd.read_csv(paths.NEATX / "NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv")
    n = n.rename(columns={"Image Index": "image_id", "Drain": "drain_neatx"})[["image_id", "drain_neatx"]]
    return d.merge(n, on="image_id", how="left")


def patient_split(df: pd.DataFrame, seed: int = 20260926, fr=(0.6, 0.2, 0.2)) -> pd.Series:
    pats = np.array(sorted(df.patient.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n = len(pats)
    cut1, cut2 = int(fr[0] * n), int((fr[0] + fr[1]) * n)
    lab = {p: ("train" if i < cut1 else "val" if i < cut2 else "test") for i, p in enumerate(pats)}
    return df.patient.map(lab)


def build_synthetic_cohort(neg_per_pos: int = 3, seed: int = 20260926) -> pd.DataFrame:
    d = nih_table()
    pos = d[(d.ptx == 1) & (d.drain_neatx == 0)]
    negs = d[(d.no_finding == 1) & ~d.patient.isin(pos.patient)]
    # one image per negative patient (limits patient clustering), deterministic
    negs = negs.sample(frac=1.0, random_state=seed).drop_duplicates("patient")
    negs = negs.sample(n=min(len(negs), neg_per_pos * len(pos)), random_state=seed)
    c = pd.concat([pos, negs]).rename(columns={"ptx": "y"})
    c["split"] = patient_split(c, seed)
    c["group"] = c.patient
    return c[["image_id", "y", "split", "group", "patient", "view"]].sort_values("image_id").reset_index(drop=True)


def load_nih_synthetic_cohort(size: int):
    """(cohort, placements, loaders, geometry) for scripts/run_synthetic.py."""
    from ..features import ImageCache  # noqa: F401  (cache built by prepare_cxr.py)

    df = pd.read_csv(PREP / "nih_ptx_cohort.csv")
    w, h = TUBE_GEOMETRY[size]
    pl = json.loads((PREP / f"nih_ptx_placements_{size}_{w}x{h}.json").read_text())
    keep = set(pd.read_csv(PREP / f"nih_ptx_common_support_{size}.csv").image_id)
    df = df[df.image_id.isin(keep)].reset_index(drop=True)
    cache_dir = PREP / f"cache_nih_ptx_{size}"
    rgb = np.load(cache_dir / "rgb.npy", mmap_mode="r")
    roi = np.load(cache_dir / "roi.npy", mmap_mode="r")
    idx = {k: j for j, k in enumerate((cache_dir / "ids.txt").read_text().split())}
    loaders = (lambda i: np.asarray(rgb[idx[i]]), lambda i: np.asarray(roi[idx[i]]))
    return Cohort("nih_ptx", df[["image_id", "y", "split", "group"]], "lung"), pl, loaders, {size: (w, h)}
