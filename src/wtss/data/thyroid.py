"""Thyroid ultrasound (TN3K images + TNCD labels) — controlled synthetic-caliper cohort.

Only marker-free images (no detected marker pixel) are used, so the only marker in any image is the synthetic
one. Split: TN3K official test -> test; trainval -> train/val (80/20) by pHash leakage group.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import paths
from ..synthetic import ARTIFACT_GEOMETRY, OVERLAPS, feasible_ids, find_best_placements_for_mask
from .isic2018 import Cohort

T = paths.DATA / "us" / "tncd"


def load_thyroid_synthetic_cohort(size: int = 518):
    assert size == 518, "cache is 518 px"
    c = pd.read_csv(T / "thyroid_cohort.csv")
    c = c[c.marker_px == 0].copy()
    c["group"] = c.group_ph8.astype(str)
    rng = np.random.default_rng(20260927)
    tv = c[c.split == "trainval"].group.unique()
    val_g = set(rng.choice(tv, int(0.2 * len(tv)), replace=False))
    c["split"] = np.where(c.split == "test", "test", np.where(c.group.isin(val_g), "val", "train"))
    cdir = T / "cache_518"
    gray = np.load(cdir / "gray.npy", mmap_mode="r")
    roi = np.load(cdir / "roi.npy", mmap_mode="r")
    idx = {k: j for j, k in enumerate((cdir / "ids.txt").read_text().split())}
    w, h = ARTIFACT_GEOMETRY[size]
    pf = T / f"synthetic_placements_{size}_{w}x{h}.json"
    if pf.exists():
        pl = json.loads(pf.read_text())
    else:
        pl = {i: find_best_placements_for_mask(np.asarray(roi[idx[i]]) > 0, i, OVERLAPS, w, h, 2500) for i in c.image_id}
        pf.write_text(json.dumps(pl))
    keep = feasible_ids(pl, OVERLAPS, 0.10)
    c = c[c.image_id.isin(keep)].reset_index(drop=True)
    loaders = (lambda i: np.repeat(np.asarray(gray[idx[i]])[..., None], 3, -1), lambda i: np.asarray(roi[idx[i]]))
    return Cohort("thyroid_synth", c[["image_id", "y", "split", "group"]], "nodule"), pl, loaders, {size: (w, h)}


def load_ovary_synthetic_cohort(size: int = 518):
    """MMOTU marker-free images (docs/PREREGISTRATION_OVARY_TRAPS.md), split 60/20/20 by pHash group."""
    assert size == 518
    OV = paths.DATA / "ovary"
    c = pd.read_csv(OV / "ovary_cohort.csv")
    c = c[c.marker_px == 0].copy()
    c["group"] = c.group.astype(str)
    g = np.array(sorted(c.group.unique()), dtype=object)
    np.random.default_rng(20260927).shuffle(g)
    n = len(g)
    split = {k: ("train" if j < 0.6 * n else "val" if j < 0.8 * n else "test") for j, k in enumerate(g)}
    c["split"] = c.group.map(split)
    cdir = OV / "cache_518"
    gray = np.load(cdir / "gray.npy", mmap_mode="r")
    roi = np.load(cdir / "roi.npy", mmap_mode="r")
    idx = {k: j for j, k in enumerate((cdir / "ids.txt").read_text().split())}
    w, h = ARTIFACT_GEOMETRY[size]
    pf = OV / f"synthetic_placements_{size}_{w}x{h}.json"
    if pf.exists():
        pl = json.loads(pf.read_text())
    else:
        pl = {i: find_best_placements_for_mask(np.asarray(roi[idx[i]]) > 0, i, OVERLAPS, w, h, 2500) for i in c.image_id}
        pf.write_text(json.dumps(pl))
    keep = feasible_ids(pl, OVERLAPS, 0.10)
    c = c[c.image_id.isin(keep)].reset_index(drop=True)
    loaders = (lambda i: np.repeat(np.asarray(gray[idx[i]])[..., None], 3, -1), lambda i: np.asarray(roi[idx[i]]))
    return Cohort("ovary_synth", c[["image_id", "y", "split", "group"]], "tumour"), pl, loaders, {size: (w, h)}
