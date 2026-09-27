"""Capsule endoscopy (SEE-AI) — controlled synthetic-debris cohort (docs/PREREGISTRATION_CAPSULE_TRAPS.md).

Frames with contamination < 5 % only; split 60/20/20 by leakage group (seeded). Debris box 48x48 at 518 px.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import paths
from ..synthetic import OVERLAPS, feasible_ids, find_best_placements_for_mask
from .isic2018 import Cohort

CAP = paths.DATA / "capsule"
DEBRIS_GEOMETRY = {518: (48, 48)}


def load_capsule_synthetic_cohort(size: int = 518):
    assert size == 518
    c = pd.read_csv(CAP / "capsule_cohort.csv")
    c = c[c.contam_frac < 0.05].copy()
    c["group"] = c.group.astype(str)
    g = np.array(sorted(c.group.unique()), dtype=object)
    rng = np.random.default_rng(20260927)
    rng.shuffle(g)
    n = len(g)
    split = {k: ("train" if j < 0.6 * n else "val" if j < 0.8 * n else "test") for j, k in enumerate(g)}
    c["split"] = c.group.map(split)
    cdir = CAP / "cache_518"
    rgb = np.load(cdir / "rgb.npy", mmap_mode="r")
    roi = np.load(cdir / "roi.npy", mmap_mode="r")
    idx = {k: j for j, k in enumerate((cdir / "ids.txt").read_text().split())}
    w, h = DEBRIS_GEOMETRY[size]
    pf = CAP / f"synthetic_placements_{size}_{w}x{h}.json"
    if pf.exists():
        pl = json.loads(pf.read_text())
    else:
        pl = {i: find_best_placements_for_mask(np.asarray(roi[idx[i]]) > 0, i, OVERLAPS, w, h, 2500) for i in c.image_id}
        pf.write_text(json.dumps(pl))
    keep = feasible_ids(pl, OVERLAPS, 0.10)
    c = c[c.image_id.isin(keep)].reset_index(drop=True)
    loaders = (lambda i: np.asarray(rgb[idx[i]]), lambda i: np.asarray(roi[idx[i]]))
    return Cohort("capsule_synth", c[["image_id", "y", "split", "group"]], "lesion"), pl, loaders, {size: (w, h)}
