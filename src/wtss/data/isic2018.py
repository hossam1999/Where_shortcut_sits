"""ISIC 2018 Task 1/2 pilot cohort (synthetic-ruler experiments).

Uses the *frozen* leakage-safe manifest (pHash groups; three 2016/2017 label
conflicts dropped) and the frozen 224/518 multi-resolution common support (2,437
images). Machine-specific paths in the manifest are ignored and rebuilt from WTSS_DATA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import paths
from ..utils import load_mask, load_rgb


@dataclass
class Cohort:
    name: str
    df: pd.DataFrame  # image_id, y, split, group (+ extra)
    roi_name: str

    def ids(self, split=None):
        d = self.df if split is None else self.df[self.df.split == split]
        return d.image_id.tolist()

    def y(self, split=None):
        d = self.df if split is None else self.df[self.df.split == split]
        return d.y.to_numpy().astype(int)


def load_isic2018_pilot(common_support: bool = True) -> Cohort:
    m = pd.read_csv(paths.FROZEN / "isic2018_pilot_manifest.frozen.csv")
    m = m.rename(columns={"MEL": "y", "phash_group": "group"})
    m["y"] = (m.y.astype(float) >= 0.5).astype(int)
    if common_support:
        keep = set(pd.read_csv(paths.FROZEN / "multires_common_support_ids.csv").image_id)
        m = m[m.image_id.isin(keep)]
    # leakage check: no pHash group spans splits
    bad = m.groupby("group").split.nunique()
    assert (bad <= 1).all(), "pHash group crosses splits"
    m = m[["image_id", "y", "split", "group"]].reset_index(drop=True)
    return Cohort("isic2018_pilot", m, "lesion")


def isic2018_image_path(image_id: str):
    return paths.ISIC2018_IMAGES / f"{image_id}.jpg"


def isic2018_mask_path(image_id: str):
    return paths.ISIC2018_MASKS / f"{image_id}_segmentation.png"


def isic2018_loaders(size: int):
    return (lambda i: np.asarray(load_rgb(isic2018_image_path(i), size), dtype=np.uint8),
            lambda i: load_mask(isic2018_mask_path(i), size))
