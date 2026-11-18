"""Lesion ROI per ISIC 2019 image for P2 arms b and c (section 3: "lesion mask (HAM10000 manual; U-Net for BCN/MSK, as in
E13)"), with the rule of scripts/data/prepare_isic2019.py: manual mask (HAM10000, else ISIC 2018 Task 1) where one
exists, otherwise the lesion U-Net prediction (scripts/data/train_lesion_unet.py --stage predict). Masks are resized to
224 x 224 (nearest), the geometry of P2's rendered images (their Resize((224, 224)) of the original). CPU only.

    python scripts/replication/p2_lesion_rois.py
Writes $WTSS_DATA/replication/pahde/lesion_roi224.npz (ids, packed bits) and results/replication/pahde/lesion_roi_sources.csv.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wtss import paths  # noqa: E402

D = os.environ.get("WTSS_DATA", "/workspace/data")
OUT = Path(f"{D}/replication/pahde/lesion_roi224.npz")
UNET = paths.DATA / "isic2019" / "unet_masks"

ham = {p.name.split("_segmentation")[0]: p for p in paths.HAM_SEG.parent.rglob("*_segmentation.png")}
i18 = {p.name.split("_segmentation")[0]: p for p in paths.ISIC2018_MASKS.glob("*_segmentation.png")}
ids = pd.read_csv(paths.ISIC2019_GT).image.tolist()


def one(i):
    src, p = ("ham", ham[i]) if i in ham else ("isic2018", i18[i]) if i in i18 else ("unet", UNET / f"{i}_unet.png")
    with Image.open(p) as m:
        r = np.asarray(m.convert("L").resize((224, 224), Image.NEAREST)) >= 128
    return src, r


with ThreadPoolExecutor(32) as ex:
    res = list(ex.map(one, ids))
src = np.array([s for s, _ in res])
R = np.stack([r for _, r in res])
np.savez_compressed(OUT, ids=np.array(ids, dtype="U"), bits=np.packbits(R, axis=-1))
t = pd.DataFrame({"source": src, "roi_frac": R.mean((1, 2))}).groupby("source").agg(n=("roi_frac", "size"),
    mean_roi_frac=("roi_frac", "mean"), empty=("roi_frac", lambda x: int((x == 0).sum()))).reset_index()
t.to_csv(ROOT / "results/replication/pahde/lesion_roi_sources.csv", index=False)
print(t.to_string(index=False))
