"""Build cohort_spec.csv and roi_spec.npy (518) for the spec E13/E15 replication.

Lesion masks: HAM manual; all other images: spec U-Net (256). r = |hair ∩ lesion| / |hair| computed at 256 px
(r_spec) and at native resolution (r_native: lesion mask upsampled with nearest neighbour).
"""
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from wtss import paths

Image.MAX_IMAGE_PIXELS = None
P = paths.DATA / "isic2019" / "prepared"
U = paths.DATA / "isic2019" / "unet_spec"
S = 256


def key(n):
    return f"ISIC_{int(re.match(r'ISIC_(\d+)', n).group(1)):07d}"


HAIR = {key(p.name): p for p in (paths.ARTIFACT_MASKS / "hair_ruler").rglob("*.tif")}
HAM = {p.name.split("_segmentation")[0]: p for p in paths.HAM_SEG.parent.rglob("*_segmentation.png")}
UIDS = (U / "ids.txt").read_text().split()
UIDX = {k: j for j, k in enumerate(UIDS)}


def one(i):
    um = np.load(U / "masks_256.npy", mmap_mode="r")
    with Image.open(HAIR[key(i)]) as im:
        h_nat = np.asarray(im.convert("L")) >= 128
    H, W = h_nat.shape
    if i in HAM:
        with Image.open(HAM[i]) as m:
            les_img = m.convert("L")
            l256 = np.asarray(les_img.resize((S, S), Image.NEAREST)) >= 128
            l_nat = np.asarray(les_img.resize((W, H), Image.NEAREST)) >= 128
        src = "manual_ham"
    else:
        l256 = np.asarray(um[UIDX[i]]).astype(bool)
        l_nat = np.asarray(Image.fromarray(l256.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) >= 128
        src = "unet_spec"
    h256 = np.asarray(Image.fromarray(h_nat.astype(np.uint8) * 255).resize((S, S), Image.NEAREST)) >= 128
    r256 = (h256 & l256).sum() / h256.sum() if h256.sum() else np.nan
    rnat = (h_nat & l_nat).sum() / h_nat.sum() if h_nat.sum() else np.nan
    l518 = np.asarray(Image.fromarray(l256.astype(np.uint8) * 255).resize((518, 518), Image.NEAREST)) >= 128
    return i, src, float(l256.mean()), float(r256), float(rnat), float(h256.mean()), l518


if __name__ == "__main__":
    base = pd.read_csv(P / "cohort_518.csv")
    base["key"] = base.image_id.map(key)
    base = base.merge(pd.read_csv(P / "hair_native.csv"), on="key")
    base["source"] = base.source.replace({"ISIC_legacy": "MSK"})
    ids = (P / "cache_518" / "ids.txt").read_text().split()
    pos = {k: j for j, k in enumerate(ids)}
    roi_old = np.load(P / "cache_518" / "roi.npy", mmap_mode="r")
    roi = np.lib.format.open_memmap(P / "cache_518" / "roi_spec.npy", "w+", np.uint8, roi_old.shape)
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for i, src, lf, r256, rnat, hf256, l518 in ex.map(one, base.image_id, chunksize=32):
            rows.append((i, src, lf, r256, rnat, hf256))
            # HAM: manual mask at 518 is already in the cache; others: spec U-Net upsampled
            roi[pos[i]] = np.asarray(roi_old[pos[i]]) if src == "manual_ham" else l518
    roi.flush()
    d = pd.DataFrame(rows, columns=["image_id", "lesion_mask_spec", "lesion_frac_spec", "r_spec", "r_native", "hair_frac_256"])
    c = base.merge(d, on="image_id")
    c["qc_ok"] = c.lesion_frac_spec > 0
    c.to_csv(P / "cohort_spec.csv", index=False)
    print(c.lesion_mask_spec.value_counts(), "\nqc fail:", int((~c.qc_ok).sum()))
