"""Lesion-mask source sensitivity (thesis §6.1 / §11).

On ISIC 2019 images that have a manual mask AND were held out from U-Net training:
  * Dice(U-Net, manual);
  * hair-lesion overlap r under each mask source, and the fraction of hair-present images whose
    trap stratum (Trap A: r>=0.5 / donor: 0.1<=r<0.5 / Trap B: r<0.1) changes when the source is swapped.
"""
import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from wtss import paths
from wtss.data.isic2019 import HAIR_MIN, R_IN, R_OUT

P = paths.DATA / "isic2019"
coh = pd.read_csv(P / "prepared" / "cohort_518.csv")
split = pd.read_csv(P / "unet" / "unet_split.csv")
held = set(split[split.split == "heldout"].image_id)
cdir = P / "prepared" / "cache_518"
ids = (cdir / "ids.txt").read_text().split()
idx = {k: j for j, k in enumerate(ids)}
roi = np.load(cdir / "roi.npy", mmap_mode="r")
hair = np.load(cdir / "hair.npy", mmap_mode="r")
sel = coh[coh.image_id.isin(held) & coh.lesion_mask_source.str.startswith("manual")]


def stratum(r):
    return np.where(np.isnan(r), "none", np.where(r >= R_IN, "A", np.where(r < R_OUT, "B", "donor")))


rows = []
for i in tqdm(sel.image_id, desc="mask source"):
    m = np.asarray(roi[idx[i]]).astype(bool)
    with Image.open(P / "unet_masks" / f"{i}_unet.png") as im:
        u = np.asarray(im.convert("L").resize((518, 518), Image.NEAREST)) >= 128
    h = np.asarray(hair[idx[i]]).astype(bool)
    d = 2 * (m & u).sum() / max(1, m.sum() + u.sum())
    hs = h.sum()
    rows.append({"image_id": i, "dice": d, "hair_frac": h.mean(),
                 "r_manual": (h & m).sum() / hs if hs else np.nan, "r_unet": (h & u).sum() / hs if hs else np.nan})
df = pd.DataFrame(rows)
hp = df[df.hair_frac >= HAIR_MIN].copy()
hp["s_manual"], hp["s_unet"] = stratum(hp.r_manual.to_numpy()), stratum(hp.r_unet.to_numpy())
rep = {"n_heldout_manual": int(len(df)), "dice_mean": float(df.dice.mean()), "dice_median": float(df.dice.median()),
       "n_hair_present": int(len(hp)), "stratum_change_rate": float((hp.s_manual != hp.s_unet).mean()),
       "crosstab": pd.crosstab(hp.s_manual, hp.s_unet).to_dict()}
out = paths.ensure(paths.RESULTS / "verification")
df.to_csv(out / "mask_source_per_image.csv", index=False)
(out / "MASK_SOURCE_SENSITIVITY.json").write_text(json.dumps(rep, indent=2, default=str))
print(json.dumps(rep, indent=2, default=str))
