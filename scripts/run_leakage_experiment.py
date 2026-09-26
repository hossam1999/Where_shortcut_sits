"""Controlled leakage experiment (replaces the unsupported '27% inflation' claim, thesis §6.1).

Same 2,437-image cohort, same cached features, same protocol; only the split changes:
  grouped   : the frozen pHash/lesion-grouped split
  random_r  : image-level random splits with identical split sizes per class (r = 0..4)
Reports ERM shortcut gap (corr - rev AUROC), clean AUROC and mask-ERM on reversed test at 0/100%,
plus the number of near-duplicate pairs (pHash <= 8) crossing train/test in each split.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from wtss import paths
from wtss.data.isic2018 import Cohort, isic2018_loaders, load_isic2018_pilot
from wtss.experiments.synthetic import SynthConfig, run_synthetic
from wtss.synthetic import ARTIFACT_GEOMETRY


def random_split(df: pd.DataFrame, r: int) -> pd.Series:
    rng = np.random.default_rng(1000 + r)
    out = pd.Series(index=df.index, dtype=object)
    for y in (0, 1):
        idx = df.index[df.y == y].to_numpy()
        sizes = df[df.y == y].split.value_counts()
        perm = rng.permutation(idx)
        a, b = sizes["train"], sizes["train"] + sizes["val"]
        out[perm[:a]], out[perm[a:b]], out[perm[b:]] = "train", "val", "test"
    return out


def cross_pairs(df: pd.DataFrame) -> int:
    m = pd.read_csv(paths.FROZEN / "isic2018_pilot_manifest.frozen.csv")[["image_id", "phash_hex"]]
    d = df.merge(m, on="image_id")
    h = np.array([int(x, 16) for x in d.phash_hex], dtype=np.uint64)
    x = h[:, None] ^ h[None, :]
    pc = np.zeros(x.shape, np.uint8)
    for s in range(64):
        pc += ((x >> np.uint64(s)) & np.uint64(1)).astype(np.uint8)
    tr = (d.split == "train").to_numpy(); te = (d.split == "test").to_numpy()
    return int((pc[np.ix_(tr, te)] <= 8).sum())


def main():
    base = load_isic2018_pilot()
    pl = json.loads((paths.FROZEN / "placements_518_65x19.json").read_text())
    cfg = SynthConfig(overlaps=[0.0, 1.0], arms=["erm", "mask"], workers=4)
    rows = []
    for name in ["grouped"] + [f"random_{r}" for r in range(5)]:
        df = base.df.copy()
        if name != "grouped":
            df["split"] = random_split(df, int(name.split("_")[1]))
        coh = Cohort("isic2018_pilot", df, "lesion")  # same name => cached views are reused
        out = paths.RESULTS / "leakage" / name
        if not (out / "metrics.csv").exists():
            run_synthetic(coh, "dino518", pl, out, cfg, paths.CACHE / "images", isic2018_loaders(518),
                          paths.CACHE / "features", torch.device("cuda"))
        m = pd.read_csv(out / "metrics.csv")
        for ov in (0.0, 1.0):
            q = m[m.overlap == ov].groupby(["method", "env"]).auc.mean()
            rows.append({"split": name, "overlap": ov, "cross_split_near_dup_pairs": cross_pairs(df),
                         "erm_clean": q["erm", "clean"], "erm_corr": q["erm", "test_corr"], "erm_rev": q["erm", "test_rev"],
                         "erm_gap": q["erm", "test_corr"] - q["erm", "test_rev"],
                         "mask_minus_erm_rev": q["mask", "test_rev"] - q["erm", "test_rev"]})
    r = pd.DataFrame(rows)
    r.to_csv(paths.RESULTS / "leakage" / "LEAKAGE_SUMMARY.csv", index=False)
    g = r[r.split == "grouped"].set_index("overlap")
    rr = r[r.split != "grouped"].groupby("overlap")[["erm_gap", "mask_minus_erm_rev", "erm_clean", "cross_split_near_dup_pairs"]].agg(["mean", "std"])
    print(r.round(3).to_string()); print(rr.round(3))
    print("gap inflation random/grouped:", (r[r.split != "grouped"].groupby("overlap").erm_gap.mean() / g.erm_gap).round(3).to_dict())


if __name__ == "__main__":
    main()
