"""Thyroid ultrasound marker traps (docs/PREREGISTRATION_THYROID_TRAPS.md).

  python scripts/run_thyroid_traps.py --counts_only
  python scripts/run_thyroid_traps.py --backbone dino518                     # template arms (real markers)
  python scripts/run_thyroid_traps.py --backbone dino518 --generic --tag universal
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from wtss import paths
from wtss.data.isic2019_spec import build_spec_envs, matched_pool
from wtss.experiments.real_traps import RealCache
from wtss.experiments.spec_traps import ARMS, run_spec
from wtss.stats import difference_of_deltas, hierarchical_paired_bootstrap, safe_auc, slim

BDIR = {"dino518": "dinov2_b14_518", "medsiglip448": "medsiglip_448", "convnext384": "convnext_b_384"}
T = paths.DATA / "us" / "tncd"
CAP = paths.DATA / "capsule"
OV = paths.DATA / "ovary"


def ovary_cohort() -> pd.DataFrame:
    """MMOTU ovarian tumours: cystic benign vs solid-component; calipers (docs/PREREGISTRATION_OVARY_TRAPS.md)."""
    c = pd.read_csv(OV / "ovary_cohort.csv")
    c["source"] = "MMOTU"
    c["lesion_id"] = c.image_id
    c["A0"] = c.marker_px == 0
    pres = c.marker_px >= 15
    c["trapA_A1"] = pres & (c.r >= 0.5)
    c["trapB_A1"] = pres & (c.r < 0.1)
    c["group"] = c.group.astype(str)
    return c


def capsule_cohort() -> pd.DataFrame:
    """SEE-AI erosion vs polyp-like; contamination = debris / bubbles (docs/PREREGISTRATION_CAPSULE_TRAPS.md)."""
    c = pd.read_csv(CAP / "capsule_cohort.csv")
    c["source"] = "SEE-AI"
    c["lesion_id"] = c.image_id
    c["A0"] = c.contam_frac < 0.03
    pres = c.contam_frac >= 0.10
    c["trapA_A1"] = pres & (c.r >= 0.5)
    c["trapB_A1"] = pres & (c.r < 0.1)
    c["group"] = c.group.astype(str)
    return c


def cohort() -> pd.DataFrame:
    c = pd.read_csv(T / "thyroid_cohort.csv")
    c["source"] = "TN3K"
    c["lesion_id"] = c.image_id
    c["A0"] = c.marker_px == 0
    pres = c.marker_px >= 15
    c["trapA_A1"] = pres & (c.r >= 0.5)
    c["trapB_A1"] = pres & (c.r < 0.1)
    c["group"] = c.group_ph8.astype(str)
    return c


def _b(j):
    q, arm, env, s = j
    return hierarchical_paired_bootstrap(q, arm, "erm", env, 10000, s, fast=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dino518")
    ap.add_argument("--cohort", default="thyroid", choices=("thyroid", "capsule", "ovary"))
    ap.add_argument("--generic", action="store_true")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--counts_only", action="store_true")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--save_val", action="store_true", help="also save val_groups predictions (adaptive selection)")
    ap.add_argument("--min_px", type=int, default=15, help="sensitivity: minimum detected marker pixels for A=1")
    ap.add_argument("--rA", type=float, default=0.5, help="sensitivity: Trap A overlap threshold (r >=)")
    ap.add_argument("--rB", type=float, default=0.1, help="sensitivity: Trap B overlap threshold (r <)")
    a = ap.parse_args()
    c = {"thyroid": cohort, "capsule": capsule_cohort, "ovary": ovary_cohort}[a.cohort]()
    if a.cohort in ("thyroid", "ovary") and (a.min_px, a.rA, a.rB) != (15, 0.5, 0.1):  # sensitivity analysis
        pres = c.marker_px >= a.min_px
        c["trapA_A1"] = pres & (c.r >= a.rA)
        c["trapB_A1"] = pres & (c.r < a.rB)
    envs = build_spec_envs(c, group_col="group")
    rows = []
    for trap in ("trapA", "trapB"):
        cells = matched_pool(c, trap).groupby(["a", "y"]).size()
        rows.append({"trap": trap, **{f"A{x}_Y{y}": int(cells.get((x, y), 0)) for x in (0, 1) for y in (0, 1)},
                     "rev_pos_seed42": int(sum(envs[(trap, 42, k, "test_rev")].y.sum() for k in range(5)))})
    cnt = pd.DataFrame(rows)
    out = paths.ensure(paths.RESULTS / a.cohort / f"{a.backbone}_{a.tag}")
    cnt.to_csv(out / "counts.csv", index=False)
    print(cnt.to_string(), flush=True)
    if a.counts_only:
        return
    if a.cohort == "thyroid":
        cache = RealCache(T / "cache_518", roi_file="roi.npy", art_file="marker.npy")
        donors = c[(c.marker_px >= 15) & c.r.between(0.1, 0.5, inclusive="left")].image_id.tolist()
    elif a.cohort == "ovary":
        cache = RealCache(OV / "cache_518", roi_file="roi.npy", art_file="marker.npy")
        donors = c[(c.marker_px >= 15) & c.r.between(0.1, 0.5, inclusive="left")].image_id.tolist()
    else:
        cache = RealCache(CAP / "cache_518", roi_file="roi.npy", art_file="contam.npy")
        donors = c[(c.contam_frac >= 0.10) & c.r.between(0.1, 0.5, inclusive="left")].image_id.tolist()
    kw = {}
    if a.generic:
        from PIL import Image as _I
        from wtss.synthetic import draw_generic_artifact
        kw = dict(insert_fn=lambda i, rgb, roi: np.asarray(draw_generic_artifact(_I.fromarray(rgb), roi, f"u|{i}")),
                  insert_tag="_generic", arms=("erm", "mask", "balanced", "dfr", "i2e", "i2e_balanced", "mte", "mte_balanced"))
    if a.arms:
        kw["arms"] = tuple(a.arms)
    if not (out / "predictions.csv.gz").exists():
        run_spec(envs, cache, a.backbone, out, paths.CACHE / "features" / a.cohort / BDIR[a.backbone], donors,
                 device=torch.device("cuda"), save_val=a.save_val, **kw)
    preds = pd.read_csv(out / "predictions.csv.gz")
    arms = [m for m in preds.method.unique() if m != "erm"]
    jobs, keys = [], []
    for trap in ("trapA", "trapB"):
        q = preds[preds.trap == trap]
        for arm in arms:
            for env in ("test_rev", "clean"):
                jobs.append((slim(q, env, (arm, "erm")), arm, env, 20260918 + sum(map(ord, arm + trap + env))))
                keys.append((trap, arm, env))
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_b, jobs))
    boot = pd.DataFrame([{"trap": t, "arm": m, "env": e, **r} for (t, m, e), r in zip(keys, res)])
    boot.to_csv(out / "bootstrap_vs_erm.csv", index=False)
    cr = difference_of_deltas(preds[preds.trap == "trapB"], preds[preds.trap == "trapA"], "mask", "erm", "test_rev", 10000, 11)
    (out / "T3_crossover.json").write_text(json.dumps(cr, indent=2, default=float))
    extra = {}
    if "mte" in arms:
        for trap in ("trapA", "trapB"):
            r = hierarchical_paired_bootstrap(slim(preds[preds.trap == trap], "test_rev", ("mte", "mask")), "mte", "mask",
                                              "test_rev", 10000, 3, fast=True)
            extra[f"{trap}_mte_minus_mask"] = {k: r[k] for k in ("seed_delta_mean", "ci95_lo", "ci95_hi")}
    (out / "EXTRA.json").write_text(json.dumps(extra, indent=2, default=float))
    auc = preds.groupby(["trap", "method", "env", "seed"]).apply(lambda q: safe_auc(q.y, q.prob), include_groups=False)
    auc.rename("auc").reset_index().to_csv(out / "metrics_per_seed.csv", index=False)
    print(boot[boot.env == "test_rev"][["trap", "arm", "seed_delta_mean", "ci95_lo", "ci95_hi"]].round(3).to_string())
    print("crossover", {k: round(cr[k], 3) for k in ("seed_delta_mean", "ci95_lo", "ci95_hi")}, json.dumps(extra, default=float))
    print(auc.groupby(["trap", "method", "env"]).mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
