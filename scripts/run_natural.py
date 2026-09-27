"""Natural-distribution evaluation (no trap resampling; docs/PREREGISTRATION_NATURAL.md).

Thyroid: train on the TN3K official trainval split with its natural caliper–label association; test on the untouched
official TN3K test split. 5 seeds of a group-safe 80/20 train/validation split of trainval = bootstrap clusters.
Metrics on the natural test set: AUROC; cross-group AUROC min(AUROC(Y1A0 vs Y0A1), AUROC(Y1A1 vs Y0A0)) with
A = in-ROI caliper (r >= 0.5, >= 15 px); AUROC within the in-ROI-caliper subgroup.

  python scripts/run_natural.py --cohort thyroid
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wtss import paths
from wtss.experiments.real_traps import RealCache
from wtss.experiments.spec_traps import run_spec
from wtss.stats import hierarchical_paired_bootstrap, safe_auc
from wtss.synthetic import draw_generic_artifact
from wtss.utils import stable_int

spec = importlib.util.spec_from_file_location("rt", Path(__file__).with_name("run_thyroid_traps.py"))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
ARMS = ("erm", "mask", "balanced", "mask_balanced", "mte", "mte_protect", "mte_balanced", "mte_protect_balanced",
        "dfr", "mask_dfr", "jtt")
SEEDS = (42, 123, 456, 789, 2026)


def cross_group(y, a, p):
    from sklearn.metrics import roc_auc_score
    out = []
    for hp, hn in ((0, 1), (1, 0)):
        s = ((y == 1) & (a == hp)) | ((y == 0) & (a == hn))
        out.append(roc_auc_score(y[s], p[s]) if len(np.unique(y[s])) == 2 else np.nan)
    return float(np.nanmin(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="thyroid", choices=("thyroid",))
    a = ap.parse_args()
    c = rt.cohort()
    c["a"] = ((c.marker_px >= 15) & (c.r >= 0.5)).astype(int)  # A = in-ROI caliper
    tv, te = c[c.split == "trainval"], c[c.split == "test"]
    envs = {}
    for s in SEEDS:
        g = tv.group.unique()
        val_g = {x for x in g if stable_int("natural_val", s, x) % 5 == 0}
        va, tr = tv[tv.group.isin(val_g)], tv[~tv.group.isin(val_g)]
        cols = ["image_id", "y", "a", "source"]
        e = {"train_corr": tr[cols], "val_clean": va[cols], "val_groups": va[cols], "train_all": tr[cols],
             "test_corr": te[cols], "test_rev": te[cols], "clean": te[cols]}
        for k, d in e.items():
            envs[("natural", s, 0, k)] = d.reset_index(drop=True)
    out = paths.ensure(paths.RESULTS / "natural" / f"{a.cohort}_dino518")
    cache = RealCache(rt.T / "cache_518", roi_file="roi.npy", art_file="marker.npy")
    ins = lambda i, rgb, roi: np.asarray(draw_generic_artifact(Image.fromarray(rgb), roi, f"u|{i}"))
    if not (out / "predictions.csv.gz").exists():
        run_spec(envs, cache, "dino518", out, paths.CACHE / "features" / "thyroid_all" / "dinov2_b14_518", [],
                 arms=ARMS, traps=("natural",), device=torch.device("cuda"), insert_fn=ins, insert_tag="_generic",
                 folds=[0])
    p = pd.read_csv(out / "predictions.csv.gz")
    p = p[p.env == "clean"].copy()  # the natural test set (identical in all three test slots)
    sub = te.set_index("image_id")
    p["a_in"] = sub.loc[p.image_id, "a"].to_numpy()
    rows = []
    for (m, s), q in p.groupby(["method", "seed"]):
        y, pr, ai = q.y.to_numpy(), q.prob.to_numpy(), q.a_in.to_numpy()
        rows.append({"method": m, "seed": s, "auc": safe_auc(y, pr), "cross_group": cross_group(y, ai, pr),
                     "auc_in_roi_caliper": safe_auc(y[ai == 1], pr[ai == 1])})
    m = pd.DataFrame(rows).groupby("method").mean(numeric_only=True).drop(columns="seed").round(3)
    m.to_csv(out / "natural_metrics.csv")
    print(m.sort_values("cross_group", ascending=False).to_string())
    boots = {}
    for arm in [x for x in ("mask", "mte_protect", "mte_balanced", "balanced", "dfr") if x in set(p.method)]:
        ref = "erm" if arm == "mask" else "mask"
        r = hierarchical_paired_bootstrap(p.assign(env="clean"), arm, ref, "clean", 10000, 5, fast=True)
        boots[f"{arm}-{ref}"] = [round(r[k], 3) for k in ("seed_delta_mean", "ci95_lo", "ci95_hi")]
    (out / "natural_boot.json").write_text(json.dumps(boots, indent=1))
    print("overall AUROC deltas:", boots)


if __name__ == "__main__":
    main()
