"""Natural-distribution evaluation (no trap resampling; docs/PREREGISTRATION_NATURAL.md).

Cohorts
  thyroid        train: TN3K official trainval (natural caliper–label association); test: untouched official test.
  isic_<SRC>     leave-one-hospital-out (SRC in BCN, HAM, MSK): train on the other two ISIC 2019 sources, test on SRC.
  capsule        SEE-AI; per seed a group-safe 80/20 train/test split of frame-block groups.
5 seeds (validation split and, for capsule, the test split) = bootstrap clusters. A = in-ROI artifact.
Metrics: AUROC; cross-group AUROC; paired bootstraps on shortcut-conflicting (hard) and -consistent (easy) pairs.

  python scripts/run_natural.py --cohort thyroid | isic_HAM | capsule
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
COLS = ["image_id", "y", "a", "source"]


def cross_group(y, a, p):
    from sklearn.metrics import roc_auc_score
    out = []
    for hp, hn in ((0, 1), (1, 0)):
        s = ((y == 1) & (a == hp)) | ((y == 0) & (a == hn))
        out.append(roc_auc_score(y[s], p[s]) if len(np.unique(y[s])) == 2 else np.nan)
    return float(np.nanmin(out))


def load(cohort):
    """(cohort df with y, a, group, source), cache, feature dir, function(seed) -> (train+val df, test df)."""
    if cohort == "thyroid":
        c = rt.cohort(); c["a"] = ((c.marker_px >= 15) & (c.r >= 0.5)).astype(int)
        cache = RealCache(rt.T / "cache_518", roi_file="roi.npy", art_file="marker.npy")
        split = lambda s: (c[c.split == "trainval"], c[c.split == "test"])
        return c, cache, paths.CACHE / "features" / "thyroid_all" / "dinov2_b14_518", split
    if cohort.startswith("isic_"):
        src = cohort.split("_", 1)[1]
        c = pd.read_csv(paths.DATA / "isic2019" / "prepared" / "cohort_spec.csv")
        c = c[c.qc_ok.astype(bool)].copy()
        c["a"] = ((c.hair_px_native > 30) & (c.r_spec >= 0.5)).astype(int)
        c["group"] = c.group.astype(str)
        cache = RealCache(paths.DATA / "isic2019" / "prepared" / "cache_518", roi_file="roi_spec.npy", art_file="hair.npy")
        split = lambda s: (c[c.source != src], c[c.source == src])
        return c, cache, paths.CACHE / "features" / "isic_all" / "dinov2_b14_518", split
    if cohort == "capsule":
        c = rt.capsule_cohort(); c["a"] = ((c.contam_frac >= 0.10) & (c.r >= 0.5)).astype(int)
        cache = RealCache(rt.CAP / "cache_518", roi_file="roi.npy", art_file="contam.npy")

        def split(s):
            te_g = {g for g in c.group.unique() if stable_int("natural_test", s, g) % 5 == 0}
            return c[~c.group.isin(te_g)], c[c.group.isin(te_g)]
        return c, cache, paths.CACHE / "features" / "capsule" / "dinov2_b14_518", split
    raise ValueError(cohort)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="thyroid")
    a = ap.parse_args()
    c, cache, fdir, split = load(a.cohort)
    envs, tests = {}, {}
    for s in SEEDS:
        tv, te = split(s)
        g = tv.group.unique()
        val_g = {x for x in g if stable_int("natural_val", s, x) % 5 == 0}
        va, tr = tv[tv.group.isin(val_g)], tv[~tv.group.isin(val_g)]
        e = {"train_corr": tr, "val_clean": va, "val_groups": va, "train_all": tr, "test_corr": te, "test_rev": te, "clean": te}
        for k, d in e.items():
            envs[("natural", s, 0, k)] = d[COLS].reset_index(drop=True)
    out = paths.ensure(paths.RESULTS / "natural" / f"{a.cohort}_dino518")
    ins = lambda i, rgb, roi: np.asarray(draw_generic_artifact(Image.fromarray(rgb), roi, f"u|{i}"))
    if not (out / "predictions.csv.gz").exists():
        run_spec(envs, cache, "dino518", out, fdir, [], arms=ARMS, traps=("natural",), device=torch.device("cuda"),
                 insert_fn=ins, insert_tag="_generic", folds=[0])
    p = pd.read_csv(out / "predictions.csv.gz")
    p = p[p.env == "clean"].copy()
    amap = dict(zip(c.image_id, c.a))
    p["a_in"] = p.image_id.map(amap)
    rows = []
    for (m, s), q in p.groupby(["method", "seed"]):
        y, pr, ai = q.y.to_numpy(), q.prob.to_numpy(), q.a_in.to_numpy()
        rows.append({"method": m, "seed": s, "auc": safe_auc(y, pr), "cross_group": cross_group(y, ai, pr),
                     "auc_in_roi_artifact": safe_auc(y[ai == 1], pr[ai == 1])})
    m = pd.DataFrame(rows).groupby("method").mean(numeric_only=True).drop(columns="seed").round(3)
    m.to_csv(out / "natural_metrics.csv")
    print(m.sort_values("cross_group", ascending=False).to_string())
    # which pairing conflicts with the natural shortcut: the one where the artifact is rarer in the class it marks
    te_all = c[c.image_id.isin(p.image_id)]
    pa1, pa0 = te_all[te_all.y == 1].a.mean(), te_all[te_all.y == 0].a.mean()
    hard = ((p.y == 1) & (p.a_in == 1)) | ((p.y == 0) & (p.a_in == 0)) if pa1 < pa0 else \
        ((p.y == 1) & (p.a_in == 0)) | ((p.y == 0) & (p.a_in == 1))
    res = {"P(a|y=1)": round(float(pa1), 3), "P(a|y=0)": round(float(pa0), 3)}
    for name, sel in (("hard", hard), ("easy", ~hard), ("all", hard | ~hard)):
        q = p[sel]
        for a1, a0 in (("mask", "erm"), ("mte", "mask"), ("mte_protect", "mask"), ("balanced", "mask"), ("mte_balanced", "mask")):
            r = hierarchical_paired_bootstrap(q, a1, a0, "clean", 10000, 5, fast=True)
            res[f"{name} | {a1}-{a0}"] = [round(r[k], 3) for k in ("seed_delta_mean", "ci95_lo", "ci95_hi")]
            print(name, a1, a0, res[f"{name} | {a1}-{a0}"], flush=True)
    (out / "natural_boot.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
