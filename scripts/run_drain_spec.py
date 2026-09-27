"""Real chest-drain trap (NIH pneumothorax; drains lie inside the lung ROI) with the current arm set
(docs/PREREGISTRATION_CXR_DRAIN.md). The 5 grouped folds of wtss.data.cxr_drain act as bootstrap clusters.

  python scripts/run_drain_spec.py --backbone raddino518
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wtss import paths
from wtss.data.cxr_drain import build_drain_envs, load_drain_cohort
from wtss.experiments.real_traps import RealCache
from wtss.experiments.spec_traps import run_spec
from wtss.stats import hierarchical_paired_bootstrap, safe_auc, slim
from wtss.synthetic import draw_generic_artifact

BDIR = {"raddino518": "raddino_518", "dino518": "dinov2_b14_518", "medsiglip448": "medsiglip_448"}
ARMS = ("erm", "mask", "balanced", "dfr", "mte", "mte_balanced", "mte_aug", "mte_protect", "mte_protect_balanced",
        "mask_dfr", "jtt", "mask_jtt")
COMP = [("mask", "erm"), ("mte", "mask"), ("mte", "mte_aug"), ("mte_protect", "mask"), ("mte_balanced", "balanced"),
        ("mask_dfr", "dfr"), ("jtt", "erm"), ("mte", "jtt")]


def _b(j):
    q, a1, a0, env, s = j
    return hierarchical_paired_bootstrap(q, a1, a0, env, 10000, s, fast=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="raddino518")
    ap.add_argument("--tag", default="universal")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--save_val", action="store_true", help="also save val_groups predictions (adaptive selection)")
    a = ap.parse_args()
    df = load_drain_cohort()
    envs0, counts = build_drain_envs(df)
    ren = {"clean_test": "clean", "clean_val": "val_clean"}
    envs = {}
    for (t, k, e), d in envs0.items():  # fold k -> cluster "seed" k with a single fold 0
        d = d.copy()
        if "source" not in d:
            d["source"] = "NIH"
        envs[("trapA", k, 0, ren.get(e, e))] = d
    out = paths.ensure(paths.RESULTS / "cxr_drain" / f"{a.backbone}_{a.tag}")
    counts.to_csv(out / "counts.csv", index=False)
    cache = RealCache(paths.DATA / "cxr" / "prepared" / "cache_nih_drain_518")
    ins = lambda i, rgb, roi: np.asarray(draw_generic_artifact(Image.fromarray(rgb), roi, f"u|{i}"))
    if not (out / "predictions.csv.gz").exists():
        run_spec(envs, cache, a.backbone, out, paths.CACHE / "features" / "cxr_drain" / BDIR[a.backbone], [],
                 arms=tuple(a.arms) if a.arms else ARMS, traps=("trapA",), device=torch.device("cuda"), insert_fn=ins, insert_tag="_generic",
                 folds=[0], save_val=a.save_val)
    preds = pd.read_csv(out / "predictions.csv.gz")
    from concurrent.futures import ProcessPoolExecutor
    jobs, keys = [], []
    have = set(preds.method.unique())
    for a1, a0 in [c for c in COMP + [("mte_balanced", "mask_splice"), ("mte_protect", "mask_splice"), ("mask_splice", "mask")]
                   if c[0] in have and c[1] in have]:
        for env in ("test_rev", "clean"):
            jobs.append((slim(preds, env, (a1, a0)), a1, a0, env, 20260927 + sum(map(ord, a1 + a0 + env))))
            keys.append((a1, a0, env))
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_b, jobs))
    boot = pd.DataFrame([{"arm": a1, "ref": a0, "env": e, **{k: r[k] for k in ("seed_delta_mean", "ci95_lo", "ci95_hi")}}
                         for (a1, a0, e), r in zip(keys, res)])
    boot.to_csv(out / "paired_deltas.csv", index=False)
    auc = preds.groupby(["method", "env", "seed"]).apply(lambda q: safe_auc(q.y, q.prob), include_groups=False)
    auc.rename("auc").reset_index().to_csv(out / "metrics_per_seed.csv", index=False)
    print(boot.round(3).to_string())
    print(auc.groupby(["method", "env"]).mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
