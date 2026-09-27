"""End-to-end fine-tuning check on the thyroid / capsule traps (docs/PREREGISTRATION_FINETUNE.md).

Seed 42, folds 0-4 of the spec protocol act as 5 bootstrap clusters. ResNet-50 (ImageNet), 224 px, 8 epochs.
  python scripts/run_finetune_spec.py --cohort thyroid
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wtss import paths
from wtss.data.isic2019_spec import build_spec_envs
from wtss.evaluation import evaluate
from wtss.experiments.finetune import train_eval
from wtss.experiments.real_traps import RealCache, make_renderers
from wtss.stats import hierarchical_paired_bootstrap, slim
from wtss.synthetic import draw_generic_artifact

spec = importlib.util.spec_from_file_location("rt", Path(__file__).with_name("run_thyroid_traps.py"))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
COMP = [("mask", "erm"), ("mte_ft", "mask"), ("mask_balanced", "balanced"), ("mte_ft", "erm")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="thyroid", choices=("thyroid", "capsule"))
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--arms", nargs="+", default=["erm", "mask", "balanced", "mask_balanced", "mte_ft"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    if a.cohort == "thyroid":
        c = rt.cohort(); cache = RealCache(rt.T / "cache_518", roi_file="roi.npy", art_file="marker.npy")
    else:
        c = rt.capsule_cohort(); cache = RealCache(rt.CAP / "cache_518", roi_file="roi.npy", art_file="contam.npy")
    envs = build_spec_envs(c, seeds=(42,), group_col="group")
    ins = lambda i, rgb, roi: np.asarray(draw_generic_artifact(Image.fromarray(rgb), roi, f"u|{i}"))
    render = make_renderers(cache, 518, [], ins)
    out = paths.ensure(paths.RESULTS / "finetune" / a.cohort / a.arch)
    pf = out / "predictions.csv.gz"
    done = pd.read_csv(pf) if pf.exists() else None
    frames = [] if done is None else [done]
    for trap in ("trapA", "trapB"):
        for k in a.folds:
            E = {e: envs[(trap, 42, k, e)] for e in ("train_corr", "test_corr", "test_rev")}
            E["clean_test"], E["clean_val"] = envs[(trap, 42, k, "clean")], envs[(trap, 42, k, "val_clean")]
            for arm in a.arms:
                if done is not None and ((done.trap == trap) & (done.seed == k) & (done.method == arm)).any():
                    continue
                res = train_eval(arm, E, render, arch=a.arch, epochs=a.epochs, seed=k, device=torch.device("cuda"), workers=6)
                meta = {"cohort": a.cohort, "backbone": f"ft_{a.arch}", "trap": trap, "seed": k, "method": arm}
                msg = []
                for env, (clf, thr, d) in res.items():
                    r, f = evaluate(clf, thr, None, d.y.to_numpy(), d.image_id.to_numpy(), d.a.to_numpy(),
                                    {**meta, "env": "clean" if env == "clean_test" else env})
                    frames.append(f); msg.append(f"{r['env']}={r['auc']:.3f}")
                print(f"[ft] {a.cohort} {trap} fold {k} {arm}: " + " ".join(msg), flush=True)
                pd.concat(frames, ignore_index=True).to_csv(pf, index=False, compression="gzip")
    preds = pd.read_csv(pf)
    rows = []
    for trap in ("trapA", "trapB"):
        q = preds[preds.trap == trap]
        for a1, a0 in COMP:
            if {a1, a0} <= set(q.method):
                r = hierarchical_paired_bootstrap(slim(q, "test_rev", (a1, a0)), a1, a0, "test_rev", 10000, 7, fast=True)
                rows.append({"trap": trap, "arm": a1, "ref": a0, **{k: r[k] for k in ("seed_delta_mean", "ci95_lo", "ci95_hi")}})
    b = pd.DataFrame(rows); b.to_csv(out / "paired_deltas.csv", index=False)
    print(b.round(3).to_string())


if __name__ == "__main__":
    main()
