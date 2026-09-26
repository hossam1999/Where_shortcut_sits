"""Real-artifact traps: ISIC 2019 hair (Trap A in-ROI, Trap B out-of-ROI) and NIH chest drains.

  python scripts/run_traps.py --cohort isic2019 --backbone dino518
  python scripts/run_traps.py --cohort isic2019 --backbone dermlip224
  python scripts/run_traps.py --cohort isic2019 --counts_only     # power check before any model
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from wtss import paths
from wtss.experiments.real_traps import RealCache, TrapConfig, analyse_traps, run_traps


def isic2019(args):
    from wtss.data.isic2019 import build_trap_envs, load_cohort, trap_count_table

    df = load_cohort(518, exclude_vignetting=args.exclude_vignetting)
    if getattr(args, "matched", False):
        from wtss.data.isic2019 import add_match_strata

        df = add_match_strata(df)
        envs = build_trap_envs(df, match_col="stratum")
    else:
        envs = build_trap_envs(df)
    counts = trap_count_table(envs)
    donors = df[df.group_A == "donor"].image_id.tolist()
    cache = RealCache(paths.DATA / "isic2019" / "prepared" / "cache_518")
    return df, envs, counts, donors, cache, None


def nih_drain(args):
    from wtss.data.cxr_drain import build_drain_envs, drain_insert_fn, load_drain_cohort

    df = load_drain_cohort()
    envs, counts = build_drain_envs(df)
    cache = RealCache(paths.DATA / "cxr" / "prepared" / "cache_nih_drain_518")
    return df, envs, counts, [], cache, drain_insert_fn()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="isic2019", choices=["isic2019", "nih_drain"])
    ap.add_argument("--backbone", default="dino518")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--traps", nargs="*", default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--exclude_vignetting", action="store_true")
    ap.add_argument("--matched", action="store_true", help="metadata-matched follow-up (docs/PRECOMMIT_MATCHED_TRAPS.md)")
    ap.add_argument("--counts_only", action="store_true")
    ap.add_argument("--analyse_only", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    df, envs, counts, donors, cache, insert_fn = {"isic2019": isic2019, "nih_drain": nih_drain}[a.cohort](a)
    out = paths.RESULTS / "real" / a.cohort / f"{a.backbone}_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = open(out / ".lock", "w")
    try:  # one writer per experiment directory (queues may overlap)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[traps] {out} is being written by another process; exiting")
        return
    counts.to_csv(out / "trap_counts.csv", index=False)
    agg = counts.groupby(["trap", "env"]).agg(n=("n", "sum"), n_mel=("n_mel", "sum"), n_art=("n_art", "sum")).reset_index()
    print(agg.to_string())
    if a.counts_only:
        return
    cfg = TrapConfig(workers=a.workers)
    if a.arms:
        cfg.arms = a.arms
    if a.traps:
        cfg.traps = a.traps
    elif a.cohort == "nih_drain":
        cfg.traps = ("drain",)
    if a.cohort == "nih_drain" and not a.arms:
        # no drain pixel masks exist: removal-based arms (oracle inpaint, removal-paired LEACE) are undefined
        cfg.arms = tuple(x for x in cfg.arms if x not in ("inpaint", "leace_paired"))
    if not a.analyse_only:
        run_traps(a.cohort, df, envs, cache, a.backbone, out, paths.CACHE / "features" / f"real_{a.cohort}", cfg,
                  donors, insert_fn, torch.device("cuda"))
    res = analyse_traps(out, sources=("HAM", "BCN") if a.cohort == "isic2019" else ())
    b = res["boot"]
    print(b[(b.env == "test_rev") & (b.source == "all")][["trap", "arm", "seed_delta_mean", "ci95_lo", "ci95_hi"]].round(3).to_string())
    if len(res["cross"]):
        print(res["cross"][["arm", "seed_delta_mean", "ci95_lo", "ci95_hi"]].round(3).to_string())


if __name__ == "__main__":
    main()
