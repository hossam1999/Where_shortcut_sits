"""E13 (DINOv2@518) and E15 (DermLIP@224) in the author's protocol, compared with the expected numbers.

  python scripts/run_spec_e13.py --counts_only
  python scripts/run_spec_e13.py --backbone dino518
  python scripts/run_spec_e13.py --backbone dermlip224
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from wtss import paths
from wtss.data.isic2019_spec import build_spec_envs, load_spec_cohort, spec_counts
from wtss.experiments.real_traps import RealCache
from wtss.experiments.spec_traps import run_spec
from wtss.stats import difference_of_deltas, hierarchical_paired_bootstrap, slim

BACKBONE_DIR = {"dino518": "dinov2_b14_518", "dermlip224": "dermlip_panderm_224", "dino224": "dinov2_b14_224",
                "dinos518": "dinov2_s14_518", "dinol518": "dinov2_l14_518"}  # s/l: scale test (PREREGISTRATION_SCALE.md)

EXPECTED = {  # docs/REPLICATION_SPEC.md E13 / E15: (trap, arm) -> (clean, rev, delta, lo, hi)
    "dino518": {("trapA", "erm"): (0.782, 0.516, None, None, None), ("trapB", "erm"): (None, 0.510, None, None, None),
                ("trapA", "mask"): (0.769, 0.452, -0.064, -0.079, -0.049), ("trapB", "mask"): (None, None, 0.124, 0.105, 0.143),
                ("trapA", "inpaint"): (0.811, 0.616, 0.101, 0.094, 0.107), ("trapB", "inpaint"): (None, None, 0.108, 0.098, 0.117),
                ("trapA", "balanced"): (0.812, 0.729, 0.213, 0.190, 0.235), ("trapB", "balanced"): (None, None, 0.234, 0.217, 0.252),
                ("trapA", "dfr"): (0.724, 0.718, 0.202, 0.176, 0.228), ("trapB", "dfr"): (None, None, 0.192, 0.168, 0.216),
                ("trapA", "leace_paired"): (0.800, 0.574, 0.058, 0.053, 0.063), ("trapB", "leace_paired"): (None, None, 0.078, 0.070, 0.085),
                ("trapA", "leace_unpaired"): (0.736, 0.831, 0.316, None, None)},
    "dermlip224": {("trapA", "erm"): (0.845, 0.677, None, None, None),
                   ("trapA", "mask"): (0.800, 0.608, -0.069, -0.088, -0.051),
                   ("trapA", "balanced"): (0.858, 0.803, 0.126, 0.115, 0.138),
                   ("trapA", "dfr"): (0.827, 0.792, 0.114, 0.091, 0.138),
                   ("trapA", "leace_paired"): (0.856, 0.708, 0.031, 0.028, 0.034),
                   ("trapA", "leace_unpaired"): (0.764, 0.869, 0.192, None, None)},
}
EXPECTED_COUNTS = {"trapA": dict(A0_Y0=708, A0_Y1=157, A1_Y0=2903, A1_Y1=1144, eligible=4912),
                   "trapB": dict(A0_Y0=708, A0_Y1=157, A1_Y0=4093, A1_Y1=809, eligible=5767)}


def _b(j):
    q, a, env, s = j
    return hierarchical_paired_bootstrap(q, a, "erm", env, 10000, s, fast=True)


def verdict(exp, got, lo, hi, elo, ehi):
    if exp is None or got is None or np.isnan(got):
        return ""
    ok_sign = np.sign(exp) == np.sign(got)
    ok_ci = True if elo is None else ((elo > 0 or ehi < 0) == (lo > 0 or hi < 0))
    ok_pt = abs(exp - got) <= 0.02
    return "MATCH" if ok_sign and ok_ci and ok_pt else ("SIGN+CI match" if ok_sign and ok_ci else "MISMATCH")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dino518")
    ap.add_argument("--counts_only", action="store_true")
    ap.add_argument("--r_col", default="r_spec")
    ap.add_argument("--tag", default="spec")
    ap.add_argument("--generic", action="store_true", help="artifact-agnostic insertion library (U-I2E / U-MtE)")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--lama", action="store_true", help="LaMa-inpainted hair cache (docs/PREREGISTRATION_INPAINT_LAMA.md)")
    ap.add_argument("--text_dirs", default=None, help="npz from scripts/make_text_directions.py (text-prompted arms)")
    ap.add_argument("--save_val", action="store_true", help="also save val_groups predictions (adaptive selection)")
    ap.add_argument("--match", action="store_true", help="covariate-matched trap cells (docs/PREREGISTRATION_REVIEW2.md, R1)")
    a = ap.parse_args()
    if a.match and a.tag == "spec":
        raise SystemExit("the matched variant must use its own --tag (it would overwrite the spec results)")
    c = load_spec_cohort(r_col=a.r_col)
    match_rep = None
    if a.match:
        from wtss.matching import match_traps
        c, match_rep = match_traps("isic", c, strata=("source",))
    envs = build_spec_envs(c)
    cnt = spec_counts(c, envs)
    for t, e in EXPECTED_COUNTS.items():
        for k, v in e.items():
            cnt.loc[cnt.trap == t, f"exp_{k}"] = v
    out = paths.ensure(paths.RESULTS / "spec_e13" / f"{a.backbone}_{a.tag}")
    cnt.to_csv(out / "counts_vs_expected.csv", index=False)
    if match_rep is not None:
        for k, v in match_rep.items():
            v.to_csv(out / f"match_{k}.csv", index=False)
    print(cnt.T.to_string())
    if a.counts_only:
        return
    cache = RealCache(paths.DATA / "isic2019" / "prepared" / ("cache_518_lama" if a.lama else "cache_518"), roi_file="roi_spec.npy")
    donors = c[~c.A0 & (c[a.r_col] >= 0.1) & (c[a.r_col] < 0.5)].image_id.tolist()
    if not (out / "predictions.csv.gz").exists():
        kw = {}
        if a.generic:
            from wtss.synthetic import draw_generic_artifact
            from PIL import Image as _I
            kw = dict(insert_fn=lambda i, rgb, roi: np.asarray(draw_generic_artifact(_I.fromarray(rgb), roi, f"u|{i}")),
                      insert_tag="_generic")
        if a.arms:
            kw["arms"] = tuple(a.arms)
        run_spec(envs, cache, a.backbone, out, paths.CACHE / "features" / ("spec_isic2019_lama" if a.lama else "spec_isic2019") / BACKBONE_DIR[a.backbone], donors,
                 device=torch.device("cuda"), save_val=a.save_val, **kw,
                 extra_ctx={"text_U": np.load(a.text_dirs)["U"]} if a.text_dirs else None)
    preds = pd.read_csv(out / "predictions.csv.gz")
    arms = [m for m in preds.method.unique() if m != "erm"]
    jobs, keys = [], []
    for trap in ("trapA", "trapB"):
        q = preds[preds.trap == trap]
        for arm in arms:
            for env in ("test_rev", "clean"):
                jobs.append((slim(q, env, (arm, "erm")), arm, env, 20260918 + sum(map(ord, arm + trap + env)))); keys.append((trap, arm, env, "all"))
        for s in ("HAM", "BCN"):
            jobs.append((slim(q[q.source == s], "test_rev", ("mask", "erm")), "mask", "test_rev", 7)); keys.append((trap, "mask", "test_rev", s))
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_b, jobs))
    boot = pd.DataFrame([{"trap": t, "arm": m, "env": e, "source": s, **r} for (t, m, e, s), r in zip(keys, res)])
    boot.to_csv(out / "bootstrap_vs_erm.csv", index=False)
    cross = difference_of_deltas(preds[preds.trap == "trapB"], preds[preds.trap == "trapA"], "mask", "erm", "test_rev", 10000, 11)
    per = pd.read_csv(out / "metrics_per_seed.csv")
    auc = per.groupby(["trap", "method", "env"]).auc.mean().unstack()
    rows = []
    for (trap, arm), (ec, er, ed, elo, ehi) in EXPECTED.get(a.backbone, {}).items():
        if (trap, arm) not in auc.index:
            continue
        b = boot[(boot.trap == trap) & (boot.arm == arm) & (boot.env == "test_rev") & (boot.source == "all")]
        d, lo, hi = (b.seed_delta_mean.iloc[0], b.ci95_lo.iloc[0], b.ci95_hi.iloc[0]) if len(b) else (np.nan,) * 3
        rows.append({"trap": trap, "arm": arm, "clean_exp": ec, "clean_got": round(auc.loc[(trap, arm), "clean"], 3),
                     "corr_got": round(auc.loc[(trap, arm), "test_corr"], 3),
                     "rev_exp": er, "rev_got": round(auc.loc[(trap, arm), "test_rev"], 3),
                     "delta_exp": ed, "ci_exp": f"[{elo}, {ehi}]" if elo is not None else "",
                     "delta_got": round(d, 3) if not np.isnan(d) else None, "ci_got": f"[{lo:+.3f}, {hi:+.3f}]" if not np.isnan(d) else "",
                     "verdict": verdict(ed, d, lo, hi, elo, ehi)})
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "COMPARISON.csv", index=False)
    s = boot[(boot.env == "test_rev") & (boot.source != "all")]
    summary = {"crossover_B_minus_A_mask": {k: cross[k] for k in ("seed_delta_mean", "ci95_lo", "ci95_hi", "seed_deltas_json")},
               "source_stratified_trapA_mask": s[s.trap == "trapA"][["source", "seed_delta_mean", "ci95_lo", "ci95_hi"]].to_dict("records")}
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=float))
    print(comp.to_string()); print(json.dumps(summary, indent=1, default=float))
    print(auc.round(3).to_string())


if __name__ == "__main__":
    main()
