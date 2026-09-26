"""Chest radiography real-device traps (docs/PREREGISTRATION_CXR_DEVICE_TRAPS.md).

  python scripts/run_cxr_traps.py --counts_only
  python scripts/run_cxr_traps.py --backbone raddino518
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
from wtss.experiments.spec_traps import run_spec
from wtss.stats import difference_of_deltas, hierarchical_paired_bootstrap, slim

DISEASES = ("Infiltration", "Effusion", "Atelectasis", "Consolidation")
TRAPS = {"trapA": ("CVC", 4), "trapB": ("ETT", 1)}  # trap -> (device, bit in dev.npy)
BDIR = {"raddino518": "raddino_518", "medsiglip448": "medsiglip_448", "dino518": "dinov2_b14_518"}
PREP = paths.DATA / "cxr" / "prepared"


DEVICE_MATCH = False  # follow-up 1 (--device_matched)


def cohort(disease: str) -> pd.DataFrame:
    c = pd.read_csv(PREP / "clip_cohort.csv").rename(columns={"StudyInstanceUID": "image_id"})
    c["y"] = c.findings.str.contains(disease).astype(int)
    c["source"] = c.view
    c["patient"] = c.nih_patient.astype(str)
    c["trapA_A0"] = c.has_CVC == 0
    c["trapA_A1"] = (c.has_CVC == 1) & (c.r_CVC >= 0.5)
    c["trapB_A0"] = c.has_ETT == 0
    c["trapB_A1"] = (c.has_ETT == 1) & (c.r_ETT < 0.1)
    c["A0"] = False
    if DEVICE_MATCH:  # follow-up 1: match on the other devices, so only the trap's device differs
        c["trapA_source"] = c.view + "|ETT" + c.has_ETT.astype(str) + "|NGT" + c.has_NGT.astype(str)
        c["trapB_source"] = c.view + "|CVC" + c.has_CVC.astype(str) + "|NGT" + c.has_NGT.astype(str)
    return c[c.view == "AP"].reset_index(drop=True)  # Amendment 1: AP radiographs only


def donors(c: pd.DataFrame, trap: str):
    if trap == "trapA":
        return c[(c.has_CVC == 1) & c.r_CVC.between(0.1, 0.5, inclusive="left")].image_id.tolist()
    return c[(c.has_ETT == 1) & (c.r_ETT >= 0.1)].image_id.tolist()


def _b(j):
    q, a, env, s = j
    return hierarchical_paired_bootstrap(q, a, "erm", env, 10000, s, fast=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="raddino518")
    ap.add_argument("--diseases", nargs="+", default=list(DISEASES))
    ap.add_argument("--counts_only", action="store_true")
    ap.add_argument("--device_matched", action="store_true", help="follow-up 1")
    a = ap.parse_args()
    global DEVICE_MATCH
    DEVICE_MATCH = a.device_matched
    out_root = paths.ensure(paths.RESULTS / "cxr_traps" / (a.backbone + ("_devmatched" if a.device_matched else "")))
    rows = []
    for dis in a.diseases:
        c = cohort(dis)
        for trap in TRAPS:
            pool = matched_pool(c, trap)
            cells = pool.groupby(["a", "y"]).size()
            rows.append({"disease": dis, "trap": trap, **{f"A{x}_Y{y}": int(cells.get((x, y), 0)) for x in (0, 1) for y in (0, 1)},
                         "rev_pos_est": int(cells.get((0, 1), 0) / 0.9)})
    cnt = pd.DataFrame(rows)
    cnt.to_csv(out_root / "counts.csv", index=False)
    print(cnt.to_string())
    if a.counts_only:
        return
    for dis in a.diseases:
        out = paths.ensure(out_root / dis)
        if (out / "bootstrap_vs_erm.csv").exists():
            continue
        c = cohort(dis)
        ok = cnt[(cnt.disease == dis)].set_index("trap")
        traps_ok = tuple(t for t in TRAPS if min(ok.loc[t, ["A0_Y0", "A0_Y1", "A1_Y0", "A1_Y1"]]) >= 25)
        for t in set(TRAPS) - set(traps_ok):
            print(f"[gate] {dis} {t}: a matched cell < 25 -> underpowered, not fitted", flush=True)
        if not traps_ok:
            continue
        envs = build_spec_envs(c, traps=traps_ok, group_col="patient")
        frames = []
        for trap, (dev, bit) in [(t, TRAPS[t]) for t in traps_ok]:
            cache = RealCache(PREP / "cache_clip_518", roi_file="roi.npy", art_file="dev.npy", art_bit=bit)
            sub = paths.ensure(out / trap)
            run_spec(envs, cache, a.backbone, sub, paths.CACHE / "features" / "cxr_clip" / trap / BDIR[a.backbone],
                     donors(c, trap), traps=(trap,), device=torch.device("cuda"), extra_meta={"disease": dis})
            frames.append(pd.read_csv(sub / "predictions.csv.gz"))
        preds = pd.concat(frames, ignore_index=True)
        preds.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
        arms = [m for m in preds.method.unique() if m != "erm"]
        jobs, keys = [], []
        for trap in traps_ok:
            q = preds[preds.trap == trap]
            for arm in arms:
                for env in ("test_rev", "clean"):
                    jobs.append((slim(q, env, (arm, "erm")), arm, env, 20260918 + sum(map(ord, arm + trap + env + dis))))
                    keys.append((trap, arm, env))
        with ProcessPoolExecutor(4) as ex:
            res = list(ex.map(_b, jobs))
        boot = pd.DataFrame([{"disease": dis, "trap": t, "arm": m, "env": e, **r} for (t, m, e), r in zip(keys, res)])
        boot.to_csv(out / "bootstrap_vs_erm.csv", index=False)
        cr = {"seed_delta_mean": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan")}
        if len(traps_ok) == 2:
            cr = difference_of_deltas(preds[preds.trap == "trapB"], preds[preds.trap == "trapA"], "mask", "erm", "test_rev", 10000, 11)
        (out / "X3_crossover.json").write_text(json.dumps(cr, indent=2, default=float))
        from wtss.stats import safe_auc
        auc = preds.groupby(["trap", "method", "env", "seed"]).apply(lambda q: safe_auc(q.y, q.prob), include_groups=False)
        auc.rename("auc").reset_index().to_csv(out / "metrics_per_seed.csv", index=False)
        s = boot[boot.env == "test_rev"][["trap", "arm", "seed_delta_mean", "ci95_lo", "ci95_hi"]]
        print(f"== {dis}\n", s.round(3).to_string(), "\ncrossover", {k: round(cr[k], 3) for k in ("seed_delta_mean", "ci95_lo", "ci95_hi")}, flush=True)


if __name__ == "__main__":
    main()
