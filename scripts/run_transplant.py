"""Real-artifact transplant experiment (docs/PREREGISTRATION_REVIEW2.md, R2).

The same real artifact instance is pasted inside (overlap >= 0.95) or outside (overlap 0) the ROI of the same
artifact-free image; controlled-sweep environments (0.9/0.1 train and correlated test, 0.1/0.9 reversed test, clean
without artifact); 5-fold grouped CV, fold predictions pooled per presence seed (seed = bootstrap cluster).

  python scripts/run_transplant.py --cohort thyroid        # thyroid | ovary | capsule | isic
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from wtss import paths
from wtss.data.isic2018 import Cohort
from wtss.data.isic2019_spec import load_spec_cohort, pool_groups
from wtss.experiments.real_traps import RealCache
from wtss.experiments.synthetic import SynthConfig, analyse_synthetic, run_synthetic
from wtss.stats import safe_auc
from wtss.transplant import extract_instance, make_drawer, neutral_instance, place

spec = importlib.util.spec_from_file_location("rt", Path(__file__).with_name("run_thyroid_traps.py"))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
SEEDS = (42, 123, 456, 789, 2026)
ARMS = ("erm", "mask", "inpaint", "balanced", "dfr")
FOLD_SEED = 20260928


def setup(name: str):
    """(recipients df[image_id, y, group], donor ids, RealCache, artifact kind, ROI name)."""
    if name in ("thyroid", "ovary"):
        c = rt.cohort() if name == "thyroid" else rt.ovary_cohort()
        cache = RealCache((rt.T if name == "thyroid" else rt.OV) / "cache_518", roi_file="roi.npy", art_file="marker.npy")
        rec, don, kind, roi = c[c.marker_px == 0], c[c.marker_px >= 15], "caliper", "nodule" if name == "thyroid" else "tumour"
    elif name == "capsule":
        c = rt.capsule_cohort()
        cache = RealCache(rt.CAP / "cache_518", roi_file="roi.npy", art_file="contam.npy")
        rec, don, kind, roi = c[c.contam_frac < 0.03], c[c.contam_frac >= 0.10], "debris", "lesion"
    elif name == "isic":
        c = load_spec_cohort()
        cache = RealCache(paths.DATA / "isic2019" / "prepared" / "cache_518", roi_file="roi_spec.npy", art_file="hair.npy")
        rec, don, kind, roi = c[c.A0].copy(), c[~c.A0], "hair", "lesion"
        rec["group"] = pool_groups(rec.reset_index(drop=True), 8)
    else:
        raise ValueError(name)
    rec = rec[["image_id", "y", "group"]].astype({"image_id": str, "group": str}).sort_values("image_id").reset_index(drop=True)
    return rec, sorted(don.image_id.astype(str)), cache, kind, roi


_G: dict = {}


def _inst(d):
    if d not in _G["memo"]:
        rgb, _, art = _G["cache"].get(d)
        _G["memo"][d] = extract_instance(art, rgb, _G["kind"])
    return _G["memo"][d]


class _Lazy(dict):
    def __missing__(self, d):
        return _inst(d)


def _place(i):
    rgb, roi, _ = _G["cache"].get(i)
    valid = _G["valid"]
    return i, place(i, rgb, roi, valid, _Lazy())


def placements(name, rec, donors, cache, kind, pf: Path):
    if pf.exists():
        return json.loads(pf.read_text())
    _G.update(cache=cache, kind=kind, memo={})
    # donor validity (instance within the size limits) — outcome-free, computed once
    with mp.get_context("fork").Pool(32) as pool:
        ok = pool.map(_valid_one, donors, chunksize=16)
    _G["valid"] = [d for d, v in zip(donors, ok) if v]
    print(f"[transplant] {name}: {len(_G['valid'])}/{len(donors)} donors have a usable instance", flush=True)
    with mp.get_context("fork").Pool(32) as pool:
        res = dict(pool.map(_place, rec.image_id.tolist(), chunksize=4))
    pl = {i: p for i, p in res.items() if p is not None}
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(json.dumps(pl))
    return pl


def _valid_one(d):
    return _inst(d) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=("thyroid", "ovary", "capsule", "isic"))
    ap.add_argument("--backbone", default="dino518")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--placements_only", action="store_true")
    ap.add_argument("--neutral", action="store_true",
                    help="paste-edge control: same masks/positions/compositing, neutral tissue (PREREGISTRATION_REVIEW3 R5)")
    a = ap.parse_args()
    rec, donors, cache, kind, roi_name = setup(a.cohort)
    wd = paths.ensure(paths.DATA / "review2")
    pl = placements(a.cohort, rec, donors, cache, kind, wd / f"transplant_{a.cohort}_placements.json")
    out = paths.ensure(paths.RESULTS / "review2" / "transplant" / (f"{a.cohort}_{a.backbone}" + ("_neutral" if a.neutral else "")))
    rec["feasible"] = rec.image_id.isin(pl)
    cnt = rec.groupby(["y", "feasible"]).size().unstack(fill_value=0)
    cnt.to_csv(out / "feasibility_by_label.csv")
    print(cnt.to_string(), flush=True)
    d = rec[rec.feasible].drop(columns="feasible").reset_index(drop=True)
    used = sorted({p["donor"] for p in pl.values()})
    inst = {k: extract_instance(cache.get(k)[2], cache.get(k)[0], kind) for k in used}
    if a.neutral:  # per recipient: the donor's mask filled with neutral tissue (keyed by recipient -> own instance)
        free = rec.image_id.tolist()
        fb = [cache.get(free[j])[0] for j in np.random.default_rng(20260928).choice(len(free), 20, replace=False)]
        ninst, src = {}, []
        for i, p in pl.items():
            dn = p["donor"]
            drgb, _, dart = cache.get(dn)
            ninst[i] = neutral_instance(inst[dn], drgb, dart, f"{i}|{dn}", fb)
            src.append(ninst[i]["source"])
        pl = {i: {**p, "donor": i, "orig_donor": p["donor"]} for i, p in pl.items()}  # drawer looks up by "donor"
        inst = ninst
        pd.Series(src).value_counts().to_csv(out / "neutral_source_counts.csv")
    pd.DataFrame([{"image_id": i, **{k: v for k, v in p.items() if k in ("donor", "op", "n_px")},
                   "ov_in": p["1.00"]["achieved"], "ov_out": p["0.00"]["achieved"]} for i, p in pl.items()]
                 ).to_csv(out / "placements.csv", index=False)
    (out / "design.json").write_text(json.dumps({"recipients_feasible": len(d), "recipients_total": len(rec),
                                                  "donors_used": len(used), "donors_total": len(donors),
                                                  "n_pos": int(d.y.sum()), "kind": kind}, indent=1))
    if a.placements_only:
        return
    fold = np.zeros(len(d), int)
    sg = StratifiedGroupKFold(5, shuffle=True, random_state=FOLD_SEED)
    for k, (_, te) in enumerate(sg.split(d, d.y, d.group)):
        fold[te] = k
    loaders = (lambda i: cache.get(i)[0], lambda i: cache.get(i)[1])
    frames, metrics = [], []
    for k in range(5):
        fdir = out / f"fold{k}"
        if not (fdir / "predictions.csv.gz").exists():
            df = d.assign(split=np.where(fold == k, "test", np.where(fold == (k + 1) % 5, "val", "train")))
            cfg = SynthConfig(phase="corr", overlaps=(0.0, 1.0), seeds=SEEDS, arms=a.arms,
                              artifact="transplant_neutral" if a.neutral else "transplant",
                              drawer=make_drawer(pl, inst), workers=8, batch_size=64)
            run_synthetic(Cohort(f"{a.cohort}_transplant", df, roi_name), a.backbone, pl, fdir, cfg,
                          paths.CACHE / "images", loaders, paths.CACHE / "features", torch.device("cuda"))
        frames.append(pd.read_csv(fdir / "predictions.csv.gz").assign(fold=k))
        metrics.append(pd.read_csv(fdir / "metrics.csv").assign(fold=k))
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    pd.concat(metrics, ignore_index=True).to_csv(out / "metrics.csv", index=False)
    analyse_synthetic(out, n_boot=10000)
    # pooled-per-seed AUROC (every recipient is a test image once per seed)
    auc = (preds.groupby(["method", "overlap", "env", "seed"]).apply(lambda q: safe_auc(q.y, q.prob), include_groups=False)
           .rename("auc").reset_index())
    auc.to_csv(out / "auc_per_seed.csv", index=False)
    print(auc.groupby(["method", "overlap", "env"]).auc.mean().unstack().round(3).to_string())
    li = pd.read_csv(out / "location_interaction.csv")
    print(li[["method", "seed_delta_mean", "ci95_lo", "ci95_hi"]].round(3).to_string())


if __name__ == "__main__":
    main()
