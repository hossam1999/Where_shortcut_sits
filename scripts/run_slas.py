"""SLAS on the E13 spec traps (docs/PREREGISTRATION_SLAS.md).

One DINOv2@518 forward per image; four pooled patch-token views (+ oracle ceiling):
  tok_all (ERM analogue) | tok_roi (masking analogue) | tok_clean (SLAS) | tok_roiclean (mask-then-SLAS)
  tok_roiclean_oracle    (ROI patches minus Wegley-mask patches; ceiling, uses the real mask at test time)
The probe is fitted on k annotated DONOR images (0.1 <= r < 0.5; never in either trap pool).

  python scripts/run_slas.py --k 50
  python scripts/run_slas.py --k 50 --cohort thyroid     # same code, ultrasound markers
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wtss import paths
from wtss.backbones import load_dino
from wtss.data.isic2019_spec import build_spec_envs, load_spec_cohort
from wtss.experiments import spec_traps
from wtss.experiments.real_traps import RealCache
from wtss.methods.token_suppression import fit_probe, pooled_views
from wtss.stats import hierarchical_paired_bootstrap, safe_auc, slim

VIEWS = ("tok_all", "tok_roi", "tok_clean", "tok_roiclean", "tok_roiclean_oracle")
ARMS = [("tok_erm", "tok_all", False), ("tok_mask", "tok_roi", False), ("slas", "tok_clean", False),
        ("mts", "tok_roiclean", False), ("mts_oracle", "tok_roiclean_oracle", False),
        ("tok_balanced", "tok_all", True), ("mts_balanced", "tok_roiclean", True)]
COMPARISONS = [("tok_mask", "tok_erm"), ("slas", "tok_erm"), ("mts", "tok_mask"), ("mts", "tok_erm"),
               ("mts_oracle", "tok_mask"), ("mts_balanced", "tok_balanced"), ("tok_balanced", "tok_erm")]


def cohort(name):
    if name == "isic":
        c = load_spec_cohort()
        cache = RealCache(paths.DATA / "isic2019" / "prepared" / "cache_518", roi_file="roi_spec.npy", art_file="hair.npy")
        donors = c[~c.A0 & (c.r_spec >= 0.1) & (c.r_spec < 0.5) & (c.has_hair_mask == True) & (c.hair_frac >= 0.02)]
        return c, build_spec_envs(c), cache, donors.image_id.to_numpy()
    import importlib.util
    spec = importlib.util.spec_from_file_location("rt", Path(__file__).resolve().parent / "run_thyroid_traps.py")
    rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
    c = rt.cohort()
    cache = RealCache(rt.T / "cache_518", roi_file="roi.npy", art_file="marker.npy")
    donors = c[(c.marker_px >= 15) & c.r.between(0.1, 0.5, inclusive="left")].image_id.to_numpy()
    return c, build_spec_envs(c, group_col="group"), cache, donors


def _b(j):
    q, a1, a0, env, s = j
    return hierarchical_paired_bootstrap(q, a1, a0, env, 10000, s, fast=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="isic", choices=("isic", "thyroid"))
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--tau", type=float, default=0.5)
    a = ap.parse_args()
    dev = torch.device("cuda")
    c, envs, cache, donors = cohort(a.cohort)
    out = paths.ensure(paths.RESULTS / "slas" / f"{a.cohort}_k{a.k}")
    fdir = paths.ensure(paths.CACHE / "features" / "slas" / f"{a.cohort}_k{a.k}")
    pool = sorted(set().union(*[set(d.image_id) for d in envs.values()]))
    assert not set(pool) & set(donors), "probe images must be outside the trap pools"
    pg = set(c.set_index("image_id").loc[pool, "group"])
    donors = np.array([d for d in donors if c.set_index("image_id").at[d, "group"] not in pg])  # no shared leakage group
    if not (fdir / "views.npz").exists():
        bk = load_dino(518, dev)
        rng = np.random.default_rng(20260927)
        pids = rng.choice(donors, min(a.k, len(donors)), replace=False)
        real = lambda i: (Image.fromarray(cache.get(i)[0]), (cache.get(i)[2] > 0).astype(np.uint8))
        probe = fit_probe(bk.model, dev, pids, real, bk.preprocess, n_tokens=max(30000, 150 * a.k), seed=a.k)
        plain = lambda i: Image.fromarray(cache.get(i)[0])
        roi = lambda i: cache.get(i)[1] > 0
        art = lambda i: (cache.get(i)[2] > 0).astype(np.uint8)
        v = pooled_views(bk.model, dev, pool, plain, bk.preprocess, roi, probe, tau=a.tau, art_fn=art)
        np.savez(fdir / "loc.npz", score=v["loc_score"], cover=v["loc_cover"], roi=v["loc_roi"])
        np.savez(fdir / "views.npz", ids=np.array(pool), probe_w=probe[0], probe_b=probe[1], probe_ids=pids,
                 **{k: v[k] for k in VIEWS})
        del bk; torch.cuda.empty_cache()
    z = np.load(fdir / "views.npz")
    assert list(z["ids"]) == pool
    spec_traps._CTX = dict(V={**{k: z[k] for k in VIEWS}, "erm": z["tok_all"]}, pos={k: j for j, k in enumerate(pool)}, envs=envs, arms=(),
                           backend_name="dinov2_b14_518_tok", extra_meta={"k": a.k}, view_arms=ARMS)
    seeds = sorted({k[1] for k in envs})
    jobs = [(t, s, f) for t in ("trapA", "trapB") for s in seeds for f in range(5)]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(4) as p:
        frames = [f for fr in p.imap(spec_traps._fold_job, jobs) for f in fr]
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    auc = preds.groupby(["trap", "method", "env", "seed"]).apply(lambda q: safe_auc(q.y, q.prob), include_groups=False)
    auc.rename("auc").reset_index().to_csv(out / "metrics_per_seed.csv", index=False)
    jobs, keys = [], []
    for trap in ("trapA", "trapB"):
        q = preds[preds.trap == trap]
        for a1, a0 in COMPARISONS:
            for env in ("test_rev", "clean"):
                jobs.append((slim(q, env, (a1, a0)), a1, a0, env, 20260927 + sum(map(ord, a1 + a0 + trap + env))))
                keys.append((trap, a1, a0, env))
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_b, jobs))
    boot = pd.DataFrame([{"trap": t, "arm": a1, "ref": a0, "env": e, **r} for (t, a1, a0, e), r in zip(keys, res)])
    boot.to_csv(out / "bootstrap.csv", index=False)
    print(boot[["trap", "arm", "ref", "env", "seed_delta_mean", "ci95_lo", "ci95_hi"]].round(3).to_string())
    print(auc.groupby(["trap", "method", "env"]).mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
