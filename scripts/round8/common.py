"""Shared pieces of round 8 (docs/ROUND8_DESIGN.md; later docs/PREREGISTRATION_ROUND8.md).

`joint_replicates` is the crossed seed x image bootstrap of wtss.stats_crossed, generalised so that one set of
replicate weights (one multinomial draw over clusters and one Poisson(1) weight per test image, shared by every arm,
environment and subset) yields the cluster-weighted mean AUROC of many keys at once. Any contrast between keys
(arm - mask, min(rev, corr) of an arm minus that of masking) is then a function of the same replicates, which is what
wtss.stats_crossed does for a single contrast.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Dict, Hashable, List, Tuple

import numpy as np
import pandas as pd

from wtss import paths
from wtss.stats import _WeightedAUC, binary_auc

ROOT = paths.REPO_ROOT
OUT = ROOT / "results" / "round8"
SMOKE_OUT = OUT / "_smoke"
FEAT = paths.CACHE / "features" / "round8"
RERUN = ROOT / "results" / "rerun_2026-09-28"
N_BOOT = 10000


def out_dir(smoke: bool) -> Path:
    d = SMOKE_OUT if smoke else OUT
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(**kw):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **kw}, default=str), flush=True)


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def load_round4_common():
    spec = importlib.util.spec_from_file_location("round4_common", ROOT / "scripts" / "round4" / "common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, k in enumerate(o):
        run = max(run, (len(p) - r) * p[k])
        adj[k] = min(run, 1.0)
    return adj


# terms: key -> list of (cluster, image_ids, y, scores)
Terms = Dict[Hashable, List[Tuple[Hashable, list, np.ndarray, np.ndarray]]]


def joint_replicates(terms: Terms, n_boot: int, seed: int, chunk: int = 250):
    """Cluster-weighted mean AUROC of every key under shared crossed replicates.

    Returns (point, reps): point[key] = mean over clusters of the AUROC; reps[key] = array (n_boot,) with, per
    replicate, sum_c w_c AUROC_c / sum_c w_c over the clusters c that the key has (w_c multinomial cluster counts;
    image weights Poisson(1), one per image identifier, shared across keys and clusters). NaN where a replicate
    leaves a key without a valid cluster.
    """
    ids = sorted({str(i) for lst in terms.values() for t in lst for i in t[1]})
    pos = {k: j for j, k in enumerate(ids)}
    clusters = sorted({t[0] for lst in terms.values() for t in lst}, key=str)
    cpos = {c: j for j, c in enumerate(clusters)}
    prep, point = {}, {}
    for key, lst in terms.items():
        prep[key] = []
        pts = []
        for c, im, y, s in lst:
            y, s = np.asarray(y), np.asarray(s, float)
            if len(np.unique(y)) < 2:
                continue
            prep[key].append((cpos[c], np.array([pos[str(i)] for i in im]), _WeightedAUC(y, s)))
            pts.append(binary_auc(y, s))
        point[key] = float(np.mean(pts)) if pts else float("nan")
    rng = np.random.default_rng(seed)
    reps = {k: [] for k in terms}
    for b0 in range(0, n_boot, chunk):
        B = min(chunk, n_boot - b0)
        W = rng.poisson(1.0, size=(B, len(ids))).astype(np.float64)
        cw = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=B).astype(float)
        for key, lst in prep.items():
            num, den = np.zeros(B), np.zeros(B)
            for ci, idx, f in lst:
                v = f(W[:, idx])
                ok = np.isfinite(v)
                num += np.where(ok, v, 0) * cw[:, ci]
                den += np.where(ok, cw[:, ci], 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                reps[key].append(num / den)
    return point, {k: np.concatenate(v) if v else np.array([]) for k, v in reps.items()}


def ci(arr: np.ndarray) -> Tuple[float, float]:
    a = arr[np.isfinite(arr)]
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def p_two_sided(arr: np.ndarray) -> float:
    a = arr[np.isfinite(arr)]
    if len(a) == 0:
        return float("nan")
    p = 2 * min((a <= 0).mean(), (a >= 0).mean())
    return float(min(1.0, max(p, 1 / len(a))))


def p_greater(arr: np.ndarray, margin: float = 0.0) -> float:
    """One-sided bootstrap p for H1: statistic > margin (floored at 1/n)."""
    a = arr[np.isfinite(arr)]
    if len(a) == 0:
        return float("nan")
    return float(max((a <= margin).mean(), 1 / len(a)))


def terms_from_predictions(p: pd.DataFrame, key_cols: List[str], cluster_col: str = "seed") -> Terms:
    """Group a predictions frame into bootstrap terms, one term per (key, cluster); folds are pooled per cluster."""
    out: Terms = {}
    for key, q in p.groupby(key_cols + [cluster_col], sort=True):
        *k, c = key if isinstance(key, tuple) else (key,)
        k = tuple(k) if len(k) > 1 else k[0]
        out.setdefault(k, []).append((c, q.image_id.astype(str).tolist(), q.y.to_numpy(), q.prob.to_numpy()))
    return out
