"""Crossed (cluster x image) bootstrap versions of every estimator in wtss.stats.

Selected with the environment variable WTSS_BOOTSTRAP=crossed (docs/PREREGISTRATION_REVIEW3.md, R6;
docs/PREREGISTRATION_FINAL.md). Each replicate resamples clusters (training seeds or folds) with replacement and draws
ONE Poisson(1) weight per test-image identifier, shared by every term and every cluster in which that image appears.
The original estimator resampled images independently within each sampled cluster, which understates the test-set
sampling variance whenever clusters share test images (a fixed test split scored by several seeds).

Return dictionaries carry the same keys as the functions they replace, plus "estimator".
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd

ESTIMATOR = "crossed cluster x image (Poisson) bootstrap"


def _stats():
    from . import stats  # late import: stats imports this module lazily
    return stats


def replicates(terms: List[tuple], n_boot: int, seed: int, chunk: int = 250):
    """terms: (cluster, coef, image_ids, y, scores); statistic per cluster = sum coef * AUROC.
    Returns (per-cluster point estimates dict, replicate array)."""
    st = _stats()
    rng = np.random.default_rng(seed)
    ids = sorted({str(i) for t in terms for i in t[2]})
    pos = {k: j for j, k in enumerate(ids)}
    clusters = sorted({t[0] for t in terms})
    prep = {c: [] for c in clusters}
    point = {c: 0.0 for c in clusters}
    for c, coef, im, y, s in terms:
        y, s = np.asarray(y), np.asarray(s, float)
        prep[c].append((coef, np.array([pos[str(i)] for i in im]), st._WeightedAUC(y, s)))
        point[c] += coef * st.binary_auc(y, s)
    out = []
    for b0 in range(0, n_boot, chunk):
        B = min(chunk, n_boot - b0)
        W = rng.poisson(1.0, size=(B, len(ids))).astype(np.float64)
        cw = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=B).astype(float)
        vals = np.zeros((B, len(clusters)))
        for k, c in enumerate(clusters):
            v = np.zeros(B)
            for coef, idx, f in prep[c]:
                v += coef * f(W[:, idx])
            vals[:, k] = v
        ok = ~np.isnan(vals)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(ok, vals, 0).__mul__(cw).sum(1) / np.where(ok, cw, 0).sum(1)
        out.append(r)
    arr = np.concatenate(out)
    return point, arr[np.isfinite(arr)]


def _summary(point: Dict, arr: np.ndarray, extra: Dict) -> Dict:
    st = _stats()
    lo, hi = st._ci(arr)
    pts = [float(point[c]) for c in sorted(point)]
    return {**extra, "delta_mean_bootstrap": float(arr.mean()), "ci95_lo": lo, "ci95_hi": hi, "n_boot_valid": int(len(arr)),
            "bootstrap_seed_clusters_per_rep": len(pts), "seed_delta_mean": float(np.mean(pts)),
            "seed_delta_sd": float(np.std(pts, ddof=1)) if len(pts) > 1 else float("nan"),
            "seed_deltas_json": json.dumps(pts), "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "p_boot_two_sided": st.boot_p(arr), "estimator": ESTIMATOR}


def _pair_terms(predictions, method_a, method_b, env, cluster_col, coef=1.0, prefix=""):
    st = _stats()
    by, _ = st.paired_by_cluster(predictions, method_a, method_b, env, cluster_col)
    terms = []
    for c, p in by.items():
        ids = [prefix + str(i) for i in p.image_id]
        terms += [(c, coef, ids, p.y.to_numpy(), p.pa.to_numpy()), (c, -coef, ids, p.y.to_numpy(), p.pb.to_numpy())]
    return terms


def hierarchical_paired_bootstrap(predictions, method_a, method_b, env, n_boot=10000, seed=20260918,
                                  cluster_col="seed", fast=False) -> Dict:
    point, arr = replicates(_pair_terms(predictions, method_a, method_b, env, cluster_col), n_boot, seed)
    return _summary(point, arr, {"method_a": method_a, "method_b": method_b, "env": env,
                                 "estimand": "mean_seed_specific_paired_auc_delta"})


def hierarchical_auc_bootstrap(predictions, method, env, n_boot=10000, seed=20260918, cluster_col="seed", fast=True) -> Dict:
    x = predictions[(predictions.env == env) & (predictions.method == method)]
    terms = [(int(s), 1.0, q.image_id.astype(str).tolist(), q.y.to_numpy(), q.prob.to_numpy())
             for s, q in x.groupby(cluster_col) if q.y.nunique() == 2]
    point, arr = replicates(terms, n_boot, seed)
    lo, hi = _stats()._ci(arr)
    return {"method": method, "env": env, "auc_mean": float(np.mean(list(point.values()))), "ci95_lo": lo, "ci95_hi": hi,
            "estimator": ESTIMATOR}


def hierarchical_interaction(predictions, method, baseline, env, key_col, lo_val, hi_val, n_boot=10000,
                             seed=20260918, cluster_col="seed", fast=False) -> Dict:
    x = predictions[predictions["env"] == env]
    sel = lambda v: (np.isclose(x[key_col].astype(float), float(v)) if not isinstance(v, str) else (x[key_col] == v))
    lo_df, hi_df = x[sel(lo_val)], x[sel(hi_val)]
    # the same images appear at both levels: identical identifiers -> shared weights
    terms = _pair_terms(lo_df, method, baseline, env, cluster_col, +1.0) + _pair_terms(hi_df, method, baseline, env, cluster_col, -1.0)
    point, arr = replicates(terms, n_boot, seed)
    return _summary(point, arr, {"method": method, "baseline": baseline, "env": env, "lo": lo_val, "hi": hi_val,
                                 "estimand": "mean_seed_specific_interaction"})


def difference_of_deltas(pred_1, pred_2, method, baseline, env, n_boot=10000, seed=20260918, cluster_col="seed") -> Dict:
    terms = _pair_terms(pred_1, method, baseline, env, cluster_col, +1.0) + _pair_terms(pred_2, method, baseline, env, cluster_col, -1.0)
    point, arr = replicates(terms, n_boot, seed)
    return _summary(point, arr, {"method": method, "baseline": baseline, "env": env,
                                 "estimand": "crossover_set1_minus_set2"})


def hierarchical_paired_mean_bootstrap(paired_rows, value_a, value_b, n_boot=10000, seed=20260918, cluster_col="seed") -> Dict:
    """Paired mean effect (A - B) with shared per-image Poisson weights (image_id if present, else row identity)."""
    d = paired_rows.dropna(subset=[value_a, value_b]).copy()
    d["_id"] = d.image_id.astype(str) if "image_id" in d else [f"row{k}" for k in range(len(d))]
    ids = sorted(d._id.unique()); pos = {k: j for j, k in enumerate(ids)}
    by = {c: (np.array([pos[i] for i in q._id]), (q[value_a] - q[value_b]).to_numpy(float)) for c, q in d.groupby(cluster_col)}
    clusters = sorted(by)
    point = {c: float(by[c][1].mean()) for c in clusters}
    rng = np.random.default_rng(seed)
    out = []
    for b0 in range(0, n_boot, 500):
        B = min(500, n_boot - b0)
        W = rng.poisson(1.0, size=(B, len(ids))).astype(float)
        cw = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=B).astype(float)
        vals = np.stack([(W[:, idx] * v).sum(1) / np.maximum(W[:, idx].sum(1), 1e-12) for idx, v in (by[c] for c in clusters)], 1)
        out.append((vals * cw).sum(1) / cw.sum(1))
    arr = np.concatenate(out)
    return _summary(point, arr, {"estimand": "mean_seed_specific_paired_mean_delta"})
