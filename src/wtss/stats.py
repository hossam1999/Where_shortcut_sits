"""Uncertainty estimators.

Primary CI (pilot v3 estimator, unchanged): hierarchical paired bootstrap that
resamples training-seed (or CV-fold) clusters with replacement, then paired
test images within each sampled cluster, computes the AUROC delta *inside* each
cluster, and averages cluster-specific deltas per replicate. Predictions from
different trained models are never pooled into one ROC curve.

The RNG draw order is identical to ``pilot_core.py`` so CIs recomputed from the
archived predictions reproduce the archived numbers exactly.
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


KEEP = ["seed", "method", "image_id", "y", "prob"]


def _use_crossed() -> bool:
    """WTSS_BOOTSTRAP=crossed routes every estimator below to wtss.stats_crossed (corrected, shared-test-set bootstrap)."""
    import os
    return os.environ.get("WTSS_BOOTSTRAP", "").lower() == "crossed"


def slim(preds: pd.DataFrame, env: str, methods, cluster_col: str = "seed") -> pd.DataFrame:
    """Rows/columns needed by one paired bootstrap (keeps worker payloads small)."""
    cols = [c for c in dict.fromkeys(KEEP + [cluster_col]) if c in preds.columns] + ["env"]
    return preds.loc[(preds.env == env) & preds.method.isin(list(methods)), cols]


def safe_auc(y, prob) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")


def binary_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC (same estimand as sklearn, no pandas; handles ties)."""
    pos = s[y == 1]
    neg = s[y == 0]
    n1, n0 = pos.size, neg.size
    if n1 == 0 or n0 == 0:
        return float("nan")
    neg = np.sort(neg)
    left = np.searchsorted(neg, pos, side="left")
    right = np.searchsorted(neg, pos, side="right")
    return float((left + 0.5 * (right - left)).sum() / (n1 * n0))


def _ci(arr: np.ndarray) -> Tuple[float, float]:
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def paired_by_cluster(predictions: pd.DataFrame, method_a: str, method_b: str, env: str,
                      cluster_col: str = "seed") -> Tuple[Dict[int, pd.DataFrame], List[float]]:
    x = predictions[predictions["env"] == env]
    by: Dict[int, pd.DataFrame] = {}
    deltas: List[float] = []
    for s in sorted(x[cluster_col].unique().tolist()):
        ss = x[x[cluster_col] == s]
        a = ss[ss["method"] == method_a][["image_id", "y", "prob"]].rename(columns={"prob": "pa"})
        b = ss[ss["method"] == method_b][["image_id", "y", "prob"]].rename(columns={"prob": "pb"})
        p = a.merge(b, on=["image_id", "y"], how="inner")
        if len(p) == 0 or p.y.nunique() < 2:
            continue
        by[int(s)] = p
        deltas.append(safe_auc(p.y.to_numpy(), p.pa.to_numpy()) - safe_auc(p.y.to_numpy(), p.pb.to_numpy()))
    if not by:
        raise ValueError(f"No paired predictions for {method_a} vs {method_b} on {env}")
    return by, deltas


class _WeightedAUC:
    """AUROC of a fixed score vector under many resampling-weight vectors, without re-sorting.

    With multinomial weights w (a bootstrap resample), Mann-Whitney AUROC is
        sum_i w_i [W_neg(< s_i) + 0.5 W_neg(= s_i)] / (W_pos W_neg),
    and W_neg(< s_i) is a gather from the cumulative weights of the (once-)sorted negatives.
    """

    def __init__(self, y: np.ndarray, s: np.ndarray):
        self.pos, self.neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        o = np.argsort(s[self.neg], kind="stable")
        self.neg_sorted = self.neg[o]
        sn = s[self.neg_sorted]
        sp = s[self.pos]
        self.left = np.searchsorted(sn, sp, "left")
        self.right = np.searchsorted(sn, sp, "right")

    def __call__(self, w: np.ndarray) -> np.ndarray:  # w: (B, n) float
        wn = w[:, self.neg_sorted]
        cum = np.concatenate([np.zeros((len(w), 1)), np.cumsum(wn, 1)], 1)
        below = cum[:, self.left]
        eq = cum[:, self.right] - below
        wp = w[:, self.pos]
        num = (wp * (below + 0.5 * eq)).sum(1)
        den = wp.sum(1) * wn.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)


def _multinomial_counts(rng, n: int, b: int) -> np.ndarray:
    """(b, n) bootstrap resample counts (multinomial; benchmarked faster than index+bincount here)."""
    return rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float64)


def _cluster_bootstrap_fast(arrays: Dict[int, tuple], stat: str, n_boot: int, seed: int, chunk: int = 500) -> np.ndarray:
    """Vectorised equivalent of _cluster_bootstrap for AUROC statistics.

    stat: 'auc' (arrays: y, a), 'delta' (y, a, b), 'inter' (y, a0, b0, a1, b1).
    Same estimand (resample clusters, then images within cluster via multinomial counts, average the
    cluster statistics); different RNG stream from the legacy loop. One-class resamples are dropped.
    """
    rng = np.random.default_rng(seed)
    keys = np.array(sorted(arrays))
    clusters = rng.choice(keys, size=(n_boot, len(keys)), replace=True)
    vals = np.full(clusters.shape, np.nan)
    for c in keys:
        arr = arrays[int(c)]
        y = arr[0]
        n = len(y)
        aucs = [_WeightedAUC(y, a) for a in arr[1:]]
        rows, cols = np.nonzero(clusters == c)
        for k in range(0, len(rows), chunk):
            rr, cc = rows[k:k + chunk], cols[k:k + chunk]
            w = _multinomial_counts(rng, n, len(rr))
            a = [f(w) for f in aucs]
            v = a[0] if stat == "auc" else a[0] - a[1] if stat == "delta" else (a[0] - a[1]) - (a[2] - a[3])
            vals[rr, cc] = v
    out = np.nanmean(vals, axis=1)
    return out[np.isfinite(out)]


def _cluster_bootstrap(arrays: Dict[int, tuple], stat_fn, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keys = np.array(sorted(arrays))
    boots: List[float] = []
    for _ in range(n_boot):
        vals = []
        for s in rng.choice(keys, size=len(keys), replace=True):
            arr = arrays[int(s)]
            y = arr[0]
            n = y.shape[0]
            d = None
            for _attempt in range(10):
                idx = rng.integers(0, n, size=n)
                yy = y[idx]
                if yy.min() == yy.max():
                    continue
                d = stat_fn(yy, *(a[idx] for a in arr[1:]))
                break
            if d is not None and np.isfinite(d):
                vals.append(float(d))
        if vals:
            boots.append(float(np.mean(vals)))
    out = np.asarray(boots, dtype=float)
    if len(out) == 0:
        raise ValueError("No valid bootstrap replicates")
    return out


def boot_p(arr: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: delta = 0 (percentile method), floored at 1/n_boot."""
    arr = np.asarray(arr, float)
    p = 2 * min((arr <= 0).mean(), (arr >= 0).mean())
    return float(min(1.0, max(p, 1.0 / len(arr))))


def hierarchical_paired_bootstrap(predictions: pd.DataFrame, method_a: str, method_b: str, env: str,
                                  n_boot: int = 10000, seed: int = 20260918,
                                  cluster_col: str = "seed", fast: bool = False) -> Dict:
    """Mean cluster-specific paired AUROC delta (method_a - method_b) with hierarchical 95% CI.

    fast=False reproduces the pilot's RNG stream exactly; fast=True is the vectorised equivalent.
    """
    if _use_crossed():
        from . import stats_crossed as _x
        return _x.hierarchical_paired_bootstrap(predictions, method_a, method_b, env, n_boot, seed, cluster_col)
    by, deltas = paired_by_cluster(predictions, method_a, method_b, env, cluster_col)
    arrays = {s: (p.y.to_numpy(), p.pa.to_numpy(), p.pb.to_numpy()) for s, p in by.items()}
    arr = (_cluster_bootstrap_fast(arrays, "delta", n_boot, seed) if fast else
           _cluster_bootstrap(arrays, lambda y, a, b: binary_auc(y, a) - binary_auc(y, b), n_boot, seed))
    lo, hi = _ci(arr)
    return {
        "method_a": method_a, "method_b": method_b, "env": env,
        "estimand": "mean_seed_specific_paired_auc_delta",
        "delta_mean_bootstrap": float(arr.mean()), "ci95_lo": lo, "ci95_hi": hi,
        "n_boot_valid": int(len(arr)), "bootstrap_seed_clusters_per_rep": len(arrays),
        "seed_delta_mean": float(np.mean(deltas)),
        "seed_delta_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan"),
        "seed_deltas_json": json.dumps([float(x) for x in deltas]),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "p_boot_two_sided": boot_p(arr),
    }


def hierarchical_auc_bootstrap(predictions: pd.DataFrame, method: str, env: str, n_boot: int = 10000,
                               seed: int = 20260918, cluster_col: str = "seed", fast: bool = True) -> Dict:
    """Mean cluster-specific AUROC of one method with hierarchical CI."""
    if _use_crossed():
        from . import stats_crossed as _x
        return _x.hierarchical_auc_bootstrap(predictions, method, env, n_boot, seed, cluster_col)
    x = predictions[(predictions.env == env) & (predictions.method == method)]
    arrays = {int(s): (q.y.to_numpy(), q.prob.to_numpy()) for s, q in x.groupby(cluster_col) if q.y.nunique() == 2}
    arr = (_cluster_bootstrap_fast(arrays, "auc", n_boot, seed) if fast else
           _cluster_bootstrap(arrays, lambda y, a: binary_auc(y, a), n_boot, seed))
    lo, hi = _ci(arr)
    point = float(np.mean([binary_auc(*v) for v in arrays.values()]))
    return {"method": method, "env": env, "auc_mean": point, "ci95_lo": lo, "ci95_hi": hi}


def hierarchical_interaction(predictions: pd.DataFrame, method: str, baseline: str, env: str,
                             key_col: str, lo_val, hi_val, n_boot: int = 10000, seed: int = 20260918,
                             cluster_col: str = "seed", fast: bool = False) -> Dict:
    """[method-baseline]_{lo} - [method-baseline]_{hi} on the *same* image IDs.

    Used for the synthetic location interaction (key_col='overlap', same test images at
    0% and 100%). For real traps, whose test sets differ, use ``difference_of_deltas``.
    """
    if _use_crossed():
        from . import stats_crossed as _x
        return _x.hierarchical_interaction(predictions, method, baseline, env, key_col, lo_val, hi_val, n_boot, seed, cluster_col)
    x = predictions[predictions["env"] == env]
    by: Dict[int, pd.DataFrame] = {}
    effects: List[float] = []
    for s in sorted(x[cluster_col].unique()):
        ss = x[x[cluster_col] == s]
        pieces = {}
        for v, tag in [(lo_val, "lo"), (hi_val, "hi")]:
            sel = np.isclose(ss[key_col].astype(float), float(v)) if not isinstance(v, str) else (ss[key_col] == v)
            for m, name in [(method, "arm"), (baseline, "base")]:
                q = ss[(ss["method"] == m) & sel]
                pieces[f"{name}_{tag}"] = q[["image_id", "y", "prob"]].rename(columns={"prob": f"p_{name}_{tag}"})
        z = pieces["arm_lo"].merge(pieces["base_lo"], on=["image_id", "y"])
        z = z.merge(pieces["arm_hi"], on=["image_id", "y"]).merge(pieces["base_hi"], on=["image_id", "y"])
        if z.empty or z.y.nunique() < 2:
            continue
        d_lo = safe_auc(z.y, z.p_arm_lo) - safe_auc(z.y, z.p_base_lo)
        d_hi = safe_auc(z.y, z.p_arm_hi) - safe_auc(z.y, z.p_base_hi)
        by[int(s)] = z
        effects.append(float(d_lo - d_hi))
    if not by:
        raise ValueError(f"No paired rows for interaction {method} vs {baseline}")
    arrays = {s: (z.y.to_numpy(), z.p_arm_lo.to_numpy(), z.p_base_lo.to_numpy(),
                  z.p_arm_hi.to_numpy(), z.p_base_hi.to_numpy()) for s, z in by.items()}
    fn = lambda y, a0, b0, a1, b1: (binary_auc(y, a0) - binary_auc(y, b0)) - (binary_auc(y, a1) - binary_auc(y, b1))
    arr = _cluster_bootstrap_fast(arrays, "inter", n_boot, seed) if fast else _cluster_bootstrap(arrays, fn, n_boot, seed)
    lo, hi = _ci(arr)
    return {"method": method, "baseline": baseline, "env": env, "lo": lo_val, "hi": hi_val,
            "estimand": "mean_seed_specific_interaction", "delta_mean_bootstrap": float(arr.mean()),
            "ci95_lo": lo, "ci95_hi": hi, "n_boot_valid": int(len(arr)),
            "seed_delta_mean": float(np.mean(effects)),
            "seed_delta_sd": float(np.std(effects, ddof=1)) if len(effects) > 1 else float("nan"),
            "seed_deltas_json": json.dumps(effects), "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "p_boot_two_sided": boot_p(arr)}


def difference_of_deltas(pred_1: pd.DataFrame, pred_2: pd.DataFrame, method: str, baseline: str, env: str,
                         n_boot: int = 10000, seed: int = 20260918, cluster_col: str = "seed") -> Dict:
    """Crossover between two *independent* test sets (e.g. Trap B vs Trap A):
    [method-baseline]_{set1} - [method-baseline]_{set2}.

    Each replicate resamples clusters (shared ids, e.g. CV folds), then images independently within
    each set (multinomial weights), and averages cluster-specific crossovers. Vectorised.
    """
    if _use_crossed():
        from . import stats_crossed as _x
        return _x.difference_of_deltas(pred_1, pred_2, method, baseline, env, n_boot, seed, cluster_col)
    by1, _ = paired_by_cluster(pred_1, method, baseline, env, cluster_col)
    by2, _ = paired_by_cluster(pred_2, method, baseline, env, cluster_col)
    keys = np.array(sorted(set(by1) & set(by2)))
    if len(keys) == 0:
        raise ValueError("No shared clusters")
    point = []
    rng = np.random.default_rng(seed)
    clusters = rng.choice(keys, size=(n_boot, len(keys)), replace=True)
    vals = np.full(clusters.shape, np.nan)
    for c in keys:
        eff = []
        for by in (by1, by2):
            p = by[int(c)]
            y = p.y.to_numpy()
            fa, fb = _WeightedAUC(y, p.pa.to_numpy()), _WeightedAUC(y, p.pb.to_numpy())
            eff.append((fa, fb, len(y)))
            point.append(None)
        point[-2:] = [binary_auc(by1[int(c)].y.to_numpy(), by1[int(c)].pa.to_numpy()) - binary_auc(by1[int(c)].y.to_numpy(), by1[int(c)].pb.to_numpy()),
                      binary_auc(by2[int(c)].y.to_numpy(), by2[int(c)].pa.to_numpy()) - binary_auc(by2[int(c)].y.to_numpy(), by2[int(c)].pb.to_numpy())]
        rows, cols = np.nonzero(clusters == c)
        for k in range(0, len(rows), 500):  # chunked: bounded memory (n can be ~10^4 per cluster)
            rr, cc = rows[k:k + 500], cols[k:k + 500]
            d = []
            for fa, fb, n in eff:
                w = _multinomial_counts(rng, n, len(rr))
                d.append(fa(w) - fb(w))
            vals[rr, cc] = d[0] - d[1]
    arr = np.nanmean(vals, 1)
    arr = arr[np.isfinite(arr)]
    pts = [point[i] - point[i + 1] for i in range(0, len(point), 2)]
    lo, hi = _ci(arr)
    return {"method": method, "baseline": baseline, "env": env, "estimand": "crossover_set1_minus_set2",
            "p_boot_two_sided": boot_p(arr),
            "delta_mean_bootstrap": float(arr.mean()), "ci95_lo": lo, "ci95_hi": hi,
            "seed_delta_mean": float(np.mean(pts)), "seed_deltas_json": json.dumps([float(x) for x in pts]),
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def hierarchical_paired_mean_bootstrap(paired_rows: pd.DataFrame, value_a: str, value_b: str,
                                       n_boot: int = 10000, seed: int = 20260918,
                                       cluster_col: str = "seed") -> Dict:
    """Hierarchical CI for an image-level paired mean effect (A - B), e.g. |Δp| differences."""
    if _use_crossed():
        from . import stats_crossed as _x
        return _x.hierarchical_paired_mean_bootstrap(paired_rows, value_a, value_b, n_boot, seed, cluster_col)
    by = {int(s): q.dropna(subset=[value_a, value_b]) for s, q in paired_rows.groupby(cluster_col)}
    by = {s: q for s, q in by.items() if len(q)}
    if not by:
        raise ValueError("No paired rows for mean bootstrap")
    effects = [float((q[value_a] - q[value_b]).mean()) for q in by.values()]
    keys = np.array(sorted(by))
    arrays = {s: (q[value_a].to_numpy(), q[value_b].to_numpy()) for s, q in by.items()}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        e = []
        for s in rng.choice(keys, size=len(keys), replace=True):
            a, b = arrays[int(s)]
            idx = rng.integers(0, len(a), size=len(a))
            e.append(float((a[idx] - b[idx]).mean()))
        boots.append(float(np.mean(e)))
    arr = np.asarray(boots)
    lo, hi = _ci(arr)
    return {"delta_mean_bootstrap": float(arr.mean()), "ci95_lo": lo, "ci95_hi": hi,
            "n_boot_valid": int(len(arr)), "seed_delta_mean": float(np.mean(effects)),
            "seed_delta_sd": float(np.std(effects, ddof=1)) if len(effects) > 1 else float("nan")}


def fmt_ci(d: Dict, key: str = "delta_mean_bootstrap", point_key: str = "seed_delta_mean", signed: bool = True) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(d[point_key])} [{f.format(d['ci95_lo'])}, {f.format(d['ci95_hi'])}]"


# --------------------------------------------------------------------------- crossed (cluster x image) bootstrap
def crossed_auc_bootstrap(terms: List[tuple], n_boot: int = 10000, seed: int = 20260928, chunk: int = 250) -> Dict:
    """Bootstrap for statistics that are signed sums of AUROCs, when clusters (training seeds / folds) share test
    images (docs/PREREGISTRATION_REVIEW3.md, R6).

    terms: (cluster, coef, image_ids, y, scores). Per cluster the statistic is sum_j coef_j * AUROC_j; the estimate is
    the mean over clusters. Each replicate resamples clusters with replacement and draws ONE Poisson(1) weight per
    image identifier, shared by every term and cluster in which that image appears (crossed design), instead of
    resampling images independently within each cluster (which understates test-set sampling variance when the
    clusters share test images).
    """
    rng = np.random.default_rng(seed)
    ids = sorted({i for t in terms for i in t[2]})
    pos = {k: j for j, k in enumerate(ids)}
    clusters = sorted({t[0] for t in terms})
    prep = {c: [] for c in clusters}
    point = {c: 0.0 for c in clusters}
    for c, coef, im, y, s in terms:
        y, s = np.asarray(y), np.asarray(s, float)
        prep[c].append((coef, np.array([pos[i] for i in im]), _WeightedAUC(y, s)))
        point[c] += coef * binary_auc(y, s)
    est = float(np.mean([point[c] for c in clusters]))
    out = []
    for b0 in range(0, n_boot, chunk):
        B = min(chunk, n_boot - b0)
        W = rng.poisson(1.0, size=(B, len(ids))).astype(np.float64)
        cw = np.stack([rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters))) for _ in range(B)])
        vals = np.zeros((B, len(clusters)))
        for k, c in enumerate(clusters):
            v = np.zeros(B)
            for coef, idx, f in prep[c]:
                v += coef * f(W[:, idx])
            vals[:, k] = v
        with np.errstate(invalid="ignore"):
            r = np.nansum(vals * cw, 1) / np.where(np.isnan(vals), 0, cw).sum(1)
        out.append(r)
    arr = np.concatenate(out)
    arr = arr[np.isfinite(arr)]
    lo, hi = _ci(arr)
    return {"estimate": est, "ci95_lo": lo, "ci95_hi": hi, "p_boot_two_sided": boot_p(arr), "n_boot_valid": int(len(arr)),
            "ci_excludes_zero": bool(lo > 0 or hi < 0), "n_images": len(ids), "n_clusters": len(clusters),
            "estimator": "crossed cluster x image (Poisson) bootstrap"}


def paired_terms(preds: pd.DataFrame, a: str, b: str, env: str, coef: float = 1.0, cluster_col: str = "seed",
                 cluster_tag: str = "") -> List[tuple]:
    """+coef*AUROC(a) - coef*AUROC(b) on the images both arms scored, per cluster."""
    out = []
    x = preds[preds.env == env]
    for c, q in x.groupby(cluster_col):
        qa = q[q.method == a][["image_id", "y", "prob"]]
        qb = q[q.method == b][["image_id", "y", "prob"]]
        z = qa.merge(qb, on=["image_id", "y"], suffixes=("_a", "_b"))
        if z.y.nunique() < 2:
            continue
        ids = (cluster_tag + z.image_id.astype(str)).tolist() if cluster_tag else z.image_id.astype(str).tolist()
        out += [(c, coef, ids, z.y.to_numpy(), z.prob_a.to_numpy()), (c, -coef, ids, z.y.to_numpy(), z.prob_b.to_numpy())]
    return out
