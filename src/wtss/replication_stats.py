"""Paired crossed (seed x image) bootstrap for the three-paper replication, for any per-image metric
(docs/PREREGISTRATION_REPLICATION.md, section 2 and amendment A3).

Same estimator as wtss.stats_crossed.replicates: each replicate resamples seed clusters (multinomial) and draws ONE
Poisson(1) weight per test-image id, shared by every seed and both methods; the statistic is the mean over sampled seeds
of metric(ours) - metric(theirs) on the weighted images. With one seed this is the image-only paired bootstrap of A3
(training-seed variance not included). Random draws are made in the same order as stats_crossed.replicates, so for
AUROC the two give identical replicates (tests/test_replication_stats.py).

Metrics (weighted by the per-image bootstrap weights W, shape (B, n)):
  auroc        binary AUROC of a score (y in {0, 1})
  macro_auroc  mean one-vs-rest AUROC over classes of the softmax probabilities (scores: (n, K) logits)
  accuracy     share of argmax == y
  macro_f1     mean over classes of 2 TP / (2 TP + FP + FN)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .stats import _WeightedAUC, _ci, boot_p, binary_auc


def _metric(name: str, y: np.ndarray, s: np.ndarray) -> Tuple[Callable[[np.ndarray], np.ndarray], float]:
    """(weighted metric W -> (B,), point value) for one method's scores on one seed's test images."""
    y = np.asarray(y).astype(int)
    if name == "auroc":
        f = _WeightedAUC(y, np.asarray(s, float))
        return f, binary_auc(y, np.asarray(s, float))
    s = np.asarray(s, float)
    K = s.shape[1]
    if name == "macro_auroc":
        s = np.exp(s - s.max(1, keepdims=True)); s = s / s.sum(1, keepdims=True)   # logits -> softmax probabilities
        fs = [_WeightedAUC((y == k).astype(int), s[:, k]) for k in range(K)]
        return (lambda W: np.mean([f(W) for f in fs], 0)), float(np.mean([binary_auc((y == k).astype(int), s[:, k])
                                                                          for k in range(K)]))
    pred = s.argmax(1)
    if name == "accuracy":
        c = (pred == y).astype(float)
        return (lambda W: (W * c).sum(1) / W.sum(1)), float(c.mean())
    if name == "macro_f1":
        tp = [((pred == k) & (y == k)).astype(float) for k in range(K)]
        fp = [((pred == k) & (y != k)).astype(float) for k in range(K)]
        fn = [((pred != k) & (y == k)).astype(float) for k in range(K)]

        def f1(W):
            out = []
            for k in range(K):
                a, b, c = (W * tp[k]).sum(1), (W * fp[k]).sum(1), (W * fn[k]).sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out.append(np.where(2 * a + b + c > 0, 2 * a / (2 * a + b + c), 0.0))
            return np.mean(out, 0)
        return f1, float(f1(np.ones((1, len(y))))[0])
    raise ValueError(name)


def paired_bootstrap(seeds: Dict[object, Tuple[Sequence[str], np.ndarray, np.ndarray, np.ndarray]], metric: str,
                     n_boot: int = 10000, seed: int = 20260918, chunk: int = 250) -> Dict:
    """seeds: {seed: (image_ids, y, scores_ours, scores_theirs)}; ours and theirs scored on the same images.
    Returns the mean paired difference (ours - theirs) over seeds, its 95% percentile CI, two-sided bootstrap p,
    per-seed values, and both methods' mean metric."""
    rng = np.random.default_rng(seed)
    ids = sorted({str(i) for v in seeds.values() for i in v[0]})
    pos = {k: j for j, k in enumerate(ids)}
    clusters = sorted(seeds)
    prep, point, ma, mb = {}, {}, {}, {}
    for c in clusters:
        im, y, a, b = seeds[c]
        fa, pa = _metric(metric, y, a)
        fb, pb = _metric(metric, y, b)
        prep[c] = (np.array([pos[str(i)] for i in im]), fa, fb)
        point[c], ma[c], mb[c] = pa - pb, pa, pb
    out = []
    for b0 in range(0, n_boot, chunk):
        B = min(chunk, n_boot - b0)
        W = rng.poisson(1.0, size=(B, len(ids))).astype(np.float64)
        cw = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=B).astype(float)
        vals = np.zeros((B, len(clusters)))
        for k, c in enumerate(clusters):
            idx, fa, fb = prep[c]
            vals[:, k] = fa(W[:, idx]) - fb(W[:, idx])
        ok = ~np.isnan(vals)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(ok, vals, 0).__mul__(cw).sum(1) / np.where(ok, cw, 0).sum(1))
    arr = np.concatenate(out)
    arr = arr[np.isfinite(arr)]
    lo, hi = _ci(arr)
    pts = [point[c] for c in clusters]
    return {"metric": metric, "n_seeds": len(clusters), "ours_mean": float(np.mean(list(ma.values()))),
            "theirs_mean": float(np.mean(list(mb.values()))), "delta": float(np.mean(pts)),
            "ci95_lo": lo, "ci95_hi": hi, "p_boot_two_sided": boot_p(arr), "seed_deltas": pts,
            "seed_variance_included": len(clusters) > 1, "n_boot_valid": int(len(arr))}


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def beaten(res: Dict, p_holm: float, alpha: float = 0.05) -> bool:
    """Registered rule: higher mean, 95% CI of the paired difference above 0, Holm-adjusted p < alpha."""
    return res["ours_mean"] > res["theirs_mean"] and res["ci95_lo"] > 0 and p_holm < alpha
