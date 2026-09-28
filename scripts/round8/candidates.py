"""Round 8 candidates on frozen features (docs/PREREGISTRATION_ROUND8.md, section 3).

Every function is a pure function of feature arrays and labels, so each arm can be unit-tested without images.
Heads are logistic regressions with C chosen from wtss.heads.CS by AUROC on the C-selection split (val_clean in the
traps, the validation fold on natural data), exactly as every earlier arm; hyperparameters beyond C are chosen by the
registered rule in common.select_setting on the selection split (val_groups).

Primary candidates
  mask_bal(λ)   masked features; sample weight ∝ (1/n_y)·(n_{a,y}/n_y)^(−λ): λ = 0 is the masking head (class-balanced),
                λ = 1 is mask_balanced (equal weight per artifact × label group).                     [published: group
                reweighting, Sagawa et al. 2020 / Idrissi et al. 2022; the λ-path is an adaptation]
  mask_cmc      masked features projected onto the orthogonal complement of the within-class mean differences
                δ_y = μ(a=1, y) − μ(a=0, y) (pooled: one direction, the mean of δ_0 and δ_1; per_class: both), then an
                ordinary head: w·δ_y = 0, so within each class the score has no mean artifact effect.  [adaptation:
                first-moment, linear-score case of the conditional-MMD regulariser of Veitch et al. 2021]
  full_cmc      the same on unmasked features (keeps the context masking removes).                        [adaptation]
  locrand       masked head trained after pasting real artifact instances into artifact-free training images, with
                class-specific probabilities that equalise P(A = 1 | Y) and inside / outside the ROI with probability ½
                each (features of the pasted images come from scripts/round8/locrand.py).   [adaptation of copy-paste
                augmentation, Ghiasi et al. 2021]
Descriptive: locrand_loc (in- and out-of-ROI presence equalised separately), AFR (Qiu et al. 2023), GroupDRO (Sagawa
et al. 2020) on masked features, mask_cfs (counterfactual-sensitivity generalised ridge; N2), logit ensembles of mask
and mask_balanced, U-MtE family, mask + DFR, and the references erm, mask, balanced, mask_balanced.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from wtss import heads as H
from wtss.utils import stable_int

LAMBDAS = (0.25, 0.5, 0.75, 1.0)
CMC_MODES = ("pooled", "per_class")
AFR_GAMMAS = (0.5, 1.0, 2.0, 4.0)
CFS_MUS = (1.0, 10.0, 100.0)
ENS_ALPHAS = (0.75, 0.5, 0.25)  # weight of the masking head's logit


# ----------------------------------------------------------------------------------------------------------- heads
class Head:
    """Probability of y = 1 from features of one view, optionally after a transform."""

    def __init__(self, clf, transform: Optional[Callable] = None):
        self.clf, self.transform = clf, transform

    def prob(self, X: np.ndarray) -> np.ndarray:
        Z = self.transform(X) if self.transform is not None else X
        return self.clf.predict_proba(Z)[:, 1]

    def logit(self, X: np.ndarray) -> np.ndarray:
        Z = self.transform(X) if self.transform is not None else X
        return self.clf.decision_function(Z)


class EnsembleHead:
    """α · logit(head_a) + (1 − α) · logit(head_b) on the same view."""

    def __init__(self, ha: Head, hb: Head, alpha: float):
        self.ha, self.hb, self.alpha = ha, hb, alpha

    def prob(self, X):
        z = self.alpha * self.ha.logit(X) + (1 - self.alpha) * self.hb.logit(X)
        return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def _lr(C: float, seed: int, class_weight=None):
    return LogisticRegression(C=C, class_weight=class_weight, max_iter=3000, solver="liblinear", random_state=seed)


def fit_weighted(Xtr, ytr, w, Xv, yv, seed, transform=None) -> Tuple[Head, float, float]:
    """Logistic head with sample weights; C by validation AUROC (wtss.heads.select_C)."""
    Zt = transform(Xtr) if transform is not None else Xtr
    Zv = transform(Xv) if transform is not None else Xv
    clf, C, v = H.select_C(lambda C: _lr(C, seed).fit(Zt, ytr, sample_weight=w), Zv, yv)
    return Head(clf, transform), C, v


def class_balanced_weights(y) -> np.ndarray:
    y = np.asarray(y).astype(int)
    n = np.bincount(y, minlength=2).astype(float)
    w = 1.0 / n[y]
    return w / w.mean()


# ------------------------------------------------------------------------------------------------ C1 mask_bal(λ)
def mask_bal_weights(y, a, lam: float) -> np.ndarray:
    """(1/n_y) · (n_{a,y} / n_y)^(−λ), normalised to mean 1. λ = 0: class-balanced (the masking head's weighting);
    λ = 1: 1/n_{a,y}, equal total weight per artifact × label group (wtss.heads.group_weights)."""
    y, a = np.asarray(y).astype(int), np.asarray(a).astype(int)
    ny = np.bincount(y, minlength=2).astype(float)
    g = 2 * a + y
    ng = np.bincount(g, minlength=4).astype(float)
    frac = ng[g] / ny[y]
    w = (1.0 / ny[y]) * np.power(frac, -lam)
    return w / w.mean()


# ------------------------------------------------------------------------------------------------------ N1 CMC
class Projector:
    """x -> x − (x·Q) Qᵀ for an orthonormal basis Q (d × k): the projection onto the complement of span(Q)."""

    def __init__(self, Q: np.ndarray):
        self.Q = np.asarray(Q, np.float64)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64)
        return X - (X @ self.Q) @ self.Q.T


def cmc_directions(X, y, a, mode: str = "pooled", ids: Optional[Sequence[str]] = None,
                   allowed: Optional[set] = None) -> np.ndarray:
    """Orthonormal basis of the within-class mean differences δ_y = μ(a=1,y) − μ(a=0,y).
    pooled: the single direction (δ_0 + δ_1)/2 (classes with an empty group are skipped); per_class: span(δ_0, δ_1).
    `allowed` (image ids) restricts the images the means are estimated from (the dermoscopy sensitivity run)."""
    X, y, a = np.asarray(X, np.float64), np.asarray(y).astype(int), np.asarray(a).astype(int)
    keep = np.ones(len(y), bool) if allowed is None else np.array([str(i) in allowed for i in ids])
    deltas = []
    for c in (0, 1):
        m1, m0 = keep & (y == c) & (a == 1), keep & (y == c) & (a == 0)
        if m1.sum() >= 2 and m0.sum() >= 2:
            deltas.append(X[m1].mean(0) - X[m0].mean(0))
    if not deltas:
        return np.zeros((X.shape[1], 0))
    if mode == "pooled":
        D = np.mean(deltas, 0)[:, None]
    elif mode == "per_class":
        D = np.stack(deltas, 1)
    else:
        raise ValueError(mode)
    Q, R = np.linalg.qr(D)
    keep_cols = np.abs(np.diag(R)) > 1e-10 * max(1.0, np.abs(R).max())
    return Q[:, keep_cols]


def fit_cmc(Xtr, ytr, atr, Xv, yv, seed, mode="pooled", ids=None, allowed=None) -> Tuple[Head, float, float]:
    P = Projector(cmc_directions(Xtr, ytr, atr, mode, ids, allowed))
    return fit_weighted(Xtr, ytr, class_balanced_weights(ytr), Xv, yv, seed, transform=P)


# ------------------------------------------------------------------------------------------------ N3 locrand
def paste_plan(tr: pd.DataFrame, rng: np.random.Generator, mode: str = "locrand") -> Dict[str, str]:
    """Which artifact-free training images receive a pasted instance, and where ('in' / 'out').

    tr columns: image_id, y, a_any (artifact present anywhere), recipient (artifact-free and a pasted version with a
    training-set donor exists); for mode 'loc_matched' also a_in, a_out.
    locrand: target t = max_y P(a_any = 1 | y); class y receives round(t·n_y − n_{a_any=1,y}) pastes among its
    recipients (capped by their number), each inside or outside the ROI with probability ½. After pasting
    P(A = 1 | Y = 1) = P(A = 1 | Y = 0) (up to rounding and the cap, both reported).
    loc_matched (descriptive): the same separately for in-ROI and out-of-ROI presence, so P(A_in | Y) and P(A_out | Y)
    are both class-independent; in-ROI pastes are allocated first."""
    plan: Dict[str, str] = {}
    y = tr.y.to_numpy().astype(int)
    rec = tr.recipient.to_numpy().astype(bool)
    if mode == "locrand":
        present = tr.a_any.to_numpy().astype(int)
        t = max(present[y == c].mean() for c in (0, 1) if (y == c).any())
        for c in (0, 1):
            n_c = int((y == c).sum())
            need = int(round(t * n_c - present[y == c].sum()))
            pool = tr.image_id[(y == c) & rec].astype(str).to_numpy()
            k = max(0, min(need, len(pool)))
            if k:
                for i in rng.choice(pool, k, replace=False):
                    plan[str(i)] = "in" if rng.random() < 0.5 else "out"
        return plan
    if mode == "loc_matched":
        free = {c: list(rng.permutation(tr.image_id[(y == c) & rec].astype(str).to_numpy())) for c in (0, 1)}
        for loc, col in (("in", "a_in"), ("out", "a_out")):
            pres = tr[col].to_numpy().astype(int)
            t = max(pres[y == c].mean() for c in (0, 1) if (y == c).any())
            for c in (0, 1):
                need = int(round(t * int((y == c).sum()) - pres[y == c].sum()))
                for _ in range(max(0, min(need, len(free[c])))):
                    plan[free[c].pop()] = loc
        return plan
    raise ValueError(mode)


def paste_features(ids: Sequence[str], X_base: np.ndarray, plan: Dict[str, str], version: Dict[str, int],
                   paste_X: Callable[[int, str, str], np.ndarray]) -> np.ndarray:
    """Replace the rows of pasted images by the features of their pasted version (paste_X(k, loc, image_id))."""
    X = np.array(X_base, copy=True)
    for j, i in enumerate(ids):
        loc = plan.get(str(i))
        if loc is not None:
            X[j] = paste_X(version[str(i)], loc, str(i))
    return X


def plan_balance(tr: pd.DataFrame, plan: Dict[str, str]) -> dict:
    """P(A = 1 | Y) before and after pasting (reported for every training set)."""
    y = tr.y.to_numpy().astype(int)
    a = tr.a_any.to_numpy().astype(int)
    pasted = tr.image_id.astype(str).isin(plan).to_numpy()
    out = {}
    for c in (0, 1):
        m = y == c
        out[f"p_a_y{c}_before"] = float(a[m].mean()) if m.any() else np.nan
        out[f"p_a_y{c}_after"] = float(np.maximum(a, pasted)[m].mean()) if m.any() else np.nan
        out[f"n_pasted_y{c}"] = int((pasted & m).sum())
    return out


# ---------------------------------------------------------------------------------------------- descriptive arms
def fit_afr(Xtr, ytr, ids, Xv, yv, seed, gammas=AFR_GAMMAS) -> Dict[str, Head]:
    """Automatic feature reweighting (Qiu et al. 2023) for a linear probe: an ERM head on 80 % of the training images
    (stable hash split), then a head on the other 20 % with weights exp(−γ·p_true) of the first head, normalised per
    class to equal total weight. No group label in training."""
    ids = np.asarray([str(i) for i in ids])
    ytr = np.asarray(ytr).astype(int)
    rw = np.array([stable_int("afr_split", i) % 5 == 0 for i in ids])
    if rw.sum() < 10 or (~rw).sum() < 10 or len(np.unique(ytr[rw])) < 2 or len(np.unique(ytr[~rw])) < 2:
        return {}
    base = H.fit_erm(Xtr[~rw], ytr[~rw], Xv, yv, seed)[0]
    pt = base.predict_proba(Xtr[rw])[np.arange(rw.sum()), ytr[rw]]
    out = {}
    for g in gammas:
        w = np.exp(-g * pt)
        for c in (0, 1):
            m = ytr[rw] == c
            if m.any():
                w[m] = w[m] / w[m].sum()
        out[f"g{g:g}"] = fit_weighted(Xtr[rw], ytr[rw], w / w.mean(), Xv, yv, seed)[0]
    return out


def cfs_transform(X0: np.ndarray, X1: np.ndarray, mu: float) -> Callable:
    """Generalised ridge along the counterfactual (overlay-insertion) shifts: a head fitted with an ordinary L2 penalty
    on Z = X A^(−1/2), A = I + μ Σ̃, Σ̃ = cov(X1 − X0) / its largest eigenvalue, has the penalty wᵀ A w on the original
    features — the squared prediction change under inserted overlays, weighted by μ (N2)."""
    D = np.asarray(X1, np.float64) - np.asarray(X0, np.float64)
    S = D.T @ D / max(len(D), 1)
    evals, evecs = np.linalg.eigh(S)
    top = max(float(evals.max()), 1e-12)
    inv_sqrt = 1.0 / np.sqrt(1.0 + mu * np.clip(evals, 0, None) / top)
    M = (evecs * inv_sqrt) @ evecs.T
    return lambda X: np.asarray(X, np.float64) @ M


def fit_groupdro_cpu(Xtr, ytr, atr, Xv, yv, seed):
    import torch
    torch.set_num_threads(1)
    clf, C, v = H.fit_groupdro(Xtr, ytr, atr, Xv, yv, seed, torch.device("cpu"))
    return Head(clf), C, v


def val_auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
