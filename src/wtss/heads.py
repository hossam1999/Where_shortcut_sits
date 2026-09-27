"""Linear heads on frozen features (ERM and mitigations). Ported from the pilot.

All hyper-parameters (C, λ) are selected on *clean* validation AUROC only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .utils import seed_all

CS = (0.01, 0.1, 1.0, 10.0)


@dataclass
class TorchLinearHead:
    w: np.ndarray
    b: float

    def predict_proba(self, X):
        z = X @ self.w + self.b
        p = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
        return np.c_[1 - p, p]


class TransformHead:
    """Linear head applied after a frozen feature transform (e.g. an eraser)."""

    def __init__(self, transform: Callable[[np.ndarray], np.ndarray], clf):
        self.transform, self.clf = transform, clf

    def predict_proba(self, X):
        return self.clf.predict_proba(self.transform(X))


def four_group_ids(y, a):
    return (2 * np.asarray(a).astype(int) + np.asarray(y).astype(int)).astype(int)


def _logreg(C, seed, class_weight="balanced"):
    return LogisticRegression(C=C, class_weight=class_weight, max_iter=3000, solver="liblinear", random_state=seed)


def select_C(fit_fn, Xval, yval, Cs=CS):
    best = None
    for C in Cs:
        clf = fit_fn(C)
        s = roc_auc_score(yval, clf.predict_proba(Xval)[:, 1])
        if best is None or s > best[0]:
            best = (s, C, clf)
    return best[2], best[1], best[0]


def fit_erm(Xtr, ytr, Xval, yval, seed):
    """ERM: logistic regression, class_weight on the *label* only."""
    return select_C(lambda C: _logreg(C, seed).fit(Xtr, ytr), Xval, yval)


def group_weights(y, a):
    g = four_group_ids(y, a)
    counts = np.bincount(g, minlength=4).astype(np.float64)
    counts[counts == 0] = np.nan
    w = 1.0 / counts[g]
    return w / np.nanmean(w)


def fit_balanced(Xtr, ytr, atr, Xval, yval, seed):
    """Equal total weight on the four (artifact x label) groups. Needs image-level A at train."""
    w = group_weights(ytr, atr)
    return select_C(lambda C: _logreg(C, seed, None).fit(Xtr, ytr, sample_weight=w), Xval, yval)


def group_balanced_indices(y, a, seed):
    g = four_group_ids(y, a)
    rng = np.random.default_rng(seed)
    sizes = [int((g == k).sum()) for k in range(4)]
    if min(sizes) <= 0:
        raise RuntimeError(f"DFR requires all four groups; counts={sizes}")
    return np.concatenate([rng.choice(np.flatnonzero(g == k), size=min(sizes), replace=False) for k in range(4)])


def fit_dfr(Xg, yg, ag, Xval, yval, seed):
    """Deep feature reweighting: retrain the last layer on a group-balanced held-out subset."""
    idx = group_balanced_indices(yg, ag, seed)
    return select_C(lambda C: _logreg(C, seed, None).fit(Xg[idx], yg[idx]), Xval, yval)


def fit_groupdro(Xtr, ytr, atr, Xval, yval, seed, device, lr=0.01, dro_eta=0.01, epochs=500, weight_decay=1e-4):
    seed_all(seed)
    x = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    y = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    g = torch.as_tensor(four_group_ids(ytr, atr), dtype=torch.long, device=device)
    model = torch.nn.Linear(Xtr.shape[1], 1).to(device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    q = torch.ones(4, device=device) / 4.0
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    best_state, best_loss, stale = None, float("inf"), 0
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        per = bce(model(x).squeeze(1), y)
        gl = torch.stack([per[g == k].mean() if (g == k).any() else per.new_zeros(()) for k in range(4)])
        q = q * torch.exp(dro_eta * gl.detach())
        q = q / q.sum()
        loss = (q * gl).sum()
        loss.backward()
        opt.step()
        v = float(loss.detach())
        if v < best_loss - 1e-7:
            best_loss, stale = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 80 and epoch >= 150:
                break
    model.load_state_dict(best_state)
    head = TorchLinearHead(model.weight.detach().cpu().numpy().reshape(-1), float(model.bias.detach().cpu()))
    return head, None, float(roc_auc_score(yval, head.predict_proba(Xval)[:, 1]))


# ----------------------------------------------------------------------------- erasure
def fit_leace(X0: np.ndarray, X1: np.ndarray):
    """LEACE with z = 0 for X0 rows, 1 for X1 rows (paired views => z independent of Y)."""
    from concept_erasure import LeaceEraser

    x = torch.as_tensor(np.vstack([X0, X1]), dtype=torch.float64)
    z = torch.as_tensor(np.r_[np.zeros(len(X0)), np.ones(len(X1))], dtype=torch.float64)
    return LeaceEraser.fit(x, z)


def fit_leace_labels(X: np.ndarray, z: np.ndarray):
    """Unpaired LEACE on image-level concept labels (the variant the thesis shows can be gamed)."""
    from concept_erasure import LeaceEraser

    return LeaceEraser.fit(torch.as_tensor(X, dtype=torch.float64), torch.as_tensor(z, dtype=torch.float64))


def eraser_fn(eraser) -> Callable[[np.ndarray], np.ndarray]:
    def f(X):
        return eraser(torch.as_tensor(X, dtype=torch.float64)).numpy().astype(np.float32)

    return f


def fit_on_transformed(transform, Xtr, ytr, Xval, yval, seed, atr=None, balanced=False):
    Xt, Xv = transform(Xtr), transform(Xval)
    if balanced:
        clf, C, s = fit_balanced(Xt, ytr, atr, Xv, yval, seed)
    else:
        clf, C, s = fit_erm(Xt, ytr, Xv, yval, seed)
    return TransformHead(transform, clf), C, s


# ----------------------------------------------------------------------------- consistency (Phase 2)
def fit_consistency(Xo, Xi, y, Xval, yval, seed, lam, device, epochs=500, lr=0.03, weight_decay=1e-4):
    """BCE(orig) + BCE(repaired) + λ·mean((p_orig - p_rep)^2); deploy on original features.

    Identical to the pilot's Phase-2 head (zero init, label pos_weight, loss-plateau stop).
    lam=0 gives the λ=0 control.
    """
    seed_all(seed + int(lam * 10000))
    xo = torch.as_tensor(Xo, dtype=torch.float32, device=device)
    xi = torch.as_tensor(Xi, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    model = torch.nn.Linear(Xo.shape[1], 1).to(device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    npos, nneg = max(1, float(y.sum())), max(1, float(len(y) - y.sum()))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([nneg / npos], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state, best_loss, stale = None, float("inf"), 0
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        lo, li = model(xo).squeeze(1), model(xi).squeeze(1)
        loss = bce(lo, yt) + bce(li, yt) + lam * torch.mean((torch.sigmoid(lo) - torch.sigmoid(li)) ** 2)
        loss.backward()
        opt.step()
        v = float(loss.detach())
        if v < best_loss - 1e-7:
            best_loss, stale = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 80 and epoch >= 150:
            break
    model.load_state_dict(best_state)
    head = TorchLinearHead(model.weight.detach().cpu().numpy().reshape(-1), float(model.bias.detach().cpu()))
    return head, lam, float(roc_auc_score(yval, head.predict_proba(Xval)[:, 1]))


def fit_consistency_select(Xo, Xi, y, Xval, yval, seed, device, lambdas=(0.1, 0.5, 1.0)):
    best = None
    for lam in lambdas:
        head, _, auc = fit_consistency(Xo, Xi, y, Xval, yval, seed, lam, device)
        if best is None or (auc, -lam) > best[0]:
            best = ((auc, -lam), head, lam, auc)
    return best[1], best[2], best[3]


class PrevalenceCalibrated:
    """Post-hoc prevalence-equalised recalibration (Kina & Petersen 2026, arXiv:2609.07922), linear-probe form.

    The ERM head implicitly calibrates to the training prevalence *within* each shortcut group a.
    We remove the group-specific prior: logit'(x) = logit(x) - [logit P(Y=1|A=a)_train - logit P(Y=1)_train].
    Needs the image-level artifact label A at *test* time.
    """

    def __init__(self, clf, ytr, atr):
        self.clf = clf
        p = np.clip(np.mean(ytr), 1e-4, 1 - 1e-4)
        base = np.log(p / (1 - p))
        self.shift = {}
        for a in (0, 1):
            q = np.clip(np.mean(ytr[atr == a]) if (atr == a).any() else p, 1e-4, 1 - 1e-4)
            self.shift[a] = np.log(q / (1 - q)) - base
        self.a_test = None

    def predict_proba_groups(self, X, a):
        pr = np.clip(self.clf.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
        z = np.log(pr / (1 - pr)) - np.where(np.asarray(a) == 1, self.shift[1], self.shift[0])
        p = 1 / (1 + np.exp(-z))
        return np.c_[1 - p, p]


def fit_leace_conditional(X: np.ndarray, a: np.ndarray, y: np.ndarray):
    """Class-conditional LEACE: erase the artifact concept *within each diagnosis class*.

    Features are centred on their class mean before fitting, so the eraser removes directions along which
    the artifact varies given Y and cannot remove the between-class (diagnostic) mean difference that
    unpaired erasure removes when A and Y are correlated. Needs image-level A at train only.
    """
    from concept_erasure import LeaceEraser

    Xc = X.astype(np.float64).copy()
    for c in np.unique(y):
        Xc[y == c] -= Xc[y == c].mean(0)
    # balance the concept within class: weight-free approximation by using centred features
    return LeaceEraser.fit(torch.as_tensor(Xc), torch.as_tensor(a, dtype=torch.float64))


def pseudo_artifact_labels(X_orig, X_ins, X_train, a_true=None, seed=0):
    """Annotation-free artifact labels: logistic 'overlay detector' on (original, inserted) feature pairs,
    applied to the training images; binarised by Otsu's threshold on the logits.
    Returns (pseudo labels, AUROC vs the true labels (diagnostic only; nan if unknown))."""
    Xd = np.vstack([X_orig, X_ins])
    yd = np.r_[np.zeros(len(X_orig)), np.ones(len(X_ins))]
    det = _logreg(1.0, seed, None).fit(Xd, yd)
    s = det.decision_function(X_train)
    # Otsu on a 256-bin histogram of the logits
    hist, edges = np.histogram(s, 256)
    mid = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mid) / np.maximum(w0, 1); m1 = (np.sum(hist * mid) - np.cumsum(hist * mid)) / np.maximum(w1, 1)
    t = mid[np.argmax(w0 * w1 * (m0 - m1) ** 2)]
    pa = (s > t).astype(int)
    auc = float("nan")
    if a_true is not None and len(np.unique(a_true)) == 2:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(a_true, s))
    return pa, auc


def fit_jtt(Xtr, ytr, Xval, yval, seed, ups=(5.0, 20.0, 50.0)):
    """Just Train Twice (Liu et al. 2021), group-label-free: ERM -> training errors (threshold selected on clean
    validation) -> retrain with errors upweighted by λ; λ and C chosen by clean-validation AUROC."""
    from .evaluation import select_threshold_clean_val
    first, _, _ = fit_erm(Xtr, ytr, Xval, yval, seed)
    thr, _ = select_threshold_clean_val(yval, first.predict_proba(Xval)[:, 1])
    err = (first.predict_proba(Xtr)[:, 1] >= thr).astype(int) != ytr
    best = None
    for lam in ups:
        w = np.where(err, lam, 1.0)
        clf, C, s = select_C(lambda C: _logreg(C, seed, "balanced").fit(Xtr, ytr, sample_weight=w), Xval, yval)
        if best is None or s > best[2]:
            best = (clf, C, s)
    return best


class SpliceProjection:
    """SPLICE-style task-preserving linear concept removal (oblique projection; 'Preserving task-relevant
    information under linear concept removal', 2025). With a = Cov(X, z) (concept) and b = Cov(X, y) (task),
    P = I − a wᵀ / (wᵀ a) with w = a − (aᵀb / bᵀb) b, so that P a = 0 (no linear covariance with z) and
    P b = b (covariance with the task label preserved). Needs concept (artifact) labels."""

    def __init__(self, X, z, y):
        X = np.asarray(X, np.float64)
        self.mu = X.mean(0)
        Xc = X - self.mu
        a = Xc.T @ (np.asarray(z, float) - np.mean(z)) / len(X)
        b = Xc.T @ (np.asarray(y, float) - np.mean(y)) / len(X)
        w = a - (a @ b) / (b @ b) * b
        self.a, self.w, self.den = a, w, float(w @ a)

    def __call__(self, X):
        Xc = np.asarray(X, np.float64) - self.mu
        return (Xc - np.outer(Xc @ self.w, self.a) / self.den + self.mu).astype(np.float32)
