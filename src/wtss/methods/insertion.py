"""Insert-to-Erase (I2E): localisation-free shortcut erasure by counterfactual artifact *insertion*.

Motivation (thesis §10): removing an in-ROI artifact needs its pixels (a detector reached IoU 0.22),
and paired erasure from *removal* pairs is rank-1 and data-starved. Insertion needs only a template
library: any image can receive an artifact anywhere — including inside the ROI — so every training
image yields a counterfactual pair (x, x+) whose feature difference Δ = f(x+) - f(x) is, by
construction, independent of the label.

I2E erases the *subspace* spanned by these differences rather than a single mean direction:

    Δ_i = f(x_i^+) - f(x_i),   Σ_Δ = E[Δ Δ^T],   U_k = top-k eigenvectors of Σ_Δ
    P(x) = x - U_k U_k^T (x - μ)

k is chosen WITHOUT labels: the smallest k whose held-out paired-difference energy remaining,
E||P Δ||² / E||Δ||², is ≤ 1 - energy (default energy 0.90). A linear head is then fitted on P(f(x))
(ERM; or group-balanced when image-level artifact labels exist: "i2e+balanced").

Also provided (ablations):
  * i2e_rank1  — LEACE on inserted pairs (binary concept; the rank-1 special case);
  * insert_aug — "correlation breaking" (thesis WP3 route 2): add inserted copies of a random,
                 label-independent 50% of training images to the head's training set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from .. import heads as H


@dataclass
class SubspaceEraser:
    U: np.ndarray  # (d, k)
    mu: np.ndarray  # (d,)
    k: int
    energy_curve: np.ndarray  # held-out energy remaining for k = 0..max_k

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self.k == 0:
            return X.astype(np.float32)
        Z = X - self.mu
        return (X - (Z @ self.U) @ self.U.T).astype(np.float32)


def fit_difference_subspace(X0: np.ndarray, X1: np.ndarray, energy: float = 0.90, max_k: int = 64,
                            holdout: float = 0.2, seed: int = 0, mu: np.ndarray | None = None) -> SubspaceEraser:
    """Fit I2E on paired features X0 (original) / X1 (inserted); rows are pairs."""
    D = (X1 - X0).astype(np.float64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(D))
    n_ho = max(1, int(holdout * len(D)))
    ho, tr = D[perm[:n_ho]], D[perm[n_ho:]]
    # uncentred second moment: we erase the mean shift *and* the variation of the shift
    C = tr.T @ tr / len(tr)
    w, V = np.linalg.eigh(C)
    V = V[:, ::-1][:, :max_k]
    tot = (ho ** 2).sum()
    proj = ho @ V
    remaining = 1 - np.r_[0.0, np.cumsum((proj ** 2).sum(0))] / tot
    ok = np.flatnonzero(remaining <= 1 - energy)
    k = int(ok[0]) if len(ok) else max_k
    # refit on all pairs with the chosen k
    Call = D.T @ D / len(D)
    w, V = np.linalg.eigh(Call)
    U = V[:, ::-1][:, :k].astype(np.float32)
    mu = (X0.mean(0) if mu is None else mu).astype(np.float32)
    return SubspaceEraser(U, mu, k, remaining)


def i2e_head(X0_pairs, X1_pairs, Xtr, ytr, Xval, yval, seed, atr=None, balanced=False, energy=0.90, max_k=64):
    er = fit_difference_subspace(X0_pairs, X1_pairs, energy=energy, max_k=max_k, seed=seed)
    clf, C, vauc = H.fit_on_transformed(er, Xtr, ytr, Xval, yval, seed, atr=atr, balanced=balanced)
    return clf, C, vauc, {"k": er.k, "energy_remaining_at_k": float(er.energy_curve[min(er.k, len(er.energy_curve) - 1)])}


def rank1_head(X0_pairs, X1_pairs, Xtr, ytr, Xval, yval, seed):
    er = H.fit_leace(X0_pairs, X1_pairs)
    clf, C, vauc = H.fit_on_transformed(H.eraser_fn(er), Xtr, ytr, Xval, yval, seed)
    return clf, C, vauc, {"k": 1}


def insert_aug_head(Xtr, ytr, Xtr_ins, Xval, yval, seed, frac=0.5):
    """Train on original + inserted copies of a label-independent random subset."""
    rng = np.random.default_rng(seed + 7)
    pick = rng.random(len(ytr)) < frac
    Xa = np.vstack([Xtr, Xtr_ins[pick]])
    ya = np.r_[ytr, ytr[pick]]
    clf, C, vauc = H.fit_erm(Xa, ya, Xval, yval, seed)
    return clf, C, vauc, {"n_aug": int(pick.sum())}


# ----------------------------------------------------------------------------- synthetic hook
def synthetic_extra_arms(cfg) -> Dict[str, Callable]:
    """Arms for the synthetic driver. The insertion view is a *different* template library from
    the trap artifact: random-style rulers (or tubes) at random positions (not the trap placement)."""
    from ..features import extract_view
    from ..synthetic import draw_ruler_variable, draw_tube, sample_ruler_style
    from ..utils import stable_int

    state: Dict[str, np.ndarray] = {}

    def insertion_view(ctx):
        views = ctx["views"]
        key = f"insert_random_{cfg.artifact.split('_')[0]}"
        if key in state:
            return state[key]
        size = views.backend.size
        w, h = cfg.geometry[size]

        def render(i):
            img = views.cache.image(i)
            rng = np.random.default_rng(stable_int("i2e_insert", i))
            x = int(rng.integers(0, size - w)); y = int(rng.integers(0, size - h))
            if cfg.artifact.startswith("ruler"):
                style = sample_ruler_style(f"i2e|{i}", 0.5)
                img, _ = draw_ruler_variable(img, x, y, w, h, style)
            else:
                img, _ = draw_tube(img, x, y, w, h, image_id=f"i2e|{i}")
            return img

        X = extract_view(views.backend, views.ids, render, views.dir / f"{key}.npz", views.device,
                         cfg.batch_size, cfg.workers, desc=key)
        state[key] = X
        return X

    def _pairs(ctx):
        tr = ctx["sp"]["train"]
        Xins = insertion_view(ctx)
        return ctx["Xc"][tr], Xins[tr], Xins

    def arm_i2e(ctx):
        X0, X1, _ = _pairs(ctx)
        va = ctx["sp"]["val"]
        return i2e_head(X0, X1, ctx["Xtr"], ctx["ytr"], ctx["Xc"][va], ctx["y"][va], ctx["seed"])

    def arm_i2e_bal(ctx):
        X0, X1, _ = _pairs(ctx)
        va = ctx["sp"]["val"]
        return i2e_head(X0, X1, ctx["Xtr"], ctx["ytr"], ctx["Xc"][va], ctx["y"][va], ctx["seed"],
                        atr=ctx["a_tr"], balanced=True)

    def arm_rank1(ctx):
        X0, X1, _ = _pairs(ctx)
        va = ctx["sp"]["val"]
        return rank1_head(X0, X1, ctx["Xtr"], ctx["ytr"], ctx["Xc"][va], ctx["y"][va], ctx["seed"])

    def arm_aug(ctx):
        _, _, Xins = _pairs(ctx)
        tr, va = ctx["sp"]["train"], ctx["sp"]["val"]
        # inserted copy of the clean image (random position and style)
        return insert_aug_head(ctx["Xtr"], ctx["ytr"], Xins[tr], ctx["Xc"][va], ctx["y"][va], ctx["seed"])

    def mask_insertion_view(ctx):
        """Masked image with a random-style artifact inserted at a random position whose centre is in the ROI."""
        views = ctx["views"]
        key = f"mask_insert_roi_{cfg.artifact.split('_')[0]}"
        if key in state:
            return state[key]
        from ..ops import apply_roi_mask
        size = views.backend.size
        w, h = cfg.geometry[size]

        def render(i):
            img, roi = views.cache.image(i), views.cache.roi_mask(i)
            rng = np.random.default_rng(stable_int("mte_insert", i))
            ys, xs = np.nonzero(roi)
            if len(ys):
                j = int(rng.integers(0, len(ys)))
                x = int(np.clip(xs[j] - w // 2, 0, size - w)); y = int(np.clip(ys[j] - h // 2, 0, size - h))
            else:
                x = int(rng.integers(0, size - w)); y = int(rng.integers(0, size - h))
            if cfg.artifact.startswith("ruler"):
                img, _ = draw_ruler_variable(img, x, y, w, h, sample_ruler_style(f"mte|{i}", 0.5))
            else:
                img, _ = draw_tube(img, x, y, w, h, image_id=f"mte|{i}")
            return apply_roi_mask(img, roi)

        X = extract_view(views.backend, views.ids, render, views.dir / f"{key}.npz", views.device,
                         cfg.batch_size, cfg.workers, desc=key)
        state[key] = X
        return X

    def _mte(ctx, balanced):
        tr, va = ctx["sp"]["train"], ctx["sp"]["val"]
        Xm_c, Em = ctx["env_X"]("mask")
        Xmi = mask_insertion_view(ctx)
        er = fit_difference_subspace(Xm_c[tr], Xmi[tr], energy=0.9, seed=ctx["seed"])
        Xtr = Em[ctx["train_env"]][0][tr]
        clf, C, vauc = H.fit_on_transformed(er, Xtr, ctx["ytr"], Xm_c[va], ctx["y"][va], ctx["seed"],
                                            atr=ctx["a_tr"], balanced=balanced)
        return clf, C, vauc, {"k": er.k, "_view": "mask"}

    def generic_view(ctx, masked: bool):
        """Universal (artifact-agnostic) insertion: same procedural library for every artifact/modality."""
        views = ctx["views"]
        key = "generic_insert_masked" if masked else "generic_insert"
        if key in state:
            return state[key]
        from ..ops import apply_roi_mask
        from ..synthetic import draw_generic_artifact

        def render(i):
            img, roi = views.cache.image(i), views.cache.roi_mask(i)
            out = draw_generic_artifact(img, roi, f"u|{i}")
            return apply_roi_mask(out, roi) if masked else out

        X = extract_view(views.backend, views.ids, render, views.dir / f"{key}.npz", views.device,
                         cfg.batch_size, cfg.workers, desc=key)
        state[key] = X
        return X

    def _ui2e(ctx, balanced):
        tr, va = ctx["sp"]["train"], ctx["sp"]["val"]
        er = fit_difference_subspace(ctx["Xc"][tr], generic_view(ctx, False)[tr], energy=0.9, seed=ctx["seed"])
        clf, C, vauc = H.fit_on_transformed(er, ctx["Xtr"], ctx["ytr"], ctx["Xc"][va], ctx["y"][va], ctx["seed"],
                                            atr=ctx["a_tr"], balanced=balanced)
        return clf, C, vauc, {"k": er.k}

    def _umte(ctx, balanced):
        tr, va = ctx["sp"]["train"], ctx["sp"]["val"]
        Xm_c, Em = ctx["env_X"]("mask")
        er = fit_difference_subspace(Xm_c[tr], generic_view(ctx, True)[tr], energy=0.9, seed=ctx["seed"])
        clf, C, vauc = H.fit_on_transformed(er, Em[ctx["train_env"]][0][tr], ctx["ytr"], Xm_c[va], ctx["y"][va],
                                            ctx["seed"], atr=ctx["a_tr"], balanced=balanced)
        return clf, C, vauc, {"k": er.k, "_view": "mask"}

    return {"i2e": arm_i2e, "i2e_balanced": arm_i2e_bal, "i2e_rank1": arm_rank1, "insert_aug": arm_aug,
            "mte": lambda ctx: _mte(ctx, False), "mte_balanced": lambda ctx: _mte(ctx, True),
            "ui2e": lambda ctx: _ui2e(ctx, False), "ui2e_balanced": lambda ctx: _ui2e(ctx, True),
            "umte": lambda ctx: _umte(ctx, False), "umte_balanced": lambda ctx: _umte(ctx, True)}


# ----------------------------------------------------------------------------- disease-protected erasure
def disease_directions(Xclean: np.ndarray, yclean: np.ndarray, m: int = 5, C: float = 0.1, seed: int = 0) -> np.ndarray:
    """(d, m') orthonormal basis of the label directions of ARTIFACT-FREE images: logistic weights on m bootstrap
    resamples (captures a small disease subspace, not just one direction)."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed + 991)
    Ws = []
    for b in range(m):
        idx = rng.integers(0, len(yclean), len(yclean)) if b else np.arange(len(yclean))
        if len(np.unique(yclean[idx])) < 2:
            continue
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, solver="liblinear",
                                 random_state=seed).fit(Xclean[idx], yclean[idx])
        Ws.append(clf.coef_.ravel())
    Q, R = np.linalg.qr(np.stack(Ws, 1))
    keep = np.abs(np.diag(R)) > 1e-6 * np.abs(np.diag(R)).max()
    return Q[:, keep].astype(np.float32)


def protect(er: SubspaceEraser, W: np.ndarray) -> SubspaceEraser:
    """Remove the disease subspace W from the erased subspace U: U' = orth((I - W W^T) U).
    Guarantees w^T P(x) = w^T x for every w in span(W): a head fitted on artifact-free images is untouched."""
    if er.k == 0:
        return er
    Up = er.U - W @ (W.T @ er.U)
    Q, R = np.linalg.qr(Up)
    keep = np.abs(np.diag(R)) > 1e-6
    return SubspaceEraser(Q[:, keep].astype(np.float32), er.mu, int(keep.sum()), er.energy_curve)
