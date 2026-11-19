"""Exploratory track (docs/PREREGISTRATION_NEWMETHOD.md, DRAFT): development of candidate (a), class-conditional LEACE,
against mask_cmc. VALIDATION ONLY: features of training and validation images (cached); no test set is read.

Setting: P1 location test, tube inside the ROI (A15/A16): training presence 90/10, validation reversed (10/90), so a method
that removes the tube shortcut scores higher on validation. Encoders with cached train/valid features: arm a (their
network, circle-masked), arm b (DINOv2-518, lungs+heart-masked), arm d (EfficientNet-B0 trained on masked images).

CC-LEACE (Belrose et al. 2023, LEACE, with class-conditional moments): Σ_w = Σ_y π_y Cov(X | y), C_w = Σ_y π_y Cov(X, A | y);
W = (Σ_w + εI)^{-1/2}; r(x) = x − W⁺ P_{W C_w} W (x − μ). Head: class-balanced logistic regression, C in the A21 grid,
selected on the same validation split it is reported on (development only; the selection rule for registration is in
the draft).
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "round8"), str(ROOT / "scripts" / "replication")]
import candidates as CA  # noqa: E402

D = "/workspace/data"
CS = (0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0)


def ccleace(X, y, a, eps_rel=1e-3):
    X, a = np.asarray(X, np.float64), np.asarray(a, np.float64)
    d = X.shape[1]; Sw = np.zeros((d, d)); Cw = np.zeros((d, 1)); n = len(y)
    for c in np.unique(y):
        m = y == c
        if m.sum() < 3 or a[m].std() == 0:
            continue
        Xc = X[m] - X[m].mean(0); ac = a[m] - a[m].mean()
        Sw += Xc.T @ Xc / n; Cw += (Xc.T @ ac[:, None]) / n
    ev, U = np.linalg.eigh(Sw + eps_rel * np.trace(Sw) / d * np.eye(d))
    W = U @ np.diag(ev ** -0.5) @ U.T; Wi = U @ np.diag(ev ** 0.5) @ U.T
    v = W @ Cw; v /= np.linalg.norm(v); P = v @ v.T
    M = Wi @ P @ W; mu = X.mean(0)
    return lambda Z: np.asarray(Z, np.float64) - (np.asarray(Z, np.float64) - mu) @ M.T


def within_class_probe(Ztr, ytr, atr, Zv, yv, av):
    """AUROC of a within-class probe for the artifact on validation (average over classes; 0.5 = erased)."""
    out = []
    for c in np.unique(ytr):
        m, mv = ytr == c, yv == c
        if len(set(atr[m])) < 2 or len(set(av[mv])) < 2:
            continue
        p = LogisticRegression(C=1.0, max_iter=3000).fit(Ztr[m], atr[m])
        out.append(roc_auc_score(av[mv], p.decision_function(Zv[mv])))
    return float(np.mean(out))


def head(Ztr, ytr, Zv, yv):
    best = None
    for C in CS:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, solver="liblinear").fit(Ztr, ytr)
        v = roc_auc_score(yv, clf.decision_function(Zv))
        if best is None or v > best[0]:
            best = (v, C)
    return best


def load_p1loc():
    import armd_train as A
    S = A.P1LocInside()
    y = {e: S.labels(e)[0] for e in ("train", "valid")}; a = {e: S.labels(e)[1] for e in ("train", "valid")}
    L = Path(f"{D}/replication/viviano/location/inside")
    feats = {"a (their net)": {e: np.load(L / "features_masked.npz")[e] for e in ("train", "valid")},
             "b (DINOv2-518)": {e: np.load(L / f"arms/dino518_{e}.npz")["X"] for e in ("train", "valid")},
             "d (EfficientNet-B0)": {e: np.load(f"{D}/replication/armd/p1_loc_inside/features.npz")[e] for e in ("train", "valid")}}
    return feats, y, a


def main():
    feats, y, a = load_p1loc()
    rows = []
    for enc, F in feats.items():
        for name, make in (("none", lambda X, yy, aa: (lambda Z: Z)),
                           ("mask_cmc pooled", lambda X, yy, aa: CA.Projector(CA.cmc_directions(X, yy, aa, "pooled"))),
                           ("CC-LEACE", lambda X, yy, aa: ccleace(X, yy, aa))):
            T = make(F["train"], y["train"], a["train"])
            Ztr, Zv = T(F["train"]), T(F["valid"])
            v, C = head(Ztr, y["train"], Zv, y["valid"])
            pr = within_class_probe(Ztr, y["train"], a["train"], Zv, y["valid"], a["valid"])
            rows.append({"setting": "P1 location, inside", "encoder": enc, "method": name, "val_auroc (reversed)": round(v, 3),
                         "C": C, "within-class tube probe AUROC (val)": round(pr, 3)})
            print(rows[-1], flush=True)
    out = ROOT / "results/newmethod"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "dev_ccleace_p1loc_validation.csv", index=False)


if __name__ == "__main__" and not os.environ.get("ITER"):
    main()


def iterative(X, y, a, Xv, yv, av, rounds=12, eps_rel=1e-3):
    """CC-LEACE, then repeated rounds: fit a within-class tube probe (pooled over classes, class-centred features) on the
    erased training features, erase its direction in the whitened metric. Returns per-round (val AUROC, probe AUROC)."""
    T0 = ccleace(X, y, a, eps_rel)
    Z, Zv = T0(X), T0(Xv)
    out = []
    for r in range(rounds + 1):
        v, C = head(Z, y, Zv, yv)
        out.append({"round": r, "val_auroc": round(v, 3), "probe": round(within_class_probe(Z, y, a, Zv, yv, av), 3)})
        Zc = Z.copy()
        for c in np.unique(y):
            Zc[y == c] -= Z[y == c].mean(0)
        w = LogisticRegression(C=1.0, max_iter=3000).fit(Zc, a).coef_[0][:, None]
        d = Z.shape[1]; Sw = Zc.T @ Zc / len(Zc)
        ev, U = np.linalg.eigh(Sw + eps_rel * np.trace(Sw) / d * np.eye(d))
        W = U @ np.diag(ev ** -0.5) @ U.T; Wi = U @ np.diag(ev ** 0.5) @ U.T
        u = Wi @ w; u /= np.linalg.norm(u)   # the probe direction expressed in the whitened coordinates
        M = Wi @ (u @ u.T) @ W; mu = Z.mean(0)
        Z, Zv = Z - (Z - mu) @ M.T, Zv - (Zv - mu) @ M.T
    return out


if __name__ == "__main__" and os.environ.get("ITER"):
    feats, y, a = load_p1loc()
    rows = []
    for enc in ("a (their net)", "d (EfficientNet-B0)"):
        F = feats[enc]
        for r in iterative(F["train"], y["train"], a["train"], F["valid"], y["valid"], a["valid"]):
            rows.append({"encoder": enc, **r}); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/newmethod/dev_ccleace_iterative_p1loc_validation.csv", index=False)
