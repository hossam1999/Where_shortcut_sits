"""SLAS step 2 — how many annotated images does the token probe need? (learning curve)

Probe = logistic regression on DINOv2@518 patch tokens, trained on k images with real Wegley hair/ruler masks
(k in 5..500, 5 random draws each), evaluated on 1000 held-out images from disjoint leakage groups.
Also: generic-overlay probe and generic + k real (does synthetic pre-training help at low k?).

  python scripts/analysis/slas_fewshot.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from wtss import paths
from wtss.backbones import load_dino
from wtss.experiments.real_traps import RealCache
from wtss.methods.token_suppression import _Imgs, _patch_frac, dino_tokens

C = paths.DATA / "isic2019" / "prepared"
KS = (5, 10, 25, 50, 100, 250, 500)


@torch.inference_mode()
def tokens(model, dev, ids, cache, bs=32):
    """(n, N, D) fp16 tokens, (n, N) artifact coverage, (n, N) ROI flag."""
    from PIL import Image

    def rend(i):
        return Image.fromarray(cache.get(i)[0])

    ds = _Imgs(ids, rend, load_dino.__globals__["dino_preprocess"], lambda i: cache.get(i)[1] > 0,
               lambda i: (cache.get(i)[2] > 0).astype("uint8"))
    T, V, R = [], [], []
    for x, roi, art, _ in DataLoader(ds, batch_size=bs, num_workers=4):
        with torch.autocast("cuda", dtype=torch.float16):
            t = dino_tokens(model, x.to(dev))
        g = int(round(t.shape[1] ** 0.5))
        T.append(t.half().cpu().numpy()); V.append(_patch_frac(art.to(dev), g).cpu().numpy())
        R.append((_patch_frac(roi.to(dev), g) >= 0.5).cpu().numpy())
    return np.concatenate(T), np.concatenate(V), np.concatenate(R)


def sample(T, V, rng, per_img=150):
    X, Y = [], []
    for t, v in zip(T, V):
        pos, neg = np.flatnonzero(v >= 0.2), np.flatnonzero(v == 0)
        if len(pos) == 0:
            continue
        p = rng.choice(pos, min(len(pos), per_img // 2), replace=False)
        q = rng.choice(neg, min(len(neg), per_img // 2), replace=False)
        X.append(t[np.r_[p, q]].astype(np.float32)); Y.append(np.r_[np.ones(len(p)), np.zeros(len(q))])
    return np.concatenate(X), np.concatenate(Y)


def score(w, b, T, V, R):
    s = (T.reshape(-1, T.shape[-1]).astype(np.float32) @ w + b).reshape(V.shape)
    lab = np.where(V >= 0.2, 1, np.where(V == 0, 0, -1))
    out = {}
    for scope, sel in (("all", lab >= 0), ("roi", (lab >= 0) & R)):
        y, p = lab[sel], s[sel]
        # IoU at the threshold giving 5 % FPR on clean patches (threshold calibrated on the TRAIN images)
        out[scope] = {"auroc": float(roc_auc_score(y, p))}
    return out, s


def main():
    dev = torch.device("cuda")
    c = pd.read_csv(C / "cohort_spec.csv")
    cache = RealCache(C / "cache_518", roi_file="roi_spec.npy", art_file="hair.npy")
    rng = np.random.default_rng(2026)
    groups = c.group.unique()
    rng.shuffle(groups)
    g_tr = set(groups[: len(groups) // 2])  # same split as slas_localisation.py
    has = (c.hair_frac >= 0.02) & (c.has_hair_mask == True)
    ev_pool = c[~c.group.isin(g_tr) & has].image_id.to_numpy()
    ev = rng.choice(ev_pool, 1000, replace=False)
    tr_pool = c[c.group.isin(g_tr) & has].image_id.to_numpy()
    bk = load_dino(518, dev)
    Te, Ve, Re = tokens(bk.model, dev, ev, cache)
    tr_all = np.random.default_rng(7).choice(tr_pool, max(KS), replace=False)
    Tt, Vt, _ = tokens(bk.model, dev, tr_all, cache)
    rows = []
    for k in KS:
        for rep in range(5 if k < max(KS) else 1):
            r = np.random.default_rng(100 * k + rep)
            idx = r.choice(len(tr_all), k, replace=False)
            X, Y = sample(Tt[idx], Vt[idx], r, per_img=max(150, 30000 // k))
            clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(X, Y)
            sc, _ = score(clf.coef_.ravel().astype(np.float32), float(clf.intercept_[0]), Te, Ve, Re)
            rows.append({"k": k, "rep": rep, "auroc_all": sc["all"]["auroc"], "auroc_roi": sc["roi"]["auroc"]})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    out = paths.ensure(paths.RESULTS / "slas")
    df.to_csv(out / "fewshot_isic_hair.csv", index=False)
    print(df.groupby("k")[["auroc_all", "auroc_roi"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
