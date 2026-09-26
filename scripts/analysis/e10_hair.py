"""E10 — real hair: linear decodability (Bissoto labels), paired LEACE on Mendeley manual hair masks, and the
pre-committed DullRazor-style detector (kernel 17, threshold 10, 3x3 open). DINOv2 ViT-B/14 @518.

Spec: decodability test AUROC 0.943 [0.919, 0.964] (n=356, base rate 0.567); LEACE rank 1/768; A-decoder
0.968 -> 0.500; clean melanoma AUROC 0.812 -> 0.814; melanoma-head |Δp| 0.072 -> 0.058; detector IoU mean 0.22,
median 0.18, precision 0.32, recall 0.46.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from wtss import heads as H
from wtss import paths
from wtss.backbones import load_backend
from wtss.data.isic2018 import isic2018_image_path, load_isic2018_pilot
from wtss.features import extract_view
from wtss.ops import apply_inpaint
from wtss.utils import load_rgb

S = 518
MH = next((paths.DATA / "mendeley_hair").glob("A skin lesion*"))
BISSOTO = Path("/root/isic_pcam_code_results/isic_overlap_pilot_patch_v3/results_v3/hair_linear_leace/external/bissoto_isic_bias.csv")
out = paths.ensure(paths.RESULTS / "analysis" / "E10")
dev = torch.device("cuda")


def boot_auc(y, s, n=10000, seed=0):
    rng = np.random.default_rng(seed); b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def detect(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    _, b = cv2.threshold(bh, 10, 255, cv2.THRESH_BINARY)
    return (cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0).astype(np.uint8)


coh = load_isic2018_pilot(common_support=True).df
F = paths.CACHE / "features" / "isic2018_pilot" / "dinov2_b14_518" / "ruler_fixed" / "erm_clean.npz"
z = np.load(F); X = z["X"]; pos = {k: j for j, k in enumerate(z["ids"].astype(str))}
res = {}
# ---------- H1: decodability of Bissoto hair presence
bi = pd.read_csv(BISSOTO, sep=";")  # Bissoto et al. isic_bias.csv (semicolon-separated)
bi["image_id"] = bi["image"].astype(str).str.extract(r"(ISIC_\d+)")[0]
d = coh.merge(bi[["image_id", "hair"]], on="image_id")
sp = {s: d[d.split == s] for s in ("train", "val", "test")}
Xs = {s: X[[pos[i] for i in q.image_id]] for s, q in sp.items()}
best = max(((roc_auc_score(sp["val"].hair, LogisticRegression(C=C, max_iter=3000, solver="liblinear").fit(Xs["train"], sp["train"].hair).predict_proba(Xs["val"])[:, 1]), C)
            for C in (0.01, 0.1, 1, 10)))
clf = LogisticRegression(C=best[1], max_iter=3000, solver="liblinear").fit(Xs["train"], sp["train"].hair)
pt = clf.predict_proba(Xs["test"])[:, 1]; yt = sp["test"].hair.to_numpy()
res["H1"] = {"auroc": roc_auc_score(yt, pt), "ci": boot_auc(yt, pt), "n_test": int(len(yt)), "base_rate": float(yt.mean()),
             "n_train": int(len(sp["train"])), "n_val": int(len(sp["val"])),
             "P_hair_mel": float(sp["test"][sp["test"].y == 1].hair.mean()), "P_hair_ben": float(sp["test"][sp["test"].y == 0].hair.mean())}
print("H1", res["H1"], flush=True)

# ---------- Mendeley pairs + detector
man = pd.read_csv(paths.FROZEN / "isic2018_pilot_manifest.frozen.csv")
mids = sorted(p.stem for p in (MH / "hair_mask").glob("*"))
join = man[man.image_id.isin(mids)][["image_id", "split", "MEL"]].rename(columns={"MEL": "y"}).reset_index(drop=True)
mask_file = {p.stem: p for p in (MH / "hair_mask").glob("*")}
rgb = {i: np.asarray(load_rgb(isic2018_image_path(i), S)) for i in join.image_id}
gt = {i: (np.asarray(Image.open(mask_file[i]).convert("L").resize((S, S), Image.NEAREST)) >= 128).astype(np.uint8) for i in join.image_id}
det_rows = []
for i in join.image_id:
    p, g = detect(rgb[i]), gt[i]
    inter, union = int((p & g).sum()), int((p | g).sum())
    det_rows.append({"image_id": i, "iou": inter / union if union else 1.0, "precision": inter / p.sum() if p.sum() else np.nan,
                     "recall": inter / g.sum() if g.sum() else np.nan, "common_support": i in pos})
dd = pd.DataFrame(det_rows); dcs = dd[dd.common_support]
res["detector"] = {"n": int(len(dcs)), "mean_iou": float(dcs.iou.mean()), "median_iou": float(dcs.iou.median()),
                   "precision": float(dcs.precision.mean()), "recall": float(dcs.recall.mean()), "frac_iou_lt_0.3": float((dcs.iou < 0.3).mean())}
print("detector", res["detector"], flush=True)
be = load_backend("dino518", dev)
ids = join.image_id.tolist()
Xo = extract_view(be, ids, lambda i: Image.fromarray(rgb[i]), out / "mendeley_orig.npz", dev, 32, 0)
Xi = extract_view(be, ids, lambda i: apply_inpaint(Image.fromarray(rgb[i]), gt[i]), out / "mendeley_inpaint.npz", dev, 32, 0)
tr, va, te = (np.flatnonzero(join.split == s) for s in ("train", "val", "test"))
er = H.fit_leace(Xi[tr], Xo[tr])
E = H.eraser_fn(er)
P = E(np.eye(768, dtype=np.float32)) - E(np.zeros((1, 768), np.float32))
rank = int(np.linalg.matrix_rank(np.eye(768) - P, tol=1e-4))


def decoder(T):
    Z = lambda idx: (np.vstack([T(Xi[idx]), T(Xo[idx])]), np.r_[np.zeros(len(idx)), np.ones(len(idx))])
    Ztr, ytr = Z(tr); Zva, yva = Z(va); Zte, yte = Z(te)
    C = max(((roc_auc_score(yva, LogisticRegression(C=C, max_iter=3000).fit(Ztr, ytr).predict_proba(Zva)[:, 1]), C) for C in (0.01, 0.1, 1, 10)))[1]
    return roc_auc_score(yte, LogisticRegression(C=C, max_iter=3000).fit(Ztr, ytr).predict_proba(Zte)[:, 1])


# clean melanoma utility (common-support clean features) and melanoma-head |Δp| on test hair pairs
cs = {s: coh[coh.split == s] for s in ("train", "val", "test")}
Xc = {s: X[[pos[i] for i in q.image_id]] for s, q in cs.items()}
util, dps = {}, {}
for name, T in (("raw", lambda A: A), ("leace", E)):
    clf = H.fit_on_transformed(T, Xc["train"], cs["train"].y.to_numpy(), Xc["val"], cs["val"].y.to_numpy(), 42)[0]
    util[name] = roc_auc_score(cs["test"].y, clf.predict_proba(Xc["test"])[:, 1])
    dps[name] = float(np.abs(clf.predict_proba(Xo[te])[:, 1] - clf.predict_proba(Xi[te])[:, 1]).mean())
res["H2"] = {"n_join": len(join), "n_train_pairs": int(len(tr)), "n_test_pairs": int(len(te)), "leace_rank": rank,
             "A_decoder_raw": decoder(lambda A: A), "A_decoder_leace": decoder(E),
             "clean_auroc_raw": util["raw"], "clean_auroc_leace": util["leace"], "abs_dp_raw": dps["raw"], "abs_dp_leace": dps["leace"]}
print("H2", res["H2"], flush=True)
(out / "E10.json").write_text(json.dumps(res, indent=2, default=float))
dd.to_csv(out / "detector_per_image.csv", index=False)
