"""Our arms in the P1 location test (amendments A15, A16), one position at a time:
  a) mask_cmc on their trained Standard network's features of circle-masked images (saved by p1_location.py score);
  b) DINOv2 ViT-B/14 @ 518 on lungs u heart-masked images (tube drawn as in their run) + mask_cmc;
  c) U-MtE (ROI = lungs u heart), label-free, class-balanced logistic head.
mask_cmc artifact label = the pasted-tube indicator; mode and C chosen on P1's validation split (reversed presence) by AUROC.
Environments: train (tube 90/10), valid (10/90), test sets corr (90/10), rev (10/90), clean (none); presence and placement
are the same stable hashes as p1_location.py (checked against its saved tube column).

    python scripts/replication/p1_location_arms.py --position inside|outside
    python scripts/replication/p1_location_arms.py --position ... --model last   # A17 (post hoc): arm a on the epoch-100 net only
Writes $WTSS_DATA/replication/viviano/location/<position>/arms/preds_<arm>.csv.gz (test sets) and summary.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "round8")]
D = os.environ.get("WTSS_DATA", "/workspace/data")
L = Path(f"{D}/replication/viviano/location")
SEED, W, H = 1234, 43, 6
P_TUBE = {"train": (0.1, 0.9), "valid": (0.9, 0.1), "corr": (0.1, 0.9), "rev": (0.9, 0.1), "clean": (0.0, 0.0)}
SPLIT = {"train": "train", "valid": "valid", "corr": "test", "rev": "test", "clean": "test"}
CS = [0.01, 0.1, 1.0, 10.0]


def presence(keys, ys, env):
    from wtss.utils import stable_int
    p0, p1 = P_TUBE[env]
    return np.array([stable_int("p1loc", k, int(y), SEED, env) / (2**32 - 1) < (p1 if y == 1 else p0) for k, y in zip(keys, ys)])


def gray(path):
    a = cv2.imread(f"{D}/cxr_at/{path}", cv2.IMREAD_UNCHANGED)
    a = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if a.dtype != np.uint8 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", choices=["inside", "outside"], required=True)
    ap.add_argument("--model", choices=["best", "last"], default="best")   # last: A17, arm a only (b, c do not use their net)
    args = ap.parse_args()
    if args.model == "best":   # A17 (last) runs arm a only: a CPU head on saved features
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRACTION", "0.3")))
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import candidates as CA
    from wtss.backbones import load_backend
    from wtss.features import extract_view
    from wtss.ops import apply_roi_mask
    from wtss.synthetic import draw_tube
    from wtss.umte import UMtE

    pos = args.position
    sfx = "" if args.model == "best" else "_last"
    out = L / pos / f"arms{sfx}"; out.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(L / "nospc_1234.csv").sort_values(["split", "pos"])
    sp["key"] = sp.site.astype(str) + "/" + sp.image_id
    pl = pd.read_csv(L / "placements_1234.csv").set_index("key")
    z = np.load(L / "roi224_1234.npz"); roi = dict(zip(z["ids"], z["masks"]))
    env = {e: sp[sp.split == SPLIT[e]].reset_index(drop=True) for e in P_TUBE}
    for e, d in env.items():
        d["tube"] = presence(d.key, d.y, e)
    their = pd.read_csv(L / pos / "scores.csv.gz")
    for e in ("corr", "rev", "clean"):      # same presence and image order as their run
        t = their[(their.method == "masked") & (their.test_set == e)]
        assert (t.image_id.to_numpy() == env[e].key.to_numpy()).all() and (t.tube.to_numpy() == env[e].tube.astype(int)).all(), e
    y = {e: d.y.to_numpy() for e, d in env.items()}
    a = {e: d.tube.astype(int).to_numpy() for e, d in env.items()}
    summary = {"position": pos}

    def save(arm, prob_by_env, extra):
        rows = [pd.DataFrame({"image_id": env[e].key, "label": y[e], "tube": a[e], "method": arm, "test_set": e,
                              "score": prob_by_env[e]}) for e in ("corr", "rev", "clean")]
        pd.concat(rows).to_csv(out / f"preds_{arm}.csv.gz", index=False)
        summary[arm] = {**{f"{e}_auroc": float(roc_auc_score(y[e], prob_by_env[e])) for e in ("corr", "rev", "clean")}, **extra}
        print(arm, summary[arm], flush=True)

    def mask_cmc(X):
        best = None
        for mode in CA.CMC_MODES:
            h, C, v = CA.fit_cmc(X["train"], y["train"], a["train"], X["valid"], y["valid"], SEED, mode)
            if best is None or v > best[3]:
                best = (h, mode, C, v)
        return best

    # a) their network, circle-masked features (every split), mask_cmc
    F = np.load(L / pos / f"features_masked{sfx}.npz")
    h, mode, C, v = mask_cmc({e: F[e] for e in P_TUBE})
    save("a_mask_cmc_their_net", {e: h.prob(F[e]) for e in ("corr", "rev", "clean")}, {"mode": mode, "C": C, "val_auroc": v})
    if args.model == "last":
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        return

    # b) DINOv2 @ 518 on lungs u heart-masked images with the tube drawn as in their run
    dev = torch.device("cuda")
    bb = load_backend("dino518", dev)

    def render(e):
        d = env[e]
        def r(i):
            k = int(i); row = d.iloc[k]
            im = Image.fromarray(gray(row.path)).convert("RGB")
            if row.tube:
                p = pl.loc[row.key]
                x, yy = (int(p.in_x), int(p.in_y)) if pos == "inside" else (int(p.out_x), int(p.out_y))
                im, _ = draw_tube(im, x, yy, W, H, row.key)
            im = im.resize((518, 518), Image.BICUBIC)
            m = cv2.resize(roi[row.key], (518, 518), interpolation=cv2.INTER_NEAREST).astype(bool)
            return apply_roi_mask(im, m)
        return r
    X = {e: extract_view(bb, [str(k) for k in range(len(env[e]))], render(e), out / f"dino518_{e}.npz", dev,
                         batch_size=16, num_workers=12) for e in P_TUBE}
    h, mode, C, v = mask_cmc(X)
    save("b_dino518_mask_cmc", {e: h.prob(X[e]) for e in ("corr", "rev", "clean")}, {"mode": mode, "C": C, "val_auroc": v})

    # c) U-MtE, label-free: fit on training images (tubes as in their run) + ROIs; class-balanced head, C on validation
    def pil(e):
        ims = []
        for _, row in env[e].iterrows():
            im = Image.fromarray(gray(row.path)).convert("RGB")
            if row.tube:
                p = pl.loc[row.key]
                x, yy = (int(p.in_x), int(p.in_y)) if pos == "inside" else (int(p.out_x), int(p.out_y))
                im, _ = draw_tube(im, x, yy, W, H, row.key)
            ims.append(im)
        return ims
    rois = {e: [roi[k].astype(bool) for k in env[e].key] for e in P_TUBE}
    um = UMtE(bb, device="cuda", seed=SEED, batch_size=16)
    um.fit(pil("train"), rois["train"], keys=list(env["train"].key))
    Z = {"train": um.eraser(um._train_features), **{e: um.transform(pil(e), rois[e]) for e in ("valid", "corr", "rev", "clean")}}
    best = None
    for c in CS:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=SEED).fit(Z["train"], y["train"])
        vv = roc_auc_score(y["valid"], clf.predict_proba(Z["valid"])[:, 1])
        if best is None or vv > best[2]:
            best = (clf, c, vv)
    save("c_umte", {e: best[0].predict_proba(Z[e])[:, 1] for e in ("corr", "rev", "clean")}, {"C": best[1], "val_auroc": best[2], "k": int(um.eraser.k)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
