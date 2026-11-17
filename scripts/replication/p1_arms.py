"""Our arms on one P1 (Viviano et al. 2021) seed: a) mask_cmc on their trained network, b) DINOv2 ViT-B/14 @ 518 on
ROI-masked images + mask_cmc, c) U-MtE (label-free). docs/PREREGISTRATION_REPLICATION.md, section 3 and A5.

Inputs (from scripts/replication/viviano_score.py, run in P1's environment):
  <score dir>/<run>.csv.gz        image_id, split, site, y, logit0, logit1   (their model; one file per method)
  <score dir>/<arm-a run>_features.npz   penultimate features of their network on masked images, every split
The artifact label used at training time only: site (cardiomegaly: 0 NIH / 1 PadChest) or view (RSNA: 0 PA / 1 AP).
ROIs (label-free): TorchXRayVision PSPNet, lungs (RSNA) or lungs + heart (cardiomegaly), p > 0.5.
Selection: mask_cmc mode {pooled, per_class} and C {0.01, 0.1, 1, 10} by validation AUROC (P1's own validation split).
Outputs: $WTSS_DATA/replication/viviano/arms/<task>_<seed>/preds_<arm>.csv.gz (test scores) and a summary json.

    python scripts/replication/p1_arms.py --task xray|rsna --seed 1234 --arm_a_features <npz> --split_csv <csv.gz>
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "round8"), str(ROOT / "scripts" / "data")]
D = os.environ.get("WTSS_DATA", "/workspace/data")
CS = [0.01, 0.1, 1.0, 10.0]


def image_path(task, image_id, site):
    if task == "rsna":
        return f"{D}/cxr_at/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{image_id}.jpg"
    return f"{D}/cxr_at/{'NIH' if site == 0 else 'PC'}/images-224/{image_id}"


def gray(p):
    a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if a.dtype != np.uint8:
        a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return a


def rois(task, paths, cache: Path):
    """PSPNet ROI at 512 px (lungs; + heart for cardiomegaly), cached as uint8 (n, 512, 512)."""
    if cache.exists():
        return np.load(cache)
    import torchxrayvision as xrv
    m = xrv.baseline_models.chestx_det.PSPNet().eval().cuda()
    keep = [m.targets.index(t) for t in (["Left Lung", "Right Lung"] + (["Heart"] if task == "xray" else []))]
    out = np.zeros((len(paths), 512, 512), np.uint8)
    with torch.inference_mode():
        for k in range(0, len(paths), 8):
            g = np.stack([cv2.resize(gray(p), (512, 512), interpolation=cv2.INTER_AREA) for p in paths[k:k + 8]])
            x = torch.from_numpy(g.astype(np.float32) / 255 * 2048 - 1024)[:, None].cuda()
            out[k:k + 8] = (torch.sigmoid(m(x))[:, keep].amax(1) > 0.5).cpu().numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def masked_render(paths, R, S=518):
    from wtss.ops import apply_roi_mask

    def render(i):
        k = int(i)
        im = Image.fromarray(gray(paths[k])).convert("RGB").resize((S, S), Image.BICUBIC)
        roi = cv2.resize(R[k], (S, S), interpolation=cv2.INTER_NEAREST).astype(bool)
        return apply_roi_mask(im, roi)
    return render


def fit_mask_cmc(Xtr, ytr, atr, Xva, yva, seed):
    import candidates as CA
    best = None
    for mode in CA.CMC_MODES:
        head, C, v = CA.fit_cmc(Xtr, ytr, atr, Xva, yva, seed, mode)
        if best is None or v > best[3]:
            best = (head, mode, C, v)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["xray", "rsna"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split_csv", required=True, help="viviano_score output of any of their runs for this seed")
    ap.add_argument("--arm_a_features", default=None, help="viviano_score features npz (masked, every split)")
    ap.add_argument("--stage", choices=["rois", "arms"], default="arms",
                    help="rois: PSPNet ROIs only, plus 224-px copies keyed by image id for viviano_score --external_roi")
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from wtss.backbones import load_backend
    from wtss.features import extract_view
    from wtss.umte import UMtE

    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRACTION", "0.30")))  # shared GPU
    out = Path(f"{D}/replication/viviano/arms/{args.task}_{args.seed}"); out.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(args.split_csv)
    Fa = np.load(args.arm_a_features) if args.stage == "arms" else None
    S = {s: sp[sp.split == s].reset_index(drop=True) for s in ("train", "valid", "test")}
    paths = {s: [image_path(args.task, i, int(c)) for i, c in zip(d.image_id, d.site)] for s, d in S.items()}
    y = {s: d.y.to_numpy().astype(int) for s, d in S.items()}
    a = {s: d.site.to_numpy().astype(int) for s, d in S.items()}
    summary = {"task": args.task, "seed": args.seed, "n": {s: len(d) for s, d in S.items()}}
    if args.stage == "rois":
        for s in ("train", "valid", "test"):
            R = rois(args.task, paths[s], out / f"roi_{s}.npy")
            m224 = np.stack([cv2.resize(r, (224, 224), interpolation=cv2.INTER_NEAREST) for r in R])
            np.savez_compressed(out / f"roi224_{s}.npz", ids=np.asarray(S[s].image_id.astype(str), dtype="U"), masks=m224)
            print(s, len(R), "mean ROI fraction", round(float(m224.mean()), 3), flush=True)
        return

    def save(arm, prob, extra):
        S["test"].assign(score=prob)[["image_id", "y", "site", "score"]].to_csv(out / f"preds_{arm}.csv.gz", index=False)
        summary[arm] = {"test_auroc": float(roc_auc_score(y["test"], prob)), **extra}
        print(arm, summary[arm], flush=True)

    # a) mask_cmc on their network's penultimate features (masked images, every split)
    h, mode, C, v = fit_mask_cmc(Fa["train"], y["train"], a["train"], Fa["valid"], y["valid"], args.seed)
    save("a_mask_cmc_their_net", h.prob(Fa["test"]), {"mode": mode, "C": C, "val_auroc": v})

    # b) DINOv2 @ 518 on ROI-masked images + mask_cmc
    dev = torch.device("cuda")
    bb = load_backend("dino518", dev)
    X, R = {}, {}
    for s in ("train", "valid", "test"):
        R[s] = rois(args.task, paths[s], out / f"roi_{s}.npy")
        X[s] = extract_view(bb, [str(k) for k in range(len(paths[s]))], masked_render(paths[s], R[s]),
                            out / f"dino518_masked_{s}.npz", dev, batch_size=16, num_workers=12)
    h, mode, C, v = fit_mask_cmc(X["train"], y["train"], a["train"], X["valid"], y["valid"], args.seed)
    save("b_dino518_mask_cmc", h.prob(X["test"]), {"mode": mode, "C": C, "val_auroc": v})

    # c) U-MtE, label-free, exactly as the public tool: fit() on training images + ROIs (generic overlays inserted into
    #    the ROI; its own raw-feature encoding), transform() every split, class-balanced head with C on validation
    um = UMtE(bb, device="cuda", seed=args.seed, batch_size=16)
    pil = {s: [Image.fromarray(gray(p)).convert("RGB") for p in paths[s]] for s in paths}
    roi = {s: [r.astype(bool) for r in R[s]] for s in R}
    um.fit(pil["train"], roi["train"], keys=list(S["train"].image_id))
    Z = {"train": um.eraser(um._train_features), **{s: um.transform(pil[s], roi[s]) for s in ("valid", "test")}}
    best = None
    for c in CS:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, solver="liblinear",
                                 random_state=args.seed).fit(Z["train"], y["train"])
        vv = roc_auc_score(y["valid"], clf.predict_proba(Z["valid"])[:, 1])
        if best is None or vv > best[2]:
            best = (clf, c, vv)
    er = um.eraser
    save("c_umte", best[0].predict_proba(Z["test"])[:, 1], {"C": best[1], "val_auroc": best[2], "k": int(er.k)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
