"""Lesion segmentation U-Net for ISIC 2019 images without manual masks (thesis §6.1).

Train: ISIC 2018 Task 1 manual masks (2,594) + HAM10000 manual masks (10,015, Tschandl 2020).
Held-out: 10% of lesion groups (reported Dice), never used for training/selection of epochs
beyond the fixed schedule. Predict: every ISIC 2019 image lacking a manual mask.

Usage:
  python scripts/data/train_lesion_unet.py --stage train
  python scripts/data/train_lesion_unet.py --stage predict
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wtss import paths

S = 384
OUT = paths.DATA / "isic2019" / "unet"
Image.MAX_IMAGE_PIXELS = None


def load_pair(args):
    ip, mp = args
    with Image.open(ip) as im:
        im.draft("RGB", (S * 2, S * 2))
        x = np.asarray(im.convert("RGB").resize((S, S), Image.BILINEAR), np.uint8)
    with Image.open(mp) as m:
        y = (np.asarray(m.convert("L").resize((S, S), Image.NEAREST)) >= 128).astype(np.uint8)
    return x, y


def build_train_table():
    ham_meta = pd.read_csv(paths.HAM_META)
    ham_masks = {p.name.split("_segmentation")[0]: p for p in paths.HAM_SEG.parent.rglob("*_segmentation.png")}
    isic19 = paths.ISIC2019_IMAGES
    rows = []
    for r in ham_meta.itertuples():
        if r.image_id in ham_masks:
            rows.append(dict(image_id=r.image_id, img=str(isic19 / f"{r.image_id}.jpg"), mask=str(ham_masks[r.image_id]),
                             group=r.lesion_id, src="ham"))
    for p in sorted(paths.ISIC2018_MASKS.glob("*_segmentation.png")):
        i = p.name.split("_segmentation")[0]
        rows.append(dict(image_id=i, img=str(paths.ISIC2018_IMAGES / f"{i}.jpg"), mask=str(p), group=i, src="isic2018"))
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(20260926)
    groups = df.group.unique()
    held = set(rng.choice(groups, size=int(0.1 * len(groups)), replace=False))
    df["split"] = np.where(df.group.isin(held), "heldout", "train")
    return df


def dice(pred, gt, eps=1e-6):
    inter = (pred & gt).sum((1, 2))
    return (2 * inter + eps) / (pred.sum((1, 2)) + gt.sum((1, 2)) + eps)


def model_fn():
    import segmentation_models_pytorch as smp

    return smp.Unet("resnet34", encoder_weights="imagenet", classes=1)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_t(x, dev):
    return ((torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float() / 255) - MEAN.to(dev)) / STD.to(dev)


def train(epochs=25, bs=24):
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_train_table()
    df.to_csv(OUT / "unet_split.csv", index=False)
    with ThreadPoolExecutor(8) as ex:
        pairs = list(ex.map(load_pair, zip(df.img, df.mask)))
    X = np.stack([p[0] for p in pairs]); Y = np.stack([p[1] for p in pairs])
    tr, ho = np.flatnonzero(df.split == "train"), np.flatnonzero(df.split == "heldout")
    dev = torch.device("cuda")
    torch.manual_seed(0)
    net = model_fn().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * (len(tr) // bs + 1))
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr)
        tot = 0
        for k in range(0, len(perm), bs):
            b = perm[k:k + bs]
            x, y = X[b].copy(), Y[b].copy()
            # flips / 90-degree rotations (dermoscopy is orientation-free)
            if rng.random() < .5: x, y = x[:, :, ::-1], y[:, :, ::-1]
            if rng.random() < .5: x, y = x[:, ::-1], y[:, ::-1]
            kk = int(rng.integers(0, 4)); x, y = np.rot90(x, kk, (1, 2)), np.rot90(y, kk, (1, 2))
            xt = to_t(np.ascontiguousarray(x), dev)
            # colour jitter
            xt = xt * (1 + 0.1 * torch.randn(xt.shape[0], 3, 1, 1, device=dev)) + 0.1 * torch.randn(xt.shape[0], 3, 1, 1, device=dev)
            yt = torch.from_numpy(np.ascontiguousarray(y)).to(dev).float().unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.float16):
                lo = net(xt)
                p = torch.sigmoid(lo)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(lo, yt) + \
                    (1 - (2 * (p * yt).sum((2, 3)) + 1) / (p.sum((2, 3)) + yt.sum((2, 3)) + 1)).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss) * len(b)
        d = evaluate(net, X[ho], Y[ho], dev)
        print(f"[unet] epoch {ep + 1}/{epochs} loss {tot / len(tr):.4f} heldout Dice {d.mean():.4f}", flush=True)
    torch.save(net.state_dict(), OUT / "unet_resnet34_384.pt")
    d = evaluate(net, X[ho], Y[ho], dev)
    hs = df.iloc[ho].src.to_numpy()
    rep = {"heldout_dice_mean": float(d.mean()), "heldout_dice_median": float(np.median(d)),
           "heldout_n": int(len(ho)), "heldout_dice_ham": float(d[hs == "ham"].mean()),
           "heldout_dice_isic2018": float(d[hs == "isic2018"].mean()), "epochs": epochs, "input": S}
    (OUT / "UNET_REPORT.json").write_text(json.dumps(rep, indent=2))
    print(rep)


@torch.inference_mode()
def evaluate(net, X, Y, dev, bs=48):
    net.eval()
    out = []
    for k in range(0, len(X), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(net(to_t(X[k:k + bs], dev))).float()
            p = (p + torch.sigmoid(net(to_t(X[k:k + bs][:, :, ::-1].copy(), dev))).float().flip(3)) / 2
        out.append((p[:, 0] > 0.5).cpu().numpy())
    pred = np.concatenate(out)
    return dice(pred.astype(bool), Y.astype(bool))


@torch.inference_mode()
def predict():
    """Predict masks (at original resolution) for all ISIC 2019 images; keep largest component."""
    import cv2

    dev = torch.device("cuda")
    net = model_fn().to(dev)
    net.load_state_dict(torch.load(OUT / "unet_resnet34_384.pt", map_location=dev))
    net.eval()
    dst = paths.ensure(paths.DATA / "isic2019" / "unet_masks")
    ids = pd.read_csv(paths.ISIC2019_GT).image.tolist()

    def load(i):
        with Image.open(paths.ISIC2019_IMAGES / f"{i}.jpg") as im:
            W, H = im.size
            im.draft("RGB", (S * 2, S * 2))
            return np.asarray(im.convert("RGB").resize((S, S), Image.BILINEAR), np.uint8), (W, H)

    bs = 64
    with ThreadPoolExecutor(8) as ex:
        for k in range(0, len(ids), bs):
            chunk = ids[k:k + bs]
            loaded = list(ex.map(load, chunk))
            x = np.stack([l[0] for l in loaded])
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(net(to_t(x, dev))).float()
                p = (p + torch.sigmoid(net(to_t(x[:, :, ::-1].copy(), dev))).float().flip(3)) / 2
            p = (p[:, 0] > 0.5).cpu().numpy().astype(np.uint8)
            for i, m, (_, (W, H)) in zip(chunk, p, loaded):
                n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
                if n > 2:
                    m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
                Image.fromarray(m * 255).resize((W, H), Image.NEAREST).save(dst / f"{i}_unet.png")
            if k % 2048 == 0:
                print(f"[unet] predicted {k + len(chunk)}/{len(ids)}", flush=True)


def agreement():
    """Dice of U-Net vs manual masks on every ISIC 2019 image that has a manual mask (incl. training ones:
    reported separately for held-out and for all), and the overlap-stratum swap rate."""
    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "predict"], required=True)
    ap.add_argument("--epochs", type=int, default=25)
    a = ap.parse_args()
    train(a.epochs) if a.stage == "train" else predict()
