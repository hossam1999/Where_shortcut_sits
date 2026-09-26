"""Spec U-Net (docs/REPLICATION_SPEC.md §E13): classic Ronneberger U-Net, widths 64-128-256-512-1024,
trained on ISIC 2018 Task 1 manual masks only; predicts lesion masks for every non-HAM ISIC 2019 image.
Target: held-out Dice 0.876 (median 0.915); vs HAM manual masks 0.895 (median 0.947).

  python scripts/data/train_unet_spec.py --stage train
  python scripts/data/train_unet_spec.py --stage predict      # non-HAM ISIC 2019 + all HAM (for the agreement check)
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from wtss import paths

S = 256
OUT = paths.DATA / "isic2019" / "unet_spec"
Image.MAX_IMAGE_PIXELS = None
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, w=(64, 128, 256, 512, 1024)):
        super().__init__()
        self.down = nn.ModuleList([block(3, w[0])] + [block(w[i], w[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[i + 1], w[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([block(2 * w[i], w[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(w[0], 1, 1)

    def forward(self, x):
        skips = []
        for k, d in enumerate(self.down):
            x = d(x)
            if k < 4:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            x = d(torch.cat([u(x), s], 1))
        return self.head(x)


def load_pair(a):
    ip, mp = a
    with Image.open(ip) as im:
        im.draft("RGB", (S * 2, S * 2))
        x = np.asarray(im.convert("RGB").resize((S, S), Image.BILINEAR), np.uint8)
    with Image.open(mp) as m:
        y = (np.asarray(m.convert("L").resize((S, S), Image.NEAREST)) >= 128).astype(np.uint8)
    return x, y


def norm(x, dev):
    return (torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float() / 255 - MEAN.to(dev)) / STD.to(dev)


def dice(p, g, eps=1e-6):
    i = (p & g).sum((1, 2))
    return (2 * i + eps) / (p.sum((1, 2)) + g.sum((1, 2)) + eps)


@torch.inference_mode()
def predict_arr(net, X, dev, bs=64):
    net.eval()
    out = []
    for k in range(0, len(X), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append((torch.sigmoid(net(norm(X[k:k + bs], dev)).float())[:, 0] > 0.5).cpu().numpy())
    return np.concatenate(out)


def train(epochs=40, bs=16):
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(paths.FROZEN / "isic2018_pilot_manifest.frozen.csv")[["image_id", "phash_group"]]
    ids = sorted(p.name.split("_segmentation")[0] for p in paths.ISIC2018_MASKS.glob("*_segmentation.png"))
    df = pd.DataFrame({"image_id": ids}).merge(man, on="image_id", how="left")
    df["group"] = df.phash_group.fillna(-1).astype(int).astype(str)
    df.loc[df.group == "-1", "group"] = df.image_id
    rng = np.random.default_rng(20260926)
    g = df.group.unique()
    held = set(rng.choice(g, int(0.2 * len(g)), replace=False))
    df["split"] = np.where(df.group.isin(held), "heldout", "train")
    df.to_csv(OUT / "split.csv", index=False)
    with ThreadPoolExecutor(8) as ex:
        pairs = list(ex.map(load_pair, [(paths.ISIC2018_IMAGES / f"{i}.jpg", paths.ISIC2018_MASKS / f"{i}_segmentation.png") for i in df.image_id]))
    X = np.stack([p[0] for p in pairs]); Y = np.stack([p[1] for p in pairs]); del pairs
    tr, ho = np.flatnonzero(df.split == "train"), np.flatnonzero(df.split == "heldout")
    dev = torch.device("cuda"); torch.manual_seed(0)
    net = UNet().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler()
    r = np.random.default_rng(0)
    for ep in range(epochs):
        net.train()
        for k in range(0, len(tr), bs):
            b = tr[r.integers(0, len(tr), bs)]
            x, y = X[b], Y[b]
            if r.random() < .5: x, y = x[:, :, ::-1], y[:, :, ::-1]
            if r.random() < .5: x, y = x[:, ::-1], y[:, ::-1]
            xt = norm(np.ascontiguousarray(x), dev).contiguous(memory_format=torch.channels_last)
            yt = torch.from_numpy(np.ascontiguousarray(y)).to(dev).float().unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.float16):
                lo = net(xt)
            lo = lo.float(); p = torch.sigmoid(lo)
            loss = F.binary_cross_entropy_with_logits(lo, yt) + (1 - (2 * (p * yt).sum((2, 3)) + 1) / (p.sum((2, 3)) + yt.sum((2, 3)) + 1)).mean()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        d = dice(predict_arr(net, X[ho], dev), Y[ho].astype(bool))
        print(f"[unet-spec] epoch {ep + 1}/{epochs} heldout Dice mean {d.mean():.4f} median {np.median(d):.4f}", flush=True)
    torch.save(net.state_dict(), OUT / "unet_spec_256.pt")
    d = dice(predict_arr(net, X[ho], dev), Y[ho].astype(bool))
    rep = {"heldout_dice_mean": float(d.mean()), "heldout_dice_median": float(np.median(d)), "n_heldout": int(len(ho)),
           "arch": "Ronneberger U-Net 64-1024 (BN)", "input": S, "epochs": epochs, "train": "ISIC 2018 Task 1 only"}
    (OUT / "UNET_SPEC_REPORT.json").write_text(json.dumps(rep, indent=2)); print(rep)


@torch.inference_mode()
def predict():
    dev = torch.device("cuda")
    net = UNet().to(dev); net.load_state_dict(torch.load(OUT / "unet_spec_256.pt", map_location=dev)); net.eval()
    ids = pd.read_csv(paths.ISIC2019_GT).image.tolist()
    out = np.lib.format.open_memmap(OUT / "masks_256.npy", "w+", np.uint8, (len(ids), S, S))

    def load(i):
        with Image.open(paths.ISIC2019_IMAGES / f"{i}.jpg") as im:
            im.draft("RGB", (S * 2, S * 2))
            return np.asarray(im.convert("RGB").resize((S, S), Image.BILINEAR), np.uint8)

    with ThreadPoolExecutor(8) as ex:
        for k in range(0, len(ids), 256):
            x = np.stack(list(ex.map(load, ids[k:k + 256])))
            out[k:k + len(x)] = predict_arr(net, x, dev)
    out.flush()
    (OUT / "ids.txt").write_text("\n".join(ids))
    # agreement with HAM manual masks
    ham = {p.name.split("_segmentation")[0]: p for p in paths.HAM_SEG.parent.rglob("*_segmentation.png")}
    idx = [j for j, i in enumerate(ids) if i in ham]
    with ThreadPoolExecutor(8) as ex:
        gt = np.stack(list(ex.map(lambda j: (np.asarray(Image.open(ham[ids[j]]).convert("L").resize((S, S), Image.NEAREST)) >= 128), idx)))
    d = dice(np.asarray(out[idx]).astype(bool), gt)
    rep = json.loads((OUT / "UNET_SPEC_REPORT.json").read_text())
    rep.update({"ham_dice_mean": float(d.mean()), "ham_dice_median": float(np.median(d)), "n_ham": len(idx)})
    (OUT / "UNET_SPEC_REPORT.json").write_text(json.dumps(rep, indent=2)); print(rep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "predict"], required=True)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    train(a.epochs) if a.stage == "train" else predict()
