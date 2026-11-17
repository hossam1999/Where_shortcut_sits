"""P4 (Saab et al., MLHC 2022) ISIC replication: Binary-ERM and Segmentation-ERM on Bissoto's trap sets (A13).

Their code (third_party/saab @ 1739cf2) is the reference; it runs on Lightning 1.x / Meerkat 0.2 / TorchMetrics <= 0.9,
which do not install with CUDA 12.8 torch, so their pieces are driven by this loop. Port notes: third_party/saab/SOURCE.md.
Protocol: Appendix A.1 (p. 20), because the repository history starts after the paper (A13 item 6):
  Binary-ERM        their ResNet wrapper (torchvision ResNet-50, ImageNet V1 weights, dropout 0, 2 classes), lr 1e-5,
                    no augmentation, checkpoint on validation AUROC.
  Segmentation-ERM  smp U-Net, SE-ResNeXt50 (32x4d) encoder, ImageNet weights, 3 classes (0 background, 1 benign lesion,
                    2 melanoma lesion: their seg_loader), lr 5e-4, their train_transform.json augmentations (loaded with
                    albumentations 1.3.1; its CXR Resize/Normalize steps are replaced by ISIC's 512 resize and RGB
                    normalisation from their isic.py), checkpoint on validation Dice (their compute_dice_score).
  Both: cross-entropy, Adam, weight decay 0 (config default), 150 epochs, batch 16 (config), fp32 (their Lightning default).
Splits: Bissoto trap set k = their train.seed k (seeds 1-3 -> splits 1-3): train 1,822 (it contains the 259 validation
images, as their code trains on it), validation 259, test 772. Their seed_everything(k).
Score: Binary-ERM softmax[:, 1]; Segmentation-ERM the mean over pixels of the class-2 softmax (their utils.score).
Metric: robust AUROC = AUROC on the 772-image trap test (their notebook).

    python scripts/replication/p4_train.py --method erm|seg --split 1 [--epochs 150] [--max_batches N]
Outputs (never committed): $WTSS_DATA/replication/saab/runs/<method>_split<k>/{best.pt, checkpoint.pt, log.csv, preds_test.csv}
"""
import argparse
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

D = os.environ.get("WTSS_DATA", "/workspace/data")
B = "/workspace/third_party/bissoto/trap-sets"
SAAB = "/workspace/third_party/saab"
MEAN = np.array([0.6045568, 0.60814506, 0.6078789], np.float32)   # their isic.py img_loader
STD = np.array([0.18579379, 0.1863619, 0.19625686], np.float32)
S = 512
WEIGHTS = f"{D}/weights/se_resnext50_32x4d-a260b3a4.pth"


def cache(split_ids):
    """512-px RGB images and Task 1 lesion masks, resized once (their loader resizes every image to 512 x 512)."""
    c = Path(f"{D}/replication/saab/cache512"); c.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(split_ids))
    if (c / "ids.txt").exists() and (c / "ids.txt").read_text().split() == ids:
        return np.load(c / "img.npy", mmap_mode="r"), np.load(c / "mask.npy", mmap_mode="r"), ids
    img = np.lib.format.open_memmap(c / "img.npy", "w+", np.uint8, (len(ids), S, S, 3))
    msk = np.lib.format.open_memmap(c / "mask.npy", "w+", np.uint8, (len(ids), S, S))
    for k, i in enumerate(ids):
        im = cv2.cvtColor(cv2.imread(f"{D}/isic2018/ISIC2018_Task1-2_Training_Input/{i}.jpg"), cv2.COLOR_BGR2RGB)
        img[k] = cv2.resize(im, (S, S), interpolation=cv2.INTER_LINEAR)   # torchvision Resize default: bilinear
        m = cv2.imread(f"{D}/isic2018/ISIC2018_Task1_Training_GroundTruth/{i}_segmentation.png", cv2.IMREAD_GRAYSCALE)
        msk[k] = (cv2.resize(m, (S, S), interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8)
    img.flush(); msk.flush(); (c / "ids.txt").write_text("\n".join(ids))
    return img, msk, ids


def augmenter():
    import albumentations as A
    t = A.load(f"{SAAB}/src/data/train_transform.json")
    return A.Compose([x for x in t.transforms if type(x).__name__ not in ("Resize", "Normalize")])


class Data(torch.utils.data.Dataset):
    def __init__(self, img, msk, rows, pos, seg, aug):
        self.img, self.msk, self.rows, self.pos, self.seg = img, msk, rows, pos, seg
        self.aug = augmenter() if aug else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        r = self.rows.iloc[k]
        j = self.pos[r.id]
        x, m = np.asarray(self.img[j]), np.asarray(self.msk[j])
        if self.aug is not None:
            o = self.aug(image=x, mask=m)
            x, m = o["image"], o["mask"]
        x = (x.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())
        y = int(r.target)
        seg = torch.from_numpy((m.astype(np.int64) * (y + 1)))   # their seg_loader: (target + 1) * lesion mask
        return x, seg, y


def compute_dice_score(outs, targets):
    """Their src/utils.py: per image, overlap = pixels non-zero in both label maps, total = sum of the label values."""
    preds = outs.argmax(1)
    s = []
    for p, t in zip(preds, targets):
        s.append((2 * torch.logical_and(p, t).sum() / (p.sum() + t.sum())).item())
    return float(np.nanmean(s))


def build(method):
    if method == "erm":
        import torchvision
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Sequential(nn.Dropout(0.0), nn.Linear(2048, 2))   # their ResNet wrapper
        return m, None
    import segmentation_models_pytorch as smp
    m = smp.Unet("se_resnext50_32x4d", encoder_weights=None, activation=None, in_channels=3, classes=3)
    # encoder_weights="imagenet" points to data.lip6.fr (offline since 2025). The identical file is loaded from the Internet
    # Archive copy (SHA-256 a260b3a40f82dfe3..., the prefix in Cadene's file name; 331 tensors equal to smp-hub's re-host);
    # smp drops the classifier (last_linear) when loading, as here.
    sd = torch.load(WEIGHTS, map_location="cpu")
    m.encoder.load_state_dict({k: v for k, v in sd.items() if not k.startswith("last_linear")})
    fc = m.segmentation_head
    m.segmentation_head = nn.Identity()
    return m, fc


def image_scores(model, fc, x):
    out = model(x)
    if fc is None:
        return out.softmax(1)[:, 1], out
    seg = fc(out)
    return seg.softmax(1).flatten(2).mean(-1)[:, 2], seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["erm", "seg"], required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_batches", type=int, default=0, help="timing/smoke only")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    k = args.split
    random.seed(k); np.random.seed(k); torch.manual_seed(k); torch.cuda.manual_seed_all(k)   # seed_everything(k)
    torch.backends.cudnn.benchmark = False   # Lightning 1.x Trainer default (benchmark=False); fp32, no AMP
    rows = {s: pd.read_csv(f"{B}/isic_annotated_{s}{k}.csv").rename(columns={"label": "target"}) for s in ("train", "val", "test")}
    for s in rows:
        rows[s]["id"] = rows[s].image.str.split(".").str[0]
    img, msk, ids = cache([i for s in rows for i in rows[s].id])
    pos = {i: j for j, i in enumerate(ids)}
    seg = args.method == "seg"
    out = Path(f"{D}/replication/saab/runs/{args.name or f'{args.method}_split{k}'}"); out.mkdir(parents=True, exist_ok=True)
    model, fc = build(args.method)
    model.cuda(); fc = fc.cuda() if fc is not None else None
    params = list(model.parameters()) + (list(fc.parameters()) if fc is not None else [])
    opt = torch.optim.Adam(params, lr=5e-4 if seg else 1e-5, weight_decay=0.0)
    lossf = nn.CrossEntropyLoss()
    tr = Data(img, msk, rows["train"], pos, seg, aug=seg)
    va = Data(img, msk, rows["val"], pos, seg, aug=False)
    te = Data(img, msk, rows["test"], pos, seg, aug=False)
    # their train DataLoader: shuffle, drop_last=True; validation/test: in order, keep the last batch (task.py)
    loader = lambda d, sh, ep=0: torch.utils.data.DataLoader(d, batch_size=args.batch, shuffle=sh, drop_last=sh, num_workers=8,
                                                             generator=torch.Generator().manual_seed(k * 1000 + ep))
    best, log, start = -1.0, [], 0
    if (out / "checkpoint.pt").exists():
        c = torch.load(out / "checkpoint.pt", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        if fc is not None:
            fc.load_state_dict(c["fc"])
        best, log, start = c["best"], c["log"], c["epoch"] + 1
        print(f"resumed after epoch {c['epoch']}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    for ep in range(start, args.epochs):
        model.train(); fc is not None and fc.train(); t0 = time.time(); tl, n = [], 0
        for b, (x, sg, y) in enumerate(loader(tr, True, ep)):
            if args.max_batches and b >= args.max_batches:
                break
            x = x.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            _, o = image_scores(model, fc, x)
            loss = lossf(o, (sg if seg else y).cuda())
            loss.backward(); opt.step(); tl.append(float(loss)); n += len(x)
        model.eval(); fc is not None and fc.eval(); ps, ys, dice = [], [], []
        with torch.no_grad():
            for b, (x, sg, y) in enumerate(loader(va, False)):
                if args.max_batches and b >= args.max_batches:
                    break
                p, o = image_scores(model, fc, x.cuda())
                ps.append(p.cpu()); ys.append(y)
                if seg:
                    dice.append(compute_dice_score(o.cpu(), sg))
        yv, pv = torch.cat(ys).numpy(), torch.cat(ps).numpy()
        auc = roc_auc_score(yv, pv) if len(set(yv)) == 2 else float("nan")
        metric = float(np.nanmean(dice)) if seg else auc
        sec = time.time() - t0
        log.append({"epoch": ep, "train_loss": float(np.mean(tl)), "valid_auroc": auc, "valid_dice": float(np.nanmean(dice)) if seg else None,
                    "img_per_s": n / sec, "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, "minutes": sec / 60})
        pd.DataFrame(log).to_csv(out / "log.csv", index=False); print(log[-1], flush=True)
        if metric > best:
            best = metric
            torch.save({"model": model.state_dict(), "fc": fc.state_dict() if fc is not None else None, "epoch": ep}, out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "fc": fc.state_dict() if fc is not None else None,
                    "epoch": ep, "best": best, "log": log}, out / "checkpoint.tmp")
        (out / "checkpoint.tmp").replace(out / "checkpoint.pt")
    if args.max_batches:
        return
    c = torch.load(out / "best.pt", weights_only=False)
    model.load_state_dict(c["model"]); fc is not None and fc.load_state_dict(c["fc"])
    model.eval(); fc is not None and fc.eval(); ps = []
    with torch.no_grad():
        for x, _, _ in loader(te, False):
            ps.append(image_scores(model, fc, x.cuda())[0].cpu())
    rows["test"].assign(score=torch.cat(ps).numpy()).to_csv(out / "preds_test.csv", index=False)
    print(f"best epoch {c['epoch']}; robust AUROC (772-image trap test): {roc_auc_score(rows['test'].target, torch.cat(ps).numpy()):.4f}")


if __name__ == "__main__":
    main()
