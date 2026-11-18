"""Arm d (A20 with its correction and encoder change, A21): EfficientNet-B0 (torchvision `efficientnet_b0`, IMAGENET1K_V1;
5.3 M parameters) trained end-to-end on ROI-masked images, then the 1,280-d pooled penultimate features -> mask_cmc ->
linear head refit on train (CPU).

One fixed recipe for every paper (nothing tuned on a test): standard ERM on the paper's label task, the resolution their
network sees, their training augmentation, Adam lr 1e-4, batch 64, at most 50 epochs, early stopping on the paper's
validation metric (patience 5; the best-validation checkpoint is kept), seed 1234, mixed precision (fp16 autocast).
Checkpoint every epoch and resume. Inputs are ImageNet-normalised (the pretrained weights' statistics).

Stages:
    python scripts/replication/armd_train.py --setting p2 --stage train      # GPU
    python scripts/replication/armd_train.py --setting p2 --stage features   # GPU, best checkpoint, all sets
    python scripts/replication/armd_train.py --setting p2 --stage heads      # CPU
Settings implemented: p2 (P1, P3, P4 adapters are added before their runs).
Outputs: $WTSS_DATA/replication/armd/<setting>/ (checkpoint.pt, best.pt, log.csv, features.npz, preds_test.csv.gz) and
results/replication/armd/<setting>.{md,csv}.
"""
import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # DataLoader workers: one thread each
    os.environ.setdefault(_v, "1")                                         # (container task limit 2,816; 2026-11-18)
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "round8")]
D = os.environ.get("WTSS_DATA", "/workspace/data")
OUTR = ROOT / "results/replication/armd"
SEED, LR, BS, MAX_EPOCHS, PATIENCE = 1234, 1e-4, 64, 50, 5
CS = (0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0)            # A21
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------------------------------------- model
def dev():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_model(n_out, pretrained=True):
    import torch
    import torchvision
    m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, n_out)   # classifier = (Dropout 0.2, Linear 1280 -> n_out)
    return m


class EarlyStop:
    """Higher is better; stops after `patience` epochs without a strict improvement."""

    def __init__(self, patience=PATIENCE, best=-np.inf, bad=0):
        self.patience, self.best, self.bad = patience, best, bad

    def step(self, v):
        if v > self.best:
            self.best, self.bad = v, 0
            return True
        self.bad += 1
        return False

    @property
    def stop(self):
        return self.bad >= self.patience


# ------------------------------------------------------------------------------------------------------ settings
class P2:
    """P2 (Pahde): ISIC 2019, 8 classes, 224 px. Images rendered by p2_render.py exactly as their network sees them
    (resize, microscope by their rule), lesion ROI from p2_lesion_rois.py applied as in arm b (outside = mean colour),
    then their order: normalise, then `isic_augmentation` (their datasets/isic/isic.py:29-35) on training images.
    Validation metric: mean of clean and biased validation accuracy (section 3)."""
    n_out, size = 8, 224
    sets = ("train_as_trained", "val_clean", "val_biased", "test_clean", "test_biased")
    train_set, val_sets = "train_as_trained", ("val_clean", "val_biased")

    def __init__(self):
        R = Path(f"{D}/replication/pahde")
        z = np.load(R / "lesion_roi224.npz")
        self.roi = dict(zip(z["ids"].astype(str), np.unpackbits(z["bits"], axis=-1)[..., :224].astype(bool)))
        self.part = {k: pd.read_csv(R / f"render224/{k}.csv") for k in self.sets}
        self.npy = {k: str(R / f"render224/{k}.npy") for k in self.sets}

    @staticmethod
    def augmentation():
        import torchvision.transforms as T
        return T.Compose([T.RandomHorizontalFlip(p=.25), T.RandomVerticalFlip(p=.25),
                          T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=.25),
                          T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=.25),
                          T.RandomApply(transforms=[T.ColorJitter(brightness=.1, saturation=.1, hue=.1)], p=.25)])

    def dataset(self, k, train):
        return MaskedNpy(self.npy[k], self.part[k].image.astype(str).tolist(), self.part[k].label.to_numpy(), self.roi,
                         self.augmentation() if train else None)

    def val_metric(self, preds):   # preds: {set: (y, yhat_logits)}
        return float(np.mean([(p.argmax(1) == y).mean() for y, p in (preds[k] for k in self.val_sets)]))

    def labels(self, k):
        return self.part[k].label.to_numpy(), self.part[k].artifact.to_numpy()


class MaskedNpy:
    """uint8 N x S x S x 3 memmap + boolean ROI per image id -> masked, ImageNet-normalised tensor (+ their augmentation)."""

    def __init__(self, npy, ids, y, roi, aug):
        self.npy, self.ids, self.y, self.roi, self.aug = npy, ids, y, roi, aug
        self.mm = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, j):
        import torch
        from PIL import Image
        from wtss.ops import apply_roi_mask
        import torchvision.transforms.functional as TF
        if self.mm is None:
            self.mm = np.load(self.npy, mmap_mode="r")
        im = apply_roi_mask(Image.fromarray(np.asarray(self.mm[j])), self.roi[self.ids[j]])
        t = TF.normalize(TF.to_tensor(im), IMNET_MEAN, IMNET_STD)
        if self.aug is not None:
            t = self.aug(t)
        return t, torch.tensor(int(self.y[j]))


SETTINGS = {"p2": P2}


# --------------------------------------------------------------------------------------------------------- train
def _one_thread(_):
    import torch
    torch.set_num_threads(1)


def loader(ds, train, workers):
    import torch
    g = torch.Generator(); g.manual_seed(SEED)
    w = workers if train else min(workers, 6)
    return torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=train, drop_last=train, num_workers=w, pin_memory=True,
                                       persistent_workers=train and w > 0, worker_init_fn=_one_thread,
                                       generator=g if train else None)


def predict(model, dl, feats=False):
    import torch
    out, F = [], []
    # input of the classifier = the 1,280-d global-average-pooled features (dropout is identity in eval mode)
    hook = model.classifier.register_forward_hook(lambda m, i, o: F.append(i[0].float().cpu())) if feats else None
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, _ in dl:
            out.append(model(x.to(dev(), non_blocking=True)).float().cpu())
    if hook:
        hook.remove()
    return torch.cat(out).numpy(), (torch.cat(F).numpy() if feats else None)


def train(S, out, workers, max_batches=None, max_epochs=MAX_EPOCHS, pretrained=True):
    import torch
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    model = make_model(S.n_out, pretrained).to(dev())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(enabled=dev() == "cuda")
    es, log, start = EarlyStop(), [], 0
    if (out / "checkpoint.pt").exists():
        c = torch.load(out / "checkpoint.pt", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); scaler.load_state_dict(c["scaler"])
        es, log, start = EarlyStop(best=c["best"], bad=c["bad"]), c["log"], c["epoch"] + 1
        torch.set_rng_state(c["rng"])
        if dev() == "cuda" and c["cuda_rng"] is not None:
            torch.cuda.set_rng_state(c["cuda_rng"])
        print(f"resumed after epoch {c['epoch']}", flush=True)
    tr = loader(S.dataset(S.train_set, True), True, workers)
    va = {k: loader(S.dataset(k, False), False, workers) for k in S.val_sets}
    ys = {k: S.labels(k)[0] for k in S.val_sets}
    for ep in range(start, max_epochs):
        if es.stop:
            break
        model.train(); t0, n, tl = time.time(), 0, []
        for b, (x, y) in enumerate(tr):
            if max_batches and b >= max_batches:
                break
            x, y = x.to(dev(), non_blocking=True), y.to(dev(), non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl.append(float(loss)); n += len(y)
        tsec = time.time() - t0
        preds = {k: (ys[k][:len(p)], p) for k, p in ((k, predict(model, dl)[0]) for k, dl in va.items())}
        v = S.val_metric(preds)
        improved = es.step(v)
        log.append({"epoch": ep, "train_loss": float(np.mean(tl)), "val_metric": v, "train_img_per_s": n / tsec,
                    "minutes": (time.time() - t0) / 60, "best": es.best, "bad": es.bad})
        pd.DataFrame(log).to_csv(out / "log.csv", index=False); print(log[-1], flush=True)
        if improved:
            torch.save(model.state_dict(), out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": ep,
                    "best": es.best, "bad": es.bad, "log": log, "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if dev() == "cuda" else None}, out / "checkpoint.tmp")
        (out / "checkpoint.tmp").replace(out / "checkpoint.pt")
    (out / "TRAIN_DONE").write_text(json.dumps({"epochs": len(log), "best_val": es.best}))


def features(S, out, workers, pretrained=True):
    import torch
    model = make_model(S.n_out, pretrained).to(dev())
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev())); model.eval()
    F = {}
    for k in S.sets:
        logits, f = predict(model, loader(S.dataset(k, False), False, workers), feats=True)
        F[k], F[k + "_logits"] = f, logits
        print(k, f.shape, flush=True)
    np.savez_compressed(out / "features.npz", **F)


# ---------------------------------------------------------------------------------------------------------- heads
def heads_p2(S, out):
    """mask_cmc (one δ: only MEL has the microscope in training, so pooled = per_class) + class-balanced multinomial LR,
    C in the A21 grid by the validation mean of clean and biased accuracy; test scored once."""
    import candidates as CA
    from sklearn.linear_model import LogisticRegression
    F = np.load(out / "features.npz")
    ytr, atr = S.labels("train_as_trained")
    P = CA.Projector(CA.cmc_directions(F["train_as_trained"], ytr, atr, "pooled"))
    Z = {k: P(F[k]) for k in S.sets}
    sel, best = [], None
    for C in CS:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=0).fit(Z["train_as_trained"], ytr)
        acc = {k: float((clf.predict(Z[k]) == S.labels(k)[0]).mean()) for k in S.val_sets}
        v = float(np.mean(list(acc.values())))
        sel.append({"C": C, **{f"{k}_acc": a for k, a in acc.items()}, "val_mean": v, "converged": bool(clf.n_iter_.max() < 5000)})
        print(sel[-1], flush=True)
        if best is None or v > best[1]:
            best = (clf, v, C)
    clf, _, C = best
    sel = pd.DataFrame(sel).assign(chosen=lambda d: d.C == C)
    edge = "lowest" if C == CS[0] else "highest" if C == CS[-1] else "no"
    R = Path(f"{D}/replication/pahde")
    van = pd.read_csv(R / "scores/vanilla_seed0.csv.gz", usecols=["index", "split", "test_set", "pred"])
    rr = pd.read_csv(R / "scores_a18/resnet50d_RRClarc_lamb5e9_adam_lr0.0001_identity_2.csv.gz", usecols=["index", "split", "test_set", "pred"])
    rows, preds = [], []
    for ts in ("clean", "biased"):
        k = f"test_{ts}"
        p = S.part[k]
        y = p.label.to_numpy()
        pr = clf.predict(Z[k])
        ref = {n: d[(d.split == "test") & (d.test_set == ts)].set_index("index").loc[p["index"], "pred"].to_numpy() for n, d in (("van", van), ("rr", rr))}
        preds.append(pd.DataFrame({"index": p["index"], "image": p.image, "label": y, "method": "d_effb0_roi_mask_cmc", "test_set": ts, "pred": pr}))
        hit = np.stack([pr == y, ref["van"] == y, ref["rr"] == y], 1).astype(float)
        bs = boot(hit)
        q = lambda v: np.quantile(v, [0.025, 0.975])
        rows.append({"test": ts, "n": len(p), "arm_d_acc": hit[:, 0].mean(), "ci": q(bs[:, 0]), "vanilla_acc": hit[:, 1].mean(),
                     "d_van": hit[:, 0].mean() - hit[:, 1].mean(), "d_van_ci": q(bs[:, 0] - bs[:, 1]), "rrclarc_acc": hit[:, 2].mean(),
                     "d_rr": hit[:, 0].mean() - hit[:, 2].mean(), "d_rr_ci": q(bs[:, 0] - bs[:, 2])})
    pd.concat(preds).to_csv(out / "preds_test.csv.gz", index=False)
    OUTR.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUTR / "p2_selection.csv", index=False)
    lg = pd.read_csv(out / "log.csv")
    f3 = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
    f3s = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"
    md = ["# Arm d (A20/A21) on P2 — EfficientNet-B0 trained on lesion-masked ISIC 2019 images + mask_cmc, seed 1234", "",
          "One seed: 95% CIs are image-only bootstrap intervals (10,000 replicates), paired on the same test images; "
          "training-seed variance is not included. P2's RR-ClArC was not reproduced (A18; author's decision 2026-11-18), "
          "so every comparison here is descriptive (no verdict).", "",
          f"Encoder training: {len(lg)} epochs (early stopping, patience 5), best validation mean accuracy "
          f"{lg.val_metric.max():.3f} at epoch {int(lg.val_metric.idxmax())}; {lg.train_img_per_s.median():.0f} img/s.", "",
          "## Head selection (validation: mean of clean and biased accuracy; A21 grid)", "",
          "| C | val clean | val biased | mean | chosen |", "|---|---|---|---|---|"]
    md += [f"| {r.C:g} | {r.val_clean_acc:.3f} | {r.val_biased_acc:.3f} | {r.val_mean:.3f} | {'**yes**' if r.chosen else ''} |" for _, r in sel.iterrows()]
    md += ["", f"Chosen C = {C:g}; at an edge of the grid: **{edge}**.", "", "## Test (scored once)", "",
           "| test | n | arm d [95% CI] | our Vanilla | Δ vs Vanilla [95% CI] | our RR-ClArC (A18) | Δ vs RR-ClArC [95% CI] |",
           "|---|---|---|---|---|---|---|"]
    md += [f"| {r['test']} | {r['n']} | {r['arm_d_acc']:.3f} {f3(r['ci'])} | {r['vanilla_acc']:.3f} | {r['d_van']:+.3f} {f3s(r['d_van_ci'])} | "
           f"{r['rrclarc_acc']:.3f} | {r['d_rr']:+.3f} {f3s(r['d_rr_ci'])} |" for r in rows]
    (OUTR / "p2.md").write_text("\n".join(md) + "\n")
    pd.DataFrame([{k: (v if not isinstance(v, np.ndarray) else f"{v[0]:.4f},{v[1]:.4f}") for k, v in r.items()} for r in rows]).to_csv(OUTR / "p2.csv", index=False)
    print("\n".join(md))


def boot(hit, n=10000, seed=20260918):
    rng = np.random.default_rng(seed)
    out = []
    for b0 in range(0, n, 500):
        w = rng.poisson(1.0, size=(min(500, n - b0), len(hit)))
        out.append((w @ hit) / np.maximum(w.sum(1, keepdims=True), 1))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=sorted(SETTINGS), required=True)
    ap.add_argument("--stage", choices=["train", "features", "heads"], required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    S = SETTINGS[a.setting]()
    out = Path(f"{D}/replication/armd/{a.setting}"); out.mkdir(parents=True, exist_ok=True)
    if a.stage == "train":
        train(S, out, a.workers)
    elif a.stage == "features":
        features(S, out, a.workers)
    else:
        {"p2": heads_p2}[a.setting](S, out)


if __name__ == "__main__":
    main()
