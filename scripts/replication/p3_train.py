"""Train and score P3's (Bassi et al. 2024) DenseNet-121 or ISNet on the COVID-19 benchmark with synthetic marks.

Their model code is used as released (third_party/isnet: backup/ISNetLightning.IsDense121Lightning; heat=True is the
ISNet, heat=False the standard DenseNet-121). Lightning 1.6 does not install with CUDA 12.8 torch, so their
training_step / validation_step are driven by this loop with P3's recipe (Supp. Note 6.2): SGD lr 1e-3, momentum 0.99,
no weight decay, gradient-norm clip 1, mixed precision, deterministic ops, 48 epochs, batch 16 (P3: 10-16), random
initialisation, ISNet P=0.7, d=0.9, E=1, w1=1, w2=3; the epoch with the lowest validation loss (their val_loss: the ISNet
loss for ISNet, cross-entropy for DenseNet) is kept.

Data (Supp. Notes 5.1, 6.1): cache from p3_prepare.py (histogram-equalized 224-px grayscale, repeated to 3 channels,
scaled to [0, 1]). Augmentation on training images only: translation up to 28 px, rotation within +-40 degrees, horizontal
flip p = 0.5, applied jointly to image, lung mask and the mark (the mark is drawn first, as in P3).
Marks: white shapes, side 10% of the image (22 px): triangle COVID-19 (2), square normal (0), circle pneumonia (1).
  --position corner: P3's upper-left corner, box at (0, 0); the right-angled triangle and the flush position were measured
             from P3's main-text heatmap figure (page 9: mark box 0.10-0.11 of the side from pixel 0) | inside: fully inside the lungs
             (overlap >= 0.95, find_best_placements_for_mask on the image's lung mask)
  --presence all: every training/validation image carries its class mark (P3) | 9010: 90% carry it, keyed by a stable hash
             of (image id, seed) (amendment A6)
Tests (P3): biased (own-class mark on every image), standard (no mark), deceiving (circle on normal, triangle on pneumonia,
square on COVID-19), at the same position rule.
Outputs (never committed): $WTSS_DATA/replication/isnet/runs/<name>/{best.pt, train_log.csv, preds_<test>.csv.gz}.
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
D = os.environ.get("WTSS_DATA", "/workspace/data")
sys.path.insert(0, "/workspace/third_party/isnet/backup")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

S, M = 224, 22                       # image side; mark side (10%)
SHAPE = {0: "square", 1: "circle", 2: "triangle"}
DECEIVE = {0: "circle", 1: "triangle", 2: "square"}


def draw(img: np.ndarray, shape: str, x: int, y: int) -> np.ndarray:
    import cv2
    out = img.copy()
    if shape == "square":
        out[y:y + M, x:x + M] = 255
    elif shape == "circle":
        cv2.circle(out, (x + M // 2, y + M // 2), M // 2, 255, -1)
    else:
        pts = np.array([[x, y], [x + M - 1, y], [x, y + M - 1]], np.int32)   # right triangle in the corner (P3 Fig. 5)
        cv2.fillPoly(out, [pts], 255)
    return out


def placements(ids, roi, position):
    if position == "corner":
        return {i: (0, 0, 0.0) for i in ids}   # flush with the corner, as measured in P3's figure
    from wtss.synthetic import find_best_placements_for_mask
    out = {}
    for i, m in zip(ids, roi):
        p = find_best_placements_for_mask(np.asarray(m), str(i), [1.0], M, M)["1.00"]
        out[i] = (p["x"], p["y"], p["achieved"])
    return out


def present(image_id: str, seed: int, presence: str) -> bool:
    if presence == "all":
        return True
    h = int(hashlib.sha256(f"p3-9010|{image_id}|{seed}".encode()).hexdigest()[:12], 16) / 16**12
    return h < 0.9


class Data(torch.utils.data.Dataset):
    def __init__(self, gray, roi, ids, y, pl, marks, augment, seed):
        self.gray, self.roi, self.ids, self.y, self.pl, self.marks, self.aug = gray, roi, ids, y, pl, marks, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        import torchvision.transforms.v2.functional as F
        g, m = np.asarray(self.gray[k]), np.asarray(self.roi[k])
        if self.marks[k] is not None:
            x, yy, _ = self.pl[self.ids[k]]
            g = draw(g, self.marks[k], x, yy)
        g = torch.from_numpy(g)[None].float() / 255.0
        m = torch.from_numpy(m)[None].float()
        if self.aug:
            ang = float(torch.empty(1).uniform_(-40, 40))
            tx, ty = (int(v) for v in torch.randint(-28, 29, (2,)))
            g = F.affine(g, angle=ang, translate=[tx, ty], scale=1.0, shear=[0.0])
            m = F.affine(m, angle=ang, translate=[tx, ty], scale=1.0, shear=[0.0])
            if torch.rand(1) < 0.5:
                g, m = F.horizontal_flip(g), F.horizontal_flip(m)
        return g.repeat(3, 1, 1), (m > 0.5).float(), torch.tensor([self.y[k]])


def load(which):
    c = Path(f"{D}/replication/isnet/cache_{which}")
    ids = (c / "ids.txt").read_text().split()
    return np.load(c / "gray.npy", mmap_mode="r"), np.load(c / "roi.npy", mmap_mode="r"), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["isnet", "densenet"], required=True)
    ap.add_argument("--position", choices=["corner", "inside"], default="corner")
    ap.add_argument("--presence", choices=["all", "9010"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_batches", type=int, default=0, help="smoke test only")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    import ISNetLightning as L

    out = Path(f"{D}/replication/isnet/runs/{args.name}"); out.mkdir(parents=True, exist_ok=True)
    co = pd.read_csv(f"{D}/replication/isnet/cohort.csv").set_index("image_id")
    gray, roi, ids = load("cohort")
    pl = placements(ids, roi, args.position)
    y = co.loc[ids, "y"].to_numpy(); split = co.loc[ids, "split"].to_numpy()
    marks = [SHAPE[int(c)] if present(i, args.seed, args.presence) else None for i, c in zip(ids, y)]
    if args.position == "inside":
        bad = [i for i in ids if pl[i][2] < 0.95]
        print(f"inside: {len(bad)} training/validation images cannot hold the mark fully inside (kept, overlap reported)")
    pd.DataFrame({"image_id": ids, "split": split, "y": y, "mark": marks,
                  "x": [pl[i][0] for i in ids], "yy": [pl[i][1] for i in ids], "overlap": [pl[i][2] for i in ids]}
                 ).to_csv(out / "train_marks.csv", index=False)

    def subset(s, aug):
        idx = np.where(split == s)[0]
        return Data(gray, roi, [ids[k] for k in idx], y[idx], pl, [marks[k] for k in idx], aug, args.seed), idx

    tr, _ = subset("train", True)
    va, vidx = subset("valid", False)
    va.gray, va.roi = gray[vidx], roi[vidx]
    tr_idx = np.where(split == "train")[0]
    tr.gray, tr.roi = gray[tr_idx], roi[tr_idx]
    dl = lambda d, sh: torch.utils.data.DataLoader(d, batch_size=args.batch, shuffle=sh, num_workers=12, drop_last=False,
                                                   generator=torch.Generator().manual_seed(args.seed))
    heat = args.model == "isnet"
    net = L.IsDense121Lightning(multiLabel=False, multiMask=False, classes=3, heat=heat, LR=1e-3, P=0.7, Ea=1,
                                A=1, B=3, d=0.9, momentum=0.99).cuda()
    net.log = lambda *a, **k: None
    opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.99)
    scaler = torch.amp.GradScaler("cuda")
    batch = (lambda b: tuple(t.cuda() for t in b)) if heat else (lambda b: (b[0].cuda(), b[2].cuda()))
    best, log = float("inf"), []
    for ep in range(args.epochs):
        net.train(); t0 = time.time(); tl = []
        for k, b in enumerate(dl(tr, True)):
            if args.max_batches and k >= args.max_batches:
                break
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = net.training_step(batch(b), k)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update()
            tl.append(float(loss))
        net.eval(); vl = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for k, b in enumerate(dl(va, False)):
                if args.max_batches and k >= args.max_batches:
                    break
                vl.append(float(net.validation_step(batch(b), k)["loss"]))
        v = float(np.mean(vl))
        log.append({"epoch": ep, "train_loss": float(np.mean(tl)), "val_loss": v, "minutes": (time.time() - t0) / 60})
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        print(log[-1], flush=True)
        if v < best:
            best = v; torch.save(net.state_dict(), out / "best.pt")
    net.load_state_dict(torch.load(out / "best.pt")); net.eval()
    score(net, heat, out, args)


def score(net, heat, out, args):
    """Per-image logits on P3's three tests (biased / standard / deceiving), same position rule as training."""
    tc = Path(f"{D}/replication/isnet/cache_test")
    if not (tc / "ids.txt").exists():
        print("no test cache yet (BIMCV still fetching): scoring skipped"); return
    gray, roi, ids = load("test")
    t = pd.read_csv(f"{D}/replication/isnet/test_cohort.csv").drop_duplicates("image_id").set_index("image_id")
    y = t.loc[ids, "y"].to_numpy()
    pl = placements(ids, roi, args.position)
    for test, mk in (("biased", lambda c: SHAPE[c]), ("standard", lambda c: None), ("deceiving", lambda c: DECEIVE[c])):
        ds = Data(gray, roi, ids, y, pl, [mk(int(c)) for c in y], False, args.seed)
        L_, n = [], 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for b in torch.utils.data.DataLoader(ds, batch_size=64, num_workers=12):
                o = net.forward(b[0].cuda())
                L_.append((o[0] if heat else o).float().cpu())
        lg = torch.cat(L_).numpy()
        pd.DataFrame({"image_id": ids, "y": y, "logit0": lg[:, 0], "logit1": lg[:, 1], "logit2": lg[:, 2],
                      "overlap": [pl[i][2] for i in ids]}).to_csv(out / f"preds_{test}.csv.gz", index=False)
        from sklearn.metrics import f1_score
        print(test, "macro-F1", round(f1_score(y, lg.argmax(1), average="macro"), 4), flush=True)


if __name__ == "__main__":
    main()
