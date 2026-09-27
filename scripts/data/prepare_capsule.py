"""Capsule endoscopy (SEE-AI) cohort with contamination (debris / bubbles / bile) artifact masks.

  python scripts/data/prepare_capsule.py --stage probe     # contamination probe on 2,700 expert masks + held-out IoU
  python scripts/data/prepare_capsule.py --stage cohort    # cache_518 (rgb, lesion-box ROI, contamination mask) + table

Task: erosion (y=1) vs polyp-like (y=0), single-class SEE-AI frames; ROI = union of the class boxes.
Contamination masks: SLAS patch probe (DINOv2@518 tokens, wtss.slas) trained on the figshare 27645021 expert
masks (CECleanliness + Kvasir-Capsule + SEE-AI subsets); where a SEE-AI frame has an expert mask (pHash link)
the expert mask is used instead. Leakage groups: pHash <= 2 chains ∪ contiguous blocks of 100 frame numbers
(video frames of one lesion; no video ids are published).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image

from wtss import paths
from wtss.slas import SLAS

D = paths.DATA / "capsule"
FIG = D / "datasets"
SEE = D / "seeai"
NAMES = ['angiodysplasia', 'erosion', 'stenosis', 'lymphangiectasia', 'lymph follicle', 'SMT', 'polyp-like', 'bleeding',
         'diverticulum', 'erythema', 'foreign body', 'vein']
S = 518


def expert_pairs():
    out = []
    for sub in ("CECleanliness database", "Kvasir capsule endoscopy dataset", "SEE-AI project dataset"):
        for p in sorted(glob.glob(str(FIG / sub / "imgs" / "*.jpg"))):
            out.append((sub, p, p.replace("/imgs/", "/masks/")))
    return out


def contam(mpath, size=None):
    m = Image.open(mpath).convert("L")
    if size:
        m = m.resize(size, Image.NEAREST)
    return (np.asarray(m) < 128).astype(np.uint8)  # black = contaminated


def stage_probe():
    pairs = expert_pairs()
    rng = np.random.default_rng(20260927)
    idx = rng.permutation(len(pairs))
    n_te = len(pairs) // 5
    te, tr = [pairs[i] for i in idx[:n_te]], [pairs[i] for i in idx[n_te:]]
    s = SLAS("dinov2", tau=0.5, batch_size=32)
    s.fit([p for _, p, _ in tr], [contam(m) for _, _, m in tr], per_image=200)
    s.save(D / "contam_probe.pt")
    heat = s.localise([p for _, p, _ in te])
    ious, accs = [], []
    for h, (_, p, m) in zip(heat, te):
        pm = cv2.resize(h.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR) >= 0.5
        gm = contam(m, (256, 256)) > 0
        u = (pm | gm).sum()
        ious.append((pm & gm).sum() / u if u else 1.0)
        accs.append((pm == gm).mean())
    rep = {"n_train": len(tr), "n_test": len(te), "iou_mean": float(np.mean(ious)), "iou_median": float(np.median(ious)),
           "pixel_acc": float(np.mean(accs))}
    for sub in ("CECleanliness", "Kvasir", "SEE-AI"):
        k = [i for i, (sb, _, _) in enumerate(te) if sb.startswith(sub)]
        rep[f"iou_{sub}"] = float(np.mean([ious[i] for i in k]))
    (D / "contam_probe_eval.json").write_text(json.dumps(rep, indent=2))
    print(rep)


def boxes(img_no):
    f = SEE / "SEE_AI_project_all_txt" / "SEE_AI_project_all_txt" / f"image{img_no:05d}.txt"
    return [l.split() for l in open(f) if l.strip()]


def stage_cohort():
    rows = []
    for f in sorted(glob.glob(str(SEE / "SEE_AI_project_all_txt" / "SEE_AI_project_all_txt" / "*.txt"))):
        n = int(Path(f).stem.replace("image", ""))
        L = [l.split() for l in open(f) if l.strip()]
        cls = {NAMES[int(l[0])] for l in L}
        if cls == {"erosion"} or cls == {"polyp-like"}:
            rows.append({"image_id": f"see{n:05d}", "num": n, "cls": cls.pop()})
    c = pd.DataFrame(rows)
    c["y"] = (c.cls == "erosion").astype(int)
    print(c.cls.value_counts().to_dict())
    # expert masks for SEE-AI frames (pHash link of the 500 figshare SEE-AI images)
    ex = {}
    for p in sorted(glob.glob(str(FIG / "SEE-AI project dataset" / "imgs" / "*.jpg"))):
        ex[str(imagehash.phash(Image.open(p).convert("RGB").resize((256, 256))))] = p.replace("/imgs/", "/masks/")
    cdir = paths.ensure(D / "cache_518")
    ids = c.image_id.tolist()
    rgb = np.lib.format.open_memmap(cdir / "rgb.npy", "w+", np.uint8, (len(ids), S, S, 3))
    roi = np.lib.format.open_memmap(cdir / "roi.npy", "w+", np.uint8, (len(ids), S, S))
    art = np.lib.format.open_memmap(cdir / "contam.npy", "w+", np.uint8, (len(ids), S, S))
    ph, src = [], []
    s = SLAS.load(D / "contam_probe.pt")
    for j0 in range(0, len(ids), 64):
        chunk = c.iloc[j0:j0 + 64]
        ims = []
        for k, r in enumerate(chunk.itertuples()):
            im = Image.open(SEE / "SEE_AI_project_all_images" / "SEE_AI_project_all_images" / f"image{r.num:05d}.jpg").convert("RGB")
            h = imagehash.phash(im.resize((256, 256)))
            ph.append(str(h))
            W, H = im.size
            m = np.zeros((S, S), np.uint8)
            for l in boxes(r.num):
                cx, cy, bw, bh = (float(v) for v in l[1:5])
                x0, y0 = int((cx - bw / 2) * S), int((cy - bh / 2) * S)
                x1, y1 = int(np.ceil((cx + bw / 2) * S)), int(np.ceil((cy + bh / 2) * S))
                m[max(0, y0):y1, max(0, x0):x1] = 1
            im = im.resize((S, S), Image.BICUBIC)
            rgb[j0 + k] = np.asarray(im); roi[j0 + k] = m
            ims.append(im)
        heat = s.localise(ims)
        for k, (hm, r) in enumerate(zip(heat, chunk.itertuples())):
            key = ph[j0 + k]
            if key in ex:
                art[j0 + k] = contam(ex[key], (S, S)); src.append("expert")
            else:
                art[j0 + k] = (cv2.resize(hm.astype(np.float32), (S, S), interpolation=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)
                src.append("probe")
        print(f"[capsule] {min(j0 + 64, len(ids))}/{len(ids)}", flush=True)
    rgb.flush(); roi.flush(); art.flush()
    (cdir / "ids.txt").write_text("\n".join(ids))
    c["phash_hex"], c["mask_src"] = ph, src
    A = np.asarray(art).reshape(len(ids), -1).astype(bool)
    R = np.asarray(roi).reshape(len(ids), -1).astype(bool)
    c["contam_frac"] = A.mean(1)
    c["roi_frac"] = R.mean(1)
    inter = (A & R).sum(1)
    c["r"] = np.where(A.sum(1) > 0, inter / np.maximum(A.sum(1), 1), np.nan)
    c["roi_cover"] = inter / np.maximum(R.sum(1), 1)  # fraction of the lesion box covered by contamination
    c["group"] = leakage_groups(c)
    c.to_csv(D / "capsule_cohort.csv", index=False)
    print(c.describe().T.to_string())


def leakage_groups(c: pd.DataFrame) -> np.ndarray:
    n = len(c)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    blk = c.num.to_numpy() // 100  # contiguous blocks of 100 frame numbers (video frames; no video ids published)
    first = {}
    for i, b in enumerate(blk):
        union(first.setdefault(b, i), i)
    H = np.array([imagehash.hex_to_hash(h).hash.flatten() for h in c.phash_hex])
    for i in range(n):
        d = (H[i + 1:] != H[i]).sum(1)
        for j in np.flatnonzero(d <= 2):
            union(i, i + 1 + j)
    return np.array([find(i) for i in range(n)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["probe", "cohort"], required=True)
    a = ap.parse_args()
    {"probe": stage_probe, "cohort": stage_cohort}[a.stage]()
