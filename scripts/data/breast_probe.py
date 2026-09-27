"""Breast ultrasound caliper labels without manual annotation (docs/BREAST_MARKER_AUDIT.md, attempt 2).

A patch-token caliper probe (wtss.slas, DINOv2@518) is trained on thyroid + ovary images whose calipers were found by
the audited rule-based detector (95 % precision), validated on held-out thyroid/ovary images, and applied to BUSI +
BUS-BRA. Thresholds fixed a priori (before any breast label is looked at):
  caliper-free (A0): no patch with probability >= 0.30 (sensitive);
  caliper (A1):      >= 2 patches with probability >= 0.70 (strict);
  r = share of A1 patches lying on the tumour (>= 50 % tumour pixels in the patch).
  python scripts/data/breast_probe.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from wtss import paths
from wtss.experiments.real_traps import RealCache
from wtss.slas import SLAS

T_A0, T_A1, MIN_PATCHES = 0.30, 0.70, 2
S = 518


def cohort_imgs(csv, cdir, n_pos, n_neg, seed):
    c = pd.read_csv(csv)
    cache = RealCache(cdir, roi_file="roi.npy", art_file="marker.npy")
    rng = np.random.default_rng(seed)
    pos = rng.choice(c[c.marker_px >= 15].image_id.values, n_pos, replace=False)
    neg = rng.choice(c[c.marker_px == 0].image_id.values, n_neg, replace=False)
    ims, ms = [], []
    for i in list(pos) + list(neg):
        img, _, art = cache.get(i)
        ims.append(Image.fromarray(np.asarray(img))); ms.append((np.asarray(art) > 0).astype(np.uint8))
    return ims, ms


def main():
    th = (paths.DATA / "us/tncd/thyroid_cohort.csv", paths.DATA / "us/tncd/cache_518")
    ov = (paths.DATA / "ovary/ovary_cohort.csv", paths.DATA / "ovary/cache_518")
    tr_i, tr_m = [], []
    for (csv, cdir), npos, nneg in ((th, 500, 200), (ov, 300, 100)):
        i, m = cohort_imgs(csv, cdir, npos, nneg, seed=1); tr_i += i; tr_m += m
    s = SLAS("dinov2", batch_size=32).fit(tr_i, tr_m, per_image=300)
    # held-out validation against the detector masks (different seed, disjoint draw is likely but not guaranteed)
    va_i, va_m = [], []
    for (csv, cdir) in (th, ov):
        i, m = cohort_imgs(csv, cdir, 100, 50, seed=99); va_i += i; va_m += m
    H = s.localise(va_i)
    g = H.shape[1]
    cov = np.stack([torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(m.astype(np.float32))[None, None], g)[0, 0].numpy() for m in va_m])
    lab = np.where(cov >= 0.2, 1, np.where(cov == 0, 0, -1)).ravel(); sc = H.ravel()
    print("probe patch AUROC on held-out thyroid/ovary:", round(roc_auc_score(lab[lab >= 0], sc[lab >= 0]), 3), flush=True)
    # breast
    d = pd.read_csv(paths.DATA / "breast/marker_scan.csv")
    bb = pd.read_csv(paths.DATA / "breast/busbra/BUSBRA/BUSBRA/bus_data.csv")
    lab_bb = dict(zip(bb.ID.map(lambda i: str(paths.DATA / f"breast/busbra/BUSBRA/BUSBRA/Images/{i}.png")), (bb.Pathology == "malignant").astype(int)))
    case_bb = dict(zip(bb.ID.map(lambda i: str(paths.DATA / f"breast/busbra/BUSBRA/BUSBRA/Images/{i}.png")), bb.Case))
    d["y"] = [lab_bb.get(p, int("malignant" in p)) for p in d.img]
    rows = []
    for j0 in range(0, len(d), 256):
        q = d.iloc[j0:j0 + 256]
        Hq = s.localise(list(q.img))
        for (_, r), h in zip(q.iterrows(), Hq):
            roi = np.asarray(Image.open(r.msk).convert("L").resize((S, S), Image.NEAREST)) > 0
            rp = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(roi.astype(np.float32))[None, None], h.shape[0])[0, 0].numpy() >= 0.5
            hot = h >= T_A1
            rows.append({"img": r.img, "src": r.src, "y": r.y, "max_prob": float(h.max()), "n_hot": int(hot.sum()),
                         "r": float((hot & rp).sum() / hot.sum()) if hot.sum() else np.nan,
                         "case": case_bb.get(r.img, r.img)})
        print(f"[breast] {min(j0 + 256, len(d))}/{len(d)}", flush=True)
    out = pd.DataFrame(rows)
    out["A0"] = out.max_prob < T_A0
    out["A1"] = out.n_hot >= MIN_PATCHES
    out.to_csv(paths.DATA / "breast/probe_scan.csv", index=False)
    cell = np.select([out.A0, out.A1 & (out.r >= 0.5), out.A1 & (out.r < 0.1)], ["A0", "trapA", "trapB"], "other")
    print(pd.crosstab([out.src, cell], out.y))


if __name__ == "__main__":
    main()
