"""Prepare NIH ChestX-ray14 cohorts (see wtss/data/cxr.py).

  python scripts/data/prepare_cxr.py --stage synthetic   # nih_ptx: cache, lung masks, tube placements
  python scripts/data/prepare_cxr.py --stage detector    # drain detector on NEATX, scores for all NIH images
  python scripts/data/prepare_cxr.py --stage drain       # nih_drain trap cohort + cache
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from wtss import paths
from wtss.data.cxr import NIH_PNG, PREP, TUBE_GEOMETRY, build_synthetic_cohort, nih_table, patient_split
from wtss.synthetic import OVERLAPS, feasible_ids, find_best_placements_for_mask


def load_gray(i, size):
    with Image.open(NIH_PNG / i) as im:
        return np.asarray(im.convert("L").resize((size, size), Image.BICUBIC), np.uint8)


class LungSeg:
    def __init__(self, device):
        import torchxrayvision as xrv

        self.m = xrv.baseline_models.chestx_det.PSPNet().eval().to(device)
        self.L = [self.m.targets.index("Left Lung"), self.m.targets.index("Right Lung")]
        self.device = device

    @torch.inference_mode()
    def __call__(self, gray512: np.ndarray) -> np.ndarray:
        # xrv normalisation to [-1024, 1024]
        x = torch.from_numpy(gray512.astype(np.float32) / 255 * 2048 - 1024)[:, None].to(self.device)
        p = torch.sigmoid(self.m(x))[:, self.L].amax(1)
        return (p > 0.5).cpu().numpy().astype(np.uint8)


def build_cache(ids, name, size, device, workers=8, bs=32):
    cdir = paths.ensure(PREP / f"cache_{name}_{size}")
    if (cdir / "ids.txt").exists() and (cdir / "ids.txt").read_text().split() == list(ids):
        print(f"[cache] {cdir} exists"); return cdir
    n = len(ids)
    rgb = np.lib.format.open_memmap(cdir / "rgb.npy", "w+", np.uint8, (n, size, size, 3))
    roi = np.lib.format.open_memmap(cdir / "roi.npy", "w+", np.uint8, (n, size, size))
    seg = LungSeg(device)
    with ThreadPoolExecutor(workers) as ex:
        for k in tqdm(range(0, n, bs), desc=f"cache {name}"):
            chunk = ids[k:k + bs]
            g = list(ex.map(lambda i: load_gray(i, size), chunk))
            g512 = np.stack([np.asarray(Image.fromarray(x).resize((512, 512), Image.BILINEAR)) for x in g])
            m = seg(g512)
            for j, (x, mm) in enumerate(zip(g, m)):
                rgb[k + j] = np.repeat(x[..., None], 3, -1)
                roi[k + j] = np.asarray(Image.fromarray(mm * 255).resize((size, size), Image.NEAREST)) >= 128
    rgb.flush(); roi.flush()
    (cdir / "ids.txt").write_text("\n".join(ids))
    return cdir


def stage_synthetic(device, size=518):
    c = build_synthetic_cohort()
    PREP.mkdir(parents=True, exist_ok=True)
    c.to_csv(PREP / "nih_ptx_cohort.csv", index=False)
    print(c.groupby(["split", "y"]).size())
    cdir = build_cache(c.image_id.tolist(), "nih_ptx", size, device)
    roi = np.load(cdir / "roi.npy", mmap_mode="r")
    w, h = TUBE_GEOMETRY[size]
    pl = {}
    for j, i in enumerate(tqdm(c.image_id, desc="placements")):
        m = np.asarray(roi[j])
        if m.sum() < 0.05 * m.size:  # failed lung segmentation
            continue
        pl[i] = find_best_placements_for_mask(m, i, OVERLAPS, w, h, 2500)
    (PREP / f"nih_ptx_placements_{size}_{w}x{h}.json").write_text(json.dumps(pl))
    keep = feasible_ids(pl, OVERLAPS, 0.10)
    pd.DataFrame({"image_id": sorted(keep)}).to_csv(PREP / f"nih_ptx_common_support_{size}.csv", index=False)
    print(f"[synthetic] common support {len(keep)}/{len(c)}")


@torch.inference_mode()
def stage_detector(device, size=518, workers=8, bs=64):
    """RAD-DINO features of every NIH image (original view), then a drain probe trained on NEATX."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    from wtss.backbones import load_backend

    d = nih_table()
    fpath = paths.CACHE / "features" / "nih_all" / "raddino_518_erm.npz"
    if fpath.exists():
        z = np.load(fpath); X = z["X"]; assert list(z["ids"]) == d.image_id.tolist()
    else:
        be = load_backend("raddino518", device)
        X = np.zeros((len(d), 768), np.float32)
        ids = d.image_id.tolist()
        with ThreadPoolExecutor(workers) as ex:
            nxt = ex.map(lambda i: be.preprocess(Image.fromarray(np.repeat(load_gray(i, size)[..., None], 3, -1))), ids)
            buf, k0 = [], 0
            for t in tqdm(nxt, total=len(ids), desc="raddino all NIH", mininterval=30):
                buf.append(t)
                if len(buf) == bs:
                    with torch.autocast("cuda", dtype=torch.float16):
                        f = be.encode(torch.stack(buf).to(device))
                    X[k0:k0 + bs] = torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy(); k0 += bs; buf = []
            if buf:
                with torch.autocast("cuda", dtype=torch.float16):
                    f = be.encode(torch.stack(buf).to(device))
                X[k0:k0 + len(buf)] = torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy()
        fpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(fpath, X=X, ids=np.asarray(ids))
    lab = d.drain_neatx.notna().to_numpy()
    Xl, yl, gl = X[lab], d.drain_neatx[lab].astype(int).to_numpy(), d.patient[lab].to_numpy()
    oof = np.zeros(len(yl))
    for tr, te in GroupKFold(5).split(Xl, yl, gl):
        oof[te] = LogisticRegression(C=1.0, max_iter=5000).fit(Xl[tr], yl[tr]).predict_proba(Xl[te])[:, 1]
    auc = roc_auc_score(yl, oof)
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(Xl, yl)
    d["drain_score"] = clf.predict_proba(X)[:, 1]
    d.loc[lab, "drain_score"] = oof  # labelled images: out-of-fold scores only
    thr = {f"precision_{p}": float(_thr_at_precision(yl, oof, p)) for p in (0.9, 0.95)}
    rep = {"neatx_cv_auroc": float(auc), "n_labelled": int(lab.sum()), **thr,
           "note": "detector: logistic regression on frozen RAD-DINO CLS features, trained on NEATX (PTX+ only)"}
    PREP.mkdir(parents=True, exist_ok=True)
    (PREP / "DRAIN_DETECTOR.json").write_text(json.dumps(rep, indent=2))
    d[["image_id", "patient", "ptx", "no_finding", "drain_neatx", "drain_score", "view"]].to_csv(PREP / "nih_drain_scores.csv", index=False)
    print(rep)


def _thr_at_precision(y, s, p):
    o = np.argsort(-s); ys = y[o]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.flatnonzero(prec >= p)
    return s[o][ok[-1]] if len(ok) else 1.0


def stage_drain(device, size=518):
    from wtss.data.cxr_drain import make_drain_cohort

    c = make_drain_cohort()
    c.to_csv(PREP / "nih_drain_cohort.csv", index=False)
    print(pd.crosstab([c.source, c.y], c.a))
    build_cache(sorted(c.image_id.tolist()), "nih_drain", size, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["synthetic", "detector", "drain"], required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    {"synthetic": stage_synthetic, "detector": stage_detector, "drain": stage_drain}[a.stage](dev)
