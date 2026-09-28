"""Regenerate the blinded image-audit sample from the original data (fixed seed).

  python audit/make_audit_sample.py              # blinded sample + review sheet + key
  python audit/make_audit_sample.py --unblind    # after review: copy images into <cohort>/{in_roi,out_roi,artifact_free}/

Cohorts: ISIC 2019 hair, thyroid calipers, ovarian calipers, capsule debris. Cells: in-ROI artifact (Trap A,
overlap r >= 0.5), out-of-ROI artifact (Trap B, r < 0.1), artifact-free (shared group), by the labels used in the paper.
20 images per cell and cohort (10 positive, 10 negative diagnosis), seed 20260928. Images are the 518-px versions the
models saw. Overlays: ROI outline green, automatic artifact mask red.

Licences (audit/LICENCES.md): images whose redistribution is not confirmed (TN3K/TNCD; overlays drawing the ISIC 2019
hair masks) are written to audit_local/ (git-ignored) instead of audit/; the IDs are always committed.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wtss import paths  # noqa: E402
from wtss.experiments.real_traps import RealCache  # noqa: E402

SEED, PER_CELL = 20260928, 20
AUD, LOCAL = ROOT / "audit", ROOT / "audit_local"
# publishable = (raw images, overlays) may be committed
PUBLISH = {"isic_hair": (True, False), "thyroid_calipers": (False, False), "ovary_calipers": (True, True),
           "capsule_debris": (True, True)}


def cohorts():
    spec = importlib.util.spec_from_file_location("rt", ROOT / "scripts" / "run_thyroid_traps.py")
    rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
    from wtss.data.isic2019_spec import load_spec_cohort
    out = {}
    c = load_spec_cohort()
    out["isic_hair"] = (c, RealCache(paths.DATA / "isic2019" / "prepared" / "cache_518", "roi_spec.npy", "hair.npy"))
    out["thyroid_calipers"] = (rt.cohort(), RealCache(rt.T / "cache_518", "roi.npy", "marker.npy"))
    out["ovary_calipers"] = (rt.ovary_cohort(), RealCache(rt.OV / "cache_518", "roi.npy", "marker.npy"))
    out["capsule_debris"] = (rt.capsule_cohort(), RealCache(rt.CAP / "cache_518", "roi.npy", "contam.npy"))
    return out


def overlay(img, roi, art):
    a = np.ascontiguousarray(img if img.ndim == 3 else np.repeat(img[..., None], 3, -1)).copy()
    a[art > 0] = (255, 0, 0)
    cnt, _ = cv2.findContours((roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(a, cnt, -1, (0, 255, 0), 2)
    return a


def sample():
    rows = []
    rng = np.random.default_rng(SEED)
    for name, (c, cache) in cohorts().items():
        c = c.copy(); c["image_id"] = c.image_id.astype(str)
        cells = {"in_roi": c[c.trapA_A1], "out_roi": c[c.trapB_A1], "artifact_free": c[c.A0]}
        for cell, d in cells.items():
            for y in (1, 0):
                pool = d[d.y == y].image_id.tolist()
                pick = rng.choice(pool, min(PER_CELL // 2, len(pool)), replace=False)
                rows += [{"cohort": name, "cell": cell, "y": y, "image_id": i} for i in pick]
    key = pd.DataFrame(rows)
    key = key.sample(frac=1, random_state=SEED).reset_index(drop=True)
    key.insert(0, "audit_id", [f"A{k:04d}" for k in range(len(key))])
    return key


def write(key):
    caches = {k: v[1] for k, v in cohorts().items()}
    sheet = []
    for r in key.itertuples():
        raw_ok, ov_ok = PUBLISH[r.cohort]
        img, roi, art = caches[r.cohort].get(r.image_id)
        img = np.ascontiguousarray(img if img.ndim == 3 else np.repeat(img[..., None], 3, -1))
        base_i = (AUD if raw_ok else LOCAL) / r.cohort / "images"
        base_o = (AUD if ov_ok else LOCAL) / r.cohort / "overlays"
        base_i.mkdir(parents=True, exist_ok=True); base_o.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(base_i / f"{r.audit_id}.png")
        Image.fromarray(overlay(img, roi, art)).save(base_o / f"{r.audit_id}.png")
        sheet.append({"audit_id": r.audit_id, "cohort": r.cohort,
                      "image_file": str((base_i / f"{r.audit_id}.png").relative_to(ROOT)),
                      "overlay_file": str((base_o / f"{r.audit_id}.png").relative_to(ROOT)),
                      "artifact_present": "", "artifact_location": "", "mask_correct": "", "roi_mask_correct": "", "notes": ""})
        n = int((art > 0).sum())
        key.loc[key.audit_id == r.audit_id, "auto_artifact_px"] = n
        key.loc[key.audit_id == r.audit_id, "auto_overlap_r"] = float(((art > 0) & (roi > 0)).sum() / n) if n else np.nan
    pd.DataFrame(sheet).to_csv(AUD / "review_sheet.csv", index=False)
    LOCAL.mkdir(parents=True, exist_ok=True)
    key.to_csv(LOCAL / "KEY_open_after_review.csv", index=False)  # git-ignored: the repository is the reviewer's channel
    import hashlib
    (AUD / "KEY_SHA256.txt").write_text(hashlib.sha256((LOCAL / "KEY_open_after_review.csv").read_bytes()).hexdigest() +
                                        "  audit_local/KEY_open_after_review.csv (regenerate: python audit/make_audit_sample.py)\n")
    key[["audit_id", "cohort", "image_id"]].to_csv(AUD / "sample_ids.csv", index=False)
    for name in PUBLISH:
        for cell in ("in_roi", "out_roi", "artifact_free"):
            d = AUD / name / cell
            d.mkdir(parents=True, exist_ok=True)
            (d / "README.md").write_text("Filled by `python audit/make_audit_sample.py --unblind` after the review "
                                         "(kept empty so that the review is blind).\n")
    print(f"sample: {len(key)} images; sheet: audit/review_sheet.csv")


def unblind():
    key = pd.read_csv(LOCAL / "KEY_open_after_review.csv")
    for r in key.itertuples():
        raw_ok, _ = PUBLISH[r.cohort]
        src = (AUD if raw_ok else LOCAL) / r.cohort / "images" / f"{r.audit_id}.png"
        dst = (AUD if raw_ok else LOCAL) / r.cohort / r.cell / f"{r.audit_id}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print("unblinded copies written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--unblind", action="store_true"); a = ap.parse_args()
    unblind() if a.unblind else write(sample())
