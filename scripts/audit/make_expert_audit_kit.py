"""Blinded expert audit kit for the artifact labels (second review, point 6; docs/REVIEW2_RESPONSE.md).

For each cohort a stratified random sample (seed 20260928) of artifact-positive and artifact-negative images (by the
detector / probe / published masks) is rendered twice: (1) the raw image with the ROI contour only — the rater judges
whether the artifact is present and where it lies, blind to the automatic label; (2) the same image with the automatic
artifact mask in red — the rater judges mask quality. Items are shuffled; the key is written separately.

  python scripts/audit/make_expert_audit_kit.py [--n 60]        -> $WTSS_DATA/expert_audit_kit/ (images are third-party:
                                                                   the kit is shared with raters, not committed)
Score two completed sheets with scripts/audit/score_expert_audit.py.
"""
from __future__ import annotations

import argparse
import importlib.util

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from wtss import paths
from wtss.experiments.real_traps import RealCache

spec = importlib.util.spec_from_file_location("rt", paths.REPO_ROOT / "scripts" / "run_thyroid_traps.py")
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
ART = {"thyroid": "sonographer caliper / measurement mark", "ovary": "sonographer caliper / measurement mark",
       "capsule": "debris, bubbles or bile (contamination)", "isic": "hair (or ruler)"}
ROI = {"thyroid": "nodule", "ovary": "tumour", "capsule": "lesion box", "isic": "lesion"}

README = """# Expert audit of automatic artifact labels — instructions for raters

Two raters with clinical imaging experience (ultrasound: radiologist or sonographer; capsule: gastroenterologist;
dermoscopy: dermatologist) complete the sheet **independently**. Do not discuss items before both have finished.

For every item open `blind/<item>.png` (raw image; green = region of interest drawn by the dataset annotators) and fill:
- `present`: is the artifact named in the `artifact` column visible anywhere in the image? `yes` / `no` / `unsure`
- `location`: if present, where does most of it lie relative to the green region? `inside` / `outside` / `both` / `na`
Then open `masked/<item>.png` (same image; red = automatically detected artifact pixels) and fill:
- `mask_quality`: `good` (red covers the artifact, little else) / `partial` (misses a clear part) / `wrong`
  (red mostly on something that is not the artifact) / `na` (no red pixels)
- `comment`: optional.

Save your copy as `audit_sheet_<initials>.csv` and return it. Estimated time: about 20 s per item.
"""


def contour(a, roi):
    cnt, _ = cv2.findContours((roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(a, cnt, -1, (0, 255, 0), 1)
    return a


def cohorts():
    from wtss.data.isic2019_spec import load_spec_cohort
    out = {}
    c = rt.cohort(); out["thyroid"] = (c, c.marker_px >= 15, c.marker_px == 0, RealCache(rt.T / "cache_518", "roi.npy", "marker.npy"))
    c = rt.ovary_cohort(); out["ovary"] = (c, c.marker_px >= 15, c.marker_px == 0, RealCache(rt.OV / "cache_518", "roi.npy", "marker.npy"))
    c = rt.capsule_cohort(); out["capsule"] = (c, c.contam_frac >= 0.10, c.contam_frac < 0.03, RealCache(rt.CAP / "cache_518", "roi.npy", "contam.npy"))
    c = load_spec_cohort(); out["isic"] = (c, ~c.A0, c.A0, RealCache(paths.DATA / "isic2019" / "prepared" / "cache_518", "roi_spec.npy", "hair.npy"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="items per cohort (half positive, half negative)")
    a = ap.parse_args()
    root = paths.ensure(paths.DATA / "expert_audit_kit")
    (root / "blind").mkdir(exist_ok=True); (root / "masked").mkdir(exist_ok=True)
    rows = []
    for name, (c, pos, neg, cache) in cohorts().items():
        for lab, sel in ((1, pos), (0, neg)):
            ids = c[sel].image_id.astype(str).sample(a.n // 2, random_state=20260928).tolist()
            rows += [{"cohort": name, "image_id": i, "auto_label": lab} for i in ids]
    key = pd.DataFrame(rows).sample(frac=1, random_state=20260928).reset_index(drop=True)
    key.insert(0, "item", [f"item{k:04d}" for k in range(len(key))])
    caches = {n: v[3] for n, v in cohorts().items()}
    auto_loc = []
    for r in key.itertuples():
        img, roi, art = caches[r.cohort].get(r.image_id)
        a_ = np.ascontiguousarray(img if img.ndim == 3 else np.repeat(img[..., None], 3, -1))
        Image.fromarray(contour(a_.copy(), roi)).save(root / "blind" / f"{r.item}.png")
        m = a_.copy(); m[art > 0] = (255, 0, 0)
        Image.fromarray(contour(m, roi)).save(root / "masked" / f"{r.item}.png")
        n = int((art > 0).sum()); rr = float(((art > 0) & (roi > 0)).sum() / n) if n else np.nan
        auto_loc.append("na" if not n else ("inside" if rr >= 0.5 else "outside" if rr < 0.1 else "both"))
    key["auto_location"] = auto_loc
    key.to_csv(root / "KEY_do_not_share_with_raters.csv", index=False)
    sheet = key[["item", "cohort"]].assign(artifact=key.cohort.map(ART), roi=key.cohort.map(ROI), present="", location="",
                                          mask_quality="", comment="")
    sheet.to_csv(root / "audit_sheet_BLANK.csv", index=False)
    (root / "README.md").write_text(README)
    print(f"kit: {root} ({len(key)} items)")


if __name__ == "__main__":
    main()
