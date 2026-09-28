"""Real example images for the stage reports and the paper (figures/examples/), licence rule of audit/LICENCES.md.

  PYTHONPATH=src python scripts/make_examples.py

Per cohort: one in-ROI (Trap A, r >= 0.5) and one out-of-ROI (Trap B, r < 0.1) artifact-bearing image, 518 px as the
models saw it, with the ROI outline (green) and the automatic artifact mask (red). Plus one real-artifact transplant and
one neutral-paste example (same mask shape filled with the donor's own artifact-free tissue) for the CC BY 4.0
cohorts (MMOTU ovary, SEE-AI capsule).

Licence rule (unclear => do not publish):
  ISIC 2019 images CC BY-NC 4.0 -> committed with the ROI outline only; the overlay that draws the Wegley hair masks
      (no licence found) is written to figures_local/ (git-ignored).
  TN3K/TNCD thyroid (data licence not stated) -> figures_local/ only.
  MMOTU, SEE-AI (CC BY 4.0) -> committed with overlays.
Examples are drawn with a fixed seed from images that are NOT in the blinded audit sample (audit/sample_ids.csv), so
the figures cannot unblind the audit.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "audit"))
from wtss import paths  # noqa: E402
from wtss.transplant import composite, extract_instance, neutral_instance  # noqa: E402

OUT, LOCAL = ROOT / "figures" / "examples", ROOT / "figures_local" / "examples"
SEED = 20260928
# (commit raw with ROI outline, commit artifact overlay)
PUBLISH = {"isic_hair": (True, False), "thyroid_calipers": (False, False), "ovary_calipers": (True, True),
           "capsule_debris": (True, True)}


def rgb3(a):
    return np.ascontiguousarray(a if a.ndim == 3 else np.repeat(a[..., None], 3, -1)).copy()


def outline(a, roi, art=None):
    a = rgb3(a)
    if art is not None:
        a[art > 0] = (255, 0, 0)
    cnt, _ = cv2.findContours((roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(a, cnt, -1, (0, 255, 0), 2)
    return a


def cells():
    import make_audit_sample as mas
    audited = set(pd.read_csv(ROOT / "audit" / "sample_ids.csv").image_id.astype(str)) \
        if (ROOT / "audit" / "sample_ids.csv").exists() else set()
    rng = np.random.default_rng(SEED)
    rows = []
    for name, (c, cache) in mas.cohorts().items():
        c = c.copy(); c["image_id"] = c.image_id.astype(str)
        c = c[~c.image_id.isin(audited)]
        for cell, d in (("in_roi", c[c.trapA_A1]), ("out_roi", c[c.trapB_A1])):
            i = str(rng.choice(sorted(d.image_id)))
            img, roi, art = cache.get(i)
            raw_ok, ov_ok = PUBLISH[name]
            n = int((art > 0).sum())
            r = float(((art > 0) & (roi > 0)).sum() / n) if n else float("nan")
            files = {}
            for kind, arr, ok in (("roi", outline(img, roi), raw_ok), ("overlay", outline(img, roi, art), ov_ok)):
                base = OUT if ok else LOCAL
                base.mkdir(parents=True, exist_ok=True)
                f = base / f"{name}_{cell}_{kind}.png"
                Image.fromarray(arr).save(f)
                files[kind] = str(f.relative_to(ROOT))
            rows.append({"cohort": name, "cell": cell, "image_id": i, "overlap_r": r, "artifact_px": n,
                         "roi_file": files["roi"], "overlay_file": files["overlay"],
                         "committed_roi": raw_ok, "committed_overlay": ov_ok})
    return rows


def transplant_rows():
    spec = importlib.util.spec_from_file_location("t", ROOT / "scripts" / "run_transplant.py")
    t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
    rows = []
    for cohort in ("ovary", "capsule"):
        rec, donors, cache, kind, _ = t.setup(cohort)
        pl = json.loads((paths.DATA / "review2" / f"transplant_{cohort}_placements.json").read_text())
        i = sorted(pl)[len(pl) // 3]
        p = pl[i]
        rgb, roi, _ = cache.get(i)
        drgb, _, dart = cache.get(p["donor"])
        inst = extract_instance(dart, drgb, kind)
        neu = neutral_instance(inst, rgb3(drgb), dart, f"{cohort}|{p['donor']}")
        OUT.mkdir(parents=True, exist_ok=True)
        for tag, ins in (("transplant", inst), ("neutral_paste", neu)):
            im, m = composite(Image.fromarray(rgb3(rgb)), ins, p["op"], p["1.00"]["x"], p["1.00"]["y"])
            a = outline(np.asarray(im), roi)
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(a, cnt, -1, (255, 0, 0), 1)
            f = OUT / f"{cohort}_{tag}_in_roi.png"
            Image.fromarray(a).save(f)
            rows.append({"cohort": cohort, "cell": tag, "image_id": i, "donor": p["donor"], "overlap_r": 1.0,
                         "roi_file": str(f.relative_to(ROOT)), "overlay_file": str(f.relative_to(ROOT)),
                         "committed_roi": True, "committed_overlay": True})
    return rows


def main():
    rows = cells()
    try:
        rows += transplant_rows()
    except Exception as e:  # recorded, not fatal
        print("transplant examples failed:", repr(e))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "examples.csv", index=False)
    (OUT / "README.md").write_text(
        "# Example images\n\nGenerated by `scripts/make_examples.py` (seed 20260928; images outside the blinded audit "
        "sample). Green: ROI outline; red: automatic artifact mask (overlay files) or the pasted instance's outline "
        "(transplant / neutral paste). `examples.csv` lists the source image IDs and overlaps.\n\n"
        "Only licence-cleared files are here (audit/LICENCES.md): ISIC 2019 with the ROI outline only (the hair "
        "masks drawn in the overlay have no licence; overlays are in the git-ignored figures_local/), MMOTU and "
        "SEE-AI (CC BY 4.0) with overlays. Thyroid (TN3K/TNCD, licence not stated) examples are local only.\n")
    print(pd.DataFrame(rows)[["cohort", "cell", "image_id", "overlap_r", "committed_roi", "committed_overlay"]])


if __name__ == "__main__":
    main()
