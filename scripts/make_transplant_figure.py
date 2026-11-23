"""Example panels of the real-artifact transplant (docs/PREREGISTRATION_REVIEW2.md, R2) and of the neutral-paste
control (docs/PREREGISTRATION_REVIEW3.md, R5), for the openly licensed cohorts only (MMOTU ovary, SEE-AI capsule; CC BY 4.0).

  python scripts/run_transplant.py --cohort ovary --placements_only     # once per cohort, if the placement files
  python scripts/run_transplant.py --cohort capsule --placements_only   # are not under $WTSS_DATA/review2 yet
  python scripts/make_transplant_figure.py   -> paper/figures/transplant_examples.{pdf,png}, transplant_selection.csv

Columns: artifact-free image | the same real artifact instance pasted inside the ROI | outside the ROI | neutral tissue
pasted through the same mask, position and blending inside the ROI (paste-edge control).
Revision 2026-11-23: Okabe-Ito colours (ROI sky blue, pasted region outlined in orange), a magnified inset of the
pasted region (small calipers were invisible at print size), the neutral-paste column, and a deterministic selection:
per cohort the two recipients whose donor instance has the most pixels (ties by image id), so that the pasted artifact
is visible. The rule picks examples for legibility only; every placement is the registered one.
"""
from __future__ import annotations

import importlib.util
import json

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

from wtss import paths
from wtss.transplant import composite, extract_instance, neutral_instance

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

spec = importlib.util.spec_from_file_location("t", paths.REPO_ROOT / "scripts" / "run_transplant.py")
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)

ROI_RGB, ART_RGB = (86, 180, 233), (230, 159, 0)
COLS = ("Artifact-free", "Same artifact\ninside ROI", "Same artifact\noutside ROI", "Neutral paste\ninside ROI (control)")


def draw(a, roi, pasted=None):
    a = np.ascontiguousarray(a if a.ndim == 3 else np.repeat(a[..., None], 3, -1)).copy()
    cnt, _ = cv2.findContours((roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(a, cnt, -1, ROI_RGB, 3)
    if pasted is not None and pasted.any():
        ring = cv2.dilate(pasted.astype(np.uint8), np.ones((7, 7), np.uint8))
        cnt, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(a, cnt, -1, ART_RGB, 2)
    return a


def inset(a, src, pasted, size=170, pad=14):
    """Magnify the pasted region of `src` (unannotated) into the corner of `a` farthest from it."""
    if pasted is None or not pasted.any():
        return a
    ys, xs = np.nonzero(pasted)
    y0, y1 = max(0, ys.min() - pad), min(src.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(src.shape[1], xs.max() + pad + 1)
    crop = np.ascontiguousarray(src[y0:y1, x0:x1])
    crop = crop if crop.ndim == 3 else np.repeat(crop[..., None], 3, -1)
    s = size / max(crop.shape[:2])
    z = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_NEAREST)
    H, W = a.shape[:2]
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    oy = 6 if cy > H / 2 else H - z.shape[0] - 6
    ox = 6 if cx > W / 2 else W - z.shape[1] - 6
    a = a.copy()
    a[oy:oy + z.shape[0], ox:ox + z.shape[1]] = z
    cv2.rectangle(a, (ox, oy), (ox + z.shape[1] - 1, oy + z.shape[0] - 1), ART_RGB, 3)
    return a


def main():
    rows, sel = [], []
    for cohort, label in (("ovary", "Ovarian ultrasound\n(caliper)"), ("capsule", "Capsule endoscopy\n(debris)")):
        rec, donors, cache, kind, _ = t.setup(cohort)
        pl = json.loads((paths.DATA / "review2" / f"transplant_{cohort}_placements.json").read_text())
        free = rec.image_id.tolist()
        fb = [cache.get(free[j])[0] for j in np.random.default_rng(20260928).choice(len(free), 20, replace=False)]
        npx = {i: p.get("n_px", 0) for i, p in pl.items()}
        ids = sorted(pl, key=lambda i: (-npx[i], i))[:2]
        for i in ids:
            p = pl[i]
            rgb, roi, _ = cache.get(i)
            drgb, _, dart = cache.get(p["donor"])
            inst = extract_instance(dart, drgb, kind)
            neu = neutral_instance(inst, drgb, dart, f"{i}|{p['donor']}", fb)
            base = np.asarray(Image.fromarray(rgb).convert("RGB"))
            tiles = [draw(base, roi)]
            for ins, key in ((inst, "1.00"), (inst, "0.00"), (neu, "1.00")):
                im, m = composite(Image.fromarray(rgb), ins, p["op"], p[key]["x"], p[key]["y"])
                im = np.asarray(im)
                tiles.append(inset(draw(im, roi, m), im, m))
            rows.append((label, tiles))
            sel.append({"cohort": cohort, "recipient": i, "donor": p["donor"], "op": p["op"], "n_px": npx[i],
                        "ov_in": p["1.00"]["achieved"], "ov_out": p["0.00"]["achieved"]})
    fig, ax = plt.subplots(len(rows), 4, figsize=(7.2, 1.95 * len(rows) + 0.5))
    for r, (label, tiles) in enumerate(rows):
        for c, tile in enumerate(tiles):
            ax[r, c].imshow(tile); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(COLS[c], fontsize=8)
        ax[r, 0].set_ylabel(label, fontsize=8)
    fig.tight_layout()
    out = paths.REPO_ROOT / "paper" / "figures" / "transplant_examples"
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png", dpi=200)
    pd.DataFrame(sel).to_csv(paths.REPO_ROOT / "paper" / "figures" / "transplant_selection.csv", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
