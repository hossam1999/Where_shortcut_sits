"""Example panels of the real-artifact transplant (docs/PREREGISTRATION_REVIEW2.md, R2) for the openly licensed
cohorts (MMOTU ovary, SEE-AI capsule; CC BY 4.0): artifact-free image, the same real artifact instance pasted inside
the ROI, and outside it. Green: ROI contour.
  python scripts/make_transplant_figure.py   -> paper/figures/transplant_examples.pdf/.png
"""
from __future__ import annotations

import importlib.util
import json

import cv2
import matplotlib
import numpy as np
from PIL import Image

from wtss import paths
from wtss.transplant import composite, extract_instance

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

spec = importlib.util.spec_from_file_location("t", paths.REPO_ROOT / "scripts" / "run_transplant.py")
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)


def contour(a, roi):
    a = a.copy()
    cnt, _ = cv2.findContours((roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(a, cnt, -1, (0, 255, 0), 2)
    return a


def main():
    rows = []
    for cohort, label in (("ovary", "Ovarian ultrasound (caliper)"), ("capsule", "Capsule endoscopy (debris)")):
        rec, donors, cache, kind, _ = t.setup(cohort)
        pl = json.loads((paths.DATA / "review2" / f"transplant_{cohort}_placements.json").read_text())
        ids = sorted(pl)[::max(1, len(pl) // 7)][1:3]
        for i in ids:
            p = pl[i]
            rgb, roi, _ = cache.get(i)
            drgb, _, dart = cache.get(p["donor"])
            inst = extract_instance(dart, drgb, kind)
            tiles = [contour(rgb, roi)]
            for key in ("1.00", "0.00"):
                im, _ = composite(Image.fromarray(rgb), inst, p["op"], p[key]["x"], p[key]["y"])
                tiles.append(contour(np.asarray(im), roi))
            rows.append((label, tiles))
    fig, ax = plt.subplots(len(rows), 3, figsize=(7.2, 2.45 * len(rows)))
    for r, (label, tiles) in enumerate(rows):
        for c, (tile, title) in enumerate(zip(tiles, ("artifact-free", "same artifact inside ROI", "same artifact outside ROI"))):
            ax[r, c].imshow(tile); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(title, fontsize=9)
        ax[r, 0].set_ylabel(label, fontsize=8)
    fig.tight_layout()
    out = paths.REPO_ROOT / "paper" / "figures" / "transplant_examples"
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png", dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
