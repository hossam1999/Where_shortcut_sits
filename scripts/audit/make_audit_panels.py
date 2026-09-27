"""Regenerate the visual-audit panels of docs/ARTIFACT_MASK_AUDIT.md (40 random images per cell, seed 101).
Panels are written to results/audit/ (not committed: derived from third-party images)."""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from wtss import paths
from wtss.experiments.real_traps import RealCache

OUT = paths.ensure(paths.RESULTS / "audit")


def panel(cache, ids, overlay, name):
    tiles = []
    for n, i in enumerate(ids):
        img, roi, art = cache.get(i)
        a = (img if img.ndim == 3 else np.repeat(img[..., None], 3, -1)).copy()
        if overlay:
            a[art > 0] = (255, 0, 0)
        t = Image.fromarray(a).resize((200, 200), Image.NEAREST); ImageDraw.Draw(t).text((3, 3), str(n), fill=(0, 255, 0))
        tiles.append(np.asarray(t))
    Image.fromarray(np.concatenate([np.concatenate(tiles[k:k + 8], 1) for k in range(0, 40, 8)], 0)).save(OUT / name)
    pd.Series(list(ids)).to_csv(OUT / name.replace(".png", "_ids.csv"), index=False)


if __name__ == "__main__":
    for tag, csv, cdir in (("thy", paths.DATA / "us/tncd/thyroid_cohort.csv", paths.DATA / "us/tncd/cache_518"),
                           ("ov", paths.DATA / "ovary/ovary_cohort.csv", paths.DATA / "ovary/cache_518")):
        c = pd.read_csv(csv); cache = RealCache(cdir, roi_file="roi.npy", art_file="marker.npy")
        panel(cache, c[c.marker_px >= 15].sample(40, random_state=101).image_id, True, f"aud_{tag}_A1.png")
        panel(cache, c[c.marker_px == 0].sample(40, random_state=101).image_id, False, f"aud_{tag}_A0.png")
