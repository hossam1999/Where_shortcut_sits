"""TN3K/TNCD thyroid cohort + 518-px caches (gray image, nodule ROI, detected marker mask)."""
import numpy as np
import pandas as pd
from PIL import Image

from wtss import paths
from wtss.data.us_markers import marker_mask

R = paths.DATA / "us" / "tncd" / "ds" / "datasets" / "tn3k"
S = 518

if __name__ == "__main__":
    d = pd.read_csv(paths.DATA / "us" / "tncd" / "markers_stats.csv")
    d["image_id"] = d.split.str[:2] + "_" + d.file.str.replace(".jpg", "", regex=False)
    cdir = paths.ensure(paths.DATA / "us" / "tncd" / f"cache_{S}")
    n = len(d)
    gray = np.lib.format.open_memmap(cdir / "gray.npy", "w+", np.uint8, (n, S, S))
    roi = np.lib.format.open_memmap(cdir / "roi.npy", "w+", np.uint8, (n, S, S))
    mk = np.lib.format.open_memmap(cdir / "marker.npy", "w+", np.uint8, (n, S, S))
    for k, (f, sp) in enumerate(zip(d.file, d.split)):
        g = Image.open(R / f"{sp}-image" / f).convert("L")
        m = marker_mask(np.asarray(g))
        gray[k] = np.asarray(g.resize((S, S), Image.BICUBIC))
        roi[k] = np.asarray(Image.open(R / f"{sp}-mask" / f).convert("L").resize((S, S), Image.NEAREST)) > 127
        mk[k] = np.asarray(Image.fromarray(m * 255).resize((S, S), Image.NEAREST)) > 127
    gray.flush(); roi.flush(); mk.flush()
    (cdir / "ids.txt").write_text("\n".join(d.image_id))
    d.to_csv(paths.DATA / "us" / "tncd" / "thyroid_cohort.csv", index=False)
    print("cached", n)
