"""Build $WTSS_DATA/us/tncd/markers_stats.csv (input of prepare_thyroid.py).

TN3K images + nodule masks (Gong et al., TRFE-Net) with TNCD benign(0)/malignant(1) labels
(label4trainval.csv / label4test.csv, https://github.com/chenghui-666/ACL). Per image: frozen caliper detector
(wtss.data.us_markers.marker_mask, native resolution), caliper pixels, overlap r = |marker ∩ nodule| / |marker|,
nodule fraction, size, pHash and pHash-chain leakage groups at Hamming <= 2/4/6/8.
  python scripts/data/scan_thyroid.py
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from wtss import paths
from wtss.data.us_markers import marker_mask

R = paths.DATA / "us" / "tncd" / "ds" / "datasets" / "tn3k"


def one(args):
    f, sp = args
    g = Image.open(R / f"{sp}-image" / f).convert("L")
    a = np.asarray(g)
    m = marker_mask(a) > 0
    roi = np.asarray(Image.open(R / f"{sp}-mask" / f).convert("L")) > 127
    mp = int(m.sum())
    return {"marker_px": mp, "r": float((m & roi).sum() / mp) if mp else np.nan, "roi_frac": float(roi.mean()),
            "H": a.shape[0], "W": a.shape[1], "phash_hex": str(imagehash.phash(g.convert("RGB")))}


def groups(ph, t):
    H = np.array([imagehash.hex_to_hash(h).hash.flatten() for h in ph])
    par = list(range(len(ph)))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(len(ph)):
        for j in np.flatnonzero((H[i + 1:] != H[i]).sum(1) <= t):
            a, b = find(i), find(i + 1 + j)
            if a != b:
                par[b] = a
    roots = [find(i) for i in range(len(ph))]
    return pd.Series(roots).astype("category").cat.codes.to_numpy()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); a = ap.parse_args()
    rows = []
    for sp, f in (("trainval", "label4trainval.csv"), ("test", "label4test.csv")):
        lab = pd.read_csv(R / f, header=None, names=["file", "y"])
        lab["split"] = sp
        rows.append(lab)
    d = pd.concat(rows, ignore_index=True)
    with ProcessPoolExecutor(8) as ex:
        stats = list(ex.map(one, zip(d.file, d.split), chunksize=32))
    d = pd.concat([d, pd.DataFrame(stats)], axis=1)
    for t in (2, 4, 6, 8):
        d[f"group_ph{t}"] = groups(d.phash_hex.tolist(), t)
    out = a.out or (paths.DATA / "us" / "tncd" / "markers_stats.csv")
    d.to_csv(out, index=False)
    print(d.shape, "markers:", int((d.marker_px > 0).sum()), "->", out)
