"""Native-resolution hair/ruler pixel counts per ISIC 2019 image (no resizing: thin hairs survive)."""
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from wtss import paths

Image.MAX_IMAGE_PIXELS = None


def key(n):
    m = re.match(r"ISIC_(\d+)", n)
    return f"ISIC_{int(m.group(1)):07d}"


def one(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert("L")) >= 128
    return key(p.name), int(a.sum()), int(a.size), a.shape[1], a.shape[0]


if __name__ == "__main__":
    files = sorted((paths.ARTIFACT_MASKS / "hair_ruler").rglob("*.tif"))
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(one, files, chunksize=64))
    d = pd.DataFrame(rows, columns=["key", "hair_px_native", "px_native", "W", "H"])
    d["hair_frac_native"] = d.hair_px_native / d.px_native
    d.to_csv(paths.DATA / "isic2019" / "prepared" / "hair_native.csv", index=False)
    print(d.describe().round(4))
