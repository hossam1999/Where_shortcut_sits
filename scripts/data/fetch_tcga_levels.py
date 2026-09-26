"""Fetch one low-magnification pyramid level of each TCGA lung slide via HTTP range reads (no full download).

For every slide with a GrandQC mask: resolve the GDC file id, open the SVS remotely (fsspec + tifffile), read the
pyramid level whose resolution is closest to 4 µm/px, and store it as JPEG with its µm/px. Idempotent.
Output: $WTSS_DATA/path/levels/<slide>.jpg and levels.csv (slide, file_id, mpp_base, level_downsample, mpp, W, H).
"""
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

from wtss import paths

OUT = paths.ensure(paths.DATA / "path" / "levels")
TARGET_MPP = 4.0


def resolve_ids(names):
    ids = {}
    for k in range(0, len(names), 200):
        chunk = names[k:k + 200]
        q = {"op": "in", "content": {"field": "file_name", "value": chunk}}
        r = requests.post("https://api.gdc.cancer.gov/files", json={"filters": q, "fields": "file_id,file_name,file_size",
                                                                   "size": 1000, "format": "JSON"}, timeout=120)
        for h in r.json()["data"]["hits"]:
            ids[h["file_name"]] = (h["file_id"], h["file_size"])
    return ids


def fetch(args):
    name, fid = args
    slide = name.replace(".svs", "")
    dst = OUT / f"{slide}.jpg"
    if dst.exists():
        return None
    import fsspec
    import tifffile
    try:
        with fsspec.open(f"https://api.gdc.cancer.gov/data/{fid}", "rb", block_size=2 ** 22) as f:
            tf = tifffile.TiffFile(f)
            desc = tf.pages[0].tags["ImageDescription"].value
            m = re.search(r"MPP\s*=\s*([\d.]+)", desc)
            mpp0 = float(m.group(1)) if m else 0.25
            levels = tf.series[0].levels
            ds = [levels[0].shape[1] / l.shape[1] for l in levels]
            j = int(np.argmin([abs(mpp0 * d - TARGET_MPP) for d in ds]))
            arr = levels[j].asarray()
        Image.fromarray(arr).save(dst, quality=90)
        return {"slide": slide, "file_id": fid, "mpp_base": mpp0, "level_downsample": ds[j], "mpp": mpp0 * ds[j],
                "W": arr.shape[1], "H": arr.shape[0]}
    except Exception as e:  # network hiccups: logged, retried on the next run
        print(f"[fetch] {slide} failed: {e}", flush=True)
        return None


if __name__ == "__main__":
    st = pd.read_csv(paths.DATA / "path" / "prepared" / "grandqc_lung_stats.csv")
    names = [s + ".svs" for s in st.slide]
    ids = resolve_ids(names)
    print(f"[fetch] resolved {len(ids)}/{len(names)} GDC file ids; total full size {sum(v[1] for v in ids.values()) / 1e9:.0f} GB "
          f"(only one level is read)", flush=True)
    jobs = [(n, ids[n][0]) for n in names if n in ids]
    log = OUT / "levels.csv"
    rows = pd.read_csv(log).to_dict("records") if log.exists() else []
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as ex:
        for k, r in enumerate(ex.map(fetch, jobs, chunksize=1)):
            if r:
                rows.append(r)
            if k % 50 == 0:
                pd.DataFrame(rows).to_csv(log, index=False)
                print(f"[fetch] {k}/{len(jobs)}", flush=True)
    pd.DataFrame(rows).to_csv(log, index=False)
    print("[fetch] done", len(rows), flush=True)
