"""Image + lung-mask cache for the P3 (Bassi et al. 2024) replication, following P3's preprocessing (Supp. Note 6.1):
grayscale -> histogram equalization -> 224 x 224 (test images are first padded to a square with black borders; training
and validation images are not) -> stored as uint8 (P3 rescales to [0, 1] and repeats to 3 channels at load time).
DICOMs (BrixIA) are read with pydicom (MONOCHROME1 inverted); 16-bit PNGs (BIMCV) are equalized at full bit depth.

Lung masks: P3's segmentation targets came from a lung U-Net of an earlier study that is not published. We use the
TorchXRayVision PSPNet (chestx_det; union of left and right lung at p > 0.5), as in our own pipeline. Recorded deviation.

    python scripts/replication/p3_prepare.py --which cohort|test
Writes $WTSS_DATA/replication/isnet/cache_<which>/{gray.npy, roi.npy, ids.txt}.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
D = os.environ.get("WTSS_DATA", "/workspace/data")
S = 224


def read_raw(path: str) -> np.ndarray:
    """Grayscale pixel values at their stored bit depth. DICOM MONOCHROME1 is inverted (raw DICOM data). BIMCV PNGs are
    already stored in display orientation whatever their JSON flag says (checked visually), so PNGs are never inverted."""
    if path.endswith(".dcm"):
        import pydicom
        ds = pydicom.dcmread(path)
        a = ds.pixel_array
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            a = a.max() - a
        return a
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a


def equalize(a: np.ndarray) -> np.ndarray:
    """Histogram equalization to uint8 at the stored bit depth (cv2.equalizeHist for 8-bit; the same CDF mapping over
    all stored values for 16-bit and DICOM, so low-contrast 16-bit images are not quantised to 8 bits first)."""
    if a.dtype == np.uint8:
        return cv2.equalizeHist(a)
    v, inv, cnt = np.unique(a.ravel(), return_inverse=True, return_counts=True)
    cdf = np.cumsum(cnt).astype(np.float64)
    cdf = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1.0)
    return (cdf[inv] * 255).round().astype(np.uint8).reshape(a.shape)


def process(path: str, pad: bool) -> np.ndarray:
    g = equalize(read_raw(path))
    if pad:
        h, w = g.shape
        n = max(h, w)
        sq = np.zeros((n, n), np.uint8)
        sq[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = g
        g = sq
    return cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["cohort", "test"], required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    from prepare_cxr import LungSeg
    df = pd.read_csv(f"{D}/replication/isnet/{'cohort' if args.which == 'cohort' else 'test_cohort'}.csv")
    if "present" in df:
        df = df[df.present.astype(bool)]
    out = Path(f"{D}/replication/isnet/cache_{args.which}")
    out.mkdir(parents=True, exist_ok=True)
    n = len(df)
    gray = np.lib.format.open_memmap(out / "gray.npy", "w+", np.uint8, (n, S, S))
    roi = np.lib.format.open_memmap(out / "roi.npy", "w+", np.uint8, (n, S, S))
    seg = LungSeg(args.device)
    paths = [f"{D}/{p}" for p in df.path]
    pad = args.which == "test"
    with ThreadPoolExecutor(32) as ex:
        for k in range(0, n, 64):
            g = list(ex.map(lambda p: process(p, pad), paths[k:k + 64]))
            g512 = np.stack([cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR) for x in g])
            m = seg(g512)
            for j, (x, mm) in enumerate(zip(g, m)):
                gray[k + j] = x
                roi[k + j] = cv2.resize(mm, (S, S), interpolation=cv2.INTER_NEAREST)
            if k % 1280 == 0:
                print(f"{k}/{n}", flush=True)
    gray.flush(); roi.flush()
    (out / "ids.txt").write_text("\n".join(df.image_id.astype(str)))
    print(f"wrote {out}: {n} images; mean lung fraction {roi[:].mean():.3f}")


if __name__ == "__main__":
    main()
