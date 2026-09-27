"""LaMa (big-lama, Suvorov et al. 2022) inpainting of artifact pixels -> a parallel 518-px cache for the "modern
inpainting" baseline (docs/PREREGISTRATION_INPAINT_LAMA.md).

Writes <cache>_lama/{rgb.npy|gray.npy} with every image whose artifact mask is non-empty inpainted (mask dilated by
3 px); ROI / artifact masks / ids are symlinked from the original cache.
  python scripts/data/precompute_lama.py --cohort thyroid
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from wtss import paths

CACHES = {"thyroid": (paths.DATA / "us" / "tncd" / "cache_518", "marker.npy"),
          "ovary": (paths.DATA / "ovary" / "cache_518", "marker.npy"),
          "capsule": (paths.DATA / "capsule" / "cache_518", "contam.npy"),
          "isic": (paths.DATA / "isic2019" / "prepared" / "cache_518", "hair.npy")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=list(CACHES))
    a = ap.parse_args()
    src, art_file = CACHES[a.cohort]
    dst = paths.ensure(src.parent / (src.name + "_lama"))
    gray = (src / "gray.npy").exists()
    X = np.load(src / ("gray.npy" if gray else "rgb.npy"), mmap_mode="r")
    M = np.load(src / art_file, mmap_mode="r")
    out = np.lib.format.open_memmap(dst / ("gray.npy" if gray else "rgb.npy"), "w+", np.uint8, X.shape)
    for f in os.listdir(src):
        if f not in ("gray.npy", "rgb.npy") and not (dst / f).exists():
            (dst / f).symlink_to(src / f)
    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama(torch.device("cuda"))
    k = np.ones((7, 7), np.uint8)
    n_done = 0
    for j in range(len(X)):
        img = np.asarray(X[j])
        m = np.asarray(M[j]) > 0
        if not m.any():
            out[j] = img
            continue
        rgb = np.repeat(img[..., None], 3, -1) if gray else img
        mk = cv2.dilate(m.astype(np.uint8), k) * 255
        res = np.asarray(lama(Image.fromarray(rgb), Image.fromarray(mk)))[: rgb.shape[0], : rgb.shape[1]]
        out[j] = res.mean(-1).round().astype(np.uint8) if gray else res
        n_done += 1
        if n_done % 500 == 0:
            print(f"[lama] {a.cohort} {j + 1}/{len(X)} inpainted {n_done}", flush=True)
    out.flush()
    print(f"[lama] {a.cohort} done: {n_done} inpainted of {len(X)}", flush=True)


if __name__ == "__main__":
    main()
