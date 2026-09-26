"""Feature extraction with per-*view* caching.

A view is a deterministic function image_id -> PIL image (e.g. "masked image with the
ruler at 50% overlap"). Because features of a view do not depend on the training seed
or on the environment, environments are assembled by indexing cached views:

    X_env[i] = X_artifact_view[i] if present(i, seed, env) else X_clean_view[i]

This is mathematically identical to the pilot's per-(seed, env) extraction and needs
~10x fewer backbone passes.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .backbones import Backend


class ViewDataset(Dataset):
    def __init__(self, ids: Sequence[str], render: Callable[[str], Image.Image], preprocess):
        self.ids = list(ids)
        self.render = render
        self.preprocess = preprocess

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.preprocess(self.render(self.ids[i])), i


def _worker_init(_):
    import cv2

    cv2.setNumThreads(0)
    torch.set_num_threads(1)


@torch.inference_mode()
def extract_view(backend: Backend, ids: Sequence[str], render: Callable[[str], Image.Image], cache: Path,
                 device, batch_size: int = 64, num_workers: int = 6, amp: bool = True,
                 desc: str = "") -> np.ndarray:
    """Return features (len(ids), d) aligned to `ids`; cached as npz keyed by the view."""
    ids = [str(i) for i in ids]
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        cid = z["ids"].astype(str)
        if len(cid) == len(ids) and np.array_equal(cid, np.asarray(ids)):
            return z["X"]
        pos = {k: j for j, k in enumerate(cid)}
        if all(i in pos for i in ids):
            return z["X"][[pos[i] for i in ids]]
    loader = DataLoader(ViewDataset(ids, render, backend.preprocess), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, persistent_workers=False,
                        worker_init_fn=_worker_init if num_workers else None, prefetch_factor=4 if num_workers else None)
    out = None
    t0 = time.time()
    for xb, idx in tqdm(loader, desc=desc or cache.stem, leave=False, dynamic_ncols=True):
        xb = xb.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            f = backend.encode(xb)
        f = torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy()
        if out is None:
            out = np.zeros((len(ids), f.shape[1]), np.float32)
        bad = ~np.isfinite(f).all(1)
        if bad.any():  # rare fp16 overflow in ViT attention: recompute those rows in fp32
            f[bad] = torch.nn.functional.normalize(backend.encode(xb[torch.as_tensor(bad, device=device)]).float(),
                                                   dim=1).cpu().numpy()
            print(f"[features] fp32 recompute for {int(bad.sum())} image(s) in {cache.stem}", flush=True)
        out[idx.numpy()] = f
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=out, ids=np.asarray(ids))
    print(f"[features] {cache.name}: {len(ids)} imgs in {time.time() - t0:.0f}s", flush=True)
    return out


def assemble_env(X_clean: np.ndarray, X_art: np.ndarray, present: np.ndarray) -> np.ndarray:
    return np.where(present[:, None], X_art, X_clean).astype(np.float32)


class ImageCache:
    """Resized uint8 RGB (+ ROI mask) arrays for a cohort, memory-mapped from disk."""

    def __init__(self, root: Path, ids: Sequence[str], size: int,
                 load_rgb: Callable[[str], np.ndarray], load_roi: Callable[[str], np.ndarray] | None,
                 workers: int = 8):
        self.size = size
        root.mkdir(parents=True, exist_ok=True)
        ids = [str(i) for i in ids]
        ids_path, rgb_path, roi_path = root / "ids.txt", root / "rgb.npy", root / "roi.npy"
        if ids_path.exists() and ids_path.read_text().split() == ids and rgb_path.exists():
            self.rgb = np.load(rgb_path, mmap_mode="r")
            self.roi = np.load(roi_path, mmap_mode="r") if roi_path.exists() else None
        else:
            rgb = np.lib.format.open_memmap(rgb_path, mode="w+", dtype=np.uint8, shape=(len(ids), size, size, 3))
            roi = (np.lib.format.open_memmap(roi_path, mode="w+", dtype=np.uint8, shape=(len(ids), size, size))
                   if load_roi else None)
            from concurrent.futures import ThreadPoolExecutor

            def one(k):
                rgb[k] = load_rgb(ids[k])
                if roi is not None:
                    roi[k] = load_roi(ids[k])

            with ThreadPoolExecutor(workers) as ex:
                list(tqdm(ex.map(one, range(len(ids))), total=len(ids), desc=f"cache {size}px", leave=False))
            rgb.flush()
            if roi is not None:
                roi.flush()
            ids_path.write_text("\n".join(ids))
            self.rgb = np.load(rgb_path, mmap_mode="r")
            self.roi = np.load(roi_path, mmap_mode="r") if roi is not None else None
        self.index = {k: j for j, k in enumerate(ids)}

    def image(self, image_id: str) -> Image.Image:
        return Image.fromarray(np.asarray(self.rgb[self.index[image_id]]))

    def roi_mask(self, image_id: str) -> np.ndarray:
        return np.asarray(self.roi[self.index[image_id]])
