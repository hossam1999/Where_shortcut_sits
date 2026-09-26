"""Synthetic artifacts at controlled artifact-ROI overlap, and trap environments.

Ported unchanged from the pilot (``pilot_core.py`` v3): placements, ruler drawing,
style randomisation, and the RNG namespace of artifact presence. Placements are
frozen in ``frozen/placements_*.json`` and reused, never regenerated.

Added for the chest-radiograph extension: a curvilinear *tube* artifact (the
radiographic analogue of the ruler: a thin radio-opaque device), placed by the same
overlap search against the lung mask.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .utils import stable_int

OVERLAPS = [0.00, 0.25, 0.50, 0.75, 1.00]
DEFAULT_SEEDS = [42, 123, 456]
# Ruler geometry: ~0.45% of image area at both resolutions.
ARTIFACT_GEOMETRY = {224: (28, 8), 518: (65, 19)}

ENV_RATES = {
    # env: (P(A=1 | Y=1), P(A=1 | Y=0))
    "train_corr": (0.9, 0.1),
    "test_corr": (0.9, 0.1),
    "test_rev": (0.1, 0.9),
    "train_uncorr": (0.5, 0.5),
    "test_uncorr": (0.5, 0.5),
}


def artifact_present(image_id: str, y: int, seed: int, env: str) -> bool:
    """Deterministic artifact assignment (identical RNG namespace to the pilot)."""
    if env == "clean":
        return False
    if env == "always":
        return True
    p_pos, p_neg = ENV_RATES[env]
    p = p_pos if y == 1 else p_neg
    rng = np.random.default_rng(stable_int("artifact_presence", image_id, y, seed, env))
    return bool(rng.random() < p)


def presence_vector(ids: Sequence[str], y: Sequence[int], seed: int, env: str) -> np.ndarray:
    return np.array([artifact_present(i, int(t), seed, env) for i, t in zip(ids, y)], dtype=bool)


# --------------------------------------------------------------------------- placements
def integral_image(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def rect_sum(ii, x, y, w, h):
    return ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x]


def find_best_placements_for_mask(mask: np.ndarray, image_id: str, targets: Sequence[float],
                                  artifact_w: int, artifact_h: int, n_candidates: int = 2500) -> Dict:
    H, W = mask.shape
    rng = np.random.default_rng(stable_int("placement", image_id, W, H, artifact_w, artifact_h))
    n_all = (W - artifact_w + 1) * (H - artifact_h + 1)
    if n_all <= n_candidates:
        xs, ys = np.meshgrid(np.arange(W - artifact_w + 1), np.arange(H - artifact_h + 1))
        xs, ys = xs.ravel(), ys.ravel()
    else:
        xs = rng.integers(0, W - artifact_w + 1, size=n_candidates)
        ys = rng.integers(0, H - artifact_h + 1, size=n_candidates)
        step = max(2, min(W, H) // 24)
        gx, gy = np.meshgrid(np.arange(0, W - artifact_w + 1, step), np.arange(0, H - artifact_h + 1, step))
        xs = np.concatenate([xs, gx.ravel()])
        ys = np.concatenate([ys, gy.ravel()])
    ii = integral_image(mask)
    frac = rect_sum(ii, xs, ys, artifact_w, artifact_h).astype(np.float64) / float(artifact_w * artifact_h)
    out = {}
    for t in targets:
        idx = int(np.argmin(np.abs(frac - t)))
        out[f"{t:.2f}"] = {"x": int(xs[idx]), "y": int(ys[idx]), "achieved": float(frac[idx]),
                           "error": float(abs(frac[idx] - t))}
    return out


def feasible_ids(placements: Dict, overlaps=OVERLAPS, tolerance: float = 0.10) -> set:
    return {k for k, p in placements.items() if max(p[f"{x:.2f}"]["error"] for x in overlaps) <= tolerance}


# --------------------------------------------------------------------------- rulers
RULER_STYLE_PALETTE = ((232, 232, 232), (255, 255, 255), (248, 220, 64), (18, 18, 18),
                       (196, 40, 40), (36, 72, 176), (24, 148, 148), (210, 210, 210))


def sample_ruler_style(image_id: str, overlap: float) -> dict:
    rng = np.random.default_rng(stable_int("ruler_style_v1", image_id, f"{float(overlap):.2f}"))
    fill = RULER_STYLE_PALETTE[int(rng.integers(0, len(RULER_STYLE_PALETTE)))]
    luma = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
    outline = (15, 15, 15) if luma > 80 else (235, 235, 235)
    return {"fill": fill, "outline": outline, "opacity": float(rng.uniform(0.40, 1.0)),
            "n_ticks": int(rng.integers(3, 13)), "tick_width": int(rng.integers(1, 4)),
            "scale": float(rng.uniform(0.70, 1.0)), "ticks_on": str(rng.choice(["bottom", "top", "both"])),
            "tick_slant": int(rng.choice([-1, 0, 1]))}


def draw_ruler(img: Image.Image, x: int, y: int, w: int, h: int) -> Tuple[Image.Image, np.ndarray]:
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle([x, y, x + w - 1, y + h - 1], fill=(232, 232, 232), outline=(20, 20, 20))
    step = max(4, w // 7)
    for i, xx in enumerate(range(x + 2, x + w - 1, step)):
        tick = max(2, h - (2 if i % 2 == 0 else 4))
        d.line([xx, y + h - 1, xx, y + h - tick], fill=(15, 15, 15), width=1)
    m = np.zeros((img.height, img.width), dtype=np.uint8)
    m[y:y + h, x:x + w] = 1
    return out, m


def draw_ruler_variable(img: Image.Image, x: int, y: int, w: int, h: int, style: dict):
    nw = max(4, int(round(w * float(style["scale"]))))
    nh = max(3, int(round(h * float(style["scale"]))))
    x0 = int(x + (w - nw) // 2)
    y0 = int(y + (h - nh) // 2)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    fill_a = tuple(int(c) for c in style["fill"]) + (255,)
    outline_a = tuple(int(c) for c in style["outline"]) + (255,)
    d.rectangle([x0, y0, x0 + nw - 1, y0 + nh - 1], fill=fill_a, outline=outline_a)
    step = max(2, nw // max(int(style["n_ticks"]), 1))
    tw, slant, ticks_on = int(style["tick_width"]), int(style["tick_slant"]), str(style["ticks_on"])
    for i, xx in enumerate(range(x0 + 2, x0 + nw - 1, step)):
        tick_h = max(2, nh - (2 if i % 2 == 0 else 4))
        dx = slant * max(1, tick_h // 3)
        if ticks_on in ("bottom", "both"):
            d.line([xx, y0 + nh - 1, xx + dx, y0 + nh - tick_h], fill=outline_a, width=tw)
        if ticks_on in ("top", "both"):
            d.line([xx, y0, xx + dx, y0 + tick_h], fill=outline_a, width=tw)
    arr = np.asarray(overlay).copy()
    arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * float(style["opacity"])).astype(np.uint8)
    out = Image.alpha_composite(img.convert("RGBA"), Image.fromarray(arr, "RGBA")).convert("RGB")
    m = np.zeros((img.height, img.width), dtype=np.uint8)
    m[y0:y0 + nh, x0:x0 + nw] = 1
    return out, m


# --------------------------------------------------------------------------- CXR tube
def draw_tube(img: Image.Image, x: int, y: int, w: int, h: int, image_id: str = "",
              style_seed: Optional[int] = None) -> Tuple[Image.Image, np.ndarray]:
    """Radio-opaque tube segment inside box (x,y,w,h): a bright, slightly curved band
    with a radio-opaque stripe and side-hole gaps (chest-drain / catheter appearance)."""
    rng = np.random.default_rng(stable_int("tube_style_v1", image_id) if style_seed is None else style_seed)
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    m = np.zeros(arr.shape[:2], np.uint8)
    t = np.linspace(0, 1, 200)
    bend = rng.uniform(-0.35, 0.35) * h
    cx = x + t * (w - 1)
    cy = y + h / 2 + bend * np.sin(np.pi * t) * (1 if w >= h else 0)
    if h > w:  # vertical box: swap roles
        cy = y + t * (h - 1)
        cx = x + w / 2 + rng.uniform(-0.35, 0.35) * w * np.sin(np.pi * t)
    pts = np.stack([cx, cy], 1).astype(np.int32)
    thick = max(2, int(round(min(w, h) * 0.45)))
    cv2.polylines(m, [pts], False, 1, thickness=thick, lineType=cv2.LINE_8)
    stripe = np.zeros_like(m)
    cv2.polylines(stripe, [pts], False, 1, thickness=max(1, thick // 3), lineType=cv2.LINE_8)
    m = m & _box(m.shape, x, y, w, h)
    stripe = stripe & m
    val = arr.mean(-1, keepdims=True)
    bright = np.clip(val + 70, 0, 255)
    out = np.where(m[..., None] == 1, 0.35 * arr + 0.65 * bright, arr)
    out = np.where(stripe[..., None] == 1, 245.0, out)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)), m


def _box(shape, x, y, w, h):
    b = np.zeros(shape, np.uint8)
    b[y:y + h, x:x + w] = 1
    return b
