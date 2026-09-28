"""Real-artifact transplant (docs/PREREGISTRATION_REVIEW2.md, R2).

Real artifact pixels (calipers, hair, debris) are cut from artifact-bearing images with their masks and pasted
inside or outside the ROI of the *same* artifact-free image, so that the two locations differ only in where the
identical artifact instance sits.

Instance = original mask pixels inside the largest connected component of the mask after a grouping dilation
(calipers 9 px, hair/debris 5 px), cropped to its bounding box (+2 px). Placement: positions restricted to the field
of view; in-ROI = overlap >= 0.95, out-of-ROI = overlap 0; donors tried in a stable random order (<= 50 donors x 8
dihedral transforms), positions drawn at random among the admissible ones (stable seeds).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from .utils import stable_int

GROUP_DILATE = {"caliper": 9, "hair": 5, "debris": 5}
MIN_PX, MAX_SIDE = 30, 259
IN_MIN, FOV_MIN = 0.95, 0.95
MAX_DONORS, SIGMA = 50, 0.7


def extract_instance(art: np.ndarray, rgb: np.ndarray, kind: str) -> Optional[Dict]:
    """Largest grouped component of a binary artifact mask with its pixels, or None if outside the size limits."""
    m = art > 0
    if m.sum() < MIN_PX:
        return None
    k = GROUP_DILATE[kind]
    dil = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8))
    n, lab = cv2.connectedComponents(dil, connectivity=8)
    if n <= 1:
        return None
    cnt = np.bincount(lab[m], minlength=n)
    cnt[0] = 0
    inst = m & (lab == int(cnt.argmax()))
    if inst.sum() < MIN_PX:
        return None
    ys, xs = np.nonzero(inst)
    H, W = m.shape
    y0, y1 = max(0, ys.min() - 2), min(H, ys.max() + 3)
    x0, x1 = max(0, xs.min() - 2), min(W, xs.max() + 3)
    if max(y1 - y0, x1 - x0) > MAX_SIDE:
        return None
    return {"mask": inst[y0:y1, x0:x1].copy(), "pix": np.ascontiguousarray(rgb[y0:y1, x0:x1]).copy(),
            "n_px": int(inst.sum())}


def dihedral(a: np.ndarray, op: int) -> np.ndarray:
    if op & 1:
        a = a[:, ::-1]
    return np.ascontiguousarray(np.rot90(a, (op >> 1) % 4))


def field_of_view(rgb: np.ndarray) -> np.ndarray:
    g = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 5)
    fov = (g > 15).astype(np.uint8)
    return cv2.morphologyEx(fov, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def _window_fraction(field: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """For every top-left position where the kernel fits: |field ∩ kernel| / |kernel|."""
    r = cv2.matchTemplate(field.astype(np.float32), kern.astype(np.float32), cv2.TM_CCORR)
    return r / float(kern.sum())


def place(recipient_id: str, rgb: np.ndarray, roi: np.ndarray, donors: Sequence[str], instances: Dict[str, Dict]) -> Optional[Dict]:
    """First feasible (donor, transform) with an in-ROI and an out-of-ROI position; None if none of them fits."""
    fov = field_of_view(rgb)
    roi = (roi > 0).astype(np.uint8)
    rng = np.random.default_rng(stable_int("transplant_donor", recipient_id))
    order = [donors[j] for j in rng.permutation(len(donors))[:MAX_DONORS]]
    for d in order:
        inst = instances[d]
        for op in rng.permutation(8):
            k = dihedral(inst["mask"], int(op))
            if k.shape[0] >= roi.shape[0] or k.shape[1] >= roi.shape[1]:
                continue
            f_roi, f_fov = _window_fraction(roi, k), _window_fraction(fov, k)
            ok_fov = f_fov >= FOV_MIN
            pin = np.argwhere(ok_fov & (f_roi >= IN_MIN))
            pout = np.argwhere(ok_fov & (f_roi <= 1e-6))
            if len(pin) and len(pout):
                prng = np.random.default_rng(stable_int("transplant_pos", recipient_id, d, int(op)))
                yi, xi = pin[prng.integers(len(pin))]
                yo, xo = pout[prng.integers(len(pout))]
                return {"donor": d, "op": int(op), "n_px": inst["n_px"],
                        "1.00": {"x": int(xi), "y": int(yi), "achieved": float(f_roi[yi, xi]), "error": 0.0},
                        "0.00": {"x": int(xo), "y": int(yo), "achieved": float(f_roi[yo, xo]), "error": 0.0}}
    return None


def composite(img: Image.Image, inst: Dict, op: int, x: int, y: int):
    """Alpha-composite a transformed instance at (x, y); returns (image, full-size artifact mask)."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    m = dihedral(inst["mask"], op).astype(np.float32)
    p = dihedral(inst["pix"], op).astype(np.float32)
    if p.ndim == 2:
        p = np.repeat(p[..., None], 3, -1)
    a = cv2.GaussianBlur(m, (0, 0), SIGMA)[..., None]
    h, w = m.shape
    arr[y:y + h, x:x + w] = arr[y:y + h, x:x + w] * (1 - a) + p * a
    full = np.zeros(arr.shape[:2], np.uint8)
    full[y:y + h, x:x + w] = m > 0
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8)), full


def make_drawer(placements: Dict, instances: Dict[str, Dict]):
    """Drawer for wtss.experiments.synthetic (signature draw(img, image_id, placement, overlap))."""

    def draw(img, image_id, pp, ov):
        p = placements[image_id]
        return composite(img, instances[p["donor"]], p["op"], int(pp["x"]), int(pp["y"]))

    return draw


def neutral_instance(inst: Dict, donor_rgb: np.ndarray, donor_art: np.ndarray, key: str,
                     fallback_rgbs: Sequence[np.ndarray] = ()) -> Dict:
    """Paste-edge control (docs/PREREGISTRATION_REVIEW3.md, R5): the same mask as `inst`, filled with neutral tissue
    cut from the donor image at a window that does not touch its artifact (dilated 5 px), inside its field of view;
    falls back to artifact-free images of the cohort. Seam geometry and compositing stay identical."""
    h, w = inst["mask"].shape
    rng = np.random.default_rng(stable_int("neutral_window", key))
    for k, src in enumerate([donor_rgb, *fallback_rgbs]):
        art = cv2.dilate((donor_art > 0).astype(np.uint8), np.ones((11, 11), np.uint8)) if k == 0 else \
            np.zeros(src.shape[:2], np.uint8)
        if h >= src.shape[0] or w >= src.shape[1]:
            continue
        box = np.ones((h, w), np.float32)
        touch = cv2.matchTemplate(art.astype(np.float32), box, cv2.TM_CCORR)
        fov = _window_fraction(field_of_view(src), box)
        ok = np.argwhere((touch < 0.5) & (fov >= FOV_MIN))
        if len(ok):
            y, x = ok[rng.integers(len(ok))]
            return {"mask": inst["mask"], "pix": np.ascontiguousarray(src[y:y + h, x:x + w]).copy(), "n_px": inst["n_px"],
                    "source": "donor" if k == 0 else "artifact_free"}
    raise ValueError(f"no neutral window for {key}")
