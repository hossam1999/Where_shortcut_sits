"""Small shared helpers: deterministic hashing, seeding, image I/O."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
IMAGENET_STD_RGB = (0.229, 0.224, 0.225)
# Fill colour for masked-out pixels (ImageNet mean, as in the pilot).
MEAN_RGB = tuple(int(round(x * 255)) for x in IMAGENET_MEAN_RGB)


def stable_int(*parts: object) -> int:
    """Deterministic 32-bit integer from arbitrary parts (identical to the pilot)."""
    s = "||".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:16], 16) % (2**32 - 1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def normalize_image_id(x: str) -> str:
    x = Path(str(x)).name
    for suffix in ["_segmentation.png", ".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
        if x.endswith(suffix):
            return x[: -len(suffix)]
    return x


def load_rgb(path: Path, size: int) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").resize((size, size), Image.BICUBIC)


def load_mask(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        m = im.convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) >= 128).astype(np.uint8)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
