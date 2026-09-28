"""Round-4 helpers (docs/PREREGISTRATION_ROUND4.md): masking-variant renderers and the Holm adjustment."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "round4"))
import common as K  # noqa: E402
from masking_variants import variant_renderers  # noqa: E402


class _Cache:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.rgb = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
        self.roi = np.zeros((96, 96), np.uint8)
        cv2.circle(self.roi, (40, 50), 15, 1, -1)

    def get(self, i):
        return self.rgb, (self.roi if i == "a" else np.zeros_like(self.roi)), None


def test_variant_renderers():
    c = _Cache()
    R = variant_renderers(c, size=96)
    out = c.roi == 0
    black = np.asarray(R["mask_black"]("a"))
    assert (black[out] == 0).all() and (black[~out] == c.rgb[~out]).all()
    blur = np.asarray(R["mask_blur"]("a"))
    assert (blur[~out] == c.rgb[~out]).all() and blur[out].std() < c.rgb[out].std()
    for v in ("crop_box", "crop_mask"):
        assert np.asarray(R[v]("a")).shape == (96, 96, 3)
    assert (np.asarray(R["crop_box"]("empty")) == c.rgb).all()  # empty ROI: full image


def test_holm():
    adj = K.holm([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.2])
