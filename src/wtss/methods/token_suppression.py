"""Self-Localising Artifact Suppression (SLAS).

Pixel artifact detectors fail on thin in-ROI artifacts (hair: IoU 0.22). Foundation-model patch tokens are rich
enough to recognise "foreign overlay" locally. SLAS learns that recognition from *synthetic* insertions only:

  1. insert the generic artifact library (wtss.synthetic.draw_generic_artifact) into training images; the
     inserted-pixel mask gives free patch labels (patch covered >= 20 % -> artifact);
  2. fit a linear probe on patch tokens (artifact vs not) — no real artifact label or mask is used;
  3. at inference, drop patches whose probe score exceeds tau and mean-pool the remaining patch tokens.
     Combined with the ROI (mask-then-suppress), out-of-ROI artifacts are removed by the ROI and in-ROI
     artifacts by the probe.

Views produced for every image (L2-normalised pooled patch tokens):
  tok_all      mean of all patch tokens                       (ERM analogue)
  tok_roi      mean of ROI patches                            (masking analogue, same image)
  tok_clean    mean of patches the probe calls clean           (SLAS)
  tok_roiclean mean of ROI patches the probe calls clean       (Mask-then-SLAS)
plus per-image probe maps for localisation evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PATCH = 14


class _Imgs(Dataset):
    def __init__(self, ids, render, preprocess, roi_fn=None, art_fn=None):
        self.ids, self.render, self.pre, self.roi_fn, self.art_fn = list(ids), render, preprocess, roi_fn, art_fn

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        i = self.ids[k]
        out = self.render(i)
        img, art = out if isinstance(out, tuple) else (out, None)
        roi = self.roi_fn(i) if self.roi_fn else np.ones((img.size[1], img.size[0]), np.uint8)
        if art is None:
            art = self.art_fn(i) if self.art_fn else np.zeros_like(roi)
        return self.pre(img), torch.from_numpy(np.ascontiguousarray(roi)).float(), torch.from_numpy(np.ascontiguousarray(art)).float(), k


def _patch_frac(m: torch.Tensor, g: int) -> torch.Tensor:
    """(B, H, W) mask -> (B, g*g) fraction of each patch covered."""
    return torch.nn.functional.adaptive_avg_pool2d(m.unsqueeze(1), g).flatten(1)


@torch.inference_mode()
def dino_tokens(model, x):
    out = model.forward_features(x)
    return out["x_norm_patchtokens"]  # (B, N, D)


def fit_probe(model, device, ids, render_inserted, preprocess, n_tokens: int = 300_000, bs: int = 32, seed: int = 0):
    """Linear artifact-vs-clean probe on patch tokens from generic insertions. Returns (w, b) numpy."""
    from sklearn.linear_model import LogisticRegression

    dl = DataLoader(_Imgs(ids, render_inserted, preprocess), batch_size=bs, num_workers=4)
    X, Y = [], []
    per_img = max(20, n_tokens // max(1, len(ids)))
    rng = np.random.default_rng(seed)
    for x, _roi, art, _k in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            t = dino_tokens(model, x.to(device)).float()
        g = int(round(t.shape[1] ** 0.5))
        frac = _patch_frac(art.to(device), g)
        t, frac = t.cpu().numpy(), frac.cpu().numpy()
        for b in range(len(t)):
            pos = np.flatnonzero(frac[b] >= 0.2)
            neg = np.flatnonzero(frac[b] == 0)
            if len(pos) == 0:
                continue
            p = rng.choice(pos, min(len(pos), per_img // 2), replace=False)
            q = rng.choice(neg, min(len(neg), per_img // 2), replace=False)
            X.append(t[b, np.r_[p, q]]); Y.append(np.r_[np.ones(len(p)), np.zeros(len(q))])
    X, Y = np.concatenate(X), np.concatenate(Y)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, Y)
    return clf.coef_.ravel().astype(np.float32), float(clf.intercept_[0])


@torch.inference_mode()
def pooled_views(model, device, ids, render, preprocess, roi_fn, probe, tau: float = 0.5, bs: int = 32,
                 art_fn: Callable | None = None) -> Dict[str, np.ndarray]:
    """Pooled token views for every id (see module doc). If art_fn is given, also returns patch-level
    artifact coverage and probe scores for localisation evaluation."""
    w = torch.as_tensor(probe[0], device=device)
    b = float(probe[1])
    dl = DataLoader(_Imgs(ids, render, preprocess, roi_fn, art_fn), batch_size=bs, num_workers=4)
    out = {k: None for k in ("tok_all", "tok_roi", "tok_clean", "tok_roiclean")}
    if art_fn is not None:
        out["tok_roiclean_oracle"] = None
    loc = {"score": [], "cover": [], "roi": []}
    for x, roi, art, k in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            t = dino_tokens(model, x.to(device)).float()
        g = int(round(t.shape[1] ** 0.5))
        rfrac = _patch_frac(roi.to(device), g) >= 0.5
        s = torch.sigmoid(t @ w + b)  # (B, N)
        clean = s < tau
        views = [("tok_all", torch.ones_like(clean)), ("tok_roi", rfrac), ("tok_clean", clean), ("tok_roiclean", rfrac & clean)]
        if art_fn is not None:  # ceiling: drop patches the REAL artifact mask covers (same 20 % rule as the probe labels)
            cov = _patch_frac(art.to(device), g)
            views.append(("tok_roiclean_oracle", rfrac & (cov < 0.2)))
        for name, keep in views:
            kk = keep.float()
            kk = torch.where(kk.sum(1, keepdim=True) == 0, torch.ones_like(kk), kk)  # never pool an empty set
            v = (t * kk.unsqueeze(-1)).sum(1) / kk.sum(1, keepdim=True)
            v = torch.nn.functional.normalize(v, dim=1).cpu().numpy()
            if out[name] is None:
                out[name] = np.zeros((len(ids), v.shape[1]), np.float32)
            out[name][k.numpy()] = v
        if art_fn is not None:
            loc["score"].append(s.cpu().numpy().astype(np.float16))
            loc["cover"].append(cov.cpu().numpy().astype(np.float16))
            loc["roi"].append(rfrac.cpu().numpy())
    if art_fn is not None:
        out["loc_score"] = np.concatenate(loc["score"])
        out["loc_cover"] = np.concatenate(loc["cover"])
        out["loc_roi"] = np.concatenate(loc["roi"])
    return out
