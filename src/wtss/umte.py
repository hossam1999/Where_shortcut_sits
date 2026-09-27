"""U-MtE — artifact-agnostic Mask-then-Erase. Public, backbone-agnostic tool (the method evaluated in the paper).

Needs only training images + ROI masks (no artifact example, mask or label). Optional: diagnosis labels of
artifact-free images for disease protection, and image-level artifact labels for a group-balanced head.

    from wtss.umte import UMtE
    m = UMtE("dino518")                                   # dino518 | dermlip224 | raddino518 | medsiglip448 |
                                                          # convnext384 | any wtss.backbones.Backend
    m.fit(train_images, train_rois)                       # generic overlays -> masked insertion pairs -> subspace
    m.fit(train_images, train_rois, y=y, artifact_free=a0)  # + disease protection (recommended when available)
    Z = m.transform(test_images, test_rois)               # masked + erased features for ANY downstream head
    clf = m.fit_head(Z_train, y_train, artifact=a_train)  # optional: logistic head (group-balanced if artifact given)
    m.save("umte.pt"); m = UMtE.load("umte.pt")

CLI:
    python -m wtss.umte fit   --backbone dino518 --images tr/*.png --rois roi_dir/ --out umte.pt
    python -m wtss.umte embed --model umte.pt --images te/*.png --rois roi_dir/ --out feats.npz
ROI masks in a folder are matched to images by file stem (non-zero = ROI).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .backbones import BACKEND_SIZE, Backend, load_backend
from .methods.insertion import SubspaceEraser, disease_directions, fit_difference_subspace, protect
from .ops import apply_roi_mask
from .synthetic import draw_generic_artifact


def _img(x, S):
    im = x if isinstance(x, Image.Image) else Image.open(x).convert("RGB") if isinstance(x, (str, Path)) \
        else Image.fromarray(np.asarray(x).astype(np.uint8))
    return im.convert("RGB").resize((S, S), Image.BICUBIC)


def _roi(x, S):
    if x is None:
        return np.ones((S, S), bool)
    m = Image.open(x).convert("L") if isinstance(x, (str, Path)) else Image.fromarray((np.asarray(x) > 0).astype(np.uint8) * 255)
    return np.asarray(m.resize((S, S), Image.NEAREST)) > 0


class UMtE:
    def __init__(self, backbone="dino518", device: str = "cuda", energy: float = 0.90, max_k: int = 64,
                 batch_size: int = 32, seed: int = 0):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.bname = backbone if isinstance(backbone, str) else backbone.name
        self.bb: Backend = load_backend(backbone, self.device) if isinstance(backbone, str) else backbone
        self.S = BACKEND_SIZE.get(self.bname, self.bb.size)
        self.energy, self.max_k, self.bs, self.seed = energy, max_k, batch_size, seed
        self.eraser: Optional[SubspaceEraser] = None

    @torch.inference_mode()
    def _encode(self, pil_images):
        out = []
        for s in range(0, len(pil_images), self.bs):
            x = torch.stack([self.bb.preprocess(im) for im in pil_images[s:s + self.bs]]).to(self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                out.append(self.bb.encode(x).float().cpu().numpy())
        return np.concatenate(out)

    def masked_features(self, images: Sequence, rois: Optional[Sequence] = None, insert: bool = False, keys=None):
        rois = list(rois) if rois is not None else [None] * len(images)
        ims = []
        for j, (x, r) in enumerate(zip(images, rois)):
            im, roi = _img(x, self.S), _roi(r, self.S)
            if insert:  # generic overlay, then mask (same order as the experiments' mask_insert view)
                im = draw_generic_artifact(im, roi, f"u|{keys[j] if keys is not None else j}")
            ims.append(apply_roi_mask(im, roi))
        return self._encode(ims)

    def fit(self, images: Sequence, rois: Optional[Sequence] = None, y=None, artifact_free=None, keys=None):
        X0 = self.masked_features(images, rois)
        X1 = self.masked_features(images, rois, insert=True, keys=keys)
        er = fit_difference_subspace(X0, X1, energy=self.energy, max_k=self.max_k, seed=self.seed)
        if y is not None and artifact_free is not None:
            af = np.asarray(artifact_free, bool)
            er = protect(er, disease_directions(X0[af], np.asarray(y)[af], seed=self.seed))
        self.eraser = er
        self._train_features = X0
        return self

    def transform(self, images: Sequence, rois: Optional[Sequence] = None, features: Optional[np.ndarray] = None):
        assert self.eraser is not None, "call fit() or load() first"
        X = self.masked_features(images, rois) if features is None else features
        return self.eraser(X)

    @staticmethod
    def fit_head(Z, y, artifact=None, C: float = 1.0, seed: int = 0):
        """Logistic head; group-balanced over (artifact x label) when image-level artifact labels are given."""
        from sklearn.linear_model import LogisticRegression

        from .heads import group_weights
        w = group_weights(np.asarray(y), np.asarray(artifact)) if artifact is not None else None
        cw = None if artifact is not None else "balanced"
        return LogisticRegression(C=C, class_weight=cw, max_iter=3000, solver="liblinear",
                                  random_state=seed).fit(Z, y, sample_weight=w)

    def save(self, path):
        e = self.eraser
        torch.save({"backbone": self.bname, "U": e.U, "mu": e.mu, "k": e.k, "energy_curve": e.energy_curve,
                    "energy": self.energy, "max_k": self.max_k}, path)

    @classmethod
    def load(cls, path, device="cuda", backbone: Optional[Backend] = None):
        d = torch.load(path, weights_only=False)
        m = cls(backbone or d["backbone"], device, d["energy"], d["max_k"])
        m.eraser = SubspaceEraser(d["U"], d["mu"], d["k"], d["energy_curve"])
        return m


def _match(images, folder):
    if folder is None:
        return None
    idx = {Path(p).stem: p for p in glob.glob(str(Path(folder) / "*"))}
    return [idx.get(Path(i).stem) or next((v for k, v in idx.items() if k.startswith(Path(i).stem)), None) for i in images]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit"); f.add_argument("--backbone", default="dino518"); f.add_argument("--images", nargs="+", required=True)
    f.add_argument("--rois"); f.add_argument("--out", required=True)
    e = sp.add_parser("embed"); e.add_argument("--model", required=True); e.add_argument("--images", nargs="+", required=True)
    e.add_argument("--rois"); e.add_argument("--out", required=True)
    a = ap.parse_args()
    ims = sorted(sum((glob.glob(p) for p in a.images), []))
    if a.cmd == "fit":
        m = UMtE(a.backbone).fit(ims, _match(ims, a.rois), keys=[Path(i).stem for i in ims])
        m.save(a.out); print(f"U-MtE fitted on {len(ims)} images: erased rank k={m.eraser.k} -> {a.out}")
    else:
        m = UMtE.load(a.model)
        Z = m.transform(ims, _match(ims, a.rois))
        np.savez(a.out, images=np.array(ims), Z=Z); print(f"embedded {len(ims)} images -> {a.out} {Z.shape}")


if __name__ == "__main__":
    main()
