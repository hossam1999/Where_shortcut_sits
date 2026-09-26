"""SLAS — Self-Localising Artifact Suppression. Public, backbone-agnostic tool.

Annotate a handful of images (k ~ 5-50) with a binary mask of ANY artifact (hair, ruler, ink, calipers,
markers, tubes, ...). SLAS fits a linear probe on the frozen backbone's patch tokens that recognises artifact
patches, then embeds any image as the mean of the tokens the probe calls clean (optionally restricted to the ROI).
Downstream, train any head on these embeddings exactly as on ordinary frozen features.

    from wtss.slas import SLAS
    s = SLAS("dinov2")                         # or "raddino", "medsiglip", "dermlip", "timm:<name>", or a TokenBackbone
    s.fit(images, artifact_masks)              # few annotated images (PIL / paths / arrays; masks: HxW 0/1)
    E = s.embed(new_images, rois=None)         # {"all", "clean"} (+ {"roi", "roiclean"} when rois are given)
    heat = s.localise(new_images)              # (n, g, g) artifact probability per patch
    s.save("probe.pt"); s = SLAS.load("probe.pt")

CLI:
    python -m wtss.slas fit   --backbone dinov2 --images imgs/*.jpg --masks masks/ --out probe.pt
    python -m wtss.slas embed --probe probe.pt --images test/*.jpg [--rois rois/] --out feats.npz
Masks/ROIs in a folder are matched to images by file stem.
"""
from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------- backbones -> patch tokens
@dataclass
class TokenBackbone:
    """Any frozen model that maps a (B,3,S,S) tensor to patch tokens (B, N, D) on a square grid."""
    name: str
    size: int
    preprocess: Callable[[Image.Image], torch.Tensor]
    tokens: Callable[[torch.Tensor], torch.Tensor]
    module: Optional[torch.nn.Module] = None


def _norm_pre(size, mean, std):
    m, s = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)

    def pre(img: Image.Image):
        img = img.convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BICUBIC)
        return (TF.to_tensor(img) - m) / s
    return pre


def _grid(t: torch.Tensor) -> torch.Tensor:
    """Drop prefix tokens (CLS / registers) or flatten a CNN map; returns (B, g*g, D)."""
    if t.dim() == 4:  # (B, C, H, W)
        return t.flatten(2).transpose(1, 2)
    g = int(np.floor(np.sqrt(t.shape[1])))
    return t[:, t.shape[1] - g * g:]


def vit_block_tokens(vit: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Patch tokens from ANY ViT that has a `.blocks` list (timm / BEiT / CAE / open_clip `.transformer`):
    output of the last block, passed through the final norm when present."""
    blocks = vit.blocks if hasattr(vit, "blocks") else vit.transformer.resblocks
    norm = getattr(vit, "norm", None) or getattr(vit, "ln_post", None) or torch.nn.Identity()
    store = {}
    blocks[-1].register_forward_hook(lambda mod, i, o: store.__setitem__("t", o[0] if isinstance(o, tuple) else o))

    def tok(x):
        vit(x)
        t = store["t"]
        if t.shape[0] != x.shape[0]:  # (N, B, D) layout
            t = t.permute(1, 0, 2)
        return _grid(norm(t))
    return tok


def token_backbone(name: str, device="cuda", size: Optional[int] = None) -> TokenBackbone:
    dev = torch.device(device)
    if name == "dinov2":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True).eval().to(dev)
        S = size or 518
        return TokenBackbone(name, S, _norm_pre(S, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                             lambda x: m.forward_features(x)["x_norm_patchtokens"], m)
    if name == "raddino":
        from transformers import AutoModel
        m = AutoModel.from_pretrained("microsoft/rad-dino").eval().to(dev)
        S = size or 518
        return TokenBackbone(name, S, _norm_pre(S, (0.5307,) * 3, (0.2583,) * 3),
                             lambda x: _grid(m(pixel_values=x).last_hidden_state), m)
    if name == "medsiglip":
        import os
        from transformers import AutoModel
        m = AutoModel.from_pretrained("google/medsiglip-448", token=os.environ.get("HF_TOKEN")).eval().to(dev)
        S = size or 448
        return TokenBackbone(name, S, _norm_pre(S, (0.5,) * 3, (0.5,) * 3),
                             lambda x: _grid(m.vision_model(pixel_values=x).last_hidden_state), m)
    if name == "dermlip":
        from .backbones import load_dermlip
        b = load_dermlip(dev)
        return TokenBackbone(name, 224, b.preprocess, vit_block_tokens(b.model.visual), b.model)
    if name.startswith("timm:"):
        import timm
        m = timm.create_model(name[5:], pretrained=True, num_classes=0).eval().to(dev)
        cfg = timm.data.resolve_data_config({}, model=m)
        S = size or cfg["input_size"][-1]
        return TokenBackbone(name, S, _norm_pre(S, cfg["mean"], cfg["std"]), lambda x: _grid(m.forward_features(x)), m)
    raise ValueError(f"unknown backbone {name!r}")


# ---------------------------------------------------------------- IO helpers
def _img(x) -> Image.Image:
    return x if isinstance(x, Image.Image) else Image.open(x).convert("RGB") if isinstance(x, (str, Path)) \
        else Image.fromarray(np.asarray(x).astype(np.uint8))


def _mask(x, size) -> np.ndarray:
    if x is None:
        return np.ones(size[::-1], np.float32)
    m = Image.open(x).convert("L") if isinstance(x, (str, Path)) else Image.fromarray((np.asarray(x) > 0).astype(np.uint8) * 255)
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    return (np.asarray(m) > 0).astype(np.float32)


# ---------------------------------------------------------------- method
class SLAS:
    def __init__(self, backbone="dinov2", device: str = "cuda", tau: float = 0.5, cover: float = 0.2,
                 roi_frac: float = 0.5, batch_size: int = 16):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.bb = backbone if isinstance(backbone, TokenBackbone) else token_backbone(backbone, self.device)
        self.tau, self.cover, self.roi_frac, self.bs = tau, cover, roi_frac, batch_size
        self.w: Optional[np.ndarray] = None
        self.b = 0.0

    @torch.inference_mode()
    def _tokens(self, images, masks=None):
        """Yields (tokens (B,N,D) fp32 on device, mask coverage (B,N) or None) per batch."""
        for s in range(0, len(images), self.bs):
            S = self.bb.size
            imgs = [_img(x).resize((S, S), Image.BICUBIC) for x in images[s:s + self.bs]]  # square: masks stay aligned
            x = torch.stack([self.bb.preprocess(i) for i in imgs]).to(self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                t = self.bb.tokens(x).float()
            cov = None
            if masks is not None:
                g = int(round(t.shape[1] ** 0.5))
                m = torch.from_numpy(np.stack([_mask(mm, (S, S)) for mm in masks[s:s + self.bs]]))
                cov = F.adaptive_avg_pool2d(m.unsqueeze(1).to(self.device), g).flatten(1)
            yield t, cov

    def fit(self, images: Sequence, artifact_masks: Sequence, C: float = 1.0, per_image: int = 300, seed: int = 0):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(seed)
        X, Y = [], []
        for t, cov in self._tokens(list(images), list(artifact_masks)):
            t, cov = t.cpu().numpy(), cov.cpu().numpy()
            for tb, cb in zip(t, cov):
                pos, neg = np.flatnonzero(cb >= self.cover), np.flatnonzero(cb == 0)
                p = rng.choice(pos, min(len(pos), per_image // 2), replace=False)
                q = rng.choice(neg, min(len(neg), per_image // 2), replace=False)
                X.append(tb[np.r_[p, q]]); Y.append(np.r_[np.ones(len(p)), np.zeros(len(q))])
        X, Y = np.concatenate(X), np.concatenate(Y)
        if Y.min() == Y.max():
            raise ValueError("annotated images must contain both artifact and clean patches")
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X, Y)
        self.w, self.b = clf.coef_.ravel().astype(np.float32), float(clf.intercept_[0])
        return self

    def _scores(self, t):
        return torch.sigmoid(t @ torch.as_tensor(self.w, device=t.device) + self.b)

    def localise(self, images: Sequence) -> np.ndarray:
        out = []
        for t, _ in self._tokens(list(images)):
            s = self._scores(t)
            g = int(round(s.shape[1] ** 0.5))
            out.append(s.view(-1, g, g).cpu().numpy())
        return np.concatenate(out)

    def embed(self, images: Sequence, rois: Optional[Sequence] = None) -> Dict[str, np.ndarray]:
        assert self.w is not None, "call fit() or load() first"
        res: Dict[str, List[np.ndarray]] = {}
        for t, rc in self._tokens(list(images), list(rois) if rois is not None else None):
            clean = self._scores(t) < self.tau
            views = {"all": torch.ones_like(clean), "clean": clean}
            if rc is not None:
                r = rc >= self.roi_frac
                views.update(roi=r, roiclean=r & clean)
            for k, keep in views.items():
                kk = keep.float()
                kk = torch.where(kk.sum(1, keepdim=True) == 0, torch.ones_like(kk), kk)
                v = F.normalize((t * kk.unsqueeze(-1)).sum(1) / kk.sum(1, keepdim=True), dim=1)
                res.setdefault(k, []).append(v.cpu().numpy())
        return {k: np.concatenate(v) for k, v in res.items()}

    def save(self, path):
        torch.save({"backbone": self.bb.name, "size": self.bb.size, "w": self.w, "b": self.b, "tau": self.tau,
                    "cover": self.cover, "roi_frac": self.roi_frac}, path)

    @classmethod
    def load(cls, path, device="cuda", backbone: Optional[TokenBackbone] = None):
        d = torch.load(path, weights_only=False)
        s = cls(backbone or token_backbone(d["backbone"], device, d["size"]), device, d["tau"], d["cover"], d["roi_frac"])
        s.w, s.b = d["w"], d["b"]
        return s


# ---------------------------------------------------------------- CLI
def _match(images, folder):
    if folder is None:
        return None
    idx = {Path(p).stem: p for p in glob.glob(str(Path(folder) / "*"))}
    return [idx.get(Path(i).stem) or next((v for k, v in idx.items() if k.startswith(Path(i).stem)), None) for i in images]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit"); f.add_argument("--backbone", default="dinov2"); f.add_argument("--images", nargs="+", required=True)
    f.add_argument("--masks", required=True); f.add_argument("--out", required=True); f.add_argument("--tau", type=float, default=0.5)
    e = sp.add_parser("embed"); e.add_argument("--probe", required=True); e.add_argument("--images", nargs="+", required=True)
    e.add_argument("--rois"); e.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "fit":
        ims = sorted(sum((glob.glob(p) for p in a.images), []))
        ms = _match(ims, a.masks)
        keep = [(i, m) for i, m in zip(ims, ms) if m is not None]
        s = SLAS(a.backbone, tau=a.tau).fit([i for i, _ in keep], [m for _, m in keep])
        s.save(a.out); print(f"probe fitted on {len(keep)} annotated images -> {a.out}")
    else:
        ims = sorted(sum((glob.glob(p) for p in a.images), []))
        s = SLAS.load(a.probe)
        E = s.embed(ims, _match(ims, a.rois))
        np.savez(a.out, images=np.array(ims), **E); print(f"embedded {len(ims)} images -> {a.out} ({list(E)})")


if __name__ == "__main__":
    main()
