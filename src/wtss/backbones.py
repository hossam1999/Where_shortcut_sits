"""Frozen backbones. Each returns an L2-normalised global embedding.

- dino224 / dino518 : DINOv2 ViT-B/14 (torch.hub, CLS token after norm), as in the pilot.
- dermlip224        : DermLIP / PanDerm-base (Derm1M open_clip fork), dermatology foundation model.
- raddino518        : RAD-DINO (microsoft/rad-dino), chest-radiograph foundation model.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .paths import CACHE, REPO_ROOT
from .utils import IMAGENET_MEAN_RGB, IMAGENET_STD_RGB

_MEAN = torch.tensor(IMAGENET_MEAN_RGB).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD_RGB).view(3, 1, 1)


def dino_preprocess(img: Image.Image) -> torch.Tensor:
    return (TF.to_tensor(img) - _MEAN) / _STD


@dataclass
class Backend:
    name: str
    size: int  # experiment (render) resolution
    model: torch.nn.Module
    preprocess: Callable[[Image.Image], torch.Tensor]
    encode: Callable[[torch.Tensor], torch.Tensor]


def load_dino(size: int, device, arch: str = "b") -> Backend:
    """DINOv2 ViT-{s,b,l}/14 (scale ablation: docs/PREREGISTRATION_SCALE.md); the default ViT-B is the main backbone."""
    torch.hub.set_dir(str(CACHE / "torch_hub"))
    model = torch.hub.load("facebookresearch/dinov2", f"dinov2_vit{arch}14", pretrained=True).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return Backend(f"dinov2_{arch}14_{size}", size, model, dino_preprocess, lambda x: model(x))


def _derm1m_src() -> Path:
    repo = CACHE / "third_party" / "Derm1M"
    # always the official repo: the archived copy lacks the tokenizer vocab (.gz files were not archived)
    if not (repo / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz").exists():
        subprocess.run(["rm", "-rf", str(repo)], check=True)
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/SiyuanYan1/Derm1M.git", str(repo)], check=True)
    return repo / "src"


def load_dermlip(device) -> Backend:
    src = str(_derm1m_src().resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    for k in [k for k in sys.modules if k == "open_clip" or k.startswith("open_clip.")]:
        del sys.modules[k]
    import open_clip  # type: ignore  # the Derm1M fork

    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Backend("dermlip_panderm_224", 224, model, preprocess, lambda x: model.encode_image(x))


def load_raddino(device, size: int = 518) -> Backend:
    from transformers import AutoModel

    model = AutoModel.from_pretrained("microsoft/rad-dino", cache_dir=str(CACHE / "hf")).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    mean = torch.tensor([0.5307, 0.5307, 0.5307]).view(3, 1, 1)
    std = torch.tensor([0.2583, 0.2583, 0.2583]).view(3, 1, 1)

    def pre(img: Image.Image) -> torch.Tensor:
        return (TF.to_tensor(img) - mean) / std

    def enc(x):
        return model(pixel_values=x).pooler_output

    return Backend(f"raddino_{size}", size, model, pre, enc)


def load_medsiglip(device, size: int = 448) -> Backend:
    """MedSigLIP-448 (Google Health AI, 2025): SigLIP vision tower trained on medical images incl. chest X-rays.
    Gated on Hugging Face: needs HF_TOKEN with the licence accepted."""
    import os

    from transformers import AutoModel

    model = AutoModel.from_pretrained("google/medsiglip-448", cache_dir=str(CACHE / "hf"),
                                      token=os.environ.get("HF_TOKEN")).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

    def pre(img: Image.Image) -> torch.Tensor:
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        return (TF.to_tensor(img) - mean) / std

    def enc(x):
        out = model.get_image_features(pixel_values=x)
        return out.pooler_output if hasattr(out, "pooler_output") else out

    return Backend(f"medsiglip_{size}", size, model, pre, enc)


def load_convnext(device, size: int = 384) -> Backend:
    """ConvNeXt-Base (ImageNet-22k, fine-tuned 1k; timm) — a CNN backbone to show the methods are not ViT-specific.
    Views are rendered at 518 (shared caches) and resized to 384 in preprocessing; global-average-pooled features."""
    import timm

    torch.hub.set_dir(str(CACHE / "torch_hub"))
    model = timm.create_model("convnext_base.fb_in22k_ft_in1k_384", pretrained=True, num_classes=0).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)

    def pre(img: Image.Image) -> torch.Tensor:
        if img.size != (size, size):
            img = img.resize((size, size), Image.BICUBIC)
        return (TF.to_tensor(img) - mean) / std

    return Backend(f"convnext_b_{size}", size, model, pre, lambda x: model(x))


def load_backend(name: str, device) -> Backend:
    if name == "convnext384":
        return load_convnext(device, 384)
    if name == "dino224":
        return load_dino(224, device)
    if name == "dino518":
        return load_dino(518, device)
    if name in ("dinos518", "dinol518"):
        return load_dino(518, device, name[4])
    if name == "dermlip224":
        return load_dermlip(device)
    if name == "raddino518":
        return load_raddino(device, 518)
    if name == "medsiglip448":
        return load_medsiglip(device, 448)
    raise ValueError(name)


BACKEND_SIZE = {"dino224": 224, "dino518": 518, "dinos518": 518, "dinol518": 518, "dermlip224": 224, "raddino518": 518, "medsiglip448": 518,
                "convnext384": 518}
# medsiglip448 renders views at 518 (shared caches) and resizes to its native 448 in preprocessing
