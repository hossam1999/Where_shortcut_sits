"""Phase 5 (thyroid TN3K calipers, Trap A / Trap B, seed-42 folds 0-2): their methods and our arm d, trained under one rule.

Registered rules (docs/REPLICATION_PLAN.md, phase 5; fixed in docs/PREREGISTRATION_PHASE5.md before any phase-5 result):
  * fp32 everywhere (no autocast);
  * every network: early stopping on AUROC of `val_clean` (inner validation, marker present in 50 % of each class),
    patience 5, at most 48 epochs; the weights of the best validation epoch are kept;
  * one selection criterion for every tuned choice: `val_clean` AUROC (RR-ClArC lambda and layer, mask_cmc mode and C);
  * input 224 px, the thesis fine-tune augmentation (random horizontal flip + random rot90, applied jointly to the ROI);
  * 2-class softmax / cross-entropy (unweighted) for every network, as RR-ClArC's and ISNet's own trainers; score = p(malignant).

Networks:
  erm_r50d  RR-ClArC's base: timm resnet50d.a3_in1k with Pahde's hook layers (identity_0..2, last_conv; their
            models/resnet_timm.py), trained on the original images.
  rrclarc   their RR-ClArC fine-tune of erm_r50d (port of model_correction/rrc.py): CAV from `cavs_max` activations of the
            training images at the hook layer, direction `signal` (their utils/cav.compute_cav, imported unchanged), gradient
            target `all_logits_random`, aggregation `l2_mean`; loss = CE + lambda * aux; weights only from erm_r50d, fresh
            optimizer, lr = base lr / 10, 10 epochs, last epoch kept (their configs); all parameters train (A18 finding).
  isnet     their ISNet (third_party/isnet backup/ISNetLightning.IsDense121Lightning, heat=True, P3's P/d/E/A/B), LRP
            penalty outside the nodule ROI; SGD lr 1e-3, momentum 0.99, gradient-norm clip 1 (P3's recipe).
  armd      ours (A20): torchvision efficientnet_b0 (IMAGENET1K_V1) trained on ROI-masked images; penultimate 1,280-d
            features -> mask_cmc head on CPU (`armd_head`).
"""
from __future__ import annotations

import importlib.util
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0)          # A21
PAHDE = Path("/workspace/third_party/pahde")
ISNET = Path("/workspace/third_party/isnet/backup")


# --------------------------------------------------------------------------------------------------------------- data
def joint_augment(x: torch.Tensor, m: Optional[torch.Tensor], rng: np.random.Generator):
    """The thesis fine-tune augmentation (finetune.TrapImages): random horizontal flip, random rot90 - same draw for x and m."""
    flip, rot = bool(rng.integers(2)), int(rng.integers(4))
    if flip:
        x = x.flip(-1)
        m = m.flip(-1) if m is not None else None
    if rot:
        x = torch.rot90(x, rot, (-2, -1))
        m = torch.rot90(m, rot, (-2, -1)) if m is not None else None
    return x, m


class P5Images(Dataset):
    """(image, roi, label) for one environment DataFrame (image_id, y, a).

    view: renderer key ('erm' original, 'mask' ROI-masked). norm: 'imagenet' (timm / torchvision ImageNet nets) or 'unit'
    ([0, 1], ISNet's P3 input). roi_fn(image_id) -> HxW {0,1} array, resized nearest to `size` (ISNet's penalty mask)."""

    def __init__(self, df: pd.DataFrame, render: Dict[str, Callable], view: str, size: int = 224, train: bool = False,
                 norm: str = "imagenet", roi_fn: Optional[Callable] = None, seed: int = 0):
        self.ids, self.y, self.a = df.image_id.tolist(), df.y.to_numpy().astype(int), df.a.to_numpy().astype(int)
        self.render, self.view, self.size, self.train, self.norm, self.roi_fn = render, view, size, train, norm, roi_fn
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = self.render[self.view](self.ids[i]).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
        m = None
        if self.roi_fn is not None:
            r = Image.fromarray((np.asarray(self.roi_fn(self.ids[i])) > 0).astype(np.uint8) * 255)
            m = torch.from_numpy(np.asarray(r.resize((self.size, self.size), Image.NEAREST)) > 127).float()[None]
        if self.train:   # reproducible per (seed, epoch, index), independent of worker assignment
            x, m = joint_augment(x, m, np.random.default_rng([self.seed, self.epoch, i]))
        if self.norm == "imagenet":
            x = (x - MEAN) / STD
        if m is None:
            m = torch.zeros(1, self.size, self.size)
        return x, m, int(self.y[i]), int(self.a[i])


def loader(ds: P5Images, bs: int, train: bool, workers: int, epoch: int = 0, seed: int = 0):
    ds.epoch = epoch
    g = torch.Generator().manual_seed(seed * 1000 + epoch)
    return DataLoader(ds, batch_size=bs, shuffle=train, drop_last=False, num_workers=workers, generator=g,
                      persistent_workers=False, pin_memory=torch.cuda.is_available())


# ----------------------------------------------------------------------------------------------------------- models
def _pahde_forward_features(self, x):          # verbatim structure of Pahde's models/resnet_timm._forward_features
    x = self.conv1(x); x = self.bn1(x); x = self.act1(x); x = self.maxpool(x)
    x = self.layer1(x); x = self.identity_0(x)
    x = self.layer2(x); x = self.identity_1(x)
    x = self.layer3(x); x = self.identity_2(x)
    x = self.layer4(x)
    return x


def _pahde_forward(self, x):
    x = self.input_identity(x)
    x = self.forward_features(x)
    x = self.last_conv(x)
    return self.forward_head(x)


def make_resnet50d(pretrained: bool = True, n_class: int = 2):
    """timm resnet50d.a3_in1k (paper's weights, P2 p. 22; phase-5 decision) wrapped as Pahde's get_resnet."""
    import timm
    m = timm.create_model("resnet50d.a3_in1k", pretrained=pretrained, num_classes=n_class)
    for n in ("input_identity", "identity_0", "identity_1", "identity_2", "last_conv"):
        setattr(m, n, nn.Identity())
    m.forward_features = _pahde_forward_features.__get__(m)
    m.forward = _pahde_forward.__get__(m)
    return m


def make_effb0(pretrained: bool = True, n_class: int = 2):
    import torchvision
    w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.efficientnet_b0(weights=w)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_class)
    return m


def effb0_features(m, x):
    """Penultimate 1,280-d global-average-pooled features (before dropout and the linear classifier)."""
    return torch.flatten(m.avgpool(m.features(x)), 1)


def make_isnet(pretrained: bool = False, n_class: int = 2):
    import sys
    if str(ISNET) not in sys.path:
        sys.path.insert(0, str(ISNET))
    import ISNetLightning as L
    torch.backends.cudnn.deterministic = True     # their _LRPDenseNet refuses to build otherwise (P3 recipe: deterministic ops)
    net = L.IsDense121Lightning(multiLabel=False, multiMask=False, classes=n_class, heat=True, pretrained=pretrained,
                                LR=1e-3, P=0.7, Ea=1, A=1, B=3, d=0.9, momentum=0.99)
    net.log = lambda *a, **k: None
    return net


def _load_pahde_cav():
    spec = importlib.util.spec_from_file_location("pahde_cav", PAHDE / "utils" / "cav.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.compute_cav


# ---------------------------------------------------------------------------------------------------- RR-ClArC port
def rrclarc_criterion(y_hat: torch.Tensor, kind: str = "all_logits_random") -> torch.Tensor:
    """Their RRClarc.criterion_fn (model_correction/rrc.py), the options their configs use."""
    if kind == "all_logits_random":
        return (y_hat * torch.sign(0.5 - torch.rand_like(y_hat))).sum(1)
    if kind == "all_logits":
        return y_hat.sum(1)
    raise NotImplementedError(kind)


def rrclarc_aux_loss(gradient: torch.Tensor, cav: torch.Tensor, aggregation: str = "l2_mean", mode: str = "cavs_max"):
    """Their RRClarc.loss_compute, verbatim apart from the swin branch (not used)."""
    is_2dim = gradient.dim() == 2
    gradient = gradient[..., None, None] if is_2dim else gradient
    cav = cav.to(gradient)
    if "mean" in aggregation and gradient.dim() != 2:
        axes = 2 if gradient.dim() == 3 else (2, 3)
        gradient = gradient.mean(axes, keepdim=True).expand_as(gradient)
    if mode == "cavs_full":
        g_flat = gradient.flatten(1)
    else:
        axes_new = (1, 0, 2, 3) if gradient.dim() == 4 else (1, 0, 2)
        g_flat = gradient.permute(axes_new).flatten(start_dim=1).permute(1, 0)
    if "cosine" in aggregation:
        return F.cosine_similarity(g_flat, cav).abs().mean(0)
    if "l2" in aggregation:
        return ((g_flat * cav).sum(1) ** 2).mean(0)
    if "l1" in aggregation:
        return (g_flat * cav).sum(1).abs().mean(0)
    raise NotImplementedError(aggregation)


@torch.no_grad()
def collect_cavs_max(model, layer: str, dl, device) -> torch.Tensor:
    """Per image, the spatial max of the layer's output activations (their `cavs_max`)."""
    acts = []
    h = dict(model.named_modules())[layer].register_forward_hook(lambda m, i, o: acts.append(o.flatten(2).amax(2).cpu()))
    model.eval()
    try:
        for x, _, _, _ in dl:
            model(x.to(device))
    finally:
        h.remove()
    return torch.cat(acts)


def compute_rrclarc_cav(acts: torch.Tensor, artifact: np.ndarray, direction: str = "signal") -> torch.Tensor:
    """Their compute_cav (utils/cav.py, imported unchanged) on the training images' cavs_max vectors, targets = artifact."""
    return _load_pahde_cav()(acts.numpy(), np.asarray(artifact).astype(int), cav_type=direction).float()


class RRClarcStep:
    """One RR-ClArC training step on a model with a hook layer (their RRClarc.default_step)."""

    def __init__(self, model, layer: str, cav: torch.Tensor, lam: float, criterion="all_logits_random",
                 aggregation="l2_mean", mode="cavs_max"):
        self.model, self.cav, self.lam = model, cav, lam
        self.criterion, self.aggregation, self.mode = criterion, aggregation, mode
        self.intermediate = None
        dict(model.named_modules())[layer].register_forward_hook(self._hook)

    def _hook(self, m, i, o):          # their clarc_hook
        self.intermediate = o
        return o.clone()

    def loss(self, x, y):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            y_hat = self.model(x)
            yc = rrclarc_criterion(y_hat, self.criterion)
            grad = torch.autograd.grad(outputs=yc, inputs=self.intermediate, create_graph=True, retain_graph=True,
                                       grad_outputs=torch.ones_like(yc))[0]
            aux = rrclarc_aux_loss(grad, self.cav, self.aggregation, self.mode)
        ce = F.cross_entropy(y_hat, y)
        return ce + self.lam * aux, ce.detach(), aux.detach()


# -------------------------------------------------------------------------------------------------------- training
@dataclass
class EarlyStopping:
    """Keep the best validation epoch; stop after `patience` epochs without improvement or at `max_epochs`."""
    patience: int = 5
    max_epochs: int = 48
    best: float = -math.inf
    best_epoch: int = -1
    bad: int = 0

    def step(self, epoch: int, score: float) -> tuple[bool, bool]:
        improved = bool(np.isfinite(score) and score > self.best)
        if improved:
            self.best, self.best_epoch, self.bad = score, epoch, 0
        else:
            self.bad += 1
        stop = self.bad >= self.patience or epoch + 1 >= self.max_epochs
        return improved, stop


def auroc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


@torch.no_grad()
def predict(kind: str, model, dl, device) -> np.ndarray:
    model.eval(); out = []
    heat = getattr(model, "heat", None)
    if kind == "isnet":
        model.heat = False      # classification only: their forward returns self.DenseNet(x) when heat=False
    try:
        for x, _, _, _ in dl:
            z = model(x.to(device))
            z = z[0] if isinstance(z, tuple) else z
            out.append(torch.softmax(z.float(), 1)[:, 1].cpu().numpy())
    finally:
        if kind == "isnet":
            model.heat = heat
    return np.concatenate(out)


@dataclass
class TrainCfg:
    epochs: int = 48
    patience: int = 5
    bs: int = 32
    workers: int = 8
    seed: int = 0
    max_batches: int = 0          # timing / smoke only
    lr: float = 1e-4
    extra: dict = field(default_factory=dict)


def _optimizer(kind: str, model, cfg: TrainCfg):
    if kind == "erm_r50d":       # the thesis fine-tune optimiser (AdamW, head lr x10, wd 1e-4), constant lr
        head = list(model.get_classifier().parameters()); hid = {id(p) for p in head}
        body = [p for p in model.parameters() if id(p) not in hid]
        return torch.optim.AdamW([{"params": body, "lr": cfg.lr}, {"params": head, "lr": cfg.lr * 10}], weight_decay=1e-4)
    if kind == "rrclarc":        # same optimiser type as the base, lr = base lr / 10, fresh state
        head = list(model.get_classifier().parameters()); hid = {id(p) for p in head}
        body = [p for p in model.parameters() if id(p) not in hid]
        return torch.optim.AdamW([{"params": body, "lr": cfg.lr}, {"params": head, "lr": cfg.lr * 10}], weight_decay=1e-4)
    if kind == "isnet":          # P3's recipe
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.99)
    if kind == "armd":           # A20
        return torch.optim.Adam(model.parameters(), lr=cfg.lr)
    raise ValueError(kind)


def train_step(kind: str, model, opt, batch, device, rr: Optional[RRClarcStep] = None) -> float:
    x, m, y, _ = batch
    x, y = x.to(device), y.to(device)
    opt.zero_grad(set_to_none=True)
    if kind == "isnet":
        loss = model.training_step((x, m.to(device), y[:, None]), 0)
    elif kind == "rrclarc":
        loss = rr.loss(x, y)[0]
    else:
        loss = F.cross_entropy(model(x), y)
    assert loss.dtype == torch.float32, "phase 5 is fp32"
    loss.backward()
    if kind == "isnet":
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss)


def fit(kind: str, model, train_ds: P5Images, val_ds: P5Images, cfg: TrainCfg, out: Path, device,
        rr: Optional[RRClarcStep] = None, fixed_epochs: Optional[int] = None) -> dict:
    """Train with early stopping on val_clean AUROC (or `fixed_epochs`, RR-ClArC's fine-tune: last epoch kept).
    Resumes from out/checkpoint.pt; writes best.pt (or last.pt), log.csv."""
    out.mkdir(parents=True, exist_ok=True)
    opt = _optimizer(kind, model, cfg)
    es = EarlyStopping(cfg.patience, fixed_epochs or cfg.epochs)
    log, start = [], 0
    ck = out / "checkpoint.pt"
    if ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); es = c["es"]; log = c["log"]; start = c["epoch"] + 1
        es.patience, es.max_epochs = cfg.patience, fixed_epochs or cfg.epochs     # the current settings apply after a resume
    for ep in range(start, es.max_epochs):
        model.train(); t0, tl, n = time.time(), [], 0
        for k, b in enumerate(loader(train_ds, cfg.bs, True, cfg.workers, ep, cfg.seed)):
            if cfg.max_batches and k >= cfg.max_batches:
                break
            tl.append(train_step(kind, model, opt, b, device, rr)); n += len(b[0])
        vy = val_ds.y if not cfg.max_batches else None
        vp = predict(kind, model, loader(val_ds, max(cfg.bs, 64), False, cfg.workers), device) if not cfg.max_batches else None
        v = auroc(vy, vp) if vp is not None else float("nan")
        improved, stop = es.step(ep, v) if fixed_epochs is None else (False, ep + 1 >= fixed_epochs)
        sec = time.time() - t0
        log.append({"epoch": ep, "train_loss": float(np.mean(tl)) if tl else float("nan"), "val_clean_auroc": v,
                    "img_per_s": n / max(sec, 1e-9), "minutes": sec / 60,
                    "peak_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0})
        pd.DataFrame(log).to_csv(out / "log.csv", index=False)
        print(json.dumps(log[-1]), flush=True)
        if improved:
            torch.save(model.state_dict(), out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "es": es, "log": log, "epoch": ep}, out / "checkpoint.tmp")
        (out / "checkpoint.tmp").replace(ck)
        if stop or cfg.max_batches:
            break
    if fixed_epochs is not None:
        torch.save(model.state_dict(), out / "last.pt")
    return {"best_epoch": es.best_epoch, "best_val_clean_auroc": es.best, "epochs_run": len(log)}


# ------------------------------------------------------------------------------------------------ arm d head (CPU)
def armd_head(F_tr, y_tr, a_tr, F_va, y_va, modes=("pooled", "per_class"), cs=C_GRID):
    """mask_cmc on arm d's features: project out the within-class marker mean differences (round-8 candidates), then a
    class-balanced logistic regression; mode and C by val_clean AUROC (one criterion). Returns (predict_fn, choice)."""
    import sys
    from sklearn.linear_model import LogisticRegression
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "round8"))
    import candidates as CA
    best = None
    for mode in modes:
        Q = CA.cmc_directions(F_tr, y_tr, a_tr, mode)
        P = CA.Projector(Q)
        Ztr, Zva = P(F_tr), P(F_va)
        for C in cs:
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, solver="liblinear", random_state=0).fit(Ztr, y_tr)
            v = auroc(y_va, clf.decision_function(Zva))
            if best is None or v > best[0]:
                best = (v, mode, C, P, clf)
    v, mode, C, P, clf = best
    edge = "lowest" if C == cs[0] else "highest" if C == cs[-1] else ""
    return (lambda X: clf.decision_function(P(X))), {"mode": mode, "C": C, "C_edge": edge, "val_clean_auroc": v}
