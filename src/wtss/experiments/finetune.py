"""End-to-end fine-tuning check of the main findings (robustness to the frozen-probe assumption).

Arms (same trap environments as the linear-probe study):
  erm        : fine-tune on original images
  mask       : fine-tune and test on ROI-masked images
  balanced   : group-balanced loss (equal weight per A x Y group; needs image-level A)
  insert_aug : label-independent artifact insertion (p=0.5) as augmentation
  i2e_ft     : insertion invariance — BCE + λ ||g(x) - g(x+)||² on the penultimate embedding
               (end-to-end analogue of I2E: the network is trained so that inserting the artifact
               does not move the representation; no artifact label, no mask)
  mte_post   : fine-tune exactly like `mask`, then apply U-MtE to the fine-tuned network's penultimate features
               (generic-overlay difference subspace on masked training images, 90 % energy) and refit a linear head
               on clean validation (docs/PREREGISTRATION_FT_ERASE.md)
  umte_ft    : erase-while-fine-tuning (docs/PREREGISTRATION_FT_UMTE.md): masked image + generic overlay pairs in every
               batch; an EMA estimate of the overlay-shift subspace (90 % energy, <= 64 dims) of the *current*
               penultimate features is projected out before the head at every step (and at test time), plus the
               mte_ft invariance penalty. No artifact example, mask or label; any architecture.
  cons_ft    : prediction consistency — BCE + λ (logit(x) − logit(x+overlay))², masked view
Model selection: fixed epochs; decision threshold / reporting on clean validation only.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..evaluation import evaluate, select_threshold_clean_val
from ..heads import group_weights
from ..utils import seed_all

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class TrapImages(Dataset):
    def __init__(self, df: pd.DataFrame, render: Dict, view: str, size: int, train: bool, with_insert: bool = False,
                 insert_view: str = "insert"):
        self.ids, self.y, self.a = df.image_id.tolist(), df.y.to_numpy(), df.a.to_numpy()
        self.render, self.view, self.size, self.train, self.with_insert = render, view, size, train, with_insert
        self.insert_view = insert_view

    def __len__(self):
        return len(self.ids)

    def _t(self, img: Image.Image, flip: bool, rot: int):
        img = img.resize((self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
        if flip:
            x = x.flip(2)
        if rot:
            x = torch.rot90(x, rot, (1, 2))
        return (x - MEAN) / STD

    def __getitem__(self, i):
        rng = np.random.default_rng()
        flip, rot = (bool(rng.integers(2)), int(rng.integers(4))) if self.train else (False, 0)
        x = self._t(self.render[self.view](self.ids[i]), flip, rot)
        if self.with_insert:
            xi = self._t(self.render[self.insert_view](self.ids[i]), flip, rot)
            return x, xi, self.y[i], self.a[i]
        return x, x, self.y[i], self.a[i]


def make_model(arch: str):
    import timm

    m = timm.create_model(arch, pretrained=True, num_classes=0)
    head = torch.nn.Linear(m.num_features, 1)
    return m, head


def train_eval(arm: str, E: Dict[str, pd.DataFrame], render: Dict, arch: str = "resnet50", size: int = 224,
               epochs: int = 8, bs: int = 48, lr: float = 1e-4, lam: float = 1.0, seed: int = 0,
               device=None, workers: int = 6):
    device = device or torch.device("cuda")
    seed_all(seed)
    view = "mask" if arm in ("mask", "mte_ft", "mask_balanced", "mte_post", "umte_ft", "cons_ft") else "erm"
    tr = E["train_corr"]
    with_ins = arm in ("i2e_ft", "insert_aug", "mte_ft", "umte_ft", "cons_ft")
    ins_view = "mask_insert" if arm in ("mte_ft", "umte_ft", "cons_ft") else "insert"
    dl = DataLoader(TrapImages(tr, render, view, size, True, with_ins, ins_view), batch_size=bs, shuffle=True,
                    num_workers=workers, drop_last=True, persistent_workers=True)
    body, head = make_model(arch)
    body, head = body.to(device), head.to(device)
    params = [{"params": body.parameters(), "lr": lr}, {"params": head.parameters(), "lr": lr * 10}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    steps = epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 10], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    pos_w = torch.tensor([(1 - tr.y.mean()) / tr.y.mean()], device=device)
    gw = torch.as_tensor(group_weights(tr.y.to_numpy(), tr.a.to_numpy()), dtype=torch.float32)
    wmap = {(int(y), int(a)): float(w) for y, a, w in zip(tr.y, tr.a, gw)}
    proj = {"U": None, "mu": None, "C": None, "t": 0}  # umte_ft: running overlay-shift subspace

    def _refresh():
        w, V = torch.linalg.eigh(proj["C"])
        w, V = w.flip(0).clamp_min(0), V.flip(1)
        cum = torch.cumsum(w, 0) / w.sum().clamp_min(1e-12)
        k = min(int((cum < 0.9).sum().item()) + 1, 64)
        proj["U"] = V[:, :k].contiguous()

    def _project(g):  # float32 features -> overlay subspace removed
        if proj["U"] is None:
            return g
        return g - ((g - proj["mu"]) @ proj["U"]) @ proj["U"].T

    for ep in range(epochs):
        body.train(); head.train()
        for x, xi, y, a in dl:
            x, xi, y = x.to(device, non_blocking=True), xi.to(device, non_blocking=True), y.float().to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                if arm == "insert_aug":
                    pick = torch.rand(len(x), device=device) < 0.5
                    x = torch.where(pick[:, None, None, None], xi, x)
                if arm in ("i2e_ft", "mte_ft"):
                    g = body(torch.cat([x, xi]))
                    g0, g1 = g[: len(x)], g[len(x):]
                    logit = head(g0).squeeze(1)
                    inv = (F.normalize(g0.float(), dim=1) - F.normalize(g1.float(), dim=1)).pow(2).sum(1).mean()
                elif arm == "umte_ft":
                    g = body(torch.cat([x, xi]))
                    with torch.autocast("cuda", enabled=False):
                        g0, g1 = g[: len(x)].float(), g[len(x):].float()
                        d = (g1 - g0).detach()
                        Cb, mb = d.T @ d / len(d), g0.detach().mean(0)
                        if proj["C"] is None:
                            proj["C"], proj["mu"] = Cb, mb
                        else:
                            proj["C"].mul_(0.98).add_(Cb, alpha=0.02); proj["mu"].mul_(0.98).add_(mb, alpha=0.02)
                        if proj["t"] % 20 == 0:
                            _refresh()
                        proj["t"] += 1
                        logit = head(_project(g0)).squeeze(1)
                        inv = (F.normalize(g0, dim=1) - F.normalize(g1, dim=1)).pow(2).sum(1).mean()
                elif arm == "cons_ft":
                    lg = head(body(torch.cat([x, xi]))).squeeze(1).float()
                    logit, inv = lg[: len(x)], (lg[: len(x)] - lg[len(x):]).pow(2).mean()
                else:
                    logit = head(body(x)).squeeze(1)
                    inv = torch.zeros((), device=device)
                if arm in ("balanced", "mask_balanced"):
                    w = torch.tensor([wmap[(int(t), int(s))] for t, s in zip(y.cpu(), a)], device=device)
                    loss = (F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none") * w).mean()
                else:
                    loss = F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pos_w)
                loss = loss + lam * inv
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()

    if arm == "umte_ft":
        _refresh()  # final subspace from the end-of-training features

    @torch.inference_mode()
    def predict(df):
        body.eval(); head.eval()
        out = []
        for x, _, _, _ in DataLoader(TrapImages(df, render, view, size, False), batch_size=128, num_workers=workers):
            if arm == "umte_ft":
                with torch.autocast("cuda", dtype=torch.float16):
                    g = body(x.to(device))
                out.append(torch.sigmoid(head(_project(g.float()))).squeeze(1).cpu().numpy())
                continue
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(torch.sigmoid(head(body(x.to(device))).float()).squeeze(1).cpu().numpy())
        return np.concatenate(out)

    class _P:
        def __init__(self, p): self.p = p
        def predict_proba(self, _): return np.c_[1 - self.p, self.p]

    if arm == "mte_post":  # fine-tune-then-erase
        from .. import heads as H
        from ..methods.insertion import fit_difference_subspace

        @torch.inference_mode()
        def feats(df, v):
            body.eval()
            out = []
            for x, _, _, _ in DataLoader(TrapImages(df, render, v, size, False), batch_size=128, num_workers=workers):
                out.append(body(x.to(device)).float().cpu().numpy())  # fp32: fp16 features overflowed (NaN)
            return np.concatenate(out)
        F0, F1 = feats(tr, "mask"), feats(tr, "mask_insert")
        er = fit_difference_subspace(F0, F1, energy=0.9, seed=seed)
        cv = E["clean_val"]
        Fv = feats(cv, "mask")
        res = {}
        Fe = {env: feats(E[env], "mask") for env in ["clean_test", "test_corr", "test_rev"]}
        # mask_post: same fine-tuned body and head fitting without erasure (isolates the erasure step)
        for name, tf in (("mte_post", er), ("mask_post", lambda X: X)):
            clf = H.fit_on_transformed(tf, F0, tr.y.to_numpy(), Fv, cv.y.to_numpy(), seed)[0]
            thr, _ = select_threshold_clean_val(cv.y.to_numpy(), clf.predict_proba(Fv)[:, 1])
            for env, Fx in Fe.items():
                res[(name, env)] = (_P(clf.predict_proba(Fx)[:, 1]), thr, E[env])
        del body, head, opt
        gc.collect(); torch.cuda.empty_cache()
        return res
    pv = predict(E["clean_val"])
    thr, _ = select_threshold_clean_val(E["clean_val"].y.to_numpy(), pv)
    res = {}
    for env in ["clean_test", "test_corr", "test_rev"]:
        d = E[env]
        res[env] = (_P(predict(d)), thr, d)
    del body, head, opt
    gc.collect(); torch.cuda.empty_cache()
    return res


def run_finetune(name, envs, render, out_dir: Path, arms: Sequence[str], traps: Sequence[str], folds: Sequence[int],
                 arch="resnet50", epochs=8, device=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, frames = [], []
    done = pd.read_csv(out_dir / "metrics.csv") if (out_dir / "metrics.csv").exists() else None
    for trap in traps:
        for k in folds:
            E = {e: envs[(trap, k, e)] for e in ["train_corr", "test_corr", "test_rev", "clean_test", "clean_val"]}
            for arm in arms:
                if done is not None and ((done.trap == trap) & (done.seed == k) & (done.method == arm)).any():
                    continue
                res = train_eval(arm, E, render, arch=arch, epochs=epochs, seed=k, device=device)
                meta = {"cohort": name, "backbone": f"ft_{arch}", "trap": trap, "seed": k, "method": arm}
                for env, (clf, thr, d) in res.items():
                    r, f = evaluate(clf, thr, None, d.y.to_numpy(), d.image_id.to_numpy(), d.a.to_numpy(),
                                    {**meta, "env": "clean" if env == "clean_test" else env})
                    rows.append(r); frames.append(f)
                print(f"[finetune] {name} {arch} {trap} fold {k} {arm}: " +
                      " ".join(f"{r['env']}={r['auc']:.3f}" for r in rows[-3:]), flush=True)
                m = pd.DataFrame(rows)
                if done is not None:
                    m = pd.concat([done, m], ignore_index=True)
                m.to_csv(out_dir / "metrics.csv", index=False)
                p = pd.concat(frames, ignore_index=True)
                if (out_dir / "predictions.csv.gz").exists() and done is not None:
                    p = pd.concat([pd.read_csv(out_dir / "predictions.csv.gz"), p], ignore_index=True).drop_duplicates(
                        ["trap", "seed", "method", "env", "image_id"])
                p.to_csv(out_dir / "predictions.csv.gz", index=False, compression="gzip")
