"""Real-artifact traps (thesis Result 3; docs/PREREGISTRATION_ISIC2019_TRAPS.md).

Generic over cohorts: ISIC 2019 hair traps here; the CXR drain trap reuses the same machinery
(`run_traps` takes a cohort table, env dict and a RealCache).
"""
from __future__ import annotations

import gc
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .. import heads as H
from ..backbones import BACKEND_SIZE, load_backend
from ..evaluation import evaluate, same_head_counterfactual, select_threshold_clean_val
from ..features import extract_view
from ..methods.insertion import fit_difference_subspace, insert_aug_head, rank1_head
from ..ops import apply_inpaint, apply_roi_mask
from ..stats import difference_of_deltas, hierarchical_paired_bootstrap, slim
from ..utils import stable_int


class RealCache:
    """Memmapped rgb/roi/artifact-mask arrays written by scripts/data/prepare_*.py."""

    def __init__(self, cdir: Path, roi_file: str = "roi.npy", art_file: str = "hair.npy", art_bit: int | None = None):
        self.art_bit = art_bit  # bit-packed device masks (CXR): select one device type
        # grayscale caches (CXR) are stored with one channel and expanded on read
        self.gray = not (cdir / "rgb.npy").exists()
        self.rgb = np.load(cdir / ("gray.npy" if self.gray else "rgb.npy"), mmap_mode="r")
        self.roi = np.load(cdir / roi_file, mmap_mode="r")
        self.art = np.load(cdir / art_file, mmap_mode="r") if (cdir / art_file).exists() else None
        self.ids = (cdir / "ids.txt").read_text().split()
        self.index = {k: j for j, k in enumerate(self.ids)}

    def get(self, i):
        j = self.index[i]
        art = np.asarray(self.art[j]) if self.art is not None else None
        if art is not None and self.art_bit is not None:
            art = ((art & self.art_bit) > 0).astype(np.uint8)
        img = np.asarray(self.rgb[j])
        if self.gray:
            img = np.repeat(img[..., None], 3, -1)
        return img, np.asarray(self.roi[j]), art


def transplant(target: np.ndarray, donor_rgb: np.ndarray, donor_mask: np.ndarray, key: str, sigma: float = 0.7):
    """Alpha-composite donor artifact pixels onto target after a random flip/rot90 (8 dihedral ops)."""
    op = stable_int("i2e_tf", key) % 8
    m, d = donor_mask.astype(np.float32), donor_rgb
    if op & 1:
        m, d = m[:, ::-1], d[:, ::-1]
    k = (op >> 1) % 4
    m, d = np.rot90(m, k), np.rot90(d, k)
    a = cv2.GaussianBlur(np.ascontiguousarray(m), (0, 0), sigma)[..., None]
    return (target.astype(np.float32) * (1 - a) + np.ascontiguousarray(d).astype(np.float32) * a).clip(0, 255).astype(np.uint8)


def make_renderers(cache: RealCache, out_size: int, donors: Sequence[str], insert_fn: Callable | None = None):
    donors = list(donors)

    def fin(arr):
        img = Image.fromarray(arr)
        return img if img.size[0] == out_size else img.resize((out_size, out_size), Image.BICUBIC)

    def erm(i):
        return fin(cache.get(i)[0])

    def mask(i):
        rgb, roi, _ = cache.get(i)
        return fin(np.asarray(apply_roi_mask(Image.fromarray(rgb), roi)))

    def inpaint(i):
        rgb, _, art = cache.get(i)
        return fin(np.asarray(apply_inpaint(Image.fromarray(rgb), art)))

    def _insert_arr(i):
        rgb, roi, _ = cache.get(i)
        if insert_fn is not None:
            return insert_fn(i, rgb, roi), roi
        d = donors[stable_int("i2e_donor", i) % len(donors)]
        drgb, _, dmask = cache.get(d)
        return transplant(rgb, drgb, dmask, i), roi

    def insert(i):
        return fin(_insert_arr(i)[0])

    def mask_insert(i):
        """Mask-then-Erase view: insert an artifact template, then apply the ROI mask (what survives is in-ROI)."""
        arr, roi = _insert_arr(i)
        return fin(np.asarray(apply_roi_mask(Image.fromarray(arr), roi)))

    return {"erm": erm, "mask": mask, "inpaint": inpaint, "insert": insert, "mask_insert": mask_insert}


@dataclass
class TrapConfig:
    arms: Sequence[str] = ("erm", "mask", "inpaint", "balanced", "dfr", "leace_paired", "leace_unpaired",
                           "i2e", "i2e_balanced", "i2e_rank1", "insert_aug", "prevcal")
    traps: Sequence[str] = ("trapA", "trapB")
    folds: Sequence[int] = (0, 1, 2, 3, 4)
    energy: float = 0.90
    batch_size: int = 64
    workers: int = 6
    n_boot: int = 10000


def run_traps(name: str, cohort: pd.DataFrame, envs: Dict, cache: RealCache, backend_name: str, out_dir: Path,
              feat_dir: Path, cfg: TrapConfig, donors: Sequence[str], insert_fn=None, device=None) -> Path:
    device = device or torch.device("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = load_backend(backend_name, device)
    size = BACKEND_SIZE[backend_name]
    pool = sorted(set().union(*[set(d.image_id) for d in envs.values()]))
    pos = {k: j for j, k in enumerate(pool)}
    rend = make_renderers(cache, size, donors, insert_fn)
    need = {"erm"} | ({"mask"} if "mask" in cfg.arms else set()) | \
        ({"inpaint"} if {"inpaint", "leace_paired"} & set(cfg.arms) else set()) | \
        ({"insert"} if any(a.startswith(("i2e", "insert")) for a in cfg.arms) else set())
    V = {v: extract_view(backend, pool, rend[v], feat_dir / backend.name / f"{v}.npz", device,
                         cfg.batch_size, cfg.workers, desc=f"{name}/{v}") for v in sorted(need)}
    src = cohort.set_index("image_id").source
    rows, frames, heads = [], [], []

    def X(view, d):
        return V[view][[pos[i] for i in d.image_id]]

    for trap in cfg.traps:
        for k in cfg.folds:
            E = {e: envs[(trap, k, e)] for e in ["train_corr", "test_corr", "test_rev", "clean_test", "clean_val",
                                                  "val_groups", "train_all"]}
            tr, cv = E["train_corr"], E["clean_val"]
            ytr, atr, yv = tr.y.to_numpy(), tr.a.to_numpy(), cv.y.to_numpy()

            def fin(clf, method, view, C, vauc, **extra):
                if isinstance(clf, H.PrevalenceCalibrated):  # needs A at test: wrap per evaluation set
                    base = clf
                    class _G:  # noqa: N801
                        def __init__(self, a): self.a = a
                        def predict_proba(self, Xq): return base.predict_proba_groups(Xq, self.a)
                    thr, _ = select_threshold_clean_val(yv, _G(np.zeros(len(cv), int)).predict_proba(X(view, cv))[:, 1])
                    meta = {"cohort": name, "backbone": backend.name, "trap": trap, "seed": k, "method": method}
                    for env in ["clean_test", "test_corr", "test_rev"]:
                        d = E[env]
                        r, f = evaluate(_G(d.a.to_numpy()), thr, X(view, d), d.y.to_numpy(), d.image_id.to_numpy(),
                                        d.a.to_numpy(), {**meta, "env": "clean" if env == "clean_test" else env})
                        f["source"] = src.loc[f.image_id].to_numpy()
                        rows.append(r); frames.append(f)
                    heads.append({**meta, "C": C, "clean_val_auc": vauc, "thr": thr, **extra})
                    return
                thr, _ = select_threshold_clean_val(yv, clf.predict_proba(X(view, cv))[:, 1])
                meta = {"cohort": name, "backbone": backend.name, "trap": trap, "seed": k, "method": method}
                for env in ["clean_test", "test_corr", "test_rev"]:
                    d = E[env]
                    r, f = evaluate(clf, thr, X(view, d), d.y.to_numpy(), d.image_id.to_numpy(), d.a.to_numpy(),
                                    {**meta, "env": "clean" if env == "clean_test" else env})
                    f["source"] = src.loc[f.image_id].to_numpy()
                    rows.append(r); frames.append(f)
                heads.append({**meta, "C": C, "clean_val_auc": vauc, "thr": thr, **extra})

            for m in [a for a in cfg.arms if a in ("erm", "mask", "inpaint")]:
                clf, C, v = H.fit_erm(X(m, tr), ytr, X(m, cv), yv, k)
                fin(clf, m, m, C, v)
                if m == "erm" and "prevcal" in cfg.arms:
                    fin(H.PrevalenceCalibrated(clf, ytr, atr), "prevcal", "erm", C, v)
            Xtr, Xv = X("erm", tr), X("erm", cv)
            if "balanced" in cfg.arms:
                clf, C, v = H.fit_balanced(Xtr, ytr, atr, Xv, yv, k)
                fin(clf, "balanced", "erm", C, v)
            if "dfr" in cfg.arms:
                g = E["val_groups"]
                clf, C, v = H.fit_dfr(X("erm", g), g.y.to_numpy(), g.a.to_numpy(), Xv, yv, k)
                fin(clf, "dfr", "erm", C, v)
            if "groupdro" in cfg.arms:
                clf, C, v = H.fit_groupdro(Xtr, ytr, atr, Xv, yv, k, device)
                fin(clf, "groupdro", "erm", C, v)
            ta = E["train_all"]
            if "leace_paired" in cfg.arms:
                hp = ta[ta.a == 1]  # every hair-present training-fold image: (original, hair-inpainted)
                er = H.fit_leace(X("inpaint", hp), X("erm", hp))
                clf, C, v = H.fit_on_transformed(H.eraser_fn(er), Xtr, ytr, Xv, yv, k)
                fin(clf, "leace_paired", "erm", C, v, n_pairs=len(hp))
            if "leace_unpaired" in cfg.arms:
                er = H.fit_leace_labels(Xtr, atr)
                clf, C, v = H.fit_on_transformed(H.eraser_fn(er), Xtr, ytr, Xv, yv, k)
                fin(clf, "leace_unpaired", "erm", C, v)
            if any(a.startswith("i2e") for a in cfg.arms) or "insert_aug" in cfg.arms:
                X0, X1 = X("erm", ta), X("insert", ta)
                if "i2e" in cfg.arms or "i2e_balanced" in cfg.arms:
                    er = fit_difference_subspace(X0, X1, energy=cfg.energy, seed=k)
                    if "i2e" in cfg.arms:
                        clf, C, v = H.fit_on_transformed(er, Xtr, ytr, Xv, yv, k)
                        fin(clf, "i2e", "erm", C, v, k_rank=er.k)
                    if "i2e_balanced" in cfg.arms:
                        clf, C, v = H.fit_on_transformed(er, Xtr, ytr, Xv, yv, k, atr=atr, balanced=True)
                        fin(clf, "i2e_balanced", "erm", C, v, k_rank=er.k)
                if "i2e_rank1" in cfg.arms:
                    clf, C, v, ex = rank1_head(X0, X1, Xtr, ytr, Xv, yv, k)
                    fin(clf, "i2e_rank1", "erm", C, v)
                if "insert_aug" in cfg.arms:
                    clf, C, v, ex = insert_aug_head(Xtr, ytr, X("insert", tr), Xv, yv, k)
                    fin(clf, "insert_aug", "erm", C, v)
            print(f"[traps] {name} {backend.name} {trap} fold {k} done", flush=True)

    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(out_dir / "predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(heads).to_csv(out_dir / "heads.csv", index=False)
    del backend
    gc.collect(); torch.cuda.empty_cache()
    return out_dir


def _b(j):
    q, a, b, env, n, s = j
    return hierarchical_paired_bootstrap(q, a, b, env, n, s, fast=True)


def _x(j):
    p1, p2, a, b, env, n, s = j
    return difference_of_deltas(p1, p2, a, b, env, n, s)


def analyse_traps(out_dir: Path, n_boot: int = 10000, workers: int = 4, sources: Sequence[str] = ()) -> Dict:
    preds = pd.read_csv(out_dir / "predictions.csv.gz")
    arms = [a for a in preds.method.unique() if a != "erm"]
    jobs, keys = [], []
    for trap in sorted(preds.trap.unique()):
        q = preds[preds.trap == trap]
        for env in ["test_rev", "clean", "test_corr"]:
            for a in arms:
                jobs.append((slim(q, env, (a, "erm")), a, "erm", env, n_boot, 20260918 + sum(map(ord, a + trap + env))))
                keys.append((trap, env, a, "all"))
        for s in sources:
            qs = q[q.source == s]
            for a in ["mask"]:
                jobs.append((qs, a, "erm", "test_rev", n_boot, 20260918 + sum(map(ord, a + trap + s))))
                keys.append((trap, "test_rev", a, s))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_b, jobs))
    boot = pd.DataFrame([{"trap": t, "env": e, "arm": a, "source": s, **r} for (t, e, a, s), r in zip(keys, res)])
    boot.to_csv(out_dir / "bootstrap_vs_erm.csv", index=False)
    cross = []
    if {"trapA", "trapB"} <= set(preds.trap):
        pA, pB = preds[preds.trap == "trapA"], preds[preds.trap == "trapB"]
        with ProcessPoolExecutor(workers) as ex:
            cr = list(ex.map(_x, [(pB, pA, a, "erm", "test_rev", n_boot, 20260927) for a in arms]))
        cross = pd.DataFrame([{"arm": a, **r} for a, r in zip(arms, cr)])
        cross.to_csv(out_dir / "crossover_B_minus_A.csv", index=False)
    m = pd.read_csv(out_dir / "metrics.csv")
    summ = m.groupby(["trap", "method", "env"]).agg(auc=("auc", "mean"), auc_sd=("auc", "std"),
                                                  wga=("worst_group_accuracy", "mean"), n=("n", "mean"),
                                                  n_pos=("n_pos", "sum")).reset_index()
    summ.to_csv(out_dir / "summary_auc.csv", index=False)
    return {"boot": boot, "cross": cross, "summary": summ}
