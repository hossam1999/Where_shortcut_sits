"""Synthetic artifact at controlled artifact-ROI overlap (thesis Results 1-2; WP1 on CXR).

One driver for all cohorts and artifacts. Protocol (unchanged from the pilot):
  * linear heads on frozen features; C / λ chosen on clean-val AUROC;
  * decision threshold chosen on clean val by balanced accuracy, then frozen;
  * train 90/10 (P(A|Y=1)/P(A|Y=0)), correlated test 90/10, reversed test 10/90, clean test;
  * seeds 42/123/456 (seed controls artifact assignment); hierarchical paired bootstrap.
"""
from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from .. import heads as H
from ..backbones import BACKEND_SIZE, load_backend
from ..evaluation import evaluate, same_head_counterfactual, select_threshold_clean_val
from ..features import ImageCache, assemble_env, extract_view
from ..ops import apply_method
from ..stats import hierarchical_interaction, hierarchical_paired_bootstrap, hierarchical_paired_mean_bootstrap
from ..synthetic import (ARTIFACT_GEOMETRY, DEFAULT_SEEDS, draw_caliper, draw_debris, draw_ruler, draw_ruler_variable, draw_tube,
                         presence_vector, sample_ruler_style)

PIXEL_METHODS = {"erm", "mask", "inpaint", "dilate0", "dilate10", "dilate25", "dilate50"}
FEATURE_ARMS = {"balanced", "groupdro", "dfr", "leace", "inpaint_consistency", "inpaint_consistency_lam0"}


@dataclass
class SynthConfig:
    phase: str = "corr"  # "corr" (90/10 trap) or "occlusion" (50/50)
    overlaps: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)
    seeds: Sequence[int] = tuple(DEFAULT_SEEDS)
    arms: Sequence[str] = ("erm", "mask")
    artifact: str = "ruler_fixed"  # ruler_fixed | ruler_variable | tube
    geometry: Dict[int, tuple] = field(default_factory=lambda: dict(ARTIFACT_GEOMETRY))
    batch_size: int = 64
    workers: int = 6
    n_boot: int = 10000
    extra_arms: Dict[str, Callable] = field(default_factory=dict)  # name -> fn(ctx) for proposed methods


def _artifact_drawer(cfg: SynthConfig, size: int):
    w, h = cfg.geometry[size]

    def draw(img, image_id, pp, ov):
        x, y = int(pp["x"]), int(pp["y"])
        if cfg.artifact == "ruler_fixed":
            return draw_ruler(img, x, y, w, h)
        if cfg.artifact == "ruler_variable":
            return draw_ruler_variable(img, x, y, w, h, sample_ruler_style(image_id, ov))
        if cfg.artifact == "tube":
            return draw_tube(img, x, y, w, h, image_id=f"{image_id}|{ov:.2f}")
        if cfg.artifact == "debris":
            return draw_debris(img, x, y, w, h, image_id=f"{image_id}|{ov:.2f}")
        if cfg.artifact == "caliper":
            return draw_caliper(img, x, y, w, h, image_id=f"{image_id}|{ov:.2f}")
        raise ValueError(cfg.artifact)

    return draw


class SynthViews:
    """Lazily extracted, cached feature views for one cohort x backbone x artifact."""

    def __init__(self, cohort, cache: ImageCache, backend, placements, cfg: SynthConfig, feat_dir: Path, device):
        self.cohort, self.cache, self.backend, self.pl, self.cfg = cohort, cache, backend, placements, cfg
        self.dir, self.device = feat_dir, device
        self.draw = _artifact_drawer(cfg, getattr(cache, "size", backend.size))  # render size (e.g. 518 for medsiglip448)
        self.ids = cohort.ids()
        self._mem: Dict[str, np.ndarray] = {}

    def _render(self, method: str, ov):
        def r(i):
            img, roi = self.cache.image(i), self.cache.roi_mask(i)
            amask = None
            if ov is not None:
                img, amask = self.draw(img, i, self.pl[i][f"{ov:.2f}"], ov)
            return apply_method(img, method, roi, amask)

        return r

    def get(self, method: str, ov=None) -> np.ndarray:
        """Features for all cohort ids; ov=None => clean (no artifact)."""
        if method == "inpaint" and ov is None:
            method = "erm"  # inpainting an empty mask is the identity
        key = f"{method}_clean" if ov is None else f"{method}_ov{int(round(ov * 100)):03d}"
        if key not in self._mem:
            self._mem[key] = extract_view(self.backend, self.ids, self._render(method, ov), self.dir / f"{key}.npz",
                                          self.device, self.cfg.batch_size, self.cfg.workers, desc=key)
        return self._mem[key]


def run_synthetic(cohort, backend_name: str, placements: Dict, out_dir: Path, cfg: SynthConfig,
                  image_cache_root: Path, loaders, feat_root: Path, device=None) -> Path:
    device = device or torch.device("cuda")
    size = BACKEND_SIZE[backend_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = ImageCache(image_cache_root / f"{cohort.name}_{size}", cohort.ids(), size, *loaders)
    backend = load_backend(backend_name, device)
    views = SynthViews(cohort, cache, backend, placements, cfg,
                       feat_root / cohort.name / backend.name / cfg.artifact, device)

    df = cohort.df
    sp = {s: np.flatnonzero(df.split.to_numpy() == s) for s in ("train", "val", "test")}
    y_all = df.y.to_numpy().astype(int)
    ids_all = df.image_id.to_numpy().astype(str)
    train_env, test_envs = (("train_corr", ["test_corr", "test_rev"]) if cfg.phase == "corr"
                            else ("train_uncorr", ["test_uncorr"]))

    rows, frames, heads = [], [], []

    def record(clf, thr, method, ov, seed, clean_X, env_feats, extra=None):
        meta = {"phase": cfg.phase, "artifact": cfg.artifact, "backbone": backend.name, "method": method,
                "overlap": ov, "seed": seed}
        te = sp["test"]
        r, f = evaluate(clf, thr, clean_X[te], y_all[te], ids_all[te], np.zeros(len(te), int), {**meta, "env": "clean"})
        rows.append(r); frames.append(f)
        for env, (X, a) in env_feats.items():
            r, f = evaluate(clf, thr, X[te], y_all[te], ids_all[te], a[te].astype(int), {**meta, "env": env})
            rows.append(r); frames.append(f)
        heads.append({**meta, **(extra or {})})

    arms = list(cfg.arms)
    for ov in cfg.overlaps:
        for seed in cfg.seeds:
            pres = {env: presence_vector(ids_all, y_all, seed, env) for env in [train_env, *test_envs, "test_uncorr"]}
            tr, va = sp["train"], sp["val"]
            a_tr = pres[train_env][tr].astype(int)

            def env_X(method):
                Xc, Xa = views.get(method, None), views.get(method, ov)
                return Xc, {env: (assemble_env(Xc, Xa, pres[env]), pres[env]) for env in [train_env, *test_envs]}

            # --- pixel-level arms (the head sees the processed image) --------------------
            for m in [a for a in arms if a in PIXEL_METHODS]:
                Xc, E = env_X(m)
                Xtr = E[train_env][0][tr]
                clf, C, vauc = H.fit_erm(Xtr, y_all[tr], Xc[va], y_all[va], seed)
                thr, _ = select_threshold_clean_val(y_all[va], clf.predict_proba(Xc[va])[:, 1])
                record(clf, thr, m, ov, seed, Xc, {e: E[e] for e in test_envs}, {"C": C, "clean_val_auc": vauc, "thr": thr})

            need_erm = [a for a in arms if a in FEATURE_ARMS] or cfg.extra_arms
            if not need_erm:
                continue
            Xc, E = env_X("erm")
            Xtr, ytr = E[train_env][0][tr], y_all[tr]
            test_feats = {e: E[e] for e in test_envs}

            def fin(clf, name, C, vauc, **kw):
                view = kw.pop("_view", "erm")  # arms may be trained/evaluated on another pixel view (e.g. masked)
                Xcv, tf = (Xc, test_feats) if view == "erm" else (lambda c_, E_: (c_, {e: E_[e] for e in test_envs}))(*env_X(view))
                thr, _ = select_threshold_clean_val(y_all[va], clf.predict_proba(Xcv[va])[:, 1])
                record(clf, thr, name, ov, seed, Xcv, tf, {"C": C, "clean_val_auc": vauc, "thr": thr, **kw})

            if "balanced" in arms:
                clf, C, vauc = H.fit_balanced(Xtr, ytr, a_tr, Xc[va], y_all[va], seed)
                fin(clf, "balanced", C, vauc)
            if "groupdro" in arms:
                clf, C, vauc = H.fit_groupdro(Xtr, ytr, a_tr, Xc[va], y_all[va], seed, device)
                fin(clf, "groupdro", C, vauc)
            if "dfr" in arms:
                # held-out group-balanced set: validation split with a 50/50 artifact (as in the pilot)
                a_vu = pres["test_uncorr"][va]
                Xvu = assemble_env(Xc, views.get("erm", ov), pres["test_uncorr"])[va]
                clf, C, vauc = H.fit_dfr(Xvu, y_all[va], a_vu.astype(int), Xc[va], y_all[va], seed)
                fin(clf, "dfr", C, vauc)
            if "leace" in arms:
                er = H.fit_leace(Xc[tr], views.get("erm", ov)[tr])  # paired clean / always-on train renders
                clf, C, vauc = H.fit_on_transformed(H.eraser_fn(er), Xtr, ytr, Xc[va], y_all[va], seed)
                fin(clf, "leace", C, vauc)
            if "inpaint_consistency" in arms or "inpaint_consistency_lam0" in arms:
                Xi = assemble_env(views.get("erm", None), views.get("inpaint", ov), pres[train_env])[tr]
                if "inpaint_consistency" in arms:
                    clf, lam, vauc = H.fit_consistency_select(Xtr, Xi, ytr, Xc[va], y_all[va], seed, device)
                    fin(clf, "inpaint_consistency", None, vauc, lam=lam)
                if "inpaint_consistency_lam0" in arms:
                    clf, lam, vauc = H.fit_consistency(Xtr, Xi, ytr, Xc[va], y_all[va], seed, 0.0, device)
                    fin(clf, "inpaint_consistency_lam0", None, vauc, lam=0.0)
            for name, fn in cfg.extra_arms.items():
                ctx = dict(views=views, ov=ov, seed=seed, sp=sp, y=y_all, ids=ids_all, Xc=Xc, Xtr=Xtr, ytr=ytr,
                           a_tr=a_tr, pres=pres, train_env=train_env, device=device, cohort=cohort, env_X=env_X)
                clf, C, vauc, extra = fn(ctx)
                fin(clf, name, C, vauc, **extra)
        print(f"[synthetic] {backend.name} {cfg.artifact} ov={ov:.2f} done", flush=True)

    metrics = pd.DataFrame(rows)
    preds = pd.concat(frames, ignore_index=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    preds.to_csv(out_dir / "predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(heads).to_csv(out_dir / "heads.csv", index=False)
    del backend, views
    gc.collect()
    torch.cuda.empty_cache()
    return out_dir


def analyse_synthetic(out_dir: Path, baseline: str = "erm", n_boot: int = 10000, lo=0.0, hi=1.0,
                      workers: int = 4) -> pd.DataFrame:
    """Bootstrap tables: every arm vs ERM at every overlap; location interaction; same-head |Δp|."""
    from concurrent.futures import ProcessPoolExecutor

    preds = pd.read_csv(out_dir / "predictions.csv.gz")
    env = "test_rev" if "test_rev" in set(preds.env) else "test_uncorr"
    arms = [a for a in preds.method.unique() if a != baseline]
    ovs = sorted(preds.overlap.unique())
    from ..stats import slim
    jobs = [(slim(preds[np.isclose(preds.overlap, ov)], env, (a, baseline)), a, baseline, env, n_boot,
             20260918 + int(ov * 1000) + sum(map(ord, a))) for a in arms for ov in ovs]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_boot_job, jobs))
    job_ov = [ov for a in arms for ov in ovs]  # same order as `jobs`
    boot = pd.DataFrame([{"overlap": o, **r} for o, r in zip(job_ov, res)])
    boot.to_csv(out_dir / "bootstrap_vs_erm.csv", index=False)
    inter = []
    if lo in ovs and hi in ovs and env == "test_rev":
        with ProcessPoolExecutor(workers) as ex:
            cols = ["seed", "method", "image_id", "y", "prob", "overlap", "env"]
            inter = list(ex.map(_inter_job, [(preds.loc[(preds.env == env) & preds.method.isin([a, baseline]) &
                                                         preds.overlap.isin([lo, hi]), cols], a, baseline, env, lo, hi, n_boot)
                                             for a in arms]))
        pd.DataFrame(inter).to_csv(out_dir / "location_interaction.csv", index=False)
    cf = same_head_counterfactual(preds, env)
    if len(cf):
        cf.to_csv(out_dir / "counterfactual_per_image.csv.gz", index=False, compression="gzip")
        cf.groupby(["method", "overlap"]).abs_delta_p.mean().reset_index().to_csv(out_dir / "counterfactual_summary.csv", index=False)
    summ = (pd.read_csv(out_dir / "metrics.csv").groupby(["method", "overlap", "env"])
            .agg(auc=("auc", "mean"), wga=("worst_group_accuracy", "mean")).reset_index())
    summ.to_csv(out_dir / "summary_auc.csv", index=False)
    return boot


def _boot_job(j):
    q, a, b, env, n, s = j
    return hierarchical_paired_bootstrap(q, a, b, env, n, s)


def _inter_job(j):
    preds, a, b, env, lo, hi, n = j
    return hierarchical_interaction(preds, a, b, env, "overlap", lo, hi, n, 20260918)
