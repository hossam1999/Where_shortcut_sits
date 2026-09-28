"""E13 / E15 runner in the author's protocol (docs/REPLICATION_SPEC.md): seeds are the clusters, fold test
predictions are pooled per seed, every paired comparison is on identical image IDs."""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import gc
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch

from .. import heads as H
from ..backbones import BACKEND_SIZE, load_backend
from ..evaluation import evaluate, select_threshold_clean_val
from ..features import extract_view
from ..methods.insertion import fit_difference_subspace, insert_aug_head, rank1_head
from .real_traps import make_renderers

ARMS = ("erm", "mask", "inpaint", "balanced", "dfr", "leace_paired", "leace_unpaired",
        "i2e", "i2e_balanced", "i2e_rank1", "insert_aug", "prevcal", "mte", "mte_balanced")


_CTX: dict = {}


def _fold_job(job):
    """Fit every arm for one (trap, seed, fold); runs in a forked worker (reads _CTX)."""
    trap, seed, k = job
    # BLAS threads are fixed by OPENBLAS_NUM_THREADS (set before numpy loads). Calling
    # threadpoolctl here makes OpenBLAS spawn one thread per core and trips the pid limit.
    V, pos, envs, arms = _CTX["V"], _CTX["pos"], _CTX["envs"], _CTX["arms"]
    X = lambda v, d: V[v][[pos[i] for i in d.image_id]]
    frames = []
    E = {e: envs[(trap, seed, k, e)] for e in ("train_corr", "val_clean", "val_groups", "train_all",
                                             "test_corr", "test_rev", "clean")}
    tr, cv = E["train_corr"], E["val_clean"]
    ytr, atr, yv = tr.y.to_numpy(), tr.a.to_numpy(), cv.y.to_numpy()

    def fin(clf, method, view, test_a=False):
        if test_a:  # needs A at test time (prevalence calibration)
            pv = clf.predict_proba_groups(X(view, cv), cv.a.to_numpy())[:, 1]
        else:
            pv = clf.predict_proba(X(view, cv))[:, 1]
        thr, _ = select_threshold_clean_val(yv, pv)
        for env in ("clean", "test_corr", "test_rev") + (("val_groups",) if _CTX.get("save_val") else ()):
            d = E[env]
            c = clf
            if test_a:
                class _G:
                    def predict_proba(self, Xq, _a=d.a.to_numpy()):
                        return clf.predict_proba_groups(Xq, _a)
                c = _G()
            _, f = evaluate(c, thr, X(view, d), d.y.to_numpy(), d.image_id.to_numpy(), d.a.to_numpy(),
                            {"backbone": _CTX["backend_name"], "trap": trap, "seed": seed, "fold": k,
                             "method": method, "env": env, **_CTX["extra_meta"]})
            f["source"] = d.source.to_numpy()
            frames.append(f)

    for m in [a for a in arms if a in ("erm", "mask", "inpaint")]:
        clf = H.fit_erm(X(m, tr), ytr, X(m, cv), yv, seed)[0]
        fin(clf, m, m)
        if m == "erm" and "prevcal" in arms:
            fin(H.PrevalenceCalibrated(clf, ytr, atr), "prevcal", "erm", test_a=True)
    for m, view, bal in _CTX.get("view_arms", ()):  # extra (method, feature view, balanced?) arms, e.g. SLAS
        if bal:
            fin(H.fit_balanced(X(view, tr), ytr, atr, X(view, cv), yv, seed)[0], m, view)
        else:
            fin(H.fit_erm(X(view, tr), ytr, X(view, cv), yv, seed)[0], m, view)
    Xtr, Xv = X("erm", tr), X("erm", cv)
    if "jtt" in arms:
        fin(H.fit_jtt(Xtr, ytr, Xv, yv, seed)[0], "jtt", "erm")
    if _CTX.get("text_U") is not None and any(a.startswith(("text_erase", "mask_text")) for a in arms):
        # text-prompted artifact subspace (docs/PREREGISTRATION_TEXT_PROMPT.md)
        from ..methods.insertion import SubspaceEraser, disease_directions, protect
        U = np.asarray(_CTX["text_U"], np.float32)
        if "text_erase" in arms:
            er_t = SubspaceEraser(U, Xtr.mean(0).astype(np.float32), U.shape[1], np.zeros(1))
            fin(H.fit_on_transformed(er_t, Xtr, ytr, Xv, yv, seed)[0], "text_erase", "erm")
        mtr_t, mv_t = X("mask", tr), X("mask", cv)
        er_mt = SubspaceEraser(U, mtr_t.mean(0).astype(np.float32), U.shape[1], np.zeros(1))
        if "mask_text_erase" in arms:
            fin(H.fit_on_transformed(er_mt, mtr_t, ytr, mv_t, yv, seed)[0], "mask_text_erase", "mask")
        if "mask_text_erase_balanced" in arms:
            fin(H.fit_on_transformed(er_mt, mtr_t, ytr, mv_t, yv, seed, atr=atr, balanced=True)[0], "mask_text_erase_balanced", "mask")
        if "mask_text_erase_protect" in arms:
            er_p = protect(er_mt, disease_directions(mtr_t[atr == 0], ytr[atr == 0], seed=seed))
            fin(H.fit_on_transformed(er_p, mtr_t, ytr, mv_t, yv, seed)[0], "mask_text_erase_protect", "mask")
    if "mask_balanced" in arms:
        fin(H.fit_balanced(X("mask", tr), ytr, atr, X("mask", cv), yv, seed)[0], "mask_balanced", "mask")
    if "splice" in arms:  # task-preserving concept removal with artifact labels (baseline)
        sp = H.SpliceProjection(Xtr, atr, ytr)
        fin(H.fit_on_transformed(sp, Xtr, ytr, Xv, yv, seed)[0], "splice", "erm")
    if "mask_splice" in arms:
        mtr_, mv_ = X("mask", tr), X("mask", cv)
        sp = H.SpliceProjection(mtr_, atr, ytr)
        fin(H.fit_on_transformed(sp, mtr_, ytr, mv_, yv, seed)[0], "mask_splice", "mask")
    if "mask_jtt" in arms:
        fin(H.fit_jtt(X("mask", tr), ytr, X("mask", cv), yv, seed)[0], "mask_jtt", "mask")
    if "balanced" in arms:
        fin(H.fit_balanced(Xtr, ytr, atr, Xv, yv, seed)[0], "balanced", "erm")
    if "dfr_half" in arms or "mask_dfr_half" in arms:  # DFR on the selection-free half of val_groups (U14)
        from ..utils import stable_int as _si
        g = E["val_groups"]
        g = g[np.array([_si("val_half", i) % 2 == 0 for i in g.image_id])]
        if "dfr_half" in arms:
            fin(H.fit_dfr(X("erm", g), g.y.to_numpy(), g.a.to_numpy(), Xv, yv, seed)[0], "dfr_half", "erm")
        if "mask_dfr_half" in arms:
            fin(H.fit_dfr(X("mask", g), g.y.to_numpy(), g.a.to_numpy(), X("mask", cv), yv, seed)[0], "mask_dfr_half", "mask")
    if "dfr" in arms:
        g = E["val_groups"]
        fin(H.fit_dfr(X("erm", g), g.y.to_numpy(), g.a.to_numpy(), Xv, yv, seed)[0], "dfr", "erm")
    ta = E["train_all"]
    if "leace_paired" in arms:
        hp = ta[ta.a == 1]
        er = H.fit_leace(X("inpaint", hp), X("erm", hp))
        fin(H.fit_on_transformed(H.eraser_fn(er), Xtr, ytr, Xv, yv, seed)[0], "leace_paired", "erm")
    if "leace_unpaired" in arms:
        er = H.fit_leace_labels(Xtr, atr)
        fin(H.fit_on_transformed(H.eraser_fn(er), Xtr, ytr, Xv, yv, seed)[0], "leace_unpaired", "erm")
    if "mask_insert" in V:  # Mask-then-Erase: erase the insertion subspace *in the masked view*
        mtr, mv = X("mask", tr), X("mask", cv)
        # rank rule: 90 % held-out energy (main); _CTX overrides only in the U-MtE ablation (docs/PREREGISTRATION_UMTE_ABLATION.md)
        fk = _CTX.get("mte_k")
        er_m = fit_difference_subspace(X("mask", ta), X("mask_insert", ta), energy=1.0 if fk else _CTX.get("mte_energy", 0.9),
                                       max_k=fk or 64, seed=seed)
        if "mte" in arms:
            fin(H.fit_on_transformed(er_m, mtr, ytr, mv, yv, seed)[0], "mte", "mask")
        if "mte_balanced" in arms:
            fin(H.fit_on_transformed(er_m, mtr, ytr, mv, yv, seed, atr=atr, balanced=True)[0], "mte_balanced", "mask")
        if ("umte_pbal" in arms or "pbal" in arms) and "insert" in V:
            # annotation-free group balancing: pseudo artifact labels from an overlay detector trained on
            # (original, inserted) feature pairs, thresholded by Otsu on the training images' logits
            pa, pauc = H.pseudo_artifact_labels(X("erm", ta), X("insert", ta), X("erm", tr), atr, seed)
            n0 = len(frames)
            if "umte_pbal" in arms:
                fin(H.fit_on_transformed(er_m, mtr, ytr, mv, yv, seed, atr=pa, balanced=True)[0], "umte_pbal", "mask")
            if "pbal" in arms:
                fin(H.fit_balanced(Xtr, ytr, pa, Xv, yv, seed)[0], "pbal", "erm")
            for f in frames[n0:]:
                f["pseudo_a_auc"], f["pseudo_a_rate"] = pauc, float(pa.mean())
        if any(a in arms for a in ("mte_protect", "mte_protect_balanced")):
            # disease-protected erasure: erased subspace made orthogonal to label directions of artifact-free images
            from ..methods.insertion import disease_directions, protect
            W = disease_directions(mtr[atr == 0], ytr[atr == 0], seed=seed)
            er_p = protect(er_m, W)
            if "mte_protect" in arms:
                fin(H.fit_on_transformed(er_p, mtr, ytr, mv, yv, seed)[0], "mte_protect", "mask")
            if "mte_protect_balanced" in arms:
                fin(H.fit_on_transformed(er_p, mtr, ytr, mv, yv, seed, atr=atr, balanced=True)[0], "mte_protect_balanced", "mask")
        if "mte_dfr" in arms or "mask_dfr" in arms:  # erasure / masking combined with DFR (group-balanced val)
            g = E["val_groups"]
            gm, ya, aa = X("mask", g), g.y.to_numpy(), g.a.to_numpy()
            if "mte_dfr" in arms:
                fin(H.TransformHead(er_m, H.fit_dfr(er_m(gm), ya, aa, er_m(mv), yv, seed)[0]), "mte_dfr", "mask")
            if "mask_dfr" in arms:
                fin(H.fit_dfr(gm, ya, aa, mv, yv, seed)[0], "mask_dfr", "mask")
        if "umte_jtt" in arms:  # label-free group robustness on the masked + erased view
            fin(H.TransformHead(er_m, H.fit_jtt(er_m(mtr), ytr, er_m(mv), yv, seed)[0]), "umte_jtt", "mask")
        if "mte_aug" in arms:  # ablation: same overlays used as training augmentation instead of erasure
            fin(insert_aug_head(mtr, ytr, X("mask_insert", tr), mv, yv, seed)[0], "mte_aug", "mask")
    if "insert" in V:
        X0, X1 = X("erm", ta), X("insert", ta)
        er = fit_difference_subspace(X0, X1, energy=0.9, seed=seed)
        if "i2e" in arms:
            fin(H.fit_on_transformed(er, Xtr, ytr, Xv, yv, seed)[0], "i2e", "erm")
        if "i2e_balanced" in arms:
            fin(H.fit_on_transformed(er, Xtr, ytr, Xv, yv, seed, atr=atr, balanced=True)[0], "i2e_balanced", "erm")
        if "i2e_rank1" in arms:
            fin(rank1_head(X0, X1, Xtr, ytr, Xv, yv, seed)[0], "i2e_rank1", "erm")
        if "insert_aug" in arms:
            fin(insert_aug_head(Xtr, ytr, X("insert", tr), Xv, yv, seed)[0], "insert_aug", "erm")
    return frames


def run_spec(envs: Dict, cache, backend_name: str, out_dir: Path, feat_dir: Path, donors: Sequence[str],
             arms=ARMS, traps=("trapA", "trapB"), device=None, batch_size=64, workers=5, extra_meta=None,
             n_jobs: int = 4, insert_fn=None, insert_tag: str = "", folds=range(5), save_val: bool = False,
             extra_ctx: dict | None = None):
    device = device or torch.device("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = sorted(set().union(*[set(d.image_id) for k, d in envs.items() if k[0] in traps]))
    pos = {k: j for j, k in enumerate(pool)}
    need = {"erm", "mask"} | ({"inpaint"} if {"inpaint", "leace_paired"} & set(arms) else set()) | ({"insert"} if any(a.startswith(("i2e", "insert", "umte", "pbal")) for a in arms) else set()) | \
        ({"mask_insert"} if any(a.startswith(("mte", "umte", "pbal")) for a in arms) else set())
    xc = extra_ctx or {}
    need = need | set(xc.get("extra_views", ()))  # extra rendered views (docs/PREREGISTRATION_ROUND4.md, R8)
    from ..backbones import Backend  # noqa: F401
    def _covered(f):  # a cached view is usable only if it contains every image of this pool
        if not f.exists():
            return False
        ids_c = set(np.load(f, allow_pickle=False)["ids"].astype(str))
        return all(str(i) in ids_c for i in pool)
    cached = all(_covered(feat_dir / (f"{v}{insert_tag}.npz" if v in ("insert", "mask_insert") else f"{v}.npz")) for v in need)
    backend = None if cached else load_backend(backend_name, device)
    rend = make_renderers(cache, BACKEND_SIZE[backend_name], donors, insert_fn)
    rend.update(xc.get("extra_renderers", {}))
    fname = lambda v: f"{v}{insert_tag}.npz" if v in ("insert", "mask_insert") else f"{v}.npz"
    V = {v: extract_view(backend, pool, rend[v], feat_dir / fname(v), device, batch_size, workers, desc=v)
         for v in sorted(need)}
    del backend; gc.collect(); torch.cuda.empty_cache()
    cpus = os.cpu_count() or 8
    blas = max(1, min(2, cpus // max(n_jobs, 1)))  # keep total BLAS threads near the core count
    global _CTX
    _CTX = dict(V=V, pos=pos, envs=envs, arms=arms, backend_name=backend_name, extra_meta=extra_meta or {},
                save_val=save_val, blas_threads=blas, **(extra_ctx or {}))
    seeds = sorted({k[1] for k in envs})
    jobs = [(trap, seed, k) for trap in traps for seed in seeds for k in folds]
    import multiprocessing as mp
    frames = []
    # (seed, fold) fits are independent: fork-based pool shares the cached features copy-on-write.
    # Forking after torch/CUDA ran in this process can deadlock workers (OpenMP / CUDA state). Parallelise only
    # when no backbone was loaded here and this process has not initialized CUDA.
    if cached and n_jobs > 1 and not torch.cuda.is_initialized():
        with mp.get_context("fork").Pool(min(n_jobs, 16, len(jobs))) as pool:
            results = pool.imap(_fold_job, jobs)
            for (trap, seed, k), fr in zip(jobs, results):
                frames.extend(fr)
                if k == max(folds):
                    print(f"[spec] {backend_name} {trap} seed {seed} done", flush=True)
    else:
        if cached and n_jobs > 1 and torch.cuda.is_initialized():
            print("[spec] CUDA is already active in this process; fitting one job at a time. "
                  "Round 4 extracts features in a child process so the fitter can use every core.", flush=True)
        for trap, seed, k in jobs:
            frames.extend(_fold_job((trap, seed, k)))
            if k == max(folds):
                print(f"[spec] {backend_name} {trap} seed {seed} done", flush=True)
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(out_dir / "predictions.csv.gz", index=False, compression="gzip")
    from ..stats import safe_auc
    m = (preds.groupby(["trap", "method", "env", "seed"]).apply(lambda q: safe_auc(q.y, q.prob), include_groups=False)
         .rename("auc").reset_index())
    m.to_csv(out_dir / "metrics_per_seed.csv", index=False)
    return out_dir
