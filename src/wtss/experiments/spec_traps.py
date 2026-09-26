"""E13 / E15 runner in the author's protocol (docs/REPLICATION_SPEC.md): seeds are the clusters, fold test
predictions are pooled per seed, every paired comparison is on identical image IDs."""
from __future__ import annotations

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
        "i2e", "i2e_balanced", "i2e_rank1", "insert_aug", "prevcal")


def run_spec(envs: Dict, cache, backend_name: str, out_dir: Path, feat_dir: Path, donors: Sequence[str],
             arms=ARMS, traps=("trapA", "trapB"), device=None, batch_size=64, workers=5):
    device = device or torch.device("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = sorted(set().union(*[set(d.image_id) for d in envs.values()]))
    pos = {k: j for j, k in enumerate(pool)}
    backend = load_backend(backend_name, device)
    rend = make_renderers(cache, BACKEND_SIZE[backend_name], donors)
    need = {"erm", "mask", "inpaint"} | ({"insert"} if any(a.startswith(("i2e", "insert")) for a in arms) else set())
    V = {v: extract_view(backend, pool, rend[v], feat_dir / backend.name / f"{v}.npz", device, batch_size, workers,
                         desc=v) for v in sorted(need)}
    del backend; gc.collect(); torch.cuda.empty_cache()
    X = lambda v, d: V[v][[pos[i] for i in d.image_id]]
    rows, frames = [], []
    seeds = sorted({k[1] for k in envs})
    for trap in traps:
        for seed in seeds:
            for k in range(5):
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
                    for env in ("clean", "test_corr", "test_rev"):
                        d = E[env]
                        c = clf
                        if test_a:
                            class _G:
                                def predict_proba(self, Xq, _a=d.a.to_numpy()):
                                    return clf.predict_proba_groups(Xq, _a)
                            c = _G()
                        _, f = evaluate(c, thr, X(view, d), d.y.to_numpy(), d.image_id.to_numpy(), d.a.to_numpy(),
                                        {"backbone": backend_name, "trap": trap, "seed": seed, "fold": k,
                                         "method": method, "env": env})
                        f["source"] = d.source.to_numpy()
                        frames.append(f)

                for m in [a for a in arms if a in ("erm", "mask", "inpaint")]:
                    clf = H.fit_erm(X(m, tr), ytr, X(m, cv), yv, seed)[0]
                    fin(clf, m, m)
                    if m == "erm" and "prevcal" in arms:
                        fin(H.PrevalenceCalibrated(clf, ytr, atr), "prevcal", "erm", test_a=True)
                Xtr, Xv = X("erm", tr), X("erm", cv)
                if "balanced" in arms:
                    fin(H.fit_balanced(Xtr, ytr, atr, Xv, yv, seed)[0], "balanced", "erm")
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
            print(f"[spec] {backend_name} {trap} seed {seed} done", flush=True)
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(out_dir / "predictions.csv.gz", index=False, compression="gzip")
    from ..stats import safe_auc
    m = (preds.groupby(["trap", "method", "env", "seed"]).apply(lambda q: safe_auc(q.y, q.prob), include_groups=False)
         .rename("auc").reset_index())
    m.to_csv(out_dir / "metrics_per_seed.csv", index=False)
    return out_dir
