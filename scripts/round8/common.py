"""Shared pieces of round 8 (docs/ROUND8_DESIGN.md; later docs/PREREGISTRATION_ROUND8.md).

`joint_replicates` is the crossed seed x image bootstrap of wtss.stats_crossed, generalised so that one set of
replicate weights (one multinomial draw over clusters and one Poisson(1) weight per test image, shared by every arm,
environment and subset) yields the cluster-weighted mean AUROC of many keys at once. Any contrast between keys
(arm - mask, min(rev, corr) of an arm minus that of masking) is then a function of the same replicates, which is what
wtss.stats_crossed does for a single contrast.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Dict, Hashable, List, Tuple

import numpy as np
import pandas as pd

from wtss import paths
from wtss.stats import _WeightedAUC, binary_auc

ROOT = paths.REPO_ROOT
OUT = ROOT / "results" / "round8"
SMOKE_OUT = OUT / "_smoke"
FEAT = paths.CACHE / "features" / "round8"
RERUN = ROOT / "results" / "rerun_2026-09-28"
N_BOOT = 10000


def out_dir(smoke: bool) -> Path:
    d = SMOKE_OUT if smoke else OUT
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(**kw):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **kw}, default=str), flush=True)


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def load_round4_common():
    spec = importlib.util.spec_from_file_location("round4_common", ROOT / "scripts" / "round4" / "common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, k in enumerate(o):
        run = max(run, (len(p) - r) * p[k])
        adj[k] = min(run, 1.0)
    return adj


# terms: key -> list of (cluster, image_ids, y, scores)
Terms = Dict[Hashable, List[Tuple[Hashable, list, np.ndarray, np.ndarray]]]


def joint_replicates(terms: Terms, n_boot: int, seed: int, chunk: int = 250):
    """Cluster-weighted mean AUROC of every key under shared crossed replicates.

    Returns (point, reps): point[key] = mean over clusters of the AUROC; reps[key] = array (n_boot,) with, per
    replicate, sum_c w_c AUROC_c / sum_c w_c over the clusters c that the key has (w_c multinomial cluster counts;
    image weights Poisson(1), one per image identifier, shared across keys and clusters). NaN where a replicate
    leaves a key without a valid cluster.
    """
    ids = sorted({str(i) for lst in terms.values() for t in lst for i in t[1]})
    pos = {k: j for j, k in enumerate(ids)}
    clusters = sorted({t[0] for lst in terms.values() for t in lst}, key=str)
    cpos = {c: j for j, c in enumerate(clusters)}
    prep, point = {}, {}
    for key, lst in terms.items():
        prep[key] = []
        pts = []
        for c, im, y, s in lst:
            y, s = np.asarray(y), np.asarray(s, float)
            if len(np.unique(y)) < 2:
                continue
            prep[key].append((cpos[c], np.array([pos[str(i)] for i in im]), _WeightedAUC(y, s)))
            pts.append(binary_auc(y, s))
        point[key] = float(np.mean(pts)) if pts else float("nan")
    rng = np.random.default_rng(seed)
    reps = {k: [] for k in terms}
    for b0 in range(0, n_boot, chunk):
        B = min(chunk, n_boot - b0)
        W = rng.poisson(1.0, size=(B, len(ids))).astype(np.float64)
        cw = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=B).astype(float)
        for key, lst in prep.items():
            num, den = np.zeros(B), np.zeros(B)
            for ci, idx, f in lst:
                v = f(W[:, idx])
                ok = np.isfinite(v)
                num += np.where(ok, v, 0) * cw[:, ci]
                den += np.where(ok, cw[:, ci], 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                reps[key].append(num / den)
    return point, {k: np.concatenate(v) if v else np.array([]) for k, v in reps.items()}


def ci(arr: np.ndarray) -> Tuple[float, float]:
    a = arr[np.isfinite(arr)]
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def p_two_sided(arr: np.ndarray) -> float:
    a = arr[np.isfinite(arr)]
    if len(a) == 0:
        return float("nan")
    p = 2 * min((a <= 0).mean(), (a >= 0).mean())
    return float(min(1.0, max(p, 1 / len(a))))


def p_greater(arr: np.ndarray, margin: float = 0.0) -> float:
    """One-sided bootstrap p for H1: statistic > margin (floored at 1/n)."""
    a = arr[np.isfinite(arr)]
    if len(a) == 0:
        return float("nan")
    return float(max((a <= margin).mean(), 1 / len(a)))


def terms_from_predictions(p: pd.DataFrame, key_cols: List[str], cluster_col: str = "seed") -> Terms:
    """Group a predictions frame into bootstrap terms, one term per (key, cluster); folds are pooled per cluster."""
    out: Terms = {}
    for key, q in p.groupby(key_cols + [cluster_col], sort=True):
        *k, c = key if isinstance(key, tuple) else (key,)
        k = tuple(k) if len(k) > 1 else k[0]
        out.setdefault(k, []).append((c, q.image_id.astype(str).tolist(), q.y.to_numpy(), q.prob.to_numpy()))
    return out


# ----------------------------------------------------------------------------------------------- round 8, Phase 2
# docs/PREREGISTRATION_ROUND8.md. Seeds: development uses the environments of the earlier rounds (validation data only);
# confirmation uses new environment / fold / validation seeds that no earlier analysis has drawn.
DEV_SEEDS = (42, 123, 456, 789, 2026)
CONF_SEEDS = (8101, 8202, 8303, 8404, 8505)
FT_TRAP_SEEDS = (8101, 8202)          # fine-tuned traps: 2 env seeds x 5 folds = 10 clusters (as FT4 of round 6)
SMOKE_SEEDS = (99991,)
GUARD_MARGIN = 0.005                  # same-artifact validation AUROC >= masking head's - 0.005
# fixed-sequence order (Amendment 1): mask_cmc -> full_cmc -> mask_bal(lambda) -> locrand_loc
PRIMARY = ("mask_cmc", "full_cmc", "mask_bal", "locrand_loc")


def seeds_for(stage: str, smoke: bool) -> tuple:
    if smoke:
        return SMOKE_SEEDS
    return {"dev": DEV_SEEDS, "confirm": CONF_SEEDS}[stage]


def stage_dir(stage: str, smoke: bool) -> Path:
    d = out_dir(smoke) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pair_auc(sp: np.ndarray, sn: np.ndarray) -> Tuple[float, int]:
    """Mann-Whitney AUROC of positive scores sp against negative scores sn, and the number of pairs."""
    if len(sp) == 0 or len(sn) == 0:
        return float("nan"), 0
    s = np.concatenate([sp, sn])
    r = pd.Series(s).rank(method="average").to_numpy()[: len(sp)]
    return float((r.sum() - len(sp) * (len(sp) + 1) / 2) / (len(sp) * len(sn))), len(sp) * len(sn)


def pair_type_aucs(y, a, p) -> Dict[str, Tuple[float, int]]:
    """AUROC and pair count of the four positive-negative pair types by artifact status (a = 1 / 0)."""
    y, a, p = np.asarray(y).astype(int), np.asarray(a).astype(int), np.asarray(p, float)
    g = lambda yy, aa: p[(y == yy) & (a == aa)]
    return {"y1a0_y0a1": _pair_auc(g(1, 0), g(0, 1)), "y1a1_y0a0": _pair_auc(g(1, 1), g(0, 0)),
            "y1a1_y0a1": _pair_auc(g(1, 1), g(0, 1)), "y1a0_y0a0": _pair_auc(g(1, 0), g(0, 0))}


def worst_group_auc(y, a, p) -> float:
    """Selection objective (as U13): min(AUROC(Y1A0 vs Y0A1), AUROC(Y1A1 vs Y0A0)) over the cross-artifact pairs."""
    t = pair_type_aucs(y, a, p)
    v = [t["y1a0_y0a1"][0], t["y1a1_y0a0"][0]]
    v = [x for x in v if np.isfinite(x)]
    return float(min(v)) if v else float("nan")


def same_artifact_auc(y, a, p) -> float:
    """Guard statistic: AUROC over the pairs whose two images share the artifact status (both carry / both lack it),
    i.e. the pair-weighted mean of AUROC(Y1A1 vs Y0A1) and AUROC(Y1A0 vs Y0A0). Inside such a pair a linear score's
    artifact term cancels, so this measures disease ranking and is unaffected by how strongly the validation split
    associates artifact and label."""
    t = pair_type_aucs(y, a, p)
    num = den = 0.0
    for k in ("y1a1_y0a1", "y1a0_y0a0"):
        v, n = t[k]
        if n and np.isfinite(v):
            num, den = num + v * n, den + n
    return float(num / den) if den else float("nan")


def all_pairs_auc(y, p) -> float:
    y = np.asarray(y).astype(int)
    return _pair_auc(np.asarray(p, float)[y == 1], np.asarray(p, float)[y == 0])[0]


def select_setting(val: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], reference: str = "mask",
                   margin: float = GUARD_MARGIN, guard: str = "same_artifact") -> Tuple[str, pd.DataFrame]:
    """Validation-only selection rule (docs/PREREGISTRATION_ROUND8.md, section 4).

    val: setting -> (y, a, p) on the selection split; `reference` (the masking head) is always admissible.
    Objective: worst-group AUROC. Guard: same-artifact AUROC >= reference's - margin (guard="all_pairs" is the rule of
    docs/ROUND8_DESIGN.md 5d that the registration replaced; kept for the unit test that shows why).
    Ties (objective within 1e-9) go to the earlier setting in `val`'s order, which lists settings closest to masking
    first. Returns (chosen setting, table of every setting's statistics)."""
    rows = []
    stat = same_artifact_auc if guard == "same_artifact" else (lambda y, a, p: all_pairs_auc(y, p))
    ref = stat(*val[reference])
    for name, (y, a, p) in val.items():
        g = stat(y, a, p)
        rows.append({"setting": name, "objective": worst_group_auc(y, a, p), "guard_stat": g,
                     "guard_ref": ref, "admissible": bool(name == reference or (np.isfinite(g) and g >= ref - margin))})
    t = pd.DataFrame(rows)
    ok = t[t.admissible & np.isfinite(t.objective)]
    if ok.empty:
        return reference, t.assign(chosen=t.setting == reference)
    best = ok.objective.max()
    choice = ok[ok.objective >= best - 1e-9].setting.iloc[0]
    return choice, t.assign(chosen=t.setting == choice)


def smoke_envs_from_validation(E: dict, test_envs=("test_rev", "test_corr", "clean")) -> dict:
    """Smoke runs never score an image outside the run's own training/validation data: every test environment is
    replaced by the selection split (val_groups)."""
    E = dict(E)
    for e in test_envs:
        if e in E:
            E[e] = E["val_groups"]
    return E
