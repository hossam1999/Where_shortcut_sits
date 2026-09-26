"""E13/E15 exactly as in docs/REPLICATION_SPEC.md (the author's protocol).

- Sources HAM / BCN / MSK (images without a HAM/BCN lesion_id are MSK).
- Lesion masks: HAM manual (Tschandl); every other image from the spec U-Net (Ronneberger, ISIC 2018 Task 1 only).
- A = 0 (hair-free, shared by both traps): <= 30 hair/ruler pixels in the native-resolution archive mask
  (calibrated on outcome-free counts: reproduces 708/157 and all six source proportions of the spec).
- A = 1: hair with r >= 0.5 (Trap A) or r < 0.1 (Trap B); strict sensitivity 0.6 / 0.05.
- Source matching: within each label, A = 1 cells are subsampled to the A = 0 source mix.
- 5-fold group-safe CV per seed (seeds 42, 123, 456, 789, 2026); groups = lesion_id ∪ pHash ≤ 8 within the pool;
  fold test predictions are pooled per seed before AUROC (seed = bootstrap cluster).
- Environments by maximum-size subsampling: train/test_corr 0.9/0.1, test_rev 0.1/0.9,
  clean = artifact uncorrelated with the label (0.5/0.5).
"""
from __future__ import annotations

import math
import re
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .. import paths
from ..utils import stable_int

SEEDS = (42, 123, 456, 789, 2026)
A0_MAX_NATIVE_PX = 30
RATES = {"train_corr": (0.9, 0.1), "test_corr": (0.9, 0.1), "test_rev": (0.1, 0.9), "clean": (0.5, 0.5),
         "val_clean": (0.5, 0.5)}
PREP = paths.DATA / "isic2019" / "prepared"


def _key(n: str) -> str:
    return f"ISIC_{int(re.match(r'ISIC_(\d+)', n).group(1)):07d}"


def load_spec_cohort(r_in=0.5, r_out=0.1, r_col="r_spec") -> pd.DataFrame:
    c = pd.read_csv(PREP / "cohort_spec.csv")
    c["A0"] = c.hair_px_native <= A0_MAX_NATIVE_PX
    c["trapA_A1"] = ~c.A0 & (c[r_col] >= r_in)
    c["trapB_A1"] = ~c.A0 & (c[r_col] < r_out)
    return c[c.qc_ok].reset_index(drop=True)


def _max_sub(n1, n0, p):
    if p <= 0:
        return 0, n0
    if p >= 1:
        return n1, 0
    if n1 == 0 or n0 == 0:
        return 0, 0
    n = int(math.floor(min(n1 / p, n0 / (1 - p))))
    k = int(round(p * n))
    return min(k, n1), min(n - k, n0)


def matched_pool(c: pd.DataFrame, trap: str, seed: int = 20260926) -> pd.DataFrame:
    """A=0 (all) + A=1 subsampled per label to the A=0 source mix."""
    a0 = c[c[f"{trap}_A0"]] if f"{trap}_A0" in c else c[c.A0]  # per-trap artifact-free group (CXR)
    a1 = c[c[f"{trap}_A1"]]
    keep = [a0.assign(a=0)]
    scol = f"{trap}_source" if f"{trap}_source" in c else "source"  # per-trap matching strata (CXR follow-up)
    for y in (0, 1):
        mix = a0[a0.y == y][scol].value_counts(normalize=True)
        cand = a1[a1.y == y]
        have = cand[scol].value_counts()
        N = min(have.get(s, 0) / mix[s] for s in mix.index)
        for s in mix.index:
            n = int(math.floor(N * mix[s]))
            pool = cand[cand[scol] == s].image_id.to_numpy()
            rng = np.random.default_rng(stable_int("match", trap, y, s, seed))
            keep.append(c[c.image_id.isin(rng.choice(pool, n, replace=False))].assign(a=1))
    return pd.concat(keep, ignore_index=True)


def pool_groups(pool: pd.DataFrame, thr: int = 8) -> np.ndarray:
    """lesion_id ∪ pHash (<= thr) union-find within the pool."""
    n = len(pool)
    parent = np.arange(n)

    def f(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def u(a, b):
        a, b = f(a), f(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for _, idx in pool.groupby("lesion_id").indices.items():
        for j in idx[1:]:
            u(idx[0], j)
    h = np.array([int(x, 16) for x in pool.phash_hex], dtype=np.uint64)
    for i in range(0, n, 512):
        x = h[i:i + 512, None] ^ h[None, :]
        pc = np.zeros(x.shape, np.uint8)
        for s in range(64):
            pc += ((x >> np.uint64(s)) & np.uint64(1)).astype(np.uint8)
        ii, jj = np.nonzero(pc <= thr)
        for a, b in zip(ii + i, jj):
            if a < b:
                u(int(a), int(b))
    return np.array([f(i) for i in range(n)])


def _env(pool: pd.DataFrame, env: str, seed: int, fold: int, trap: str) -> pd.DataFrame:
    p1, p0 = RATES[env]
    out = []
    for y, p in ((1, p1), (0, p0)):
        d = pool[pool.y == y]
        A1, A0 = d[d.a == 1], d[d.a == 0]
        n1, n0 = _max_sub(len(A1), len(A0), p)
        rng = np.random.default_rng(stable_int("env", trap, env, seed, fold, y))
        out += [A1.iloc[rng.choice(len(A1), n1, replace=False)], A0.iloc[rng.choice(len(A0), n0, replace=False)]]
    return pd.concat(out)[["image_id", "y", "a", "source"]].reset_index(drop=True)


def build_spec_envs(c: pd.DataFrame, traps=("trapA", "trapB"), seeds=SEEDS, pht: int = 8, group_col=None) -> Dict:
    """{(trap, seed, fold, env): DataFrame}. env in train_corr, val_clean, val_groups, test_corr, test_rev, clean,
    train_all (all training-fold pool images, for paired erasure / insertion)."""
    envs = {}
    for trap in traps:
        pool = matched_pool(c, trap)
        pool["group"] = pool[group_col].to_numpy() if group_col else pool_groups(pool, pht)
        for seed in seeds:
            sg = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
            fold = np.zeros(len(pool), int)
            for k, (_, te) in enumerate(sg.split(pool, 2 * pool.a + pool.y, pool.group)):
                fold[te] = k
            for k in range(5):
                te = pool[fold == k]
                rest = pool[fold != k]
                # inner validation: 20% of the training groups
                g = rest.group.unique()
                rng = np.random.default_rng(stable_int("val", trap, seed, k))
                vg = set(rng.choice(g, max(1, int(0.2 * len(g))), replace=False))
                va, tr = rest[rest.group.isin(vg)], rest[~rest.group.isin(vg)]
                envs[(trap, seed, k, "train_corr")] = _env(tr, "train_corr", seed, k, trap)
                envs[(trap, seed, k, "val_clean")] = _env(va, "val_clean", seed, k, trap)
                envs[(trap, seed, k, "val_groups")] = va[["image_id", "y", "a", "source"]].reset_index(drop=True)
                envs[(trap, seed, k, "train_all")] = tr[["image_id", "y", "a", "source"]].reset_index(drop=True)
                for e in ("test_corr", "test_rev", "clean"):
                    envs[(trap, seed, k, e)] = _env(te, e, seed, k, trap)
    return envs


def spec_counts(c: pd.DataFrame, envs: Dict) -> pd.DataFrame:
    rows = []
    for trap in ("trapA", "trapB"):
        pool = matched_pool(c, trap)
        cells = pool.groupby(["a", "y"]).size()
        rows.append({"trap": trap, **{f"A{a}_Y{y}": int(cells.get((a, y), 0)) for a in (0, 1) for y in (0, 1)},
                     "eligible": len(pool),
                     "rev_mel_seed42": int(sum(envs[(trap, 42, k, "test_rev")].y.sum() for k in range(5)))
                     if (trap, 42, 0, "test_rev") in envs else None})
    return pd.DataFrame(rows)
