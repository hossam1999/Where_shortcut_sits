"""ISIC 2019 real-hair cohort and Trap A / Trap B environments (see docs/PREREGISTRATION_ISIC2019_TRAPS.md)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .. import paths
from ..utils import stable_int

HAIR_FREE_MAX = 0.001
HAIR_MIN = 0.005
R_IN = 0.5
R_OUT = 0.1
LESION_MIN, LESION_MAX = 0.005, 0.95
FOLD_SEED = 20260926
RATES = {"train_corr": (0.9, 0.1), "test_corr": (0.9, 0.1), "test_rev": (0.1, 0.9)}


def load_cohort(size: int = 518, hair_free_max=HAIR_FREE_MAX, hair_min=HAIR_MIN, exclude_vignetting=False) -> pd.DataFrame:
    df = pd.read_csv(paths.DATA / "isic2019" / "prepared" / f"cohort_{size}.csv")
    df["qc_ok"] = df.lesion_frac.between(LESION_MIN, LESION_MAX) & (df.ink_frac == 0) & \
        (df.lesion_mask_source != "missing") & ~df.get("ink_uncertain", False)
    if exclude_vignetting:
        df["qc_ok"] &= df.vig_frac == 0
    df["hair_free"] = df.hair_frac < hair_free_max
    df["hair_present"] = df.hair_frac >= hair_min
    df["group_A"] = np.select(
        [df.qc_ok & df.hair_free,
         df.qc_ok & df.hair_present & (df.hair_in_lesion_r >= R_IN),
         df.qc_ok & df.hair_present & (df.hair_in_lesion_r < R_OUT),
         df.qc_ok & df.hair_present],
        ["free", "trapA", "trapB", "donor"], default="excluded")
    # folds (on the whole cohort, so traps share folds)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    fold = np.zeros(len(df), int)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(df)), df.y, df.group)):
        fold[te] = k
    df["fold"] = fold
    assert (df.groupby("group").fold.nunique() == 1).all()
    return df


def _max_subsample(pos: np.ndarray, neg: np.ndarray, p: float):
    """Numbers (n_pos, n_neg) of the max-size subsample with P(A=1)=p."""
    if len(pos) == 0 or len(neg) == 0:
        return 0, 0
    n = int(math.floor(min(len(pos) / p, len(neg) / (1 - p))))
    npos = int(round(p * n))
    return min(npos, len(pos)), min(n - npos, len(neg))


@dataclass
class TrapEnvs:
    """image_id lists per (trap, fold, env) and the A label."""
    envs: Dict[tuple, pd.DataFrame]


def build_trap_envs(df: pd.DataFrame, seed: int = 20260926) -> Dict[tuple, pd.DataFrame]:
    """Source-matched environments for both traps, for each fold as test fold.

    Returns {(trap, fold, split_env): DataFrame[image_id, y, a, source]} where split_env in
    train_corr, test_corr, test_rev, clean_test, clean_val, val_groups (all four groups; DFR).
    """
    out = {}
    for k in range(5):
        role = np.where(df.fold == k, "test", np.where(df.fold == (k + 1) % 5, "val", "train"))
        for env in ["train_corr", "test_corr", "test_rev"]:
            split = "train" if env == "train_corr" else "test"
            p1, p0 = RATES[env]
            chosen = {"trapA": [], "trapB": []}
            for src in sorted(df.source.unique()):
                base = (role == split) & (df.source == src).to_numpy()
                for y, p in [(1, p1), (0, p0)]:
                    free = df[base & (df.group_A == "free").to_numpy() & (df.y == y).to_numpy()]
                    arts = {t: df[base & (df.group_A == t).to_numpy() & (df.y == y).to_numpy()] for t in ("trapA", "trapB")}
                    counts = {t: _max_subsample(arts[t].image_id.to_numpy(), free.image_id.to_numpy(), p) for t in arts}
                    npos = min(c[0] for c in counts.values())
                    nneg = min(c[1] for c in counts.values())
                    # keep the exact rate after taking the min over traps
                    if npos and nneg:
                        n = int(math.floor(min(npos / p, nneg / (1 - p))))
                        npos, nneg = min(int(round(p * n)), npos), min(n - int(round(p * n)), nneg)
                    else:
                        npos = nneg = 0
                    rng = np.random.default_rng(stable_int(seed, k, env, src, y, "free"))
                    neg_ids = rng.choice(free.image_id.to_numpy(), nneg, replace=False) if nneg else []
                    for t in arts:
                        r2 = np.random.default_rng(stable_int(seed, k, env, src, y, t))
                        pos_ids = r2.choice(arts[t].image_id.to_numpy(), npos, replace=False) if npos else []
                        chosen[t] += [(i, y, 1, src) for i in pos_ids] + [(i, y, 0, src) for i in neg_ids]
            for t in chosen:
                out[(t, k, env)] = pd.DataFrame(chosen[t], columns=["image_id", "y", "a", "source"])
        for t in ("trapA", "trapB"):
            ct = df[(role == "test") & (df.group_A == "free").to_numpy()]
            out[(t, k, "clean_test")] = pd.DataFrame({"image_id": ct.image_id, "y": ct.y, "a": 0, "source": ct.source})
            cv = df[(role == "val") & (df.group_A == "free").to_numpy()]
            out[(t, k, "clean_val")] = pd.DataFrame({"image_id": cv.image_id, "y": cv.y, "a": 0, "source": cv.source})
            vg = df[(role == "val") & df.group_A.isin(["free", t]).to_numpy()]
            out[(t, k, "val_groups")] = pd.DataFrame({"image_id": vg.image_id, "y": vg.y,
                                                      "a": (vg.group_A == t).astype(int), "source": vg.source})
            tr = df[(role == "train") & df.group_A.isin(["free", "trapA", "trapB", "donor"]).to_numpy()]
            out[(t, k, "train_all")] = pd.DataFrame({"image_id": tr.image_id, "y": tr.y,
                                                     "a": (tr.group_A != "free").astype(int), "source": tr.source,
                                                     "group_A": tr.group_A})
    return out


def trap_count_table(envs) -> pd.DataFrame:
    rows = []
    for (t, k, e), d in envs.items():
        if e in ("train_all",):
            continue
        rows.append({"trap": t, "fold": k, "env": e, "n": len(d), "n_mel": int(d.y.sum()),
                     "n_art": int(d.a.sum()), "p_art_mel": float(d[d.y == 1].a.mean()) if (d.y == 1).any() else np.nan,
                     "p_art_ben": float(d[d.y == 0].a.mean()) if (d.y == 0).any() else np.nan})
    return pd.DataFrame(rows)
