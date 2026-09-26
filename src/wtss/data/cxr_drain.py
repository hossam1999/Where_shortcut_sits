"""NIH chest-drain trap (A = chest drain, Y = pneumothorax), the radiographic in-ROI regime.

Drain labels: NEATX (human, PTX+ images); detector scores for PTX- images
(``nih_drain_scores.csv`` from scripts/data/prepare_cxr.py --stage detector).
View position (AP/PA) plays the role of 'source' for matching: drains are concentrated in supine AP
films, so environments are built within view.
"""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .. import paths
from ..utils import stable_int
from .cxr import PREP

RATES = {"train_corr": (0.9, 0.1), "test_corr": (0.9, 0.1), "test_rev": (0.1, 0.9)}
MAX_FREE_NEG = 20000


def load_drain_cohort(seed: int = 20260926) -> pd.DataFrame:
    return pd.read_csv(PREP / "nih_drain_cohort.csv")


def make_drain_cohort(seed: int = 20260926) -> pd.DataFrame:
    s = pd.read_csv(PREP / "nih_drain_scores.csv")
    det = json.loads((PREP / "DRAIN_DETECTOR.json").read_text())
    hi = det["precision_0.95"]
    lab = s[s.drain_neatx.notna()]
    lo = float(np.quantile(lab[lab.drain_neatx == 0].drain_score, 0.5))
    pos = s[(s.ptx == 1) & s.drain_neatx.notna()].assign(a=lambda d: d.drain_neatx.astype(int), a_source="neatx")
    neg = s[s.ptx == 0]
    neg_art = neg[neg.drain_score >= hi].assign(a=1, a_source="detector")
    neg_free = neg[neg.drain_score <= lo].sample(frac=1, random_state=seed).drop_duplicates("patient")
    neg_free = neg_free.head(MAX_FREE_NEG).assign(a=0, a_source="detector")
    c = pd.concat([pos, neg_art, neg_free]).rename(columns={"ptx": "y"})
    c["source"] = c.view
    c["group"] = c.patient
    sg = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    c = c.reset_index(drop=True)
    c["fold"] = 0
    for k, (_, te) in enumerate(sg.split(c, 2 * c.y + c.a, c.group)):
        c.loc[te, "fold"] = k
    c["thr_hi"], c["thr_lo"] = hi, lo
    return c


def _max_sub(npos, nneg, p):
    if npos == 0 or nneg == 0:
        return 0, 0
    n = int(math.floor(min(npos / p, nneg / (1 - p))))
    return min(int(round(p * n)), npos), min(n - int(round(p * n)), nneg)


def build_drain_envs(df: pd.DataFrame, seed: int = 20260926):
    envs, rows = {}, []
    for k in range(5):
        role = np.where(df.fold == k, "test", np.where(df.fold == (k + 1) % 5, "val", "train"))
        for env in ["train_corr", "test_corr", "test_rev"]:
            split = "train" if env == "train_corr" else "test"
            p1, p0 = RATES[env]
            ch = []
            for src in sorted(df.source.unique()):
                for y, p in [(1, p1), (0, p0)]:
                    base = df[(role == split) & (df.source == src).to_numpy() & (df.y == y).to_numpy()]
                    A1, A0 = base[base.a == 1], base[base.a == 0]
                    n1, n0 = _max_sub(len(A1), len(A0), p)
                    r = np.random.default_rng(stable_int(seed, k, env, src, y))
                    ch += [(i, y, 1, src) for i in r.choice(A1.image_id.to_numpy(), n1, replace=False)]
                    ch += [(i, y, 0, src) for i in r.choice(A0.image_id.to_numpy(), n0, replace=False)]
            envs[("drain", k, env)] = pd.DataFrame(ch, columns=["image_id", "y", "a", "source"])
        ct = df[(role == "test") & (df.a == 0).to_numpy()]
        envs[("drain", k, "clean_test")] = ct[["image_id", "y", "a", "source"]].copy()
        cv = df[(role == "val") & (df.a == 0).to_numpy()]
        envs[("drain", k, "clean_val")] = cv[["image_id", "y", "a", "source"]].copy()
        envs[("drain", k, "val_groups")] = df[role == "val"][["image_id", "y", "a", "source"]].copy()
        envs[("drain", k, "train_all")] = df[role == "train"][["image_id", "y", "a", "source"]].copy()
    for (t, k, e), d in envs.items():
        if e != "train_all":
            rows.append({"trap": t, "fold": k, "env": e, "n": len(d), "n_mel": int(d.y.sum()), "n_art": int(d.a.sum())})
    return envs, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- procedural drain
def draw_drain(rgb: np.ndarray, lung: np.ndarray, key: str) -> np.ndarray:
    """Insert a procedural intercostal chest drain: enters at the lateral chest wall of a random side,
    curves medially/upwards and ends inside that lung near the apex; bright tube body, radio-opaque
    stripe and side-hole gaps near the tip, plus external tubing. No annotation required."""
    H, W = lung.shape
    rng = np.random.default_rng(stable_int("drain_v1", key))
    side = int(rng.integers(0, 2))  # 0 = image-left, 1 = image-right
    half = np.zeros_like(lung)
    if side == 0:
        half[:, : W // 2] = 1
    else:
        half[:, W // 2:] = 1
    lm = lung.astype(bool) & half.astype(bool)
    ys, xs = np.nonzero(lm)
    if len(ys) < 50:
        ys, xs = np.nonzero(half)
    # tip: upper third of that lung; entry: lateral wall, lower half
    top = ys <= np.quantile(ys, 0.35)
    j = int(rng.integers(0, max(1, top.sum())))
    tip = np.array([xs[top][j], ys[top][j]], float)
    lat_x = xs.min() if side == 0 else xs.max()
    entry = np.array([lat_x + (-1 if side == 0 else 1) * rng.uniform(5, 25), rng.uniform(0.55, 0.8) * H], float)
    ctrl = np.array([(entry[0] + tip[0]) / 2 + (1 if side == 0 else -1) * rng.uniform(0, 0.08) * W,
                     entry[1] - rng.uniform(0.1, 0.3) * H])
    t = np.linspace(0, 1, 300)[:, None]
    pts = ((1 - t) ** 2 * entry + 2 * (1 - t) * t * ctrl + t ** 2 * tip)
    ext = entry + np.array([(-1 if side == 0 else 1) * rng.uniform(10, 40), rng.uniform(20, 80)])
    outside = np.linspace(ext, entry, 60)
    path = np.vstack([outside, pts]).astype(np.int32)
    width = int(rng.integers(4, 8)) * max(1, H // 518)
    body = np.zeros((H, W), np.uint8)
    cv2.polylines(body, [path], False, 1, thickness=width, lineType=cv2.LINE_AA)
    stripe = np.zeros((H, W), np.uint8)
    spts = pts.astype(np.int32)
    # side holes: gaps in the stripe near the tip
    n = len(spts)
    segs = [(0, int(0.78 * n))] + [(int(f * n), int((f + 0.04) * n)) for f in (0.82, 0.9)]
    for a, b in segs:
        cv2.polylines(stripe, [spts[a:b]], False, 1, thickness=max(1, width // 3), lineType=cv2.LINE_AA)
    g = rgb[..., 0].astype(np.float32)
    gain_body = rng.uniform(25, 55)
    g = np.where(body > 0, np.minimum(255, g * 0.8 + gain_body + 0.2 * 255 * 0.3), g)
    g = np.where(stripe > 0, np.minimum(255, g + rng.uniform(70, 120)), g)
    g = cv2.GaussianBlur(g, (0, 0), 0.6)
    out = np.where((body | stripe)[..., None] > 0, np.repeat(g[..., None], 3, -1), rgb.astype(np.float32))
    return out.clip(0, 255).astype(np.uint8)


def drain_insert_fn():
    def f(i, rgb, roi):
        return draw_drain(rgb, roi, i)

    return f
