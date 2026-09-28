"""Covariate balance of the location traps and propensity-matched trap cells (docs/PREREGISTRATION_REVIEW2.md, R1).

Trap A (artifact in the ROI) and Trap B (outside) select *different* artifact-bearing images. Here the artifact-
bearing images of the two traps are compared on pre-specified covariates (standardised mean differences) and matched
1:1 within label (and exactly within source for ISIC) on a logistic propensity score, greedy nearest neighbour on the
logit without replacement, calliper 0.2 SD, units in stable random order.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

from . import paths

SITE = {"anterior torso": "torso", "posterior torso": "torso", "lateral torso": "torso", "head/neck": "head/neck",
        "upper extremity": "upper extremity", "lower extremity": "lower extremity", "palms/soles": "palms/soles",
        "oral/genital": "oral/genital"}
MATCH_SEED = 20260928


def _cache_stats(cdir, ids: Sequence[str], img_file: str) -> pd.DataFrame:
    """Mean grey level of the image and normalised ROI centroid row, from the 518-px caches."""
    idx = {k: j for j, k in enumerate((cdir / "ids.txt").read_text().split())}
    img = np.load(cdir / img_file, mmap_mode="r")
    roi = np.load(cdir / "roi.npy", mmap_mode="r")
    rows = []
    for i in ids:
        a = np.asarray(img[idx[i]]).astype(np.float32)
        r = np.asarray(roi[idx[i]]) > 0
        ys = np.nonzero(r)[0]
        rows.append({"image_id": i, "mean_grey": float(a.mean()), "roi_frac_518": float(r.mean()),
                     "roi_centroid_row": float(ys.mean() / r.shape[0]) if len(ys) else np.nan})
    return pd.DataFrame(rows)


def covariates(name: str, c: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Covariate table aligned with c (index = image_id); returns (table, numeric cols, categorical cols)."""
    c = c.copy()
    c["image_id"] = c.image_id.astype(str)
    if name == "isic":
        d = pd.DataFrame({"image_id": c.image_id})
        d["lesion_area"] = c.lesion_frac_spec.to_numpy()
        d["hair_amount"] = c.hair_frac_native.to_numpy()
        age = c.age_approx.astype(float)
        d["age"] = age.fillna(age.median()).to_numpy()
        d["age_missing"] = age.isna().astype(float).to_numpy()
        d["sex"] = c.sex.fillna("missing").to_numpy()
        d["site"] = c.anatom_site_general.map(SITE).fillna("missing").to_numpy()
        dx = np.where(c.NV >= 0.5, "NV", np.where(c.BKL >= 0.5, "BKL", "other"))
        d["benign_dx"] = np.where(c.y == 1, "MEL", dx)
        d["source"] = c.source.to_numpy()  # exact matching stratum: constant within strata, reported in the SMD table
        num, cat = ["lesion_area", "hair_amount", "age", "age_missing"], ["sex", "site", "benign_dx", "source"]
    elif name in ("thyroid", "ovary"):
        cdir = (paths.DATA / "us" / "tncd" if name == "thyroid" else paths.DATA / "ovary") / "cache_518"
        d = _cache_stats(cdir, c.image_id.tolist(), "gray.npy")
        d["caliper_px_log"] = np.log1p(c.marker_px.to_numpy())
        if name == "thyroid":
            d["width"], d["height"] = c.W.to_numpy(), c.H.to_numpy()
            d["roi_area"] = c.roi_frac.to_numpy()
            num, cat = ["roi_area", "caliper_px_log", "width", "height", "mean_grey", "roi_centroid_row"], []
        else:
            root = paths.DATA / "ovary" / "mmotu" / "MMOTU" / "OTU_2d" / "images"
            wh = [Image.open(root / f).size for f in c.file]
            d["width"], d["height"] = [w for w, _ in wh], [h for _, h in wh]
            d["roi_area"] = d.roi_frac_518
            d["subtype"] = c.cls.astype(str).to_numpy()
            num, cat = ["roi_area", "caliper_px_log", "width", "height", "mean_grey"], ["subtype"]
    elif name == "capsule":
        d = _cache_stats(paths.DATA / "capsule" / "cache_518", c.image_id.tolist(), "rgb.npy")
        d["roi_area"] = c.roi_frac.to_numpy()
        d["contamination"] = c.contam_frac.to_numpy()
        d["frame_block"] = (c.num.to_numpy() // 1000).astype(str)
        num, cat = ["roi_area", "contamination", "mean_grey"], ["frame_block"]
    else:
        raise ValueError(name)
    return d.set_index("image_id"), num, cat


def design_matrix(t: pd.DataFrame, num: Sequence[str], cat: Sequence[str]) -> np.ndarray:
    parts = []
    if num:
        x = t[list(num)].astype(float).to_numpy()
        sd = x.std(0)
        parts.append((x - x.mean(0)) / np.where(sd > 0, sd, 1))
    for col in cat:
        lv = sorted(t[col].astype(str).unique())
        if len(lv) > 1:
            parts.append(np.stack([(t[col].astype(str) == v).to_numpy(float) for v in lv[1:]], 1))
    return np.concatenate(parts, 1) if parts else np.zeros((len(t), 0))


def smd(t: pd.DataFrame, g: np.ndarray, num: Sequence[str], cat: Sequence[str]) -> pd.DataFrame:
    """Standardised mean differences between g == 1 and g == 0 (categorical: per level)."""
    rows = []
    a, b = t[g == 1], t[g == 0]
    for col in num:
        xa, xb = a[col].astype(float), b[col].astype(float)
        s = np.sqrt((xa.var() + xb.var()) / 2)
        rows.append({"covariate": col, "level": "", "mean_1": xa.mean(), "mean_0": xb.mean(),
                     "smd": (xa.mean() - xb.mean()) / s if s > 0 else 0.0})
    for col in cat:
        for v in sorted(t[col].astype(str).unique()):
            pa, pb = (a[col].astype(str) == v).mean(), (b[col].astype(str) == v).mean()
            s = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
            rows.append({"covariate": col, "level": v, "mean_1": pa, "mean_0": pb, "smd": (pa - pb) / s if s > 0 else 0.0})
    return pd.DataFrame(rows)


def _greedy_match(la: np.ndarray, lb: np.ndarray, caliper: float, key: str) -> List[Tuple[int, int]]:
    """1:1 nearest neighbour on the logit without replacement; the smaller group is iterated in stable random order."""
    swap = len(la) > len(lb)
    x, y = (lb, la) if swap else (la, lb)
    order = np.random.default_rng(MATCH_SEED + sum(map(ord, key))).permutation(len(x))
    free = np.ones(len(y), bool)
    pairs = []
    for i in order:
        if not free.any():
            break
        dist = np.where(free, np.abs(y - x[i]), np.inf)
        j = int(dist.argmin())
        if dist[j] <= caliper:
            free[j] = False
            pairs.append((j, i) if swap else (i, j))
    return pairs


def match_traps(name: str, c: pd.DataFrame, strata: Sequence[str] = (), caliper_sd: float = 0.2) -> Tuple[pd.DataFrame, Dict]:
    """Returns c with trapA_A1 / trapB_A1 restricted to matched images, plus a report (balance before/after)."""
    tab, num, cat = covariates(name, c)
    c = c.copy()
    c["image_id"] = c.image_id.astype(str)
    units = c[c.trapA_A1 | c.trapB_A1]
    keepA, keepB, strata_rows = set(), set(), []
    keys = ["y", *strata]
    for key, u in units.groupby(keys):
        t = tab.loc[u.image_id]
        g = u.trapA_A1.to_numpy().astype(int)
        if g.sum() == 0 or (1 - g).sum() == 0:
            strata_rows.append({"stratum": str(key), "n_A": int(g.sum()), "n_B": int((1 - g).sum()), "pairs": 0})
            continue
        X = design_matrix(t, num, cat)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, g)
        p = np.clip(lr.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
        cal = caliper_sd * logit.std()
        ia, ib = np.flatnonzero(g == 1), np.flatnonzero(g == 0)
        pairs = _greedy_match(logit[ia], logit[ib], cal, str(key))
        ids = u.image_id.to_numpy()
        keepA |= {ids[ia[i]] for i, _ in pairs}
        keepB |= {ids[ib[j]] for _, j in pairs}
        strata_rows.append({"stratum": str(key), "n_A": len(ia), "n_B": len(ib), "pairs": len(pairs)})
    before = balance(name, units, tab, num, cat)
    c["trapA_A1"] = c.trapA_A1 & c.image_id.isin(keepA)
    c["trapB_A1"] = c.trapB_A1 & c.image_id.isin(keepB)
    after = balance(name, c[c.trapA_A1 | c.trapB_A1], tab, num, cat)
    return c, {"before": before, "after": after, "strata": pd.DataFrame(strata_rows)}


def balance(name: str, units: pd.DataFrame, tab: pd.DataFrame | None = None, num=None, cat=None) -> pd.DataFrame:
    """SMD Trap A vs Trap B artifact-bearing images, pooled and within each label."""
    if tab is None:
        tab, num, cat = covariates(name, units)
    out = []
    for lab, u in [("pooled", units), *[(f"y={y}", units[units.y == y]) for y in (0, 1)]]:
        g = u.trapA_A1.to_numpy().astype(int)
        s = smd(tab.loc[u.image_id.astype(str)], g, num, cat)
        s.insert(0, "subset", lab)
        s["n_trapA"], s["n_trapB"] = int(g.sum()), int((1 - g).sum())
        out.append(s)
    return pd.concat(out, ignore_index=True)
