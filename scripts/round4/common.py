"""Shared pieces of the round-4 analyses (docs/PREREGISTRATION_ROUND4.md, scripts/round4/README.md).

Cohort loaders reuse the existing trap scripts unchanged; `assemble_views` builds feature caches for a pool by
copying rows from existing caches of the same view and extracting only the missing images, so no existing cache
file is ever written.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wtss import paths

ROOT = paths.REPO_ROOT
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "round4"
FEAT = paths.CACHE / "features" / "round4"
REF_ROOT = ROOT / "results" / "rerun_2026-09-28"
REF = {"isic": "spec_e13/dino518_spec", "thyroid": "thyroid/dino518_main", "capsule": "capsule/dino518_main",
       "ovary": "ovary/dino518_main"}
LABEL = {"isic": "ISIC hair", "thyroid": "Thyroid calipers", "capsule": "Capsule debris", "ovary": "Ovary calipers"}
COHORTS = ("isic", "thyroid", "capsule", "ovary")
BACKBONE, BDIR = "dino518", "dinov2_b14_518"


def load_script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def log(**kw):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **kw}, default=str), flush=True)


def trap_cohort(name: str) -> dict:
    """Cohort table, cache, donors, grouping and the artifact-presence / overlap columns, exactly as the trap scripts."""
    from wtss.experiments.real_traps import RealCache
    if name == "isic":
        from wtss.data.isic2019_spec import load_spec_cohort
        c = load_spec_cohort()
        cache = RealCache(paths.DATA / "isic2019" / "prepared" / "cache_518", roi_file="roi_spec.npy")
        donors = c[~c.A0 & (c.r_spec >= 0.1) & (c.r_spec < 0.5)].image_id.tolist()
        old = [paths.CACHE / "features" / "spec_isic2019" / BDIR, paths.CACHE / "features" / "isic_all" / BDIR]
        return dict(c=c, cache=cache, donors=donors, group_col=None, present=~c.A0, r=c.r_spec, old=old)
    rt = load_script("rt", "scripts/run_thyroid_traps.py")
    c = {"thyroid": rt.cohort, "capsule": rt.capsule_cohort, "ovary": rt.ovary_cohort}[name]()
    if name == "capsule":
        cache = RealCache(rt.CAP / "cache_518", roi_file="roi.npy", art_file="contam.npy")
        present = c.contam_frac >= 0.10
        old = [paths.CACHE / "features" / "capsule" / BDIR]
    else:
        base = rt.T if name == "thyroid" else rt.OV
        cache = RealCache(base / "cache_518", roi_file="roi.npy", art_file="marker.npy")
        present = c.marker_px >= 15
        old = [paths.CACHE / "features" / name / BDIR] + ([paths.CACHE / "features" / "thyroid_all" / BDIR] if name == "thyroid" else [])
    donors = c[present & c.r.between(0.1, 0.5, inclusive="left")].image_id.tolist()
    return dict(c=c, cache=cache, donors=donors, group_col="group", present=present, r=c.r, old=old)


def smoke_subset(c: pd.DataFrame, traps, n=150, seed=0) -> pd.DataFrame:
    """Small cohort for a smoke run: up to n artifact-free and n artifact-bearing images per trap and label."""
    keep = []
    for y in (0, 1):
        keep.append(c[c.A0 & (c.y == y)].sample(frac=1, random_state=seed).head(n))
        for t in traps:
            keep.append(c[c[f"{t}_A1"] & (c.y == y)].sample(frac=1, random_state=seed).head(n))
    return c.loc[sorted(set(pd.concat(keep).index))].reset_index(drop=True)


def smoke_envs(envs: dict) -> dict:
    return {k: v for k, v in envs.items() if k[1] == 42 and k[2] == 0}


def _covered(f: Path, pool) -> bool:
    if not f.exists():
        return False
    ids = set(np.load(f, allow_pickle=False)["ids"].astype(str))
    return all(str(i) in ids for i in pool)


def assemble_views(pool, views, rend, fdir: Path, sources: dict, device, fname=lambda v: f"{v}.npz",
                   batch_size=64, workers=6):
    """Write fdir/fname(v) covering `pool` for every view. Rows are copied from `sources[v]` = [(npz path, id map)]
    when that file holds the same view of the same image; only the remaining images go through the backbone."""
    from wtss.features import extract_view
    pool = [str(i) for i in pool]
    fdir.mkdir(parents=True, exist_ok=True)
    backend = None
    for v in views:
        out = fdir / fname(v)
        if _covered(out, pool):
            log(view=v, status="cached", n=len(pool))
            continue
        pos = {k: j for j, k in enumerate(pool)}
        X, found = None, np.zeros(len(pool), bool)
        for src, idmap in [(out, lambda i: i)] + list(sources.get(v, [])):
            if not Path(src).exists() or found.all():
                continue
            z = np.load(src, allow_pickle=False)
            sp = {k: j for j, k in enumerate(z["ids"].astype(str))}
            rows = [(k, sp.get(idmap(i))) for k, i in enumerate(pool) if not found[k]]
            rows = [(k, j) for k, j in rows if j is not None]
            if rows:
                SX = z["X"]
                if X is None:
                    X = np.zeros((len(pool), SX.shape[1]), np.float32)
                kk, jj = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
                X[kk], found[kk] = SX[jj], True
                log(view=v, copied_from=str(src), n=len(rows))
        miss = [i for k, i in enumerate(pool) if not found[k]]
        if miss:
            if backend is None:
                from wtss.backbones import load_backend
                backend = load_backend(BACKBONE, device)
            tmp = fdir / f"_missing_{fname(v)}"
            log(view=v, extracting=len(miss))
            Xm = extract_view(backend, miss, rend[v], tmp, device, batch_size, workers, desc=v)
            if X is None:
                X = np.zeros((len(pool), Xm.shape[1]), np.float32)
            X[[pos[i] for i in miss]] = Xm
        np.savez(out, X=X, ids=np.asarray(pool))
        if miss:
            (fdir / f"_missing_{fname(v)}").unlink(missing_ok=True)
    if backend is not None:
        import gc
        import torch
        del backend
        gc.collect()
        torch.cuda.empty_cache()


def pool_of(envs: dict, traps) -> list:
    return sorted(set().union(*[set(d.image_id.astype(str)) for k, d in envs.items() if k[0] in traps]))


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, k in enumerate(o):
        run = max(run, (len(p) - r) * p[k])
        adj[k] = min(run, 1.0)
    return adj


def ci(r: dict) -> str:
    est = r["estimate"] if "estimate" in r else r["seed_delta_mean"]
    return f"{est:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def git_head() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def env_info() -> dict:
    info = {"commit": git_head(), "host": os.uname().nodename, "WTSS_BOOTSTRAP": os.environ.get("WTSS_BOOTSTRAP")}
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    return info

