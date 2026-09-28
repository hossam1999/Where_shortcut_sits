"""R12/R13 — robustness of the ISIC 2020 external result (docs/PREREGISTRATION_ROUND5.md).

  python scripts/round5/isic2020_robust.py --calibrate   # label-free; commit the two files before fitting
  python scripts/round5/isic2020_robust.py --smoke       # small subsets -> results/round5/_smoke/
  python scripts/round5/isic2020_robust.py               # fit all 32,997 images, check R10, then R12/R13

Reuses scripts/round4/natural_isic2020.py and scripts/round4/common.py. The ISIC 2019 pool, seeds, arms,
heads and generic overlays are those of R10. The only change is that every ISIC 2020 image with a lesion
mask is scored, and the duplicate rules below are subsets of that one set of predictions.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

R4 = Path(__file__).resolve().parents[1] / "round4"
sys.path.insert(0, str(R4))
import common as K  # noqa: E402
import natural_isic2020 as N  # noqa: E402

from wtss import paths  # noqa: E402

OUT = K.ROOT / "results" / "round5"
FEAT = paths.CACHE / "features" / "round5"
VIEWS = ("erm", "mask", "mask_insert")
TIERS = (("exact", "drop_exact"), ("d19_le2", "drop_d19_le2"), ("calibrated", "drop_calibrated"), ("d19_le8", "drop_d19_le8"))
PAIRS = (("mask", "erm"), ("balanced", "mask"), ("mte_balanced", "mask"))
OP_PAIRS = (("mask", "erm"), ("balanced", "mask"))
SUBSETS = ("hard", "easy", "all")
def _r10_boot_seed(subset, arm, ref):
    k = [("mask", "erm"), ("balanced", "mask"), ("mte_balanced", "mask"), ("mte", "mask")].index((arm, ref))
    return 20261401 + 10 * k + ("hard", "easy", "all").index(subset)


def _r10_op_seed(arm, ref):
    return 20261501 + list(OP_PAIRS).index((arm, ref))


def t_star(d0) -> tuple:
    """Largest t in 0..8 with F(t) = P(d0 <= t) <= 1%. None if no such t."""
    d0 = np.asarray(d0)
    F = {str(t): float((d0 <= t).mean()) for t in range(9)}
    ok = [t for t in range(9) if F[str(t)] <= 0.01]
    return (max(ok) if ok else None), F


def c_star(s0) -> tuple:
    """Smallest c on {0.900, 0.901, ..., 0.999} with G(c) = P(s0 >= c) <= 1%. None if no such c."""
    s0 = np.asarray(s0, float)
    grid = [i / 1000 for i in range(900, 1000)]
    G = {f"{c:.3f}": float((s0 >= c).mean()) for c in grid}
    ok = [c for c in grid if G[f"{c:.3f}"] <= 0.01]
    return (min(ok) if ok else None), G


def tier_flags(d19, s19, same_name, t_star_v, c_star_v) -> dict:
    d19 = np.asarray(d19)
    s19 = np.asarray(s19, float)
    same = np.asarray(same_name, bool)
    return {
        "drop_exact": (d19 == 0) | same,  # d19 = 0 or the image name occurs in ISIC 2019
        "drop_d19_le2": d19 <= 2,
        "drop_calibrated": (d19 <= t_star_v) | (s19 >= c_star_v) | same,
        "drop_d19_le8": (d19 <= 8) | same,  # R10's rule
    }


def h12_label(estimate, lo, hi) -> str:
    """Pre-registered R12 decision. Interval-below-zero is checked first."""
    if hi < 0:
        return "robust"
    if estimate < 0 and lo <= 0 <= hi:
        return "direction-consistent, inconclusive"
    return "not robust"


def h13_gate(n_mel_h0: int, n_ben_hin: int) -> bool:
    return int(n_mel_h0) >= 30 and int(n_ben_hin) >= 100


def hair_groups(hair_px, r, tau: float) -> np.ndarray:
    """H0 hair-free, Hin clear in-lesion hair, Hout clear hair beside the lesion, else ambiguous."""
    h = np.asarray(hair_px, float)
    r = np.asarray(r, float)
    g = np.full(len(h), "ambiguous", object)
    g[h <= tau] = "H0"
    g[(h >= 3 * tau) & (r >= 0.5)] = "Hin"
    g[(h >= 3 * tau) & (r < 0.1)] = "Hout"
    return g


def nearest_hamming(h_query, h_gallery, chunk=1024):
    """Min Hamming distance from each query hash to the gallery, and the gallery index of that neighbour."""
    h_query = np.asarray(h_query, np.uint64)
    h_gallery = np.asarray(h_gallery, np.uint64)
    best_d = np.full(len(h_query), 64, np.int32)
    best_j = np.zeros(len(h_query), np.int64)
    for k in range(0, len(h_query), chunk):
        x = N.popcount64(h_query[k:k + chunk, None] ^ h_gallery[None, :])
        j = x.argmin(1)
        best_d[k:k + chunk] = x[np.arange(len(j)), j]
        best_j[k:k + chunk] = j
    return best_d, best_j


def nearest_hamming_other(h, groups, chunk=1024):
    """Min Hamming distance to a different group (patient). Same-group pairs, including self, are ignored."""
    h = np.asarray(h, np.uint64)
    groups = np.asarray(groups)
    best = np.full(len(h), 64, np.int32)
    for k in range(0, len(h), chunk):
        x = N.popcount64(h[k:k + chunk, None] ^ h[None, :])
        x[groups[k:k + chunk, None] == groups[None, :]] = np.uint8(255)
        best[k:k + chunk] = x.min(1)
    return best


def _chunked_max_dot(query, gallery, device, batch, mask_groups=None):
    """Max float32 dot product of each query row against gallery rows. Both are L2-normalised, so this is cosine.

    mask_groups: integer group id per gallery row (gallery is the query set itself). Rows in the same group are
    excluded, which is the null similarity s0.
    """
    import torch

    query = np.ascontiguousarray(query, np.float32)
    gallery = np.ascontiguousarray(gallery, np.float32)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    tg = torch.from_numpy(gallery).to(dev)
    out = np.empty(len(query), np.float32)
    buckets = None
    if mask_groups is not None:
        inv = np.asarray(mask_groups)
        buckets = {g: np.flatnonzero(inv == g) for g in np.unique(inv)}
    for i in range(0, len(query), batch):
        q = torch.from_numpy(query[i:i + batch]).to(dev)
        sim = (q @ tg.T).float().cpu().numpy()
        del q
        if buckets is not None:
            inv_b = mask_groups[i:i + batch]
            for g in np.unique(inv_b):
                rows = np.flatnonzero(inv_b == g)
                sim[np.ix_(rows, buckets[g])] = -np.inf
        out[i:i + batch] = sim.max(1)
        del sim
    del tg
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _feat_dir(smoke: bool) -> Path:
    return FEAT / ("_smoke" if smoke else "") / "isic2020_robust" / K.BDIR


def _sources(fdir19: Path):
    """Copy existing rows before encoding. Round-4 natural cache first: those are the rows R10 was scored on."""
    f4 = paths.CACHE / "features" / "round4" / "natural_isic2020" / K.BDIR
    f20 = paths.CACHE / "features" / "isic2020" / K.BDIR

    def fname(v):
        return f"{v}_generic.npz" if v in ("insert", "mask_insert") else f"{v}.npz"

    def plain(i):
        return None if str(i).startswith(N.PFX) else str(i)

    def strip(i):
        return str(i)[len(N.PFX):] if str(i).startswith(N.PFX) else None

    srcs = {}
    for v in VIEWS:
        rows = [(f4 / fname(v), lambda i: str(i)), (fdir19 / fname(v), plain)]
        if v in ("erm", "mask"):
            rows.append((f20 / fname(v), strip))
        srcs[v] = rows
    return srcs, fname


def _out(smoke: bool) -> Path:
    p = OUT / ("_smoke" if smoke else "")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_aligned(npz, ids):
    z = np.load(npz, allow_pickle=False)
    pos = {k: j for j, k in enumerate(z["ids"].astype(str))}
    missing = [i for i in ids if i not in pos]
    if missing:
        raise SystemExit(f"{npz} is missing {len(missing)} ids (e.g. {missing[:3]})")
    return z["X"][np.array([pos[i] for i in ids])]


def _gpu_name() -> str:
    import subprocess
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()[0].strip()
    except Exception:
        return "cpu"


def run_calibrate(smoke: bool, device: str, batch_size: int, workers: int) -> str:
    t0 = time.perf_counter()
    out = _out(smoke)
    rn, c19, d, cache, fdir19, tau = N.load_all(smoke)
    if not smoke and len(d) != 32997:
        raise SystemExit(f"expected 32997 ISIC 2020 images with a lesion mask, found {len(d)}")
    ids19 = c19.image_id.astype(str).tolist()
    ids20 = d.image_id.astype(str).tolist()
    pool = sorted(set(ids19) | set(ids20))
    import torch
    from wtss.experiments.real_traps import make_renderers
    from wtss.synthetic import draw_generic_artifact
    from PIL import Image

    ins = lambda i, rgb, roi: np.asarray(draw_generic_artifact(Image.fromarray(rgb), roi, f"u|{i}"))
    rend = make_renderers(cache, 518, [], ins)
    srcs, fname = _sources(fdir19)
    fdir = _feat_dir(smoke)

    def extract():
        # Label-free: only the original-image view. Extraction runs in a child so this process can still fork later.
        K.assemble_views(pool, ("erm",), rend, fdir, srcs, torch.device("cuda"), fname=fname,
                         batch_size=batch_size, workers=workers)

    os.environ["WTSS_ROUND5_STAGE"] = "calibrate"
    try:
        status = K.gpu_then_cpu(not K.views_ready(pool, ("erm",), fdir, fname), extract, lambda: None)
    finally:
        os.environ.pop("WTSS_ROUND5_STAGE", None)
    if status == "extracted":
        return "extracted"
    X19 = _load_aligned(fdir / "erm.npz", ids19)
    X20 = _load_aligned(fdir / "erm.npz", ids20)
    err = float(max(np.abs(np.linalg.norm(X19, axis=1) - 1).max(), np.abs(np.linalg.norm(X20, axis=1) - 1).max()))
    if err > 1e-3:
        raise SystemExit(f"DINOv2 rows are not L2-normalised (max |norm-1|={err}); cosine is not the dot product")
    K.log(step="phash", n19=len(ids19), n20=len(ids20))
    threads = min(32, os.cpu_count() or 8)
    h19 = N.phash_all(cache, ids19, threads=threads)
    h20 = N.phash_all(cache, ids20, threads=threads)
    d19, j19 = nearest_hamming(h20, h19)
    codes, _ = pd.factorize(d.group.astype(str), sort=False)
    d0 = nearest_hamming_other(h20, codes)
    K.log(step="cosine", device=device, n20=len(ids20), n19=len(ids19))
    s19 = _chunked_max_dot(X20, X19, device, batch=4096)
    s0 = _chunked_max_dot(X20, X20, device, batch=2048, mask_groups=codes)
    ts, F = t_star(d0)
    cs, G = c_star(s0)
    if ts is None or cs is None:
        raise SystemExit(f"no threshold meets the 1% false-match cap (t*={ts}, c*={cs})")
    same = d.name.astype(str).isin(set(ids19)).to_numpy()
    flags = tier_flags(d19, s19, same, ts, cs)
    if not smoke:
        ref = pd.read_csv(K.ROOT / "results" / "round4" / "natural_isic2020" / "dedup.csv")
        ref["image_id"] = ref.image_id.astype(str)
        m = pd.DataFrame({"image_id": ids20, "d19": d19, "nearest": np.asarray(ids19)[j19], "same_name": same}).merge(
            ref, on="image_id", how="left", suffixes=("", "_r10"))
        bad = m[(m.hamming.isna()) | (m.d19 != m.hamming) | (m.same_name != m.same_name_r10)]
        if len(bad):
            raise SystemExit("d19 does not reproduce R10 dedup.csv; not writing calibration files:\n" + bad.head(5).to_string())
        if int((~flags["drop_d19_le8"]).sum()) != int((~ref.dropped).sum()):
            raise SystemExit("d19<=8 kept-count does not equal R10")
    tiers = pd.DataFrame({"image_id": ids20, "d19": d19, "s19": s19, "d0": d0, "s0": s0,
                          "same_name": same, "nearest_isic2019": np.asarray(ids19)[j19], **flags})
    tiers.to_csv(out / "dedup_tiers.csv", index=False)
    cal = {"n_isic2020": int(len(ids20)), "n_isic2019": int(len(ids19)), "tau_pred_hair_px": int(tau),
           "norm_max_abs_err": err, "cosine": "float32 dot product of stored L2-normalised DINOv2 features, chunked",
           "F": F, "G": G, "t_star": int(ts), "c_star": float(cs),
           "rule": "duplicate if d19 <= t* or s19 >= c* or the image name occurs in ISIC 2019",
           "n_dropped": {name: int(flags[col].sum()) for name, col in TIERS},
           "n_kept": {name: int((~flags[col]).sum()) for name, col in TIERS},
           "seconds": round(time.perf_counter() - t0, 1), "gpu": _gpu_name(), "commit": K.git_head(),
           "label_free": True}
    (out / "calibration.json").write_text(json.dumps(cal, indent=1))
    print(json.dumps({k: cal[k] for k in ("t_star", "c_star", "n_dropped", "n_kept", "seconds")}, indent=1), flush=True)
    return "calibrated"


def _envs(rn, c19, test, smoke):
    from wtss.utils import stable_int
    envs = {}
    tcols = test[rn.COLS].reset_index(drop=True)
    for s in rn.SEEDS[:1] if smoke else rn.SEEDS:
        g = c19.group.unique()
        val_g = {x for x in g if stable_int("natural_val", s, x) % 5 == 0}
        va, tr = c19[c19.group.isin(val_g)], c19[~c19.group.isin(val_g)]
        e = {"train_corr": tr, "val_clean": va, "val_groups": va, "train_all": tr}
        for k, dd in e.items():
            envs[("natural", s, 0, k)] = dd[rn.COLS].reset_index(drop=True)
        for k in ("test_corr", "test_rev", "clean"):
            envs[("natural", s, 0, k)] = tcols
    return envs


def run_fit(smoke, device, batch_size, workers, n_jobs) -> str:
    out = _out(smoke)
    if not (out / "dedup_tiers.csv").exists():
        raise SystemExit("missing dedup_tiers.csv — run --calibrate first (and commit it before the full fit)")
    if (out / "predictions.csv.gz").exists():
        K.log(step="fit", status="predictions exist")
        return "fit"
    rn, c19, d, cache, fdir19, _tau = N.load_all(smoke)
    import torch
    from wtss.experiments.real_traps import make_renderers
    from wtss.experiments.spec_traps import run_spec
    from wtss.synthetic import draw_generic_artifact
    from PIL import Image

    ins = lambda i, rgb, roi: np.asarray(draw_generic_artifact(Image.fromarray(rgb), roi, f"u|{i}"))
    dev = torch.device(device)
    fdir = _feat_dir(smoke)
    envs = _envs(rn, c19, d, smoke)
    pool = K.pool_of(envs, ("natural",))
    rend = make_renderers(cache, 518, [], ins)
    srcs, fname = _sources(fdir19)

    def extract():
        K.assemble_views(pool, VIEWS, rend, fdir, srcs, dev, fname=fname, batch_size=batch_size, workers=workers)

    def fit():
        run_spec(envs, cache, K.BACKBONE, out, fdir, [], arms=rn.ARMS, traps=("natural",), device=dev, insert_fn=ins,
                 insert_tag="_generic", folds=[0], save_val=True, batch_size=batch_size, workers=workers, n_jobs=n_jobs)

    return K.gpu_then_cpu(not K.views_ready(pool, VIEWS, fdir, fname), extract, fit)


def check_r10(out: Path) -> bool:
    """Predictions on the 25,340 R10 test images must match R10 within 1e-4. Returns False and does not continue."""
    ref_ids = pd.read_csv(K.ROOT / "results" / "round4" / "natural_isic2020" / "dedup.csv")
    ref_ids = set(ref_ids.loc[~ref_ids.dropped, "image_id"].astype(str))
    a = pd.read_csv(K.ROOT / "results" / "round4" / "natural_isic2020" / "predictions.csv.gz")
    b = pd.read_csv(out / "predictions.csv.gz")
    keys = ["image_id", "method", "seed", "fold", "env"]
    a = a[(a.env == "clean") & a.image_id.astype(str).isin(ref_ids)][keys + ["prob"]].copy()
    b = b[(b.env == "clean") & b.image_id.astype(str).isin(ref_ids)][keys + ["prob"]].copy()
    for df in (a, b):
        df["image_id"] = df.image_id.astype(str)
    m = a.merge(b, on=keys, how="outer", suffixes=("_r10", "_r5"), indicator=True)
    missing = int((m._merge != "both").sum())
    both = m[m._merge == "both"]
    delta = (both.prob_r10 - both.prob_r5).abs()
    by = both.assign(abs_d=delta).groupby("method").abs_d.max().to_dict() if len(both) else {}
    rep = {"n_r10_rows": int(len(a)), "n_matched": int(len(both)), "n_unmatched": missing,
           "max_abs": None if not len(both) else float(delta.max()),
           "max_abs_by_method": {k: float(v) for k, v in by.items()},
           "threshold": 1e-4, "pass": bool(missing == 0 and len(both) and float(delta.max()) <= 1e-4)}
    (out / "match_check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)
    if not rep["pass"]:
        print("STOP: round-5 predictions on the R10 test images do not match results/round4/natural_isic2020/predictions.csv.gz", flush=True)
    return rep["pass"]


def _boot_seed(tier, subset, arm, ref):
    if tier == "d19_le8":
        return _r10_boot_seed(subset, arm, ref)
    return 20262000 + [t[0] for t in TIERS].index(tier) * 100 + SUBSETS.index(subset) * 10 + list(PAIRS).index((arm, ref))


def _delta(job):
    q, a1, a0, seed, tag = job
    from wtss import stats_crossed as X
    point, arr = X.replicates(X._pair_terms(q, a1, a0, "clean", "seed"), N.N_BOOT, seed)
    s = X._summary(point, arr, {"method_a": a1, "method_b": a0, "env": "clean",
                                "estimand": "mean_seed_specific_paired_auc_delta"})
    p_lt = float((np.asarray(arr) >= 0).mean()) if len(arr) else float("nan")
    s["p_one_sided_lt"] = float(min(1.0, max(p_lt, 1.0 / max(len(arr), 1))))
    s["boot_seed"] = int(seed)
    s.update(tag)
    return s


def _op(job):
    pv, arm, ref, seed, tier = job
    rows = N.op_crossed(pv, arm, ref, seed)
    for r in rows:
        r["tier"] = tier
    return rows


def _hard_mask(df):
    y, a = df.y.to_numpy(), df.artifact_present.to_numpy()
    return pd.Series(((y == 1) & (a == 0)) | ((y == 0) & (a == 1)), index=df.index)


def run_analysis(smoke: bool, started: float | None = None) -> None:
    from wtss.stats import safe_auc
    out = _out(smoke)
    cal = json.loads((out / "calibration.json").read_text())
    tiers = pd.read_csv(out / "dedup_tiers.csv")
    tiers["image_id"] = tiers.image_id.astype(str)
    rn, c19, d, _cache, _fdir, tau = N.load_all(smoke)
    d = d.copy()
    d["image_id"] = d.image_id.astype(str)
    d = d.merge(tiers, on="image_id", how="inner")
    d["hair_group"] = hair_groups(d.hair_px, d.r, tau)
    p = pd.read_csv(out / "predictions.csv.gz")
    p["image_id"] = p.image_id.astype(str)
    count_rows = []
    for name, col in TIERS:
        sub = d[~d[col].astype(bool)]
        for yv, g in sub.groupby("y"):
            count_rows.append({"tier": name, "y": int(yv), "n": int(len(g)),
                               "H0": int((g.hair_group == "H0").sum()), "Hin": int((g.hair_group == "Hin").sum()),
                               "Hout": int((g.hair_group == "Hout").sum()), "ambiguous": int((g.hair_group == "ambiguous").sum()),
                               "in_lesion_hair": int(g.a.sum())})
    counts = pd.DataFrame(count_rows)
    counts.to_csv(out / "counts_by_tier.csv", index=False)
    metric_rows, boot_jobs, op_jobs = [], [], []
    cols = ["seed", "method", "env", "image_id", "y", "prob"]
    for name, col in TIERS:
        keep = set(d.loc[~d[col].astype(bool), "image_id"])
        te = p[(p.env == "clean") & p.image_id.isin(keep)].copy()
        for (m, s), q in te.groupby(["method", "seed"]):
            y, pr, ai = q.y.to_numpy(), q.prob.to_numpy(), q.artifact_present.to_numpy()
            hard = ((y == 1) & (ai == 0)) | ((y == 0) & (ai == 1))
            metric_rows.append({"tier": name, "method": m, "seed": int(s), "auc": safe_auc(y, pr),
                                "cross_group": N.cross_group(y, ai, pr),
                                "auc_in_lesion_hair": safe_auc(y[ai == 1], pr[ai == 1]),
                                "auc_hard": safe_auc(y[hard], pr[hard]), "auc_easy": safe_auc(y[~hard], pr[~hard])})
        hard_s = _hard_mask(te)
        for subset, sel in (("hard", hard_s), ("easy", ~hard_s), ("all", pd.Series(True, index=te.index))):
            for arm, ref in PAIRS:
                if {arm, ref} <= set(te.method):
                    q = te.loc[sel & te.method.isin([arm, ref]), cols]
                    if q[q.method == arm].y.nunique() < 2:
                        continue
                    boot_jobs.append((q, arm, ref, _boot_seed(name, subset, arm, ref), {"tier": name, "subset": subset, "arm": arm, "ref": ref}))
        pv = p[p.env.isin(["clean", "val_groups"]) & (p.image_id.isin(keep) | (p.env != "clean"))].copy()
        for k, (arm, ref) in enumerate(OP_PAIRS):
            if {arm, ref} <= set(pv.method):
                seed = _r10_op_seed(arm, ref) if name == "d19_le8" else 20263000 + [t[0] for t in TIERS].index(name) * 10 + k
                op_jobs.append((pv, arm, ref, seed, name))
    met = pd.DataFrame(metric_rows)
    met.to_csv(out / "tier_metrics_per_seed.csv", index=False)
    mm = met.groupby(["tier", "method"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    mm.to_csv(out / "tier_metrics.csv", index=False)
    K.log(step="bootstrap", jobs=len(boot_jobs))
    boot = _as_boot(K.parallel_map(_delta, boot_jobs))
    boot.to_csv(out / "tier_boot.csv", index=False)
    K.log(step="operating_points", jobs=len(op_jobs))
    ops = pd.DataFrame([row for part in K.parallel_map(_op, op_jobs) for row in part])
    ops.to_csv(out / "tier_operating_points.csv", index=False)
    r13 = _r13(p, d, boot_jobs)
    r13.to_csv(out / "r13_boot.csv", index=False)
    if not smoke:
        _check_r10_bootstrap(boot, ops, out)
    write_summary(out, cal, counts, mm, boot, ops, r13, d, started)
    return boot


def _as_boot(res):
    rows = []
    for s in res:
        rows.append({"tier": s["tier"], "subset": s["subset"], "arm": s["arm"], "ref": s["ref"],
                     "estimate": s["seed_delta_mean"], "ci95_lo": s["ci95_lo"], "ci95_hi": s["ci95_hi"],
                     "p_boot_two_sided": s["p_boot_two_sided"], "p_one_sided_lt": s["p_one_sided_lt"],
                     "boot_seed": s.get("boot_seed"), "n_boot": N.N_BOOT, "estimator": s.get("estimator", "")})
    return pd.DataFrame(rows)


def _pair_frame(te, pos_g, neg_g, cols):
    sel = ((te.y == 1) & (te.hair_group == pos_g)) | ((te.y == 0) & (te.hair_group == neg_g))
    return te.loc[sel, cols]


def _r13(p, d, _ignored):
    """Strict and descriptive pairs on the calibrated test set. H13 is strict-hard mask − ERM."""
    keep = d[~d.drop_calibrated.astype(bool)]
    te = p[(p.env == "clean") & p.image_id.isin(set(keep.image_id))].copy()
    te["hair_group"] = te.image_id.map(keep.set_index("image_id").hair_group)
    cols = ["seed", "method", "env", "image_id", "y", "prob"]
    specs = [("strict_hard", "H0", "Hin", "mask", "erm"), ("strict_easy", "Hin", "H0", "mask", "erm"),
             ("strict_hard", "H0", "Hin", "balanced", "mask"), ("hout_mel_vs_h0_ben", "Hout", "H0", "mask", "erm"),
             ("h0_mel_vs_hout_ben", "H0", "Hout", "mask", "erm")]
    jobs = []
    for k, (name, pos, neg, arm, ref) in enumerate(specs):
        q = _pair_frame(te, pos, neg, cols)
        q = q[q.method.isin([arm, ref])]
        if q[q.method == arm].y.nunique() < 2:
            continue
        jobs.append((q, arm, ref, 20264000 + k, {"tier": "calibrated", "subset": name, "arm": arm, "ref": ref}))
    if not jobs:
        return pd.DataFrame()
    K.log(step="r13", jobs=len(jobs))
    return _as_boot(K.parallel_map(_delta, jobs))


def _check_r10_bootstrap(boot, ops, out: Path):
    """The d19<=8 tier uses R10's seeds, so its intervals should reproduce R10 when the probabilities match."""
    ref = pd.read_csv(K.ROOT / "results" / "round4" / "natural_isic2020" / "natural_boot.csv")
    b = boot[boot.tier == "d19_le8"].merge(ref, on=["subset", "arm", "ref"], suffixes=("", "_r10"))
    ref_o = pd.read_csv(K.ROOT / "results" / "round4" / "natural_isic2020" / "operating_points.csv")
    o = ops[ops.tier == "d19_le8"].merge(ref_o, on=["arm", "ref", "op", "metric"], suffixes=("", "_r10"))
    rep = {"boot_max_abs": None if not len(b) else float((b.estimate - b.estimate_r10).abs().max()),
           "op_max_abs": None if not len(o) else float((o.delta - o.delta_r10).abs().max()),
           "n_boot": int(len(b)), "n_op": int(len(o))}
    (out / "r10_reproduce.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)
    if (rep["boot_max_abs"] or 0) > 1e-3 or (rep["op_max_abs"] or 0) > 1e-3:
        raise SystemExit("d19<=8 analysis does not reproduce R10 (max abs difference above 1e-3)")


def _ci(r) -> str:
    return f"{r['estimate']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]"


def _row(boot, **kw):
    q = boot
    for k, v in kw.items():
        q = q[q[k] == v]
    if len(q) != 1:
        raise SystemExit(f"expected one bootstrap row for {kw}, found {len(q)}")
    return q.iloc[0]


def write_summary(out, cal, counts, mm, boot, ops, r13, d, started=None):
    elapsed = None if started is None else time.time() - started
    h12 = _row(boot, tier="calibrated", subset="hard", arm="mask", ref="erm")
    label = h12_label(h12.estimate, h12.ci95_lo, h12.ci95_hi)
    cal_d = d[~d.drop_calibrated.astype(bool)]
    n_mel_h0 = int(((cal_d.y == 1) & (cal_d.hair_group == "H0")).sum())
    n_ben_hin = int(((cal_d.y == 0) & (cal_d.hair_group == "Hin")).sum())
    gate = h13_gate(n_mel_h0, n_ben_hin)
    h13 = _row(r13, subset="strict_hard", arm="mask", ref="erm") if len(r13) else None
    p_holm = None
    if h13 is not None and gate:
        p_holm = K.holm([h12.p_one_sided_lt, h13.p_one_sided_lt])
    match = json.loads((out / "match_check.json").read_text()) if (out / "match_check.json").exists() else {"pass": "smoke", "max_abs": None}
    L = ["# Round 5 — robustness of the ISIC 2020 result (R12, R13)", "",
         f"Environment: {json.dumps(K.env_info())}", "",
         "R10 is unchanged. R12 and R13 are sensitivity analyses of that one set of predictions "
         "(every ISIC 2020 image with a lesion mask).", "",
         f"Calibration (label-free, committed before fitting): t* = {cal['t_star']}, c* = {cal['c_star']}. "
         f"F(t*) = {cal['F'][str(cal['t_star'])]:.4f}, G(c*) = {cal['G'][format(cal['c_star'], '.3f')]:.4f}. "
         "Duplicate if d19 ≤ t* or s19 ≥ c* or the image name is in ISIC 2019.", ""]
    L += ["Kept images by tier:", "",
          K.md_table(pd.DataFrame([{"tier": k, "dropped": cal["n_dropped"][k], "kept": cal["n_kept"][k]} for k, _ in TIERS])), ""]
    L += ["Counts of kept images by label and hair group:", "", K.md_table(counts), "",
          f"Prediction match against R10 on the 25,340 test images: pass = {match.get('pass')}, "
          f"max |Δp| = {match.get('max_abs')}, unmatched rows = {match.get('n_unmatched', 'n/a')}.", ""]
    show = boot.assign(value=[_ci(r) for r in boot.to_dict("records")])
    L += ["Crossed bootstrap by tier (hard / easy / all; mask−ERM, balanced−mask, mte_balanced−mask):", "",
          K.md_table(show[["tier", "subset", "arm", "ref", "value"]]), ""]
    op_s = ops[(ops.metric == "sens_conflict")].copy()
    op_s["value"] = [f"{r.delta:+.3f} [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}]" for r in op_s.itertuples()]
    L += ["OP1–OP5 sensitivity among melanomas without in-lesion hair (thresholds from ISIC 2019 validation only):", "",
          K.md_table(op_s[["tier", "arm", "ref", "op", "value"]]), "",
          f"**H12** calibrated hard-pair AUROC mask − ERM < 0: {_ci(h12)} -> **{label}**.", ""]
    if h13 is None:
        L += ["**H13** not computed (a strict-hard pair had a single class).", ""]
    elif not gate:
        L += [f"**H13** strict hard-pair AUROC mask − ERM: {_ci(h13)}. Gate not met "
              f"({n_mel_h0} H0 melanomas, {n_ben_hin} Hin benign lesions; need ≥ 30 and ≥ 100) -> **descriptive only**. "
              "Holm over {H12, H13} is not applied.", ""]
    else:
        h13_plain = "SUPPORTED" if h13.ci95_hi < 0 else "NOT SUPPORTED"
        h13_holm = "SUPPORTED" if (h13.estimate < 0 and p_holm[1] < 0.05) else "NOT SUPPORTED"
        h12_holm = "rejects 0 at 0.05" if p_holm[0] < 0.05 else "does not reject 0 at 0.05"
        L += [f"**H13** strict hard-pair AUROC mask − ERM < 0: {_ci(h13)}.",
              f"Gate met ({n_mel_h0} H0 melanomas, {n_ben_hin} Hin benign lesions).",
              f"Without Holm: **{h13_plain}**.",
              f"With Holm over {{H12, H13}} (one-sided bootstrap p, H12 p={h12.p_one_sided_lt:.4f} -> {p_holm[0]:.4f}, "
              f"H13 p={h13.p_one_sided_lt:.4f} -> {p_holm[1]:.4f}): H12 {h12_holm}; H13 **{h13_holm}**.", ""]
    if len(r13):
        extra = r13.assign(value=[_ci(r) for r in r13.to_dict("records")])
        L += ["R13 descriptive pairs on the calibrated test set (strict easy; balanced−mask on strict hard; "
              "Hout melanomas or Hout benign lesions against H0):", "",
              K.md_table(extra[["subset", "arm", "ref", "value"]]), ""]
    timing = {"calibrate_seconds": cal.get("seconds"), "gpu": cal.get("gpu")}
    (out / "timing.json").write_text(json.dumps({**timing, "commit": K.git_head()}, indent=1))
    repro = ""
    if (out / "r10_reproduce.json").exists():
        r = json.loads((out / "r10_reproduce.json").read_text())
        repro = f" d19≤8 vs R10 bootstrap max |Δ| = {r.get('boot_max_abs')}, operating-point max |Δ| = {r.get('op_max_abs')}."
    L += [f"Runtime: calibration {cal.get('seconds')} s; fit and analysis {None if elapsed is None else round(elapsed, 1)} s. "
          f"GPU: {cal.get('gpu')}. Commit: {K.git_head()}.{repro}", ""]
    text = "\n".join(L)
    (out / "SUMMARY.md").write_text(text)
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--n_jobs", type=int, default=0, help="head-fitting processes (0 = up to 16)")
    a = ap.parse_args()
    a.n_jobs = min(a.n_jobs or K.n_cpus(), K.n_cpus())
    t0 = time.time()
    extracting = os.environ.get("WTSS_EXTRACT_ONLY") == "1"
    # The re-exec child of calibration must not fall through into fitting: the smoke fit has not written
    # dedup_tiers.csv yet, and this child only has to encode the erm view.
    if extracting and os.environ.get("WTSS_ROUND5_STAGE") == "calibrate":
        run_calibrate(a.smoke, "cpu" if a.smoke else a.device, a.batch_size, a.workers)
        return
    if a.calibrate:
        # Smoke calibration stays on CPU so a following fit in this process can still fork.
        dev = "cpu" if a.smoke else a.device
        if run_calibrate(a.smoke, dev, a.batch_size, a.workers) == "extracted":
            return
        return
    if a.smoke and not extracting and not (_out(True) / "dedup_tiers.csv").exists():
        if run_calibrate(True, "cpu", a.batch_size, a.workers) == "extracted":
            return
    status = run_fit(a.smoke, a.device, a.batch_size, a.workers, a.n_jobs)
    if status == "extracted" or extracting:
        return
    if not a.smoke and not check_r10(_out(False)):
        raise SystemExit(2)
    run_analysis(a.smoke, t0)


if __name__ == "__main__":
    main()
