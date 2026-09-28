"""R8 — masking implementations on the four real traps (docs/PREREGISTRATION_ROUND4.md).

  python scripts/round4/masking_variants.py --cohort ovary --smoke     # quick check, results/round4/_smoke/
  python scripts/round4/masking_variants.py --cohort ovary              # full run + analysis
  python scripts/round4/masking_variants.py --summary                   # Holm over the 16 primary tests
Outputs: results/round4/masking_variants/<cohort>/{counts.csv, repro_check.csv, crossovers.csv, contrast_vs_mask.csv,
metrics_per_seed.csv}; results/round4/masking_variants/SUMMARY.{csv,md}.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

VARIANTS = ("mask_black", "mask_blur", "crop_box", "crop_mask")
FULL, PARTIAL = ("mask_black", "crop_mask"), ("mask_blur", "crop_box")
VIEWS = ("mask",) + VARIANTS
N_BOOT = 10000


def variant_renderers(cache, size: int = 518) -> dict:
    from wtss.ops import apply_roi_mask
    from wtss.utils import MEAN_RGB
    mean = np.asarray(MEAN_RGB, np.uint8)

    def fin(arr):
        img = Image.fromarray(np.ascontiguousarray(arr))
        return img if img.size == (size, size) else img.resize((size, size), Image.BICUBIC)

    def masked(rgb, roi, fill=MEAN_RGB):
        return np.asarray(apply_roi_mask(Image.fromarray(rgb), roi, fill=fill))

    def crop(arr, roi):
        ys, xs = np.nonzero(np.asarray(roi) > 0)
        if len(ys) == 0:  # empty ROI: full image (counted in counts.csv)
            return arr
        H, W = arr.shape[:2]
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        dh, dw = 0.1 * (y1 - y0), 0.1 * (x1 - x0)
        y0, y1 = max(0, int(np.floor(y0 - dh))), min(H, int(np.ceil(y1 + dh)))
        x0, x1 = max(0, int(np.floor(x0 - dw))), min(W, int(np.ceil(x1 + dw)))
        sub = arr[y0:y1, x0:x1]
        h, w = sub.shape[:2]
        s = max(h, w)
        sq = np.empty((s, s, 3), np.uint8)
        sq[:] = mean
        oy, ox = (s - h) // 2, (s - w) // 2
        sq[oy:oy + h, ox:ox + w] = sub
        return np.asarray(Image.fromarray(sq).resize((size, size), Image.BICUBIC))

    def mask_black(i):
        rgb, roi, _ = cache.get(i)
        return fin(masked(rgb, roi, (0, 0, 0)))

    def mask_blur(i):
        rgb, roi, _ = cache.get(i)
        blur = cv2.GaussianBlur(np.ascontiguousarray(rgb), (0, 0), 16 * rgb.shape[0] / 518)
        return fin(np.where(np.asarray(roi)[..., None] > 0, rgb, blur).astype(np.uint8))

    def crop_box(i):
        rgb, roi, _ = cache.get(i)
        return fin(crop(rgb, roi))

    def crop_mask(i):
        rgb, roi, _ = cache.get(i)
        return fin(crop(masked(rgb, roi), roi))

    return {"mask_black": mask_black, "mask_blur": mask_blur, "crop_box": crop_box, "crop_mask": crop_mask}


def examples(cache, ids, rend_all, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for i in ids:
        tiles = [np.asarray(rend_all[v](i).resize((256, 256))) for v in ("erm",) + VIEWS]
        Image.fromarray(np.concatenate(tiles, 1)).save(out / f"{i}.png")
    (out / "ORDER.txt").write_text("left to right: " + ", ".join(("erm",) + VIEWS) + "\n")


def _boot(job):
    kind, args, seed = job
    from wtss import stats_crossed as X
    if kind == "cross":
        return X.difference_of_deltas(*args, "test_rev", N_BOOT, seed)
    return X.hierarchical_paired_bootstrap(*args, N_BOOT, seed)


def analyse(out: Path, cohort: str, smoke: bool):
    P = pd.read_csv(out / "predictions.csv.gz")
    per = pd.read_csv(out / "metrics_per_seed.csv")
    new = per.groupby(["trap", "method", "env"]).auc.mean()
    ref = pd.read_csv(K.REF_ROOT / K.REF[cohort] / "metrics_per_seed.csv").groupby(["trap", "method", "env"]).auc.mean()
    rows = []
    for t in ("trapA", "trapB"):
        for m in ("erm", "mask"):
            for e in ("clean", "test_rev"):
                a, b = new.get((t, m, e), np.nan), ref.get((t, m, e), np.nan)
                rows.append({"trap": t, "method": m, "env": e, "auc_new": a, "auc_stored": b, "abs_diff": abs(a - b)})
    rc = pd.DataFrame(rows)
    rc.to_csv(out / "repro_check.csv", index=False)
    worst = float(rc.abs_diff.max())
    K.log(cohort=cohort, repro_max_abs_diff=worst)
    if worst > 0.03 and not smoke:
        raise SystemExit(f"[R8] {cohort}: ERM/mask differ from the stored run by {worst:.3f} > 0.03 "
                         f"(see {out / 'repro_check.csv'}). Stopping as pre-registered.")
    P = P[["trap", "seed", "method", "env", "image_id", "y", "prob"]]
    sub = lambda t, e, ms: P[(P.trap == t) & (P.env == e) & P.method.isin(ms)]
    jobs, keys = [], []
    for k, v in enumerate(VIEWS):
        jobs.append(("cross", (sub("trapB", "test_rev", (v, "erm")), sub("trapA", "test_rev", (v, "erm")), v, "erm"), 20261001 + k))
        keys.append((v, "crossover"))
        for t in ("trapA", "trapB"):
            for e in ("test_rev", "clean"):
                jobs.append(("pair", (sub(t, e, (v, "erm")), v, "erm", e), 20261101 + 10 * k + (t == "trapB") * 2 + (e == "clean")))
                keys.append((v, f"{t}_{'delta_rev' if e == 'test_rev' else 'delta_clean'}"))
        if v != "mask":
            jobs.append(("cross", (sub("trapB", "test_rev", ("mask", v)), sub("trapA", "test_rev", ("mask", v)), "mask", v), 20261201 + k))
            keys.append((v, "C_mask_minus_C_v"))
    res = K.parallel_map(_boot, jobs)
    rows = [{"cohort": cohort, "view": v, "quantity": q, "estimate": r["seed_delta_mean"], "ci95_lo": r["ci95_lo"],
             "ci95_hi": r["ci95_hi"], "p_boot_two_sided": r["p_boot_two_sided"], "boot_seed": j[2],
             "estimator": r.get("estimator", "")} for (v, q), r, j in zip(keys, res, jobs)]
    d = pd.DataFrame(rows)
    d[d.quantity != "C_mask_minus_C_v"].to_csv(out / "crossovers.csv", index=False)
    d[d.quantity == "C_mask_minus_C_v"].to_csv(out / "contrast_vs_mask.csv", index=False)
    print(d[["view", "quantity", "estimate", "ci95_lo", "ci95_hi"]].round(3).to_string(), flush=True)


def summary(root: Path):
    frames = [pd.read_csv(root / c / "crossovers.csv") for c in K.COHORTS if (root / c / "crossovers.csv").exists()]
    cons = [pd.read_csv(root / c / "contrast_vs_mask.csv") for c in K.COHORTS if (root / c / "contrast_vs_mask.csv").exists()]
    if not frames:
        raise SystemExit("no cohort finished yet")
    d = pd.concat(frames + cons, ignore_index=True)
    prim = d[(d.quantity == "crossover") & d.view.isin(VARIANTS)].copy()
    prim["p_holm16"] = K.holm(prim.p_boot_two_sided.to_numpy()) if len(prim) == 16 else np.nan
    prim["supported"] = (prim.estimate > 0) & (prim.p_holm16 < 0.05)
    d = d.merge(prim[["cohort", "view", "quantity", "p_holm16", "supported"]], how="left", on=["cohort", "view", "quantity"])
    d.to_csv(root / "SUMMARY.csv", index=False)
    n_full = int(prim[prim.view.isin(FULL)].supported.sum())
    label = ("implementation-independent" if n_full == 8 else "largely implementation-independent" if n_full >= 6
             else "implementation-dependent") if len(prim) == 16 else "incomplete (fewer than 16 primary tests)"
    L = ["# R8 — masking implementations (DINOv2 ViT-B/14 @518, real traps)", "",
         f"Environment: {json.dumps(K.env_info())}", "",
         "Primary: crossover C_v = [rev(v) - rev(ERM)]_TrapB - [rev(v) - rev(ERM)]_TrapA, crossed bootstrap, Holm over 16.", ""]
    t = prim.assign(cohort=prim.cohort.map(K.LABEL), C_v=[K.ci(r) for r in prim.to_dict("records")])
    L += [K.md_table(t[["cohort", "view", "C_v", "p_holm16", "supported"]]), ""]
    L += [f"**H8a** (full-removal views, 8 tests): {n_full}/8 positive and Holm-significant -> decision: **{label}**.", ""]
    ref = d[(d.quantity == "crossover") & (d.view == "mask")]
    if len(ref):
        L += ["Reference (mean-colour fill, re-fitted here):", "",
              K.md_table(ref.assign(cohort=ref.cohort.map(K.LABEL), C_mask=[K.ci(r) for r in ref.to_dict("records")])[["cohort", "C_mask"]]), ""]
    c2 = d[d.quantity == "C_mask_minus_C_v"]
    if len(c2):
        c2 = c2.assign(cohort=c2.cohort.map(K.LABEL), contrast=[K.ci(r) for r in c2.to_dict("records")])
        L += ["**H8b** secondary contrast C_mask - C_v (partial-removal views expected > 0):", "",
              K.md_table(c2[["cohort", "view", "contrast"]]), ""]
        hb = prim[prim.view.isin(PARTIAL)]
        L += [f"H8b crossover part: {int(hb.supported.sum())}/{len(hb)} partial-removal crossovers positive and Holm-significant.", ""]
    sec = d[d.quantity.isin(["trapA_delta_rev", "trapB_delta_rev", "trapA_delta_clean", "trapB_delta_clean"])]
    sec = sec.assign(cohort=sec.cohort.map(K.LABEL), value=[K.ci(r) for r in sec.to_dict("records")])
    L += ["Secondary (descriptive): view minus ERM per trap, reversed (delta_rev) and clean (delta_clean) AUROC.", "",
          K.md_table(sec.pivot_table(index=["cohort", "view"], columns="quantity", values="value", aggfunc="first").reset_index()), ""]
    (root / "SUMMARY.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=K.COHORTS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)  # 24 GB GPU; 64 left most of the card idle
    ap.add_argument("--n_jobs", type=int, default=0, help="head-fitting processes (0 = up to 16)")
    a = ap.parse_args()
    a.n_jobs = min(a.n_jobs or K.n_cpus(), K.n_cpus())
    root = K.OUT / ("_smoke" if a.smoke else "") / "masking_variants"
    if a.summary:
        return summary(root)
    import torch
    from wtss.data.isic2019_spec import build_spec_envs, matched_pool
    from wtss.experiments.real_traps import make_renderers
    from wtss.experiments.spec_traps import run_spec
    dev = torch.device(a.device)
    co = K.trap_cohort(a.cohort)
    c = co["c"]
    if a.smoke:
        c = K.smoke_subset(c, ("trapA", "trapB"))
    envs = build_spec_envs(c, group_col=co["group_col"])
    if a.smoke:
        envs = K.smoke_envs(envs)
    out = K.OUT / ("_smoke" if a.smoke else "") / "masking_variants" / a.cohort
    out.mkdir(parents=True, exist_ok=True)
    pool = K.pool_of(envs, ("trapA", "trapB"))
    cache = co["cache"]
    empty = int(sum(not np.asarray(cache.roi[cache.index[i]]).any() for i in pool))
    rows = []
    for t in ("trapA", "trapB"):
        cells = matched_pool(c, t).groupby(["a", "y"]).size()
        rows.append({"trap": t, **{f"A{x}_Y{y}": int(cells.get((x, y), 0)) for x in (0, 1) for y in (0, 1)}})
    cnt = pd.DataFrame(rows).assign(pool_images=len(pool), empty_roi_images=empty)
    cnt.to_csv(out / "counts.csv", index=False)
    print(cnt.to_string(), flush=True)
    rend = make_renderers(cache, 518, co["donors"])
    rend.update(variant_renderers(cache))
    ex_ids = [i for i in pool if np.asarray(cache.roi[cache.index[i]]).any()][:3]
    if a.cohort in ("capsule", "ovary"):  # never write thyroid or ISIC images (licences)
        examples(cache, ex_ids, rend, K.OUT / "_local_examples" / f"masking_{a.cohort}")
    fdir = K.FEAT / ("_smoke" if a.smoke else "") / "masking_variants" / a.cohort / K.BDIR
    srcs = {v: [(d / f"{v}.npz", lambda i: i) for d in co["old"]] for v in ("erm", "mask")}
    views = ("erm",) + VIEWS
    if not (out / "predictions.csv.gz").exists():
        def extract():
            K.assemble_views(pool, views, rend, fdir, srcs, dev, batch_size=a.batch_size, workers=a.workers)

        def fit():
            run_spec(envs, cache, K.BACKBONE, out, fdir, co["donors"], arms=("erm", "mask"), device=dev,
                     batch_size=a.batch_size, workers=a.workers, n_jobs=a.n_jobs, folds=[0] if a.smoke else range(5),
                     extra_ctx={"extra_views": VARIANTS, "extra_renderers": variant_renderers(cache),
                                "view_arms": [(v, v, False) for v in VARIANTS]})

        if K.gpu_then_cpu(not K.views_ready(pool, views, fdir), extract, fit) == "extracted":
            return
    analyse(out, a.cohort, a.smoke)


if __name__ == "__main__":
    main()
