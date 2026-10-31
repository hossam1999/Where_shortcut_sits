"""Grouped-bootstrap sensitivity analysis: are the intervals still valid when images of one lesion / group share a weight?

The crossed seed x image bootstrap (wtss.stats_crossed) gives every test IMAGE its own Poisson weight. Several images of
one lesion (or near-duplicates) are then resampled independently, which can make intervals too narrow. This script
recomputes each headline result twice from the saved predictions, with the same replicate seed:
  image  - one weight per image (the reported method; reproduces the reported intervals)
  group  - one weight per group: the groups used to build the folds (ISIC 2019: lesion_id U pHash<=8, joined across
           both traps; thyroid / ovary / capsule: the cohort's near-duplicate group; ISIC 2018 sweep: pHash group).
Nothing is refitted; only the resampling unit changes.

  PYTHONPATH=src WTSS_BOOTSTRAP=crossed python scripts/analysis/grouped_bootstrap.py [--n-boot 10000]
  -> results/grouped_bootstrap/grouped_vs_image.csv and SUMMARY.md

p values are two-sided with the conservative formula p = 2(k+1)/(B+1), k = replicates on the far side of zero.
Holm is applied over the four registered DINOv2 crossovers (skin, thyroid, ovary, capsule).
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from wtss import paths
from wtss import stats as st
from wtss import stats_crossed as X
from wtss.data.isic2019_spec import load_spec_cohort, matched_pool, pool_groups

ROOT = Path(__file__).resolve().parents[2]
R = paths.RESULTS

# (label, run directory, cohort key, registered primary test?)
CROSSOVERS = [
    ("Skin hair/ruler, DINOv2", "spec_e13/dino518_spec", "isic2019", True),
    ("Thyroid calipers, DINOv2", "thyroid/dino518_main", "thyroid", True),
    ("Ovary calipers, DINOv2", "ovary/dino518_main", "ovary", True),
    ("Capsule debris, DINOv2", "capsule/dino518_main", "capsule", True),
    ("Thyroid calipers, MedSigLIP", "thyroid/medsiglip448_main", "thyroid", False),
    ("Ovary calipers, MedSigLIP", "ovary/medsiglip448_universal", "ovary", False),
    ("Capsule debris, MedSigLIP", "capsule/medsiglip448_main", "capsule", False),
    ("Thyroid calipers, ConvNeXt", "thyroid/convnext384_universal", "thyroid", False),
    ("Capsule debris, ConvNeXt", "capsule/convnext384_universal", "capsule", False),
    ("Skin hair/ruler, DermLIP", "spec_e13/dermlip224_spec", "isic2019", False),
]
SWEEPS = [
    ("Pasted ruler sweep, DINOv2 518 (location effect)", "synthetic/isic2018/dino518_ruler_fixed_corr_main", "isic2018"),
    ("Pasted ruler sweep, DINOv2 224 (location effect)", "synthetic/isic2018/dino224_ruler_fixed_corr_main", "isic2018"),
]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _union(pairs_by_trap) -> dict:
    """Join group labels from several traps into one image -> component id (union-find over image ids)."""
    parent = {}

    def f(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ids, groups in pairs_by_trap:
        first = {}
        for i, g in zip(ids, groups):
            f(i)
            if g in first:
                a, b = f(first[g]), f(i)
                if a != b:
                    parent[b] = a
            else:
                first[g] = i
    return {i: f"g{f(i)}" for i in parent}


def group_map(key: str) -> dict:
    if key == "isic2019":
        c = load_spec_cohort()
        per_trap = []
        for trap in ("trapA", "trapB"):
            pool = matched_pool(c, trap)
            per_trap.append((pool.image_id.astype(str).tolist(), pool_groups(pool, 8).tolist()))
        return _union(per_trap)
    if key in ("thyroid", "ovary", "capsule"):
        m = _load(ROOT / "scripts" / "run_thyroid_traps.py", "rtt")
        c = {"thyroid": m.cohort, "ovary": m.ovary_cohort, "capsule": m.capsule_cohort}[key]()
        return {str(i): f"g{g}" for i, g in zip(c.image_id, c.group)}
    if key == "isic2018":
        m = _load(ROOT / "scripts" / "run_synthetic.py", "rs")
        cohort, *_ = m.cohort_and_placements("isic2018", 518, "ruler_fixed")
        return {str(i): f"g{g}" for i, g in zip(cohort.df.image_id, cohort.df.group)}
    raise ValueError(key)


def predictions(run: str):
    """results/<run>/predictions.csv.gz, else the archived copy under results/rerun_2026-09-28/<run>/ (the archive
    restores the regenerated runs there; the reported numbers come from them)."""
    for f in (R / run / "predictions.csv.gz", R / "rerun_2026-09-28" / run / "predictions.csv.gz"):
        if f.exists():
            return f
    return None


def regroup(terms, gmap):
    missing = {i for t in terms for i in t[2] if i not in gmap}
    if missing:
        raise SystemExit(f"{len(missing)} test images have no group (e.g. {sorted(missing)[:3]})")
    return [(c, coef, [gmap[i] for i in ids], y, s) for c, coef, ids, y, s in terms]


def p_conservative(arr: np.ndarray) -> float:
    k = int(min((arr <= 0).sum(), (arr >= 0).sum()))
    return min(1.0, 2 * (k + 1) / (len(arr) + 1))


def summarise(label, estimand, terms, gmap, n_boot, seed, primary):
    point, a_img = X.replicates(terms, n_boot, seed)
    _, a_grp = X.replicates(regroup(terms, gmap), n_boot, seed)
    est = float(np.mean(list(point.values())))
    li, hi = st._ci(a_img)
    lg, hg = st._ci(a_grp)
    ids = {i for t in terms for i in t[2]}
    return {"result": label, "estimand": estimand, "primary": primary, "estimate": est,
            "image_lo": li, "image_hi": hi, "group_lo": lg, "group_hi": hg,
            "width_ratio_group_vs_image": (hg - lg) / (hi - li),
            "p_image": p_conservative(a_img), "p_group": p_conservative(a_grp),
            "group_ci_excludes_0": bool(lg > 0 or hg < 0),
            "n_test_images": len(ids), "n_test_groups": len({gmap[i] for i in ids})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    maps, rows = {}, []
    for label, run, key, primary in CROSSOVERS:
        f = predictions(run)
        if f is None:
            print(f"[skip] {label}: no predictions.csv.gz for {run}", flush=True)
            continue
        maps.setdefault(key, group_map(key))
        p = pd.read_csv(f)
        pA, pB = p[p.trap == "trapA"], p[p.trap == "trapB"]
        terms = X._pair_terms(pB, "mask", "erm", "test_rev", "seed", +1.0) + \
            X._pair_terms(pA, "mask", "erm", "test_rev", "seed", -1.0)
        rows.append({**summarise(label, "crossover (B - A), masked - standard, reversed test", terms, maps[key],
                                 a.n_boot, 20260927, primary), "predictions": str(f.relative_to(R))})
        print(rows[-1], flush=True)
    for label, run, key in SWEEPS:
        f = predictions(run)
        if f is None:
            print(f"[skip] {label}: no predictions.csv.gz for {run}", flush=True)
            continue
        maps.setdefault(key, group_map(key))
        p = pd.read_csv(f)
        p = p[p.env == "test_rev"]
        ov = p.overlap.astype(float)
        terms = X._pair_terms(p[np.isclose(ov, 0.0)], "mask", "erm", "test_rev", "seed", +1.0) + \
            X._pair_terms(p[np.isclose(ov, 1.0)], "mask", "erm", "test_rev", "seed", -1.0)
        rows.append({**summarise(label, "location effect, overlap 0% - 100%", terms, maps[key], a.n_boot, 20260918, False),
                     "predictions": str(f.relative_to(R))})
        print(rows[-1], flush=True)
    if not rows:
        raise SystemExit("no predictions found under " + str(R))
    out = pd.DataFrame(rows)
    prim = out.primary.to_numpy()
    if prim.any():
        for col in ("p_image", "p_group"):
            ps = out.loc[prim, col].to_numpy()
            order = np.argsort(ps)
            adj = np.empty_like(ps)
            run_max = 0.0
            for r, j in enumerate(order):
                run_max = max(run_max, min(1.0, (len(ps) - r) * ps[j]))
                adj[j] = run_max
            out.loc[prim, col + "_holm4"] = adj
    dest = paths.ensure(R / "grouped_bootstrap")
    out.to_csv(dest / "grouped_vs_image.csv", index=False)
    lines = ["# Grouped-bootstrap sensitivity analysis", "",
             "Same predictions and replicate seed; only the resampling unit changes (image vs fold group).", "",
             "| Result | Estimate | Image-level 95% CI | Group-level 95% CI | Width ratio | p (group) | Images / groups |",
             "|---|---|---|---|---|---|---|"]
    for r in out.itertuples():
        lines.append(f"| {r.result} | {r.estimate:+.3f} | [{r.image_lo:+.3f}, {r.image_hi:+.3f}] | "
                     f"[{r.group_lo:+.3f}, {r.group_hi:+.3f}] | {r.width_ratio_group_vs_image:.2f} | "
                     f"{r.p_group:.4f} | {r.n_test_images} / {r.n_test_groups} |")
    if "p_group_holm4" in out:
        ok = (out.loc[prim, "p_group_holm4"] < 0.05).all()
        lines += ["", f"Four registered crossovers, group-level, Holm over four: "
                      f"{'all significant' if ok else 'NOT all significant'} "
                      f"(adjusted p: {', '.join(f'{v:.4f}' for v in out.loc[prim, 'p_group_holm4'])})."]
    (dest / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
