"""Absolute AUROC of every arm on the round-8 confirmation data: the numbers behind the paper's remedies table, which
reports each arm minus masking. Reuses the registered analysis (scripts/round8/analyse.py: same cells, same hard-pair
rules, same crossed seed x image replicates for a given --n-boot), so point estimates equal those of the registered run.

Needs the git-ignored confirmation predictions, restored from wtss_outputs.tar into their original paths:
  results/round8/confirm/trap/<cohort>_dino518/predictions.csv.gz      (thyroid, capsule, ovary, isic)
  results/round8/confirm/natural/<cohort>_dino518/predictions.csv.gz   (thyroid, capsule, isic_BCN/HAM/MSK, isic2020)

  PYTHONPATH=src python scripts/round8/absolute_scores.py [--n-boot 10000] [--smoke]
    -> results/round8/confirm/absolute_auroc.csv   (source, cell, env, arm, auroc, ci95_lo, ci95_hi)
CPU only; about 10-30 minutes with the default 10,000 replicates (point estimates do not depend on --n-boot).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load(name):
    key = f"round8_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).resolve().parent / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


A = _load("analyse")
C = A.C


def jobs(conf: Path, n_boot: int):
    out, missing = [], []
    for coh in A.TRAPS:
        f = conf / "trap" / f"{coh}_dino518" / "predictions.csv.gz"
        (out.append((f"trap/{coh}", f, "trap", None, n_boot)) if f.exists() else missing.append(f))
    for coh in A.NATURAL:
        f = conf / "natural" / f"{coh}_dino518" / "predictions.csv.gz"
        rule = "hair" if coh == "isic2020" else "prevalence"
        (out.append((f"natural/{coh}", f, "natural", rule, n_boot)) if f.exists() else missing.append(f))
    return out, missing


def rows_of(label: str, R: dict) -> list:
    rows = []
    for (cell, arm, env), (pt, reps) in sorted(R.items()):
        lo, hi = C.ci(reps)
        rows.append({"source": label, "cell": cell, "env": env, "arm": arm, "auroc": pt, "ci95_lo": lo, "ci95_hi": hi})
    cells, arms = sorted({k[0] for k in R}), sorted({k[1] for k in R})
    for cell in cells:
        for arm in arms:
            a, b = R.get((cell, arm, "test_rev")), R.get((cell, arm, "test_corr"))
            if a is not None and b is not None:
                lo, hi = C.ci(np.fmin(a[1], b[1]))
                rows.append({"source": label, "cell": cell, "env": "min_rev_corr", "arm": arm,
                             "auroc": min(a[0], b[0]), "ci95_lo": lo, "ci95_hi": hi})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    conf = C.stage_dir("confirm", a.smoke)
    J, missing = jobs(conf, 200 if a.smoke else a.n_boot)
    if missing:
        print("missing prediction files (restore them from wtss_outputs.tar):")
        for f in missing:
            print("  ", f.relative_to(C.ROOT))
        sys.exit(1)
    S = dict(A.R4.parallel_map(A._source, J))
    d = pd.DataFrame([r for label, R in S.items() for r in rows_of(label, R)])
    f = conf / "absolute_auroc.csv"
    d.to_csv(f, index=False, float_format="%.4f")
    # consistency check against the committed registered differences (same point estimates by construction)
    fd = conf / "descriptive_all_cells.csv"
    if not fd.exists():
        print(f"wrote {f.relative_to(C.ROOT)}: {len(d)} rows (no registered differences to check against)")
        return
    desc = pd.read_csv(fd)
    m = d[d.arm == "mask"][["source", "cell", "env", "auroc"]].rename(columns={"auroc": "mask_auroc"})
    x = d.merge(m, on=["source", "cell", "env"]).assign(diff=lambda q: q.auroc - q.mask_auroc)
    x = x.merge(desc, on=["source", "cell", "env", "arm"], how="inner")
    worst = float((x["diff"] - x.arm_minus_mask).abs().max()) if len(x) else float("nan")
    print(f"wrote {f.relative_to(C.ROOT)}: {len(d)} rows; max |(arm - mask) - registered difference| = {worst:.5f}")
    if not a.smoke and len(x) and worst > 0.0006:
        print("WARNING: differences do not reproduce the registered analysis; do not use this file.")
        sys.exit(2)


if __name__ == "__main__":
    main()
