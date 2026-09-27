"""Paired hierarchical bootstrap for arbitrary arm comparisons on saved predictions.

  python scripts/analysis/paired_deltas.py results/capsule/dino518_protect_tmpl mte_protect:mte mte_protect:mask
  (writes <dir>/paired_deltas.csv; reversed and clean environments, both traps)
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from wtss.stats import hierarchical_paired_bootstrap, slim


def _b(j):
    q, a1, a0, env, s = j
    return hierarchical_paired_bootstrap(q, a1, a0, env, 10000, s, fast=True)


def main():
    d = Path(sys.argv[1])
    pairs = [p.split(":") for p in sys.argv[2:]]
    preds = pd.read_csv(d / "predictions.csv.gz")
    jobs, keys = [], []
    for trap in sorted(preds.trap.unique()):
        q = preds[preds.trap == trap]
        for a1, a0 in pairs:
            for env in ("test_rev", "clean"):
                jobs.append((slim(q, env, (a1, a0)), a1, a0, env, 20260927 + sum(map(ord, a1 + a0 + trap + env))))
                keys.append((trap, a1, a0, env))
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_b, jobs))
    out = pd.DataFrame([{"trap": t, "arm": a1, "ref": a0, "env": e, **{k: r[k] for k in ("seed_delta_mean", "ci95_lo", "ci95_hi")}}
                        for (t, a1, a0, e), r in zip(keys, res)])
    f = d / "paired_deltas.csv"
    if f.exists():  # merge with rows already computed for other comparisons (never drop earlier results)
        old = pd.read_csv(f)
        key = ["trap", "arm", "ref", "env"]
        old = old[~old.set_index(key).index.isin(out.set_index(key).index)]
        out = pd.concat([old, out], ignore_index=True)
    out.to_csv(f, index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
