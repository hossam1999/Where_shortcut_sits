"""Round 8, Phase 1 step 6: the proposed "dominates masking" criterion applied to the EXISTING arms (descriptive).

Reads results/round8/scoreboard_existing.csv only. For every arm run with DINOv2-B/14 (the encoder with every cohort)
it lists each criterion cell, its (arm - mask) estimate and 95% crossed interval, and whether the component is met:
    NI   lower bound > -margin      (natural/external all pairs, clean trap tests, Trap B reversed, min(rev, corr))
    SUP  lower bound > 0            (Trap A reversed, natural hard pairs where masking harms them)
The precision table reports, per cell, the half-width of the (arm - mask) interval of every existing arm, i.e. how
tight an interval a new method can expect; a margin smaller than that half-width cannot be met by a method that is
merely as good as masking.

    python scripts/round8/criterion_check.py [--smoke] [--margin 0.01]
Outputs: results/round8/criterion_existing.csv, criterion_existing_summary.csv, precision_by_cell.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _r8(name):
    """Load a round-8 module by path under a unique name (earlier rounds also have a module called `common`)."""
    import importlib.util
    key = f"round8_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).resolve().parent / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


C = _r8("common")
DINO = "DINOv2-B/14@518"
TRAP_COHORTS = ("thyroid", "capsule", "ovary", "isic_hair")
# the DINOv2 source run of each trap cohort that carries the full remedy set
TRAP_RUN = {"thyroid": "dino518_universal", "capsule": "dino518_universal", "ovary": "dino518_universal",
            "isic_hair": "dino518_spec_universal"}
NATURAL = ("thyroid", "capsule", "isic_BCN", "isic_HAM", "isic_MSK")
HARD_SUP = ("thyroid", "isic_BCN", "isic_MSK", "isic2019_to_2020")  # cohorts where masking harms hard pairs


def cells() -> list[dict]:
    """(setting, cohort, run, cell, env, kind) of every criterion component."""
    out = []
    for c in NATURAL:
        out.append(dict(setting="S3", cohort=c, run=f"{c}_dino518_repro", cell="all", env="clean", kind="NI",
                        component="natural all pairs"))
    out.append(dict(setting="S4", cohort="isic2019_to_2020", run="round5_calibrated", cell="all", env="clean",
                    kind="NI", component="external all pairs"))
    for c in HARD_SUP:
        s, run = ("S4", "round5_calibrated") if c.startswith("isic2019") else ("S3", f"{c}_dino518_repro")
        out.append(dict(setting=s, cohort=c, run=run, cell="hard", env="clean", kind="SUP", component="hard pairs"))
    for c in TRAP_COHORTS:
        r = TRAP_RUN[c]
        out += [dict(setting="S2", cohort=c, run=r, cell="trapA", env="test_rev", kind="SUP", component="Trap A reversed"),
                dict(setting="S2", cohort=c, run=r, cell="trapB", env="test_rev", kind="NI", component="Trap B reversed"),
                dict(setting="S2", cohort=c, run=r, cell="trapA", env="min_rev_corr", kind="NI", component="Trap A min(rev,corr)"),
                dict(setting="S2", cohort=c, run=r, cell="trapB", env="min_rev_corr", kind="NI", component="Trap B min(rev,corr)"),
                dict(setting="S2", cohort=c, run=r, cell="trapA", env="clean", kind="NI", component="clean (Trap A models)"),
                dict(setting="S2", cohort=c, run=r, cell="trapB", env="clean", kind="NI", component="clean (Trap B models)")]
    out += [dict(setting="S5", cohort="thyroid", run="round6_ft_natural", cell="all", env="clean", kind="NI",
                 component="fine-tuned all pairs"),
            dict(setting="S5", cohort="thyroid", run="round6_ft_natural", cell="hard", env="clean", kind="SUP",
                 component="fine-tuned hard pairs")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--margin", type=float, default=0.01)
    a = ap.parse_args()
    out = C.out_dir(a.smoke)
    sb = pd.read_csv(out / "scoreboard_existing.csv")
    dm = sb[sb.metric == "delta_vs_mask"]
    rows = []
    for c in cells():
        q = dm[(dm.setting == c["setting"]) & (dm.cohort == c["cohort"]) & (dm.run == c["run"]) & (dm.cell == c["cell"])
               & (dm.env == c["env"])]
        for r in q.itertuples():
            ok = r.ci95_lo > (-a.margin if c["kind"] == "NI" else 0.0)
            rows.append({**c, "arm": r.arm, "estimate": r.estimate, "ci95_lo": r.ci95_lo, "ci95_hi": r.ci95_hi,
                         "half_width": (r.ci95_hi - r.ci95_lo) / 2, "met": bool(ok)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "criterion_existing.csv", index=False, float_format="%.4f")
    n_cells = len(cells())
    summ = (df.groupby("arm")
              .agg(cells_run=("met", "size"), cells_met=("met", "sum"))
              .assign(cells_total=n_cells).reset_index())
    fails = df[~df.met].groupby("arm").apply(
        lambda g: "; ".join(f"{r.component} {r.cohort} {r.estimate:+.3f} [{r.ci95_lo:+.3f},{r.ci95_hi:+.3f}]"
                            for r in g.itertuples()), include_groups=False).rename("failed_cells")
    summ = summ.merge(fails, on="arm", how="left").sort_values(["cells_met", "cells_run"], ascending=False)
    summ["dominates"] = (summ.cells_met == n_cells)
    summ.to_csv(out / "criterion_existing_summary.csv", index=False)
    # precision: half-width of (arm - mask) intervals per cell, over every arm and encoder in the scoreboard
    p = dm.assign(half_width=(dm.ci95_hi - dm.ci95_lo) / 2)
    prec = (p.groupby(["setting", "cohort", "cell", "env"]).half_width
             .agg(["min", "median", "max", "size"]).reset_index())
    prec.to_csv(out / "precision_by_cell.csv", index=False, float_format="%.4f")
    C.log(cells=n_cells, arms=len(summ), dominating=int(summ.dominates.sum()))


if __name__ == "__main__":
    main()
