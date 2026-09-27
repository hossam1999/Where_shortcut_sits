"""Location-adaptive selection ("auto"): per (trap, seed, fold), pick the candidate arm with the best
group-aware validation score, then evaluate the chosen arm's test predictions. Candidates never train on
val_groups (DFR-type arms are excluded; reported separately as references).

Validation score (needs image-level A on the validation images only) = min(AUROC(Y=1&A=0 vs Y=0&A=1),
AUROC(Y=1&A=1 vs Y=0&A=0)) — the validation analogue of min(reversed, correlated).

  python scripts/analysis/adaptive_select.py results/thyroid/dino518_auto [results/... ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from wtss.stats import hierarchical_paired_bootstrap, safe_auc, slim

SPLIT = "--split" in sys.argv  # U14: score on the val half DFR never saw; DFR-half arms become candidates
CANDIDATES = (["dfr_half", "mask_dfr_half"] if "--split" in sys.argv else []) + ["erm", "mask", "balanced", "mask_balanced", "mte", "mte_protect", "mte_balanced",
              "mte_protect_balanced", "jtt", "mask_jtt", "umte_jtt"]


def val_score(q: pd.DataFrame) -> float:
    y, a, p = q.y.to_numpy(), q.artifact_present.to_numpy(), q.prob.to_numpy()
    out = []
    for hard_pos, hard_neg in ((0, 1), (1, 0)):
        sel = ((y == 1) & (a == hard_pos)) | ((y == 0) & (a == hard_neg))
        if len(np.unique(y[sel])) < 2:
            return float("nan")
        out.append(roc_auc_score(y[sel], p[sel]))
    return min(out)


def main():
    rows_all = []
    for d in map(Path, [x for x in sys.argv[1:] if not x.startswith("--")]):
        preds = pd.read_csv(d / "predictions.csv.gz")
        if "trap" not in preds:
            preds["trap"] = "trapA"
        if "fold" not in preds:
            preds["fold"] = 0
        cands = [c for c in CANDIDATES if c in set(preds.method)]
        v = preds[(preds.env == "val_groups") & preds.method.isin(cands)]
        if SPLIT:
            from wtss.utils import stable_int
            v = v[np.array([stable_int("val_half", i) % 2 == 1 for i in v.image_id])]
        sc = v.groupby(["trap", "seed", "fold", "method"]).apply(val_score, include_groups=False).rename("score").reset_index()
        pick = sc.loc[sc.groupby(["trap", "seed", "fold"]).score.idxmax()][["trap", "seed", "fold", "method"]]
        pick.to_csv(d / ("auto_split_choices.csv" if SPLIT else "auto_choices.csv"), index=False)
        chosen = preds.merge(pick, on=["trap", "seed", "fold", "method"])
        chosen = chosen[chosen.env != "val_groups"].assign(method="auto_split" if SPLIT else "auto")
        allp = pd.concat([preds[preds.env != "val_groups"], chosen], ignore_index=True)
        allp.to_csv(d / ("predictions_with_auto_split.csv.gz" if SPLIT else "predictions_with_auto.csv.gz"), index=False, compression="gzip")
        print(f"== {d}\nchoices:", pick.groupby("trap").method.value_counts().to_dict())
        for trap in sorted(allp.trap.unique()):
            q = allp[allp.trap == trap]
            auc = q.groupby(["method", "env", "seed"]).apply(lambda z: safe_auc(z.y, z.prob), include_groups=False)
            m = auc.groupby(["method", "env"]).mean().unstack()
            m["min_rev_corr"] = m[["test_rev", "test_corr"]].min(1)
            AUTO = "auto_split" if SPLIT else "auto"
            best_fixed = m.drop(index=AUTO).min_rev_corr.idxmax()
            refs = [r for r in ("mask", "balanced", "dfr", "mask_dfr", best_fixed) if r in m.index]
            for ref in dict.fromkeys(refs):
                r = hierarchical_paired_bootstrap(slim(q, "test_rev", (AUTO, ref)), AUTO, ref, "test_rev", 10000, 13, fast=True)
                rows_all.append({"run": d.name, "trap": trap, "ref": ref, **{k: r[k] for k in ("seed_delta_mean", "ci95_lo", "ci95_hi")}})
            print(trap, "best fixed (min rev,corr):", best_fixed)
            print(m.loc[[i for i in [AUTO, best_fixed, "mask", "balanced", "dfr", "mask_dfr"] if i in m.index]]
                  [["clean", "test_corr", "test_rev", "min_rev_corr"]].round(3).drop_duplicates().to_string())
    out = pd.DataFrame(rows_all)
    print(out.round(3).to_string())
    dirs = [x for x in sys.argv[1:] if not x.startswith("--")]
    if dirs:
        out.to_csv(Path(dirs[0]).parent.parent / ("adaptive_select_split_summary.csv" if SPLIT else "adaptive_select_summary.csv"), index=False)


if __name__ == "__main__":
    main()
