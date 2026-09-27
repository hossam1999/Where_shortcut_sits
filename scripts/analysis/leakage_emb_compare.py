"""Primary contrasts re-estimated with the stricter embedding groups (docs/PREREGISTRATION_EMBEDDING_GROUPS.md),
side by side with the main estimates (results/PRIMARY_CLAIMS.csv). Same bootstrap seeds as primary_claims.py.
  python scripts/analysis/leakage_emb_compare.py   # -> results/leakage/embedding_groups_compare.csv
"""
from __future__ import annotations

import pandas as pd

from wtss import paths
from wtss.stats import difference_of_deltas, hierarchical_paired_bootstrap, slim

R = paths.RESULTS
TESTS = [("P1 location crossover", "cross", "mask", "erm"), ("P2 U-MtE (protected) > mask", "pair", "mte_protect", "mask"),
         ("P3 erase > augment", "pair", "mte", "mte_aug"), ("P4 U-MtE+balanced > balanced", "pair", "mte_balanced", "balanced")]
LAB = {"thyroid": "Thyroid", "capsule": "Capsule", "ovary": "Ovary"}


def main():
    main_ = pd.read_csv(R / "PRIMARY_CLAIMS.csv")
    rows = []
    for co in ("thyroid", "ovary", "capsule"):
        f = R / co / "dino518_emb_groups" / "predictions.csv.gz"
        if not f.exists():
            print("missing", f); continue
        p = pd.read_csv(f)
        for fam, kind, a1, a0 in TESTS:
            if kind == "cross":
                r = difference_of_deltas(p[p.trap == "trapB"], p[p.trap == "trapA"], a1, a0, "test_rev", 10000, 11)
            else:
                q = p[p.trap == "trapA"]
                r = hierarchical_paired_bootstrap(slim(q, "test_rev", (a1, a0)), a1, a0, "test_rev", 10000, 29, fast=True)
            m = main_[(main_.family == fam) & main_.cohort.str.startswith(LAB[co])].iloc[0]
            rows.append({"cohort": LAB[co], "test": fam, "main": m.estimate, "main_lo": m.ci95_lo, "main_hi": m.ci95_hi,
                         "emb": r["seed_delta_mean"], "emb_lo": r["ci95_lo"], "emb_hi": r["ci95_hi"]})
            rows[-1]["robust"] = bool((rows[-1]["emb"] > 0) == (m.estimate > 0) and
                                      ((m.ci95_lo > 0) <= (r["ci95_lo"] > 0)) and ((m.ci95_hi < 0) <= (r["ci95_hi"] < 0)))
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(R / "leakage" / "embedding_groups_compare.csv", index=False)
    print(d.round(3).to_string())


if __name__ == "__main__":
    main()
