"""The twelve primary contrasts (P1-P4 x thyroid, ovary, capsule) under embedding-based leakage groups
(docs/PREREGISTRATION_EMBEDDING_GROUPS.md), crossed bootstrap.  -> <WTSS_RESULTS>/analysis/emb_groups_primary.csv"""
from __future__ import annotations

import os

import pandas as pd

os.environ["WTSS_BOOTSTRAP"] = "crossed"
from wtss import paths  # noqa: E402
from wtss.stats import difference_of_deltas, hierarchical_paired_bootstrap, slim  # noqa: E402

rows = []
for c in ("thyroid", "ovary", "capsule"):
    p = pd.read_csv(paths.RESULTS / c / "dino518_emb_groups" / "predictions.csv.gz")
    x = difference_of_deltas(p[p.trap == "trapB"], p[p.trap == "trapA"], "mask", "erm", "test_rev", 10000, 11)
    rows.append({"cohort": c, "family": "P1 location crossover", **{k: x[k] for k in ("seed_delta_mean", "ci95_lo", "ci95_hi", "p_boot_two_sided")}})
    q = p[p.trap == "trapA"]
    for fam, a1, a0 in (("P2 U-MtE (protected) > mask", "mte_protect", "mask"), ("P3 erase > augment", "mte", "mte_aug"),
                        ("P4 U-MtE+balanced > balanced", "mte_balanced", "balanced")):
        r = hierarchical_paired_bootstrap(slim(q, "test_rev", (a1, a0)), a1, a0, "test_rev", 10000, 29, fast=True)
        rows.append({"cohort": c, "family": fam, **{k: r[k] for k in ("seed_delta_mean", "ci95_lo", "ci95_hi", "p_boot_two_sided")}})
    print(rows[-4:], flush=True)
d = pd.DataFrame(rows)
d["excludes_zero_positive"] = d.ci95_lo > 0
d.to_csv(paths.ensure(paths.RESULTS / "analysis") / "emb_groups_primary.csv", index=False)
print(d.round(3).to_string())
