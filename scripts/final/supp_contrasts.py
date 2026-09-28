"""Paired contrasts quoted in the remedies supplement that no earlier script saved (Trap A, reversed test, crossed
bootstrap).  -> <WTSS_RESULTS>/analysis/supp_contrasts.csv"""
from __future__ import annotations

import os

import pandas as pd

os.environ["WTSS_BOOTSTRAP"] = "crossed"
from wtss import paths  # noqa: E402
from wtss.stats import hierarchical_paired_bootstrap, slim  # noqa: E402

JOBS = [("hair", "spec_e13/dino518_spec_universal", "mte", "mask"),
        ("hair", "spec_e13/dino518_spec_universal", "mte_protect", "mte"),
        ("thyroid", "thyroid/dino518_universal", "mte_protect", "mte"),
        ("thyroid", "thyroid/dino518_universal", "mte", "jtt"),
        ("ovary", "ovary/dino518_universal", "mte", "jtt"),
        ("thyroid MedSigLIP", "thyroid/medsiglip448_universal", "mte", "mte_aug"),
        ("thyroid ConvNeXt", "thyroid/convnext384_universal", "mte", "mte_aug"),
        ("thyroid ConvNeXt", "thyroid/convnext384_universal", "mte", "mask"),
        ("capsule", "capsule/dino518_universal", "mte", "mask"),
        ("capsule", "capsule/dino518_universal", "mte_protect", "mask")]
rows = []
for lab, rel, a1, a0 in JOBS:
    p = pd.read_csv(paths.RESULTS / rel / "predictions.csv.gz")
    q = p[p.trap == "trapA"]
    r = hierarchical_paired_bootstrap(slim(q, "test_rev", (a1, a0)), a1, a0, "test_rev", 10000, 29, fast=True)
    rows.append({"cohort": lab, "run": rel, "contrast": f"{a1} - {a0}", **{k: r[k] for k in ("seed_delta_mean", "ci95_lo", "ci95_hi", "p_boot_two_sided")}})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(paths.ensure(paths.RESULTS / "analysis") / "supp_contrasts.csv", index=False)
