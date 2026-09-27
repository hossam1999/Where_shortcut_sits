"""Recompute and SAVE every bootstrap that was first computed interactively and quoted in the paper/docs
(fine-tuning crossovers; controlled-sweep comparisons at full overlap). Deterministic seeds as originally used.
  python scripts/analysis/adhoc_bootstraps.py   # -> results/adhoc_bootstraps.json
"""
from __future__ import annotations

import json

import pandas as pd

from wtss import paths
from wtss.stats import difference_of_deltas, hierarchical_paired_bootstrap

R = paths.RESULTS
FT = {"thyroid_resnet50": "finetune/thyroid/resnet50", "capsule_resnet50": "finetune/capsule/resnet50",
      "ovary_resnet50": "finetune/ovary/resnet50",
      "thyroid_vit_s": "finetune/thyroid/vit_small_patch16_224.augreg_in21k_ft_in1k"}
SWEEPS = {"capsule_dino518": "synthetic/capsule/dino518_debris_corr_main",
          "capsule_medsiglip": "synthetic/capsule/medsiglip448_debris_corr_main",
          "capsule_medsiglip_protect": "synthetic/capsule/medsiglip448_debris_corr_protect",
          "capsule_dino518_protect": "synthetic/capsule/dino518_debris_corr_protect",
          "thyroid_dino518": "synthetic/thyroid/dino518_caliper_corr_main",
          "thyroid_medsiglip": "synthetic/thyroid/medsiglip448_caliper_corr_main",
          "thyroid_medsiglip_protect": "synthetic/thyroid/medsiglip448_caliper_corr_protect",
          "ovary_dino518": "synthetic/ovary/dino518_caliper_corr_main",
          "cxr_raddino518": "synthetic/nih_ptx/raddino518_tube_corr_main"}
PAIRS = [("umte", "mask"), ("umte_balanced", "balanced"), ("umte_balanced", "mask"), ("umte_protect", "mask"),
         ("umte_protect", "umte"), ("umte_protect_balanced", "balanced")]


def trio(r):
    return [round(r[k], 3) for k in ("seed_delta_mean", "ci95_lo", "ci95_hi")]


def main():
    out = {}
    for name, rel in FT.items():
        f = R / rel / "predictions.csv.gz"
        if f.exists():
            p = pd.read_csv(f)
            out[f"finetune|{name}|crossover mask-erm (B-A)"] = trio(
                difference_of_deltas(p[p.trap == "trapB"], p[p.trap == "trapA"], "mask", "erm", "test_rev", 10000, 11))
    for name, rel in SWEEPS.items():
        f = R / rel / "predictions.csv.gz"
        if not f.exists():
            continue
        p = pd.read_csv(f)
        for ov in (0.5, 1.0):
            q = p[p.overlap == ov]
            for a1, a0 in PAIRS:
                if {a1, a0} <= set(q.method):
                    out[f"sweep|{name}|r={ov}|{a1}-{a0}"] = trio(hierarchical_paired_bootstrap(q, a1, a0, "test_rev", 10000, 3, fast=True))
        print(name, flush=True)
    (R / "adhoc_bootstraps.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
