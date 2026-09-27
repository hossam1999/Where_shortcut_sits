"""Backbone-scale test (docs/PREREGISTRATION_SCALE.md): location crossover (S1), U-MtE_protect − mask and U-MtE − mask
in Trap A (S2) and the ERM shortcut gap (S3) for DINOv2 ViT-S/14, ViT-B/14 (main) and ViT-L/14.
  python scripts/analysis/scale_compare.py   # -> results/scale/scale_compare.csv
"""
from __future__ import annotations

import pandas as pd

from wtss import paths
from wtss.stats import difference_of_deltas, hierarchical_paired_bootstrap, slim

R = paths.RESULTS
RUNS = {"ViT-S/14": "dinos518_scale", "ViT-B/14": "dino518_universal", "ViT-L/14": "dinol518_scale"}


def main():
    rows = []
    for co in ("thyroid", "ovary", "capsule"):
        for bb, tag in RUNS.items():
            f = R / co / tag / "predictions.csv.gz"
            if not f.exists():
                print("missing", f); continue
            p = pd.read_csv(f)
            if "mte_protect" not in set(p.method):  # ViT-B: protected arm lives in the protect_generic run
                q = pd.read_csv(R / co / "dino518_protect_generic" / "predictions.csv.gz") if (R / co / "dino518_protect_generic" / "predictions.csv.gz").exists() else None
                if q is not None:
                    p = pd.concat([p, q[q.method == "mte_protect"]], ignore_index=True)
            r = {"cohort": co, "backbone": bb}
            x = difference_of_deltas(p[p.trap == "trapB"], p[p.trap == "trapA"], "mask", "erm", "test_rev", 10000, 11)
            r.update(cross=x["seed_delta_mean"], cross_lo=x["ci95_lo"], cross_hi=x["ci95_hi"])
            qa = p[p.trap == "trapA"]
            for a1 in ("mte", "mte_protect"):
                if a1 in set(qa.method):
                    b = hierarchical_paired_bootstrap(slim(qa, "test_rev", (a1, "mask")), a1, "mask", "test_rev", 10000, 29, fast=True)
                    r.update({f"{a1}_mask": b["seed_delta_mean"], f"{a1}_mask_lo": b["ci95_lo"], f"{a1}_mask_hi": b["ci95_hi"]})
            m = pd.read_csv(R / co / tag / "metrics_per_seed.csv")
            e = m[(m.trap == "trapA") & (m.method == "erm")].groupby("env").auc.mean()
            r.update(erm_gap=e["test_corr"] - e["test_rev"], erm_clean=e["clean"])
            rows.append(r); print(r, flush=True)
    d = pd.DataFrame(rows)
    out = paths.ensure(R / "scale")
    d.to_csv(out / "scale_compare.csv", index=False)
    print(d.round(3).to_string())


if __name__ == "__main__":
    main()
