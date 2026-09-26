"""Verification 2: rerun from raw images vs archived pilot results (same cohort, placements, seeds).

Compares per (method, overlap, seed, env) AUROC and the bootstrap Δ vs ERM with the archived
bridge / Phase-2 / λ=0 tables. Writes results/verification/synthetic_rerun_vs_archive.csv and .md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARCH = Path("/root/isic_pcam_code_results/isic_overlap_pilot_patch_v3")
NEW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/synthetic/isic2018/dino518_ruler_fixed_corr_main")
OUT = Path("results/verification"); OUT.mkdir(parents=True, exist_ok=True)

old = pd.read_csv(ARCH / "results_v3/bridge/metrics_all.csv")
lam0 = ARCH / "runs/pilot_patch_v3/phase2/lambda0_control/metrics.csv"
if lam0.exists():
    l0 = pd.read_csv(lam0)
    l0 = l0[l0.method.str.contains("lam0")]
    old = pd.concat([old, l0], ignore_index=True)
old_m = old.groupby(["method", "overlap", "seed", "env"]).auc.mean().rename("auc_archived")
new = pd.read_csv(NEW / "metrics.csv")
new_m = new.groupby(["method", "overlap", "seed", "env"]).auc.mean().rename("auc_rerun")
j = pd.concat([old_m, new_m], axis=1).dropna().reset_index()
j["diff"] = j.auc_rerun - j.auc_archived
j.to_csv(OUT / "synthetic_rerun_vs_archive_per_seed.csv", index=False)
agg = j.groupby(["method", "env"]).agg(n=("diff", "size"), mean_abs_diff=("diff", lambda x: np.abs(x).mean()),
                                       max_abs_diff=("diff", lambda x: np.abs(x).max())).reset_index()

ob = pd.read_csv(ARCH / "results_v3/bridge/hierarchical_bootstrap.csv")
nb = pd.read_csv(NEW / "bootstrap_vs_erm.csv")
b = ob.rename(columns={"arm": "method"}).merge(nb, left_on=["method", "overlap"], right_on=["method_a", "overlap"],
                                               suffixes=("_arch", "_new"))
b = b[["method", "overlap", "seed_delta_mean_arch", "ci95_lo_arch", "ci95_hi_arch", "seed_delta_mean_new", "ci95_lo_new", "ci95_hi_new"]]
b["delta_diff"] = b.seed_delta_mean_new - b.seed_delta_mean_arch
b["same_sign_and_ci_decision"] = ((b.ci95_lo_arch > 0) == (b.ci95_lo_new > 0)) & ((b.ci95_hi_arch < 0) == (b.ci95_hi_new < 0))
b.to_csv(OUT / "synthetic_rerun_vs_archive_bootstrap.csv", index=False)
with open(OUT / "SYNTHETIC_RERUN_VS_ARCHIVE.md", "w") as f:
    f.write("# Synthetic rerun from raw images vs archived pilot\n\n")
    f.write("Per-seed AUROC agreement (rerun − archived):\n\n" + agg.round(4).to_markdown(index=False) + "\n\n")
    f.write("Reversed-test Δ vs ERM (hierarchical bootstrap):\n\n" + b.round(3).to_markdown(index=False) + "\n")
print(agg.round(4).to_string()); print(b.round(3).to_string())
