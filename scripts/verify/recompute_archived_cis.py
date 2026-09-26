"""Verification 1: recompute every archived bridge CI from archived per-image predictions.

Proves the ported estimator (wtss.stats) is identical to the pilot's and that the
thesis Table (Result 1 + B1 interaction) follows from the archived predictions.
"""
import sys, json
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from wtss.stats import hierarchical_paired_bootstrap, hierarchical_interaction

ARCH = sys.argv[1] if len(sys.argv) > 1 else "/root/isic_pcam_code_results/isic_overlap_pilot_patch_v3/results_v3/bridge"
preds = pd.read_csv(f"{ARCH}/predictions_all.csv.gz")
old = pd.read_csv(f"{ARCH}/hierarchical_bootstrap.csv")

def job(args):
    arm, ov = args
    q = preds[np.isclose(preds.overlap, ov)]
    r = hierarchical_paired_bootstrap(q, arm, "erm", "test_rev", 10000, 20260918 + int(ov * 1000) + sum(map(ord, arm)))
    return arm, ov, r

def b1(arm):
    return arm, hierarchical_interaction(preds, arm, "erm", "test_rev", "overlap", 0.0, 1.0, 10000, 20260918)

jobs = [(a, ov) for a in sorted(set(old.arm)) for ov in (0.0, 0.5, 1.0)]
rows = []
with ProcessPoolExecutor(8) as ex:
    for arm, ov, r in ex.map(job, jobs):
        o = old[(old.arm == arm) & np.isclose(old.overlap, ov)]
        if o.empty: continue
        o = o.iloc[0]
        rows.append(dict(arm=arm, overlap=ov, new_point=r["seed_delta_mean"], old_point=o.seed_delta_mean,
                         new_lo=r["ci95_lo"], old_lo=o.ci95_lo, new_hi=r["ci95_hi"], old_hi=o.ci95_hi))
    b1rows = list(ex.map(b1, ["mask", "balanced", "dfr", "leace", "inpaint"]))
d = pd.DataFrame(rows)
d["max_abs_diff"] = d[["new_point", "old_point"]].diff(axis=1).abs().iloc[:, 1].combine(
    (d.new_lo - d.old_lo).abs(), max).combine((d.new_hi - d.old_hi).abs(), max)
print(d.round(4).to_string())
print("MAX |diff| over all archived CIs:", d.max_abs_diff.max())
for arm, r in b1rows:
    print(f"B1 {arm}: {r['seed_delta_mean']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]")
