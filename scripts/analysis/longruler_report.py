"""Collect the registered long-ruler numbers (docs/PREREGISTRATION_LONG_RULER.md) from the two finished runs.
Run with WTSS_BOOTSTRAP=crossed. Writes longruler_report.json and longruler_tables.csv next to the run dirs."""
import json
import os

import numpy as np
import pandas as pd

from wtss import paths
from wtss import stats_crossed as X
from wtss.stats import hierarchical_auc_bootstrap

assert os.environ.get("WTSS_BOOTSTRAP") == "crossed"
B = paths.RESULTS / "synthetic" / "isic2018_longruler"
RUNS = {"long": B / "dino518_ruler_long_corr_longruler", "standard": B / "dino518_ruler_fixed_corr_longruler_subset"}
ARMS = ["mask", "inpaint", "balanced"]
N = 10000
out = {"runs": {k: str(v.relative_to(paths.REPO_ROOT)) for k, v in RUNS.items()}}
rows = []
preds = {}
for r, d in RUNS.items():
    p = pd.read_csv(d / "predictions.csv.gz")
    preds[r] = p
    boot = pd.read_csv(d / "bootstrap_vs_erm.csv")
    li = pd.read_csv(d / "location_interaction.csv")
    for _, q in boot.iterrows():
        rows.append({"ruler": r, "quantity": "arm_minus_erm_rev", "method": q.method_a, "overlap": q.overlap,
                     "estimate": q.seed_delta_mean, "ci95_lo": q.ci95_lo, "ci95_hi": q.ci95_hi})
    for _, q in li.iterrows():
        rows.append({"ruler": r, "quantity": "location_interaction_0_minus_100", "method": q.method, "overlap": np.nan,
                     "estimate": q.seed_delta_mean, "ci95_lo": q.ci95_lo, "ci95_hi": q.ci95_hi})
    for m in ["erm", *ARMS]:
        for ov in (0.0, 1.0):
            for env in ("test_rev", "test_corr"):
                x = p[np.isclose(p.overlap, ov)]
                a = hierarchical_auc_bootstrap(x, m, env, n_boot=N, seed=20260918)
                rows.append({"ruler": r, "quantity": f"auc_{env}", "method": m, "overlap": ov,
                             "estimate": a["auc_mean"], "ci95_lo": a["ci95_lo"], "ci95_hi": a["ci95_hi"]})

# L3: long interaction minus standard interaction, same images and seeds -> shared image weights, seeds as clusters
for m in ARMS:
    terms = []
    for r, sign in (("long", +1.0), ("standard", -1.0)):
        p = preds[r]
        p = p[p.env == "test_rev"]
        lo, hi = p[np.isclose(p.overlap, 0.0)], p[np.isclose(p.overlap, 1.0)]
        terms += X._pair_terms(lo, m, "erm", "test_rev", "seed", +sign)
        terms += X._pair_terms(hi, m, "erm", "test_rev", "seed", -sign)
    point, arr = X.replicates(terms, N, 20260918)
    s = X._summary(point, arr, {"method": m, "estimand": "interaction_long_minus_standard"})
    rows.append({"ruler": "long_minus_standard", "quantity": "interaction_difference", "method": m, "overlap": np.nan,
                 "estimate": s["seed_delta_mean"], "ci95_lo": s["ci95_lo"], "ci95_hi": s["ci95_hi"]})

T = pd.DataFrame(rows)
T.to_csv(B / "longruler_tables.csv", index=False)


def get(r, q, m, ov=None):
    x = T[(T.ruler == r) & (T.quantity == q) & (T.method == m)]
    if ov is not None:
        x = x[np.isclose(x.overlap, ov)]
    assert len(x) == 1, (r, q, m, ov)
    return x.iloc[0][["estimate", "ci95_lo", "ci95_hi"]].astype(float).to_dict()


L1 = get("long", "location_interaction_0_minus_100", "mask")
L2 = get("long", "arm_minus_erm_rev", "mask", 1.0)
L3 = get("long_minus_standard", "interaction_difference", "mask")
out["claims"] = {
    "L1": {**L1, "supported": bool(L1["estimate"] > 0 and L1["ci95_lo"] > 0)},
    "L2": {**L2, "supported": bool(L2["estimate"] < 0 and L2["ci95_hi"] < 0)},
    "L3_descriptive": L3,
    "standard_subset_interaction": get("standard", "location_interaction_0_minus_100", "mask"),
}
(B / "longruler_report.json").write_text(json.dumps(out, indent=2))
pd.set_option("display.width", 200)
print(T.to_string())
print(json.dumps(out["claims"], indent=2))
