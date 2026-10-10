"""Average over the two ruler sizes run on the same 1,314 images (docs/PREREGISTRATION_LONG_RULER.md).
Descriptive summary added after the registered run (not a registered claim). Same crossed bootstrap as the
registered L3: one Poisson weight per test image shared by both rulers, seeds resampled as clusters.
  PYTHONPATH=src WTSS_BOOTSTRAP=crossed python scripts/analysis/ruler_sizes_average.py
    -> results/synthetic/isic2018_longruler/ruler_sizes_average.csv
"""
import os

import numpy as np
import pandas as pd

from wtss import paths
from wtss import stats_crossed as X

assert os.environ.get("WTSS_BOOTSTRAP") == "crossed"
B = paths.RESULTS / "synthetic" / "isic2018_longruler"
RUNS = [B / "dino518_ruler_long_corr_longruler", B / "dino518_ruler_fixed_corr_longruler_subset"]
P = [pd.read_csv(d / "predictions.csv.gz") for d in RUNS]
P = [p[p.env == "test_rev"] for p in P]
rows = []
for m in ["mask", "inpaint", "balanced"]:
    for name, parts in [("outside (0%)", [(0.0, +1)]), ("fully inside (100%)", [(1.0, +1)]),
                        ("location effect", [(0.0, +1), (1.0, -1)])]:
        terms = []
        for p in P:
            for ov, s in parts:
                terms += X._pair_terms(p[np.isclose(p.overlap, ov)], m, "erm", "test_rev", "seed", 0.5 * s)
        point, arr = X.replicates(terms, 10000, 20260918)
        r = X._summary(point, arr, {"method": m, "quantity": name})
        rows.append({"method": m, "quantity": name, "estimate": r["seed_delta_mean"], "ci95_lo": r["ci95_lo"],
                     "ci95_hi": r["ci95_hi"]})
T = pd.DataFrame(rows)
T.to_csv(B / "ruler_sizes_average.csv", index=False, float_format="%.4f")
print(T.round(3).to_string(index=False))
