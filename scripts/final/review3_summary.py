"""Rebuild the third-review summary tables from the regenerated results (crossed bootstrap), for make_review3_tables.py.
  -> <WTSS_RESULTS>/review3/{R5_paste_edge.csv, R7_matching_full.csv, operating_points_crossed.csv}
R5: from review3/crossed_ci.csv (Holm over the four cohorts; rule of docs/PREREGISTRATION_REVIEW3.md: attributed to
the artifact only if artifact - neutral > 0 after Holm and the neutral interaction is below a quarter of the artifact's).
R7: matching descriptors (image counts, pairs kept, standardised differences) are properties of the deterministic
matching and are carried over from the archived table; every crossover and interval is replaced by the regenerated,
crossed one (unmatched and 0.2 SD: review2/R0_R1_crossovers.csv; 0.05 SD: review3/crossed_ci.csv)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wtss import paths

R = paths.RESULTS
OLD = paths.REPO_ROOT / "results"
C = pd.read_csv(R / "review3" / "crossed_ci.csv")


def get(claim, cohort):
    q = C[(C.claim == claim) & (C.cohort == cohort)]
    return None if q.empty else q.iloc[0]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


rows = []
for c in ("ISIC hair", "Thyroid", "Ovary", "Capsule"):
    a, n, d = get("transplant interaction (artifact)", c), get("transplant interaction (neutral paste)", c), \
        get("artifact - neutral interaction (N3)", c)
    rows.append({"cohort": c, "artifact": a.estimate, "artifact_lo": a.crossed_lo, "artifact_hi": a.crossed_hi,
                 "neutral": n.estimate, "neutral_lo": n.crossed_lo, "neutral_hi": n.crossed_hi,
                 "ratio_neutral_to_artifact": n.estimate / a.estimate, "N3": d.estimate, "N3_lo": d.crossed_lo,
                 "N3_hi": d.crossed_hi, "N3_p": d.crossed_p})
r5 = pd.DataFrame(rows)
r5["N3_p_holm"] = holm(r5.N3_p)
r5["N3_supported"] = (r5.N3_lo > 0) & (r5.N3_p_holm < 0.05)
r5["attributed_to_artifact"] = r5.N3_supported & (r5.ratio_neutral_to_artifact < 0.25)
r5.to_csv(R / "review3" / "R5_paste_edge.csv", index=False)

r7 = pd.read_csv(OLD / "review3" / "R7_matching_full.csv")
r01 = pd.read_csv(R / "review2" / "R0_R1_crossovers.csv")
for k, r in r7.iterrows():
    rep = r01[(r01.cohort == r.cohort) & (r01.run == "repro")].iloc[0]
    r7.loc[k, ["crossover_unmatched", "unmatched_lo", "unmatched_hi"]] = [rep.crossover, rep.ci95_lo, rep.ci95_hi]
    if np.isclose(r.caliper_sd, 0.2):
        m = r01[(r01.cohort == r.cohort) & (r01.run == "matched")].iloc[0]
        r7.loc[k, ["crossover_matched", "matched_lo", "matched_hi"]] = [m.crossover, m.ci95_lo, m.ci95_hi]
    else:
        m = get("real-trap crossover (matched05)", r.cohort)
        r7.loc[k, ["crossover_matched", "matched_lo", "matched_hi"]] = [m.estimate, m.crossed_lo, m.crossed_hi]
r7["change_pct"] = 100 * (r7.crossover_matched / r7.crossover_unmatched - 1)
r7.to_csv(R / "review3" / "R7_matching_full.csv", index=False)

op = pd.read_csv(R / "final_op" / "operating_points_crossed_all.csv")
op["ci95_lo"], op["ci95_hi"] = op.crossed_lo, op.crossed_hi
op.to_csv(R / "review3" / "operating_points_crossed.csv", index=False)
print(r5.round(3).to_string()); print(r7[["cohort", "caliper_sd", "crossover_matched", "matched_lo", "matched_hi", "change_pct"]].round(3))
