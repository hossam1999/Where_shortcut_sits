"""Score the expert audit (scripts/audit/make_expert_audit_kit.py): inter-rater agreement and automatic-label accuracy.

  python scripts/audit/score_expert_audit.py sheet_A.csv sheet_B.csv   -> results/review2/expert_audit.csv / .json
Per cohort: Cohen's kappa between the two raters (presence, location); precision and recall of the automatic label
against the consensus (items where both raters agree; 'unsure' excluded); mask quality distribution. 95 % CIs by
bootstrap over items (2,000 replicates).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from wtss import paths


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def boot(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def main():
    fa, fb = sys.argv[1], sys.argv[2]
    key = pd.read_csv(paths.DATA / "expert_audit_kit" / "KEY_do_not_share_with_raters.csv")
    A, B = pd.read_csv(fa), pd.read_csv(fb)
    d = key.merge(A[["item", "present", "location", "mask_quality"]], on="item").merge(
        B[["item", "present", "location", "mask_quality"]], on="item", suffixes=("_a", "_b"))
    for col in ("present_a", "present_b", "location_a", "location_b", "mask_quality_a", "mask_quality_b"):
        d[col] = d[col].astype(str).str.strip().str.lower()
    rows = []
    for c, q in d.groupby("cohort"):
        q = q.reset_index(drop=True)
        r = {"cohort": c, "n_items": len(q)}
        ok = (q.present_a != "unsure") & (q.present_b != "unsure")
        qq = q[ok].reset_index(drop=True)
        r["kappa_presence"] = kappa(qq.present_a, qq.present_b)
        r["kappa_presence_ci"] = boot(lambda i: kappa(qq.present_a[i], qq.present_b[i]), len(qq))
        both = qq[qq.present_a == qq.present_b].reset_index(drop=True)
        truth = (both.present_a == "yes").astype(int).to_numpy()
        auto = both.auto_label.to_numpy()
        prec = lambda i: truth[i][auto[i] == 1].mean() if (auto[i] == 1).any() else np.nan
        rec = lambda i: auto[i][truth[i] == 1].mean() if (truth[i] == 1).any() else np.nan
        idx = np.arange(len(both))
        r.update({"n_consensus": len(both), "auto_precision": prec(idx), "auto_precision_ci": boot(prec, len(both)),
                  "auto_recall": rec(idx), "auto_recall_ci": boot(rec, len(both))})
        loc = qq[(qq.present_a == "yes") & (qq.present_b == "yes")]
        r["kappa_location"] = kappa(loc.location_a, loc.location_b) if len(loc) else np.nan
        agree = loc[loc.location_a == loc.location_b]
        r["auto_location_agreement"] = float((agree.location_a == agree.auto_location).mean()) if len(agree) else np.nan
        for k in ("good", "partial", "wrong"):
            r[f"mask_{k}"] = float(((q.mask_quality_a == k).mean() + (q.mask_quality_b == k).mean()) / 2)
        rows.append(r)
    out = paths.ensure(paths.RESULTS / "review2")
    pd.DataFrame(rows).to_csv(out / "expert_audit.csv", index=False)
    (out / "expert_audit.json").write_text(json.dumps(rows, indent=1, default=float))
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
