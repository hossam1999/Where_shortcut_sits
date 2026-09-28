"""Per-trap statistics of the ISIC 2020 external traps (claim X1, docs/PREREGISTRATION_FINAL.md A4), same analysis as
scripts/run_spec_e13.py (crossed bootstrap via WTSS_BOOTSTRAP=crossed).
  -> <WTSS_RESULTS>/external_isic2020/traps_dino518/bootstrap_vs_erm.csv, SUMMARY.json"""
from __future__ import annotations

import json
import os

import pandas as pd

os.environ["WTSS_BOOTSTRAP"] = "crossed"
from wtss import paths  # noqa: E402
from wtss.stats import hierarchical_paired_bootstrap, slim  # noqa: E402

out = paths.RESULTS / "external_isic2020" / "traps_dino518"
p = pd.read_csv(out / "predictions.csv.gz")
rows = []
for trap in ("trapA", "trapB"):
    q = p[p.trap == trap]
    for env in ("test_rev", "clean", "test_corr"):
        r = hierarchical_paired_bootstrap(slim(q, env, ("mask", "erm")), "mask", "erm", env, 10000,
                                          20260918 + sum(map(ord, "mask" + trap + env)), fast=True)
        rows.append({"trap": trap, "arm": "mask", "env": env, **{k: v for k, v in r.items() if not isinstance(v, (list, dict))}})
b = pd.DataFrame(rows)
b.to_csv(out / "bootstrap_vs_erm.csv", index=False)
m = pd.read_csv(out / "metrics_per_seed.csv")
auc = m.groupby(["trap", "method", "env"]).auc.mean().reset_index()
x = json.loads((out / "X1_crossover.json").read_text())
gate = json.loads((paths.RESULTS / "external_isic2020" / "gate.json").read_text())
summary = {"X1_crossover": {k: x[k] for k in ("seed_delta_mean", "ci95_lo", "ci95_hi", "p_boot_two_sided") if k in x},
           "X1_supported": bool(x["ci95_lo"] > 0), "gate": gate,
           "auc": auc.to_dict("records"), "n_test_rev_images": int(p[(p.env == "test_rev") & (p.method == "erm")].image_id.nunique())}
(out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, default=float))
print(b[["trap", "env", "seed_delta_mean", "ci95_lo", "ci95_hi"]].round(3).to_string()); print(auc.round(3).to_string())
