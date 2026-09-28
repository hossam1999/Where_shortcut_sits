"""Operating points on every unaltered test set with the crossed bootstrap (docs/PREREGISTRATION_FINAL.md, A1).
Uses crossed() of scripts/analysis/operating_points.py for each cohort; per-seed point values from per_seed().
  -> <WTSS_RESULTS>/final_op/operating_points_crossed_all.csv, operating_points_per_seed.csv"""
import importlib.util
from pathlib import Path

import pandas as pd

from wtss import paths

spec = importlib.util.spec_from_file_location("op", Path(__file__).resolve().parents[1] / "analysis" / "operating_points.py")
op = importlib.util.module_from_spec(spec); spec.loader.exec_module(op)
COH = ["thyroid", "isic_BCN", "isic_HAM", "isic_MSK", "capsule"]
rows, per = [], []
for c in COH:
    p, hpa = op.load(c)
    ps = op.per_seed(p, hpa); ps.insert(0, "cohort", c); per.append(ps)
    d = op.crossed(c)
    m = ps[ps.method.isin(["mask", "erm"])].groupby(["op", "method"])[["sens", "spec", "sens_conflict"]].mean()
    for r in d.itertuples():
        a, b = m.loc[(r.op, "mask"), r.metric], m.loc[(r.op, "erm"), r.metric]
        rows.append({**r._asdict(), "mask_value": a, "erm_value": b, "delta": a - b})
    print(c, "done", flush=True)
out = paths.ensure(paths.RESULTS / "final_op")
pd.DataFrame(rows).drop(columns=["Index"]).to_csv(out / "operating_points_crossed_all.csv", index=False)
pd.concat(per).to_csv(out / "operating_points_per_seed.csv", index=False)
