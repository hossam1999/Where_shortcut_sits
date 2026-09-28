"""Old (per-seed bootstrap, archived runs) versus new (crossed bootstrap, regenerated runs) intervals, row by row
(docs/PREREGISTRATION_FINAL.md, A1).

  python scripts/final/old_vs_new.py
  -> results/bootstrap_correction/sweeps_umte_old_vs_new.{md,csv}   (every regenerated result file with intervals)
Keys: the non-numeric columns of each table (plus overlap); JSON results are flattened. Verdict = CI excludes zero.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OLD, NEW = ROOT / "results", ROOT / "results" / "rerun_2026-09-28"
OUT = ROOT / "results" / "bootstrap_correction"
VAL = ("seed_delta_mean", "ci95_lo", "ci95_hi")


def flat_json(f: Path):
    j = json.loads(f.read_text())
    rows = []
    def add(k, v):
        if isinstance(v, dict) and {"ci95_lo", "ci95_hi"} <= set(v):
            est = v.get("seed_delta_mean", v.get("delta_mean_bootstrap", v.get("auc_mean")))
            rows.append({"key": k, "est": est, "lo": v["ci95_lo"], "hi": v["ci95_hi"]})
        elif isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
            rows.append({"key": k, "est": v[0], "lo": v[1], "hi": v[2]})
        elif isinstance(v, dict):
            for kk, vv in v.items():
                add(f"{k}|{kk}" if k else kk, vv)
    if isinstance(j, dict) and {"ci95_lo", "ci95_hi"} <= set(j):
        add(f.stem, j)
    elif isinstance(j, dict):
        for k, v in j.items():
            add(k, v)
    return pd.DataFrame(rows)


def flat_csv(f: Path):
    d = pd.read_csv(f)
    if not set(VAL) <= set(d.columns) and {"estimate", "ci95_lo", "ci95_hi"} <= set(d.columns):  # PRIMARY_CLAIMS layout
        d = d.rename(columns={"estimate": "seed_delta_mean"}).drop(columns=["source"], errors="ignore")
    if not set(VAL) <= set(d.columns):
        return pd.DataFrame()
    txt = lambda c: d[c].dtype == object or pd.api.types.is_string_dtype(d[c])  # pandas 3 reads text as `str`
    keys = [c for c in d.columns if (txt(c) and c not in ("seed_deltas_json", "estimand", "estimator"))
            or c in ("overlap", "lo", "hi")]
    d["key"] = d[keys].astype(str).agg("|".join, axis=1) if keys else d.index.astype(str)
    return d.rename(columns={"seed_delta_mean": "est", "ci95_lo": "lo", "ci95_hi": "hi"})[["key", "est", "lo", "hi"]]


def main():
    rows = []
    for fn in [p for p in NEW.rglob("*") if p.suffix in (".csv", ".json") and "_done" not in p.parts and "_logs" not in p.parts]:
        rel = fn.relative_to(NEW)
        fo = OLD / rel
        if not fo.exists() or fo.stat().st_size > 20_000_000:
            continue
        try:
            a, b = (flat_json(fo), flat_json(fn)) if fn.suffix == ".json" else (flat_csv(fo), flat_csv(fn))
        except Exception:
            continue
        if a.empty or b.empty:
            continue
        m = a.merge(b, on="key", suffixes=("_old", "_new"))
        for r in m.itertuples():
            rows.append({"file": str(rel), "key": r.key, "old_est": r.est_old, "old_lo": r.lo_old, "old_hi": r.hi_old,
                         "new_est": r.est_new, "new_lo": r.lo_new, "new_hi": r.hi_new})
    # review3/crossed_ci.csv carries both intervals of each quoted claim (orig_* = old estimator on the same predictions)
    cc = NEW / "review3" / "crossed_ci.csv"
    if cc.exists():
        c = pd.read_csv(cc).dropna(subset=["orig_lo", "orig_hi"])
        for r in c.itertuples():
            rows.append({"file": "review3/crossed_ci.csv", "key": f"{r.claim}|{r.cohort}", "old_est": r.orig_estimate,
                         "old_lo": r.orig_lo, "old_hi": r.orig_hi, "new_est": r.estimate, "new_lo": r.crossed_lo,
                         "new_hi": r.crossed_hi})
    d = pd.DataFrame(rows)
    if d.empty:
        print("nothing to compare yet"); return
    d["old_excludes_0"] = (d.old_lo > 0) | (d.old_hi < 0)
    d["new_excludes_0"] = (d.new_lo > 0) | (d.new_hi < 0)
    d["verdict_changed"] = d.old_excludes_0 != d.new_excludes_0
    d["width_ratio"] = (d.new_hi - d.new_lo) / (d.old_hi - d.old_lo).replace(0, np.nan)
    d["est_diff"] = d.new_est - d.old_est
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "sweeps_umte_old_vs_new.csv", index=False)
    f = lambda e, lo, hi: f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    lines = ["# Old versus new intervals (A1, docs/PREREGISTRATION_FINAL.md)", "",
             "Old: archived runs, per-seed bootstrap (images resampled independently within each seed). New: regenerated runs "
             "(results/rerun_2026-09-28/), crossed seed x image bootstrap. Verdict = CI excludes zero.", "",
             f"- rows compared: {len(d)}; verdict changed: {int(d.verdict_changed.sum())} "
             f"(lost significance: {int((d.old_excludes_0 & ~d.new_excludes_0).sum())}, gained: {int((~d.old_excludes_0 & d.new_excludes_0).sum())})",
             f"- median width ratio new/old: {d.width_ratio.median():.2f} (IQR {d.width_ratio.quantile(.25):.2f}--{d.width_ratio.quantile(.75):.2f})",
             f"- point estimates differing by more than 0.03: {int((d.est_diff.abs() > 0.03).sum())}", "",
             "## Verdict changes", "", "| file | row | old | new | change |", "|---|---|---|---|---|"]
    for r in d[d.verdict_changed].itertuples():
        lines.append(f"| {r.file} | {r.key} | {f(r.old_est, r.old_lo, r.old_hi)} | {f(r.new_est, r.new_lo, r.new_hi)} | "
                     f"{'lost' if r.old_excludes_0 else 'gained'} |")
    lines += ["", "## All rows", "", "| file | row | old | new | width ratio |", "|---|---|---|---|---|"]
    for r in d.itertuples():
        lines.append(f"| {r.file} | {r.key} | {f(r.old_est, r.old_lo, r.old_hi)} | {f(r.new_est, r.new_lo, r.new_hi)} | {r.width_ratio:.2f} |")
    (OUT / "sweeps_umte_old_vs_new.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
