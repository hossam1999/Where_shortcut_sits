"""A39 (reported): Holm-adjusted confidence intervals for the 8 P6 comparisons. Same estimator, seed and replicates as
wtss.replication_stats.paired_bootstrap (crossed seed x image bootstrap, 10,000); comparison i (ordered by p) gets the
percentile interval at level 1 - 0.05 / (8 - rank + 1) (Holm's step-down levels). Verdicts are unchanged (section 2 uses the
Holm-adjusted p). Writes results/replication/epvt/arms/comparison_holm_ci.{csv,md}."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from wtss.replication_stats import _metric
D = os.environ.get("WTSS_DATA", "/workspace/data"); RES = Path(f"{D}/results_push/results/replication/epvt/arms")


def replicates(seeds, n_boot=10000, seed=20260918, chunk=250):   # = paired_bootstrap's loop, returning the replicates
    rng = np.random.default_rng(seed)
    ids = sorted({str(i) for v in seeds.values() for i in v[0]}); pos = {k: j for j, k in enumerate(ids)}
    clusters = sorted(seeds); prep = {}
    for c in clusters:
        im, y, a, b = seeds[c]; fa, _ = _metric("auroc", y, a); fb, _ = _metric("auroc", y, b)
        prep[c] = (np.array([pos[str(i)] for i in im]), fa, fb)
    out = []
    for b0 in range(0, n_boot, chunk):
        B = min(chunk, n_boot - b0); W = rng.poisson(1.0, size=(B, len(ids))).astype(np.float64)
        cw = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=B).astype(float)
        vals = np.zeros((B, len(clusters)))
        for k, c in enumerate(clusters):
            idx, fa, fb = prep[c]; vals[:, k] = fa(W[:, idx]) - fb(W[:, idx])
        ok = ~np.isnan(vals)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(ok, vals, 0).__mul__(cw).sum(1) / np.where(ok, cw, 0).sum(1))
    arr = np.concatenate(out); return arr[np.isfinite(arr)]


s = pd.read_csv(RES / "scores_test_bias1_0.9.csv.gz"); c = pd.read_csv(RES / "comparison_bias1_0.9.csv")
c["rank"] = c.p_boot_two_sided.rank(method="first").astype(int); rows = []
for _, r in c.iterrows():
    x = s[s.bias == r.bias]; ep = x[x.method == "EPVT"]; per = {}
    for sd in range(5):
        a = x[(x.method == r.arm) & (x.seed == sd)]; e = ep[ep.seed == sd].set_index("image").loc[a.image]
        per[sd] = (a.image.tolist(), a.label.to_numpy(), a.score.to_numpy(), e.score.to_numpy())
    arr = replicates(per); alpha = 0.05 / (8 - r["rank"] + 1)
    rows.append({"bias": r.bias, "arm": r.arm, "delta": r.delta, "ci95_lo": r.ci95_lo, "ci95_hi": r.ci95_hi, "holm_level": 1 - alpha,
                 "holm_ci_lo": float(np.quantile(arr, alpha / 2)), "holm_ci_hi": float(np.quantile(arr, 1 - alpha / 2)), "p_holm": r.p_holm, "beaten": r.beaten})
o = pd.DataFrame(rows); o.to_csv(RES / "comparison_holm_ci.csv", index=False)
md = ["# P6 A39 — Holm-adjusted confidence intervals (reported; verdicts unchanged)", "",
      "| bias | arm | Δ | 95 % CI (unadjusted) | Holm level | Holm-adjusted CI | p Holm | beaten |", "|---|---|---|---|---|---|---|---|"]
md += [f"| {('1' if r.bias == 1 else r.bias)} | {r.arm} | {r.delta:+.3f} | [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}] | {r.holm_level:.4f} | "
       f"[{r.holm_ci_lo:+.3f}, {r.holm_ci_hi:+.3f}] | {r.p_holm:.4f} | {'yes' if r.beaten else 'no'} |" for _, r in o.iterrows()]
(RES / "comparison_holm_ci.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
