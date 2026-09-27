"""Numerical check of the linear-Gaussian theory (docs/THEORY.md).

Two feature blocks: true disease signal (SNR S) and artifact (SNR A); training environment P(A|Y) 0.9/0.1,
reversed test 0.1/0.9. Linear (logistic) head. Prediction: AUROC_rev = Φ((S − Ã) / sqrt(2(S + Ã))), Ã = 0.64 A / (1 + 0.09 A).
Scenarios: ERM; mask with out-of-ROI artifact (A -> 0, S -> S_m); mask with in-ROI artifact (A kept, S -> S_m);
subspace erasure (A -> 0, S -> S(1 − ρ²) when the erased direction overlaps the signal by ρ); group-balanced
reweighting (artifact–label association removed in training).
  python scripts/analysis/theory_sim.py   # -> results/theory/theory_sim.csv, paper/figures/theory.pdf
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from wtss import paths

rng = np.random.default_rng(0)


def sample(n, p1, p0, S, A, rho=0.0, d=8):
    """y ~ Bern(.5); a ~ Bern(p1 if y else p0); z = signal block (mean sqrt(S) y) + artifact block (mean sqrt(A) a)."""
    y = rng.integers(0, 2, n)
    a = np.where(y == 1, rng.random(n) < p1, rng.random(n) < p0).astype(int)
    zs = rng.standard_normal((n, d)); zs[:, 0] += np.sqrt(S) * y
    za = rng.standard_normal((n, d)); za[:, 0] += np.sqrt(A) * a
    return np.c_[zs, za], y, a


def fit_eval(S, A, balanced=False, n=20000):
    X, y, a = sample(n, .9, .1, S, A)
    w = None
    if balanced:
        g = 2 * y + a; cnt = np.bincount(g, minlength=4); w = (1 / cnt[g]) * len(g) / 4
    clf = LogisticRegression(max_iter=2000).fit(X, y, sample_weight=w)
    Xr, yr, _ = sample(n, .1, .9, S, A)
    return roc_auc_score(yr, clf.decision_function(Xr))


def theory(S, A, p1=0.9, p0=0.1):
    """LDA-optimal linear head, Gaussian features: AUROC_rev = Φ((S − Ã) / sqrt(2 (S + Ã))), where
    Ã = (p1 − p0)² A / (1 + p1 (1 − p1) A) is the artifact's effective SNR in the correlated training environment."""
    At = (p1 - p0) ** 2 * A / (1 + p1 * (1 - p1) * A)
    return norm.cdf((S - At) / np.sqrt(2 * (S + At))) if S + At > 0 else 0.5


def main():
    S, A = 1.0, 2.0
    rows = []
    for Sm in np.linspace(0.2, 1.0, 5):  # fraction of true signal kept by masking (background context removed)
        s_m = S * Sm
        rows += [{"scenario": "ERM", "S_kept": Sm, "sim": fit_eval(S, A), "theory": theory(S, A)},
                 {"scenario": "mask, artifact out-of-ROI", "S_kept": Sm, "sim": fit_eval(s_m, 0.0), "theory": theory(s_m, 0.0)},
                 {"scenario": "mask, artifact in-ROI", "S_kept": Sm, "sim": fit_eval(s_m, A), "theory": theory(s_m, A)},
                 {"scenario": "mask + erase (in-ROI)", "S_kept": Sm, "sim": fit_eval(s_m, 0.0), "theory": theory(s_m, 0.0)},
                 {"scenario": "balanced (any location)", "S_kept": Sm, "sim": fit_eval(S, A, balanced=True), "theory": norm.cdf(np.sqrt(S / 2))}]
    for rho in (0.0, 0.3, 0.6, 0.9):  # erased subspace overlapping the disease direction (debris ~ fibrin)
        s_e = S * (1 - rho ** 2)
        rows.append({"scenario": f"erase, overlap rho={rho}", "S_kept": 1.0, "sim": fit_eval(s_e, 0.0), "theory": theory(s_e, 0.0)})
    d = pd.DataFrame(rows)
    out = paths.ensure(paths.RESULTS / "theory")
    d.to_csv(out / "theory_sim.csv", index=False)
    print(d.round(3).to_string())
    print("max |sim − theory| =", float((d.sim - d.theory).abs().max().round(3)))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for sc, st in (("mask, artifact out-of-ROI", "-"), ("mask, artifact in-ROI", "--"), ("mask + erase (in-ROI)", ":"),
                       ("balanced (any location)", "-.")):
            q = d[d.scenario == sc]
            ax.plot(q.S_kept, q.theory, st, label=sc); ax.scatter(q.S_kept, q.sim, s=12)
        ax.axhline(theory(S, A), color="k", lw=.8, label="ERM")
        ax.set_xlabel("fraction of disease signal kept by the mask"); ax.set_ylabel("reversed-test AUROC")
        ax.legend(fontsize=6); fig.tight_layout()
        figd = paths.ensure(paths.REPO_ROOT / "paper" / "figures"); fig.savefig(figd / "theory.pdf"); fig.savefig(figd / "theory.png", dpi=200)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
