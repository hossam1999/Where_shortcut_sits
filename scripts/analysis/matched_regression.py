"""Regression-adjusted (doubly robust) location crossover on the matched traps (docs/PREREGISTRATION_FINAL.md, A2).

Outcome per reversed-test image: placement value under masking minus under ERM (positives: weighted share of
negatives scored below, ties 1/2; negatives: weighted share of positives scored above, complemented), so that the mean
over images of either class equals the AUROC difference and the between-trap difference of means equals the crossover.
Within each seed: weighted least squares of the outcome on [1, Trap B, label, all matching covariates]; the adjusted
crossover is the Trap-B coefficient averaged over seeds. Crossed bootstrap: seeds resampled, one Poisson weight per image
(shared across seeds and traps); placement values and regressions recomputed in every replicate.

  python scripts/analysis/matched_regression.py   (reads <WTSS_RESULTS>/{cohort}/dino518_matched/predictions.csv.gz)
  -> <WTSS_RESULTS>/final_a2/matched_regression.csv, balance_after_matching.csv
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from wtss import paths
from wtss.matching import covariates, design_matrix

R = paths.RESULTS
N_BOOT = 2000
COH = {"ISIC hair": ("isic", "spec_e13"), "Thyroid": ("thyroid", "thyroid"), "Ovary": ("ovary", "ovary")}


def cohort_table(name):
    if name == "isic":
        from wtss.data.isic2019_spec import load_spec_cohort
        return load_spec_cohort()
    spec = importlib.util.spec_from_file_location("rt", paths.REPO_ROOT / "scripts" / "run_thyroid_traps.py")
    rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
    return {"thyroid": rt.cohort, "ovary": rt.ovary_cohort}[name]()


def placement(y, s, w):
    """Weighted placement values: mean over positives (weights w) and over negatives both equal the weighted AUROC."""
    pos, neg = y == 1, y == 0
    out = np.full(len(y), np.nan)
    sn, wn = s[neg], w[neg]
    o = np.argsort(sn); sn, wn = sn[o], wn[o]
    cum = np.r_[0, np.cumsum(wn)]
    lo, hi = np.searchsorted(sn, s[pos], "left"), np.searchsorted(sn, s[pos], "right")
    out[pos] = (cum[lo] + 0.5 * (cum[hi] - cum[lo])) / max(cum[-1], 1e-12)
    sp, wp = s[pos], w[pos]
    o = np.argsort(sp); sp, wp = sp[o], wp[o]
    cum = np.r_[0, np.cumsum(wp)]
    lo, hi = np.searchsorted(sp, s[neg], "left"), np.searchsorted(sp, s[neg], "right")
    above = cum[-1] - cum[hi]
    out[neg] = (above + 0.5 * (cum[hi] - cum[lo])) / max(cum[-1], 1e-12)
    return out


def prepare(label):
    name, base = COH[label]
    p = pd.read_csv(R / base / "dino518_matched" / "predictions.csv.gz")
    p = p[p.env == "test_rev"]
    c = cohort_table(name)
    c["image_id"] = c.image_id.astype(str)
    tab, num, cat = covariates(name, c)
    blocks = []
    for (s, trap), q in p.groupby(["seed", "trap"]):
        a = q[q.method == "mask"][["image_id", "y", "prob"]].rename(columns={"prob": "pm"})
        b = q[q.method == "erm"][["image_id", "y", "prob"]].rename(columns={"prob": "pe"})
        z = a.merge(b, on=["image_id", "y"])
        z["seed"], z["trapB"] = s, int(trap == "trapB")
        blocks.append(z)
    d = pd.concat(blocks, ignore_index=True)
    d["image_id"] = d.image_id.astype(str)
    X = design_matrix(tab.loc[d.image_id].reset_index(drop=True), num, cat)
    return d, X, (num, cat)


def coef(d, X, w, cols=True):
    """Per seed: WLS Trap-B coefficient (adjusted if cols) with image weights w (aligned with d)."""
    out = {}
    for s in sorted(d.seed.unique()):
        k = np.flatnonzero(d.seed.to_numpy() == s)
        D = np.empty(len(k))
        for t in (0, 1):
            kk = k[d.trapB.to_numpy()[k] == t]
            y = d.y.to_numpy()[kk]
            D_t = placement(y, d.pm.to_numpy()[kk], w[kk]) - placement(y, d.pe.to_numpy()[kk], w[kk])
            D[np.searchsorted(k, kk)] = D_t
        ww = w[k]
        keep = ww > 0
        Z = [np.ones(len(k)), d.trapB.to_numpy()[k].astype(float)]
        if cols:
            Z += [d.y.to_numpy()[k].astype(float)] + [X[k, j] for j in range(X.shape[1])]
        Z = np.stack(Z, 1)[keep]
        sw = np.sqrt(ww[keep])
        beta = np.linalg.lstsq(Z * sw[:, None], D[keep] * sw, rcond=None)[0]
        out[s] = beta[1]
    return out


def boot(d, X, seed):
    rng = np.random.default_rng(seed)
    ids = d.image_id.to_numpy()
    uniq, inv = np.unique(ids, return_inverse=True)
    seeds = sorted(d.seed.unique())
    res = {"adj": [], "unadj": []}
    for _ in range(N_BOOT):
        w = rng.poisson(1.0, len(uniq)).astype(float)[inv]
        cw = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)))
        for key, cols in (("adj", True), ("unadj", False)):
            c = coef(d, X, w, cols)
            res[key].append(sum(cw[j] * c[s] for j, s in enumerate(seeds)) / cw.sum())
    return {k: np.asarray(v) for k, v in res.items()}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def main():
    rows = []
    for label in COH:
        d, X, _ = prepare(label)
        one = np.ones(len(d))
        pa, pu = coef(d, X, one, True), coef(d, X, one, False)
        b = boot(d, X, 20260928 + sum(map(ord, label)))
        r = {"cohort": label, "n_test_images_per_seed": int(len(d) / d.seed.nunique()), "n_covariate_columns": X.shape[1],
             "unadjusted": float(np.mean(list(pu.values()))), "adjusted": float(np.mean(list(pa.values())))}
        for k in ("unadj", "adj"):
            lo, hi = np.percentile(b[k], [2.5, 97.5])
            r[f"{k}_lo"], r[f"{k}_hi"] = float(lo), float(hi)
            r[f"{k}_p"] = float(max(2 * min((b[k] <= 0).mean(), (b[k] >= 0).mean()), 1 / N_BOOT))
        rows.append(r)
        print(r, flush=True)
    out = pd.DataFrame(rows)
    out["adj_p_holm"] = holm(out.adj_p)
    out["A2_supported"] = (out.adj_lo > 0) & (out.adj_p_holm < 0.05)
    o = paths.ensure(R / "final_a2")
    out.to_csv(o / "matched_regression.csv", index=False)
    bal = []
    for label, (name, base) in COH.items():
        f = R / base / "dino518_matched" / "match_after.csv"
        if f.exists():
            bal.append(pd.read_csv(f).assign(cohort=label))
    if bal:
        pd.concat(bal).to_csv(o / "balance_after_matching.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
