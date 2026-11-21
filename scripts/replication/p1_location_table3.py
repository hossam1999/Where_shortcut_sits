"""P1 location test, formal 3-seed analysis (A43 addendum; section 5; A15, A16, A20): seeds 1234 (committed, machine 1),
3232 and 3221 (machine 2). Per position (tube inside / outside the lungs + heart) and method: AUROC on the reversed,
correlated and clean tests (3-seed means); gain over Standard on the reversed test; paired difference vs Masked; and the
crossover = gain outside - gain inside. Crossed seed x image bootstrap (section 2; wtss.replication_stats' estimator): each
replicate resamples seeds (multinomial) and draws one Poisson(1) weight per test-image id, shared by seeds, methods and
both positions (the test images of a seed are the same at both positions); 10,000 replicates; percentile 95 % CI;
p = min(1, 2 min(P(<= 0), P(>= 0))). Registered expectations (section 5), Holm over the 5 tests of this phase:
  E-L1  Masked: crossover > 0 (CI above 0) and gain inside <= 0 or its CI includes 0;
  E-L2  mask_cmc arms a, b (and d, A20): gain inside > 0 (CI above 0);
  E-L3  U-MtE (arm c): gain inside > 0 (CI above 0).
Inputs: results/replication/viviano/scores/location_{pos}[_{arm}]_{seed}.csv.gz. Writes results/replication/viviano/location_3seed.{csv,md}.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from wtss.replication_stats import _metric, holm
from wtss.stats import _ci, boot_p
S = ROOT / "results/replication/viviano/scores"
SEEDS = (1234, 3232, 3221); POS = ("inside", "outside"); TESTS = ("rev", "corr", "clean")
ARMS = {"a_mask_cmc_their_net": "a: mask_cmc on their net", "b_dino518_mask_cmc": "b: DINOv2 + mask_cmc", "c_umte": "c: U-MtE",
        "d_effb0_mask_cmc": "d: EfficientNet-B0 + mask_cmc (A20)"}
METHODS = {"standard": "Standard (theirs)", "masked": "Masked (theirs, A5)", **ARMS}


def load():
    rows = []
    for s in SEEDS:
        for p in POS:
            t = pd.read_csv(S / f"location_{p}_{s}.csv.gz")
            for m in ("standard", "masked"):
                q = t[t.method == m]; rows.append(q[["image_id", "label", "test_set", "score"]].assign(method=m, pos=p, seed=s))
            for a in ARMS:
                q = pd.read_csv(S / f"location_{p}_{a}_{s}.csv.gz"); rows.append(q[["image_id", "label", "test_set", "score"]].assign(method=a, pos=p, seed=s))
    return pd.concat(rows)


def vectors(X, seed, test="rev"):
    """per seed: image ids (inside order) and every method x position score vector, aligned on the same images"""
    x = X[(X.seed == seed) & (X.test_set == test)]
    base = x[(x.method == "standard") & (x.pos == "inside")]
    ids = base.image_id.astype(str).to_numpy(); y = base.label.to_numpy()
    sc = {}
    for (m, p), g in x.groupby(["method", "pos"]):
        g = g.set_index(g.image_id.astype(str)).loc[ids]; assert (g.label.to_numpy() == y).all(); sc[(m, p)] = g.score.to_numpy()
    return ids, y, sc


def boot(X, terms, n_boot=10000, seed=20260918, chunk=250):
    """terms: list of (coef, method, pos); statistic = mean over seeds of sum coef * AUROC(method, pos) on the rev test."""
    V = {s: vectors(X, s) for s in SEEDS}
    ids = sorted({i for v in V.values() for i in v[0]}); pos = {k: j for j, k in enumerate(ids)}
    prep, point = {}, []
    for s, (im, y, sc) in V.items():
        fs = [(c, *_metric("auroc", y, sc[(m, p)])) for c, m, p in terms]
        prep[s] = (np.array([pos[i] for i in im]), fs); point.append(sum(c * v for c, _, v in fs))
    rng = np.random.default_rng(seed); out = []
    for b0 in range(0, n_boot, chunk):
        B = min(chunk, n_boot - b0); W = rng.poisson(1.0, size=(B, len(ids))).astype(np.float64)
        cw = rng.multinomial(len(SEEDS), np.full(len(SEEDS), 1 / len(SEEDS)), size=B).astype(float)
        vals = np.stack([sum(c * f(W[:, idx]) for c, f, _ in fs) for s, (idx, fs) in prep.items()], 1)
        out.append((vals * cw).sum(1) / cw.sum(1))
    arr = np.concatenate(out); arr = arr[np.isfinite(arr)]; lo, hi = _ci(arr)
    return {"delta": float(np.mean(point)), "ci95_lo": lo, "ci95_hi": hi, "p": boot_p(arr), "per_seed": [float(v) for v in point]}


def main():
    from sklearn.metrics import roc_auc_score
    X = load()
    au = X.groupby(["method", "pos", "test_set", "seed"]).apply(lambda g: roc_auc_score(g.label, g.score)).rename("auroc").reset_index()
    T = au.groupby(["method", "pos", "test_set"]).auroc.agg(["mean", "std"]).reset_index()
    rows = []
    for m in METHODS:
        for p in POS:
            r = {"method": m, "pos": p}
            for t in TESTS:
                q = T[(T.method == m) & (T.pos == p) & (T.test_set == t)].iloc[0]; r[f"{t}_mean"], r[f"{t}_sd"] = q["mean"], q["std"]
            if m != "standard":
                g = boot(X, [(1, m, p), (-1, "standard", p)]); r.update({f"gain_{k}": v for k, v in g.items() if k != "per_seed"})
            if m not in ("standard", "masked"):
                g = boot(X, [(1, m, p), (-1, "masked", p)]); r.update({f"vs_masked_{k}": v for k, v in g.items() if k != "per_seed"})
            rows.append(r)
        if m != "standard":
            c = boot(X, [(1, m, "outside"), (-1, "standard", "outside"), (-1, m, "inside"), (1, "standard", "inside")])
            for r in rows[-2:]:
                r.update({f"crossover_{k}": v for k, v in c.items() if k != "per_seed"})
    R = pd.DataFrame(rows)
    # registered expectations, Holm over the 5 tests (section 5)
    def get(m, p, col): return R[(R.method == m) & (R.pos == p)].iloc[0][col]
    tests = [("E-L1 Masked crossover > 0", "masked", "outside", "crossover"), ("E-L2 arm a gain inside > 0", "a_mask_cmc_their_net", "inside", "gain"),
             ("E-L2 arm b gain inside > 0", "b_dino518_mask_cmc", "inside", "gain"), ("E-L2 arm d gain inside > 0 (A20)", "d_effb0_mask_cmc", "inside", "gain"),
             ("E-L3 arm c gain inside > 0", "c_umte", "inside", "gain")]
    ph = holm([get(m, p, f"{k}_p") for _, m, p, k in tests])
    E = []
    for (name, m, p, k), h in zip(tests, ph):
        d, lo, hi = get(m, p, f"{k}_delta"), get(m, p, f"{k}_ci95_lo"), get(m, p, f"{k}_ci95_hi")
        sig = lo > 0 and h < 0.05
        if name.startswith("E-L1"):
            gi_lo, gi_hi, gi = get(m, "inside", "gain_ci95_lo"), get(m, "inside", "gain_ci95_hi"), get(m, "inside", "gain_delta")
            second = gi <= 0 or (gi_lo <= 0 <= gi_hi)
            verdict = "supported" if (sig and second) else ("contradicted" if hi < 0 else "not supported")
        else:
            verdict = "supported" if sig else ("contradicted" if hi < 0 and h < 0.05 else "not supported")
        E.append({"expectation": name, "delta": d, "ci95_lo": lo, "ci95_hi": hi, "p": get(m, p, f"{k}_p"), "p_holm": h, "verdict": verdict})
    E = pd.DataFrame(E)
    out = ROOT / "results/replication/viviano"; R.to_csv(out / "location_3seed.csv", index=False); E.to_csv(out / "location_3seed_expectations.csv", index=False)
    md = ["# P1 location test — formal 3-seed analysis (A43; section 5) — machine 2", "",
          "Seeds 1234 (machine 1, committed per-image scores; different GPU and software build), 3232 and 3221 (machine 2). Their "
          "code and Standard config; Masked = the Standard model with test masking (A5); our arms a–d. Crossed seed × image "
          "bootstrap (10,000; one Poisson weight per image shared by seeds, methods and positions). Epoch selection: their rule "
          "(registered table; A17's epoch-100 sensitivity analysis is not repeated).", "",
          "| method | position | reversed AUROC (mean ± SD) | correlated | clean | gain over Standard [95 % CI] | vs Masked [95 % CI] | crossover [95 % CI] |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        g = "—" if pd.isna(r.get("gain_delta")) else f"{r.gain_delta:+.3f} [{r.gain_ci95_lo:+.3f}, {r.gain_ci95_hi:+.3f}]"
        vm = "—" if pd.isna(r.get("vs_masked_delta")) else f"{r.vs_masked_delta:+.3f} [{r.vs_masked_ci95_lo:+.3f}, {r.vs_masked_ci95_hi:+.3f}]"
        cr = "—" if pd.isna(r.get("crossover_delta")) else f"{r.crossover_delta:+.3f} [{r.crossover_ci95_lo:+.3f}, {r.crossover_ci95_hi:+.3f}]"
        md.append(f"| {METHODS[r.method]} | {r.pos} | {r.rev_mean:.3f} ± {r.rev_sd:.3f} | {r.corr_mean:.3f} | {r.clean_mean:.3f} | {g} | {vm} | {cr} |")
    md += ["", "**Registered expectations (section 5), Holm over 5:**", "", "| expectation | Δ [95 % CI] | p | p Holm | verdict |", "|---|---|---|---|---|"]
    md += [f"| {e.expectation} | {e.delta:+.3f} [{e.ci95_lo:+.3f}, {e.ci95_hi:+.3f}] | {e.p:.4f} | {e.p_holm:.4f} | {e.verdict} |" for _, e in E.iterrows()]
    (out / "location_3seed.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
