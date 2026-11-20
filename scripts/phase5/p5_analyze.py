"""Phase 5 analysis (docs/PREREGISTRATION_PHASE5.md, approved 2026-11-20). One selection criterion: val_clean AUROC.

    features (GPU, venv-wtss): DINOv2 ViT-B/14 @ 518 (wtss.umte masked view: outside the nodule ROI = mean colour) of every
              image of the phase-5 environments, and the U-MtE insertion view (generic overlay, then mask) of training images
    analyze (CPU): per trap x fold
      RR-ClArC   (lambda, layer) of the 9 fine-tunes chosen by val_clean AUROC (their predictions)
      ISNet      as trained (ImageNet init), ERM ResNet-50d (RR-ClArC's base) reported
      arm d      EfficientNet-B0 features (p5_run) -> mask_cmc head, mode and C (A21 grid) by val_clean AUROC
      DINOv2 + mask_cmc   masked features -> the same head (A21 grid)
      U-MtE      label-free difference subspace of (masked, masked-insert) training features -> logistic head, C (A21) by val_clean
    Outcomes: AUROC on test_rev (primary), clean and test_corr. Comparisons (Trap A, Holm family of 4): arm d vs RR-ClArC,
    arm d vs ISNet, DINOv2 + mask_cmc vs RR-ClArC, DINOv2 + mask_cmc vs ISNet; crossed fold x image bootstrap
    (wtss.replication_stats.paired_bootstrap, 10,000; folds as clusters, one Poisson weight per image); "beaten" = higher
    mean, 95 % CI above 0, Holm p < 0.05, AND clean-AUROC drop <= 0.01 (mean over folds, ours vs theirs). U-MtE and every
    Trap B comparison descriptive. Outputs: $WTSS_DATA/results_push/results/replication/phase5/analysis/ (pushed).
"""
import argparse, importlib.util, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "round8")]
D = Path(os.environ.get("WTSS_DATA", "/workspace/data"))
P5 = D / "phase5"
OUT = D / "results_push/results/replication/phase5/analysis"
W = D / "phase5_arms"
ENVS = ("train_corr", "val_clean", "test_corr", "test_rev", "clean")
TESTS = ("test_rev", "clean", "test_corr")
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0)   # A21


def load_p5_run():
    spec = importlib.util.spec_from_file_location("p5r", ROOT / "scripts/phase5/p5_run.py"); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def features():
    from wtss.umte import UMtE
    m = load_p5_run(); envs, cache = m.thyroid()
    render = m.make_renderers(cache, 518, [])
    ids = sorted({i for (t, s, f, e), df in envs.items() if t in ("trapA", "trapB") and s == 42 and f in (0, 1, 2) and e in ENVS for i in df.image_id})
    tr = sorted({i for (t, s, f, e), df in envs.items() if t in ("trapA", "trapB") and s == 42 and f in (0, 1, 2) and e == "train_corr" for i in df.image_id})
    W.mkdir(parents=True, exist_ok=True); um = UMtE("dino518", device="cuda", seed=0, batch_size=16)
    for name, pool, ins in (("dino518_mask.npz", ids, False), ("dino518_mask_insert_train.npz", tr, True)):
        if (W / name).exists():
            continue
        X = []
        for s0 in range(0, len(pool), 128):
            b = pool[s0:s0 + 128]
            X.append(um.masked_features([render["erm"](i) for i in b], [np.asarray(cache.get(i)[1]) > 0 for i in b], insert=ins, keys=b if ins else None))
            print(name, s0 + len(b), "/", len(pool), flush=True)
        np.savez(W / name, ids=np.array(pool, dtype="U"), X=np.concatenate(X))


def _head(Xs, y, a, modes, grid=C_GRID):
    """mask_cmc (modes) or plain (mode None) + class-balanced logistic regression; (mode, C) by val_clean AUROC"""
    import candidates as CA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    best = None
    for mode in modes:
        P = (lambda X: X) if mode is None else CA.Projector(CA.cmc_directions(Xs["train_corr"], y["train_corr"], a["train_corr"], mode))
        Z = {k: P(v) for k, v in Xs.items()}
        for C in grid:
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, solver="liblinear", random_state=0).fit(Z["train_corr"], y["train_corr"])
            v = roc_auc_score(y["val_clean"], clf.decision_function(Z["val_clean"]))
            if best is None or v > best[0]:
                best = (v, mode or "none", C, {k: clf.decision_function(Z[k]) for k in TESTS})
    v, mode, C, sc = best
    return sc, {"mode": mode, "C": C, "C_edge": "lowest" if C == grid[0] else "highest" if C == grid[-1] else "", "val_clean_auroc": float(v)}


def analyze():
    from sklearn.metrics import roc_auc_score
    from wtss.methods.insertion import fit_difference_subspace
    zm = np.load(W / "dino518_mask.npz"); pm = {i: j for j, i in enumerate(zm["ids"].astype(str))}
    zi = np.load(W / "dino518_mask_insert_train.npz"); pi = {i: j for j, i in enumerate(zi["ids"].astype(str))}
    scores, sel = [], []
    for trap in ("trapA", "trapB"):
        for f in (0, 1, 2):
            d = P5 / f"{trap}_f{f}"
            base = pd.read_csv(d / "erm_r50d" / "preds.csv.gz")
            env = {e: base[base.env == e].reset_index(drop=True) for e in ENVS}
            y = {e: env[e].y.to_numpy() for e in ENVS}; a = {e: env[e].a.to_numpy() for e in ENVS}
            ids = {e: env[e].image_id.astype(str).to_numpy() for e in ENVS}

            def add(method, sc, info):
                for t in TESTS:
                    scores.append(pd.DataFrame({"trap": trap, "fold": f, "method": method, "test": t, "image_id": ids[t], "y": y[t], "score": sc[t]}))
                sel.append({"trap": trap, "fold": f, "method": method, **info})

            def from_preds(run):
                p = pd.read_csv(d / run / "preds.csv.gz")
                out = {}
                for t in TESTS:
                    q = p[p.env == t].set_index("image_id").loc[ids[t]]; assert (q.y.to_numpy() == y[t]).all(); out[t] = q.prob.to_numpy()
                v = p[p.env == "val_clean"]; return out, float(roc_auc_score(v.y, v.prob))
            # RR-ClArC: the 9 fine-tunes, chosen on val_clean
            cand = []
            for run in sorted(x.name for x in d.glob("rrclarc_lam*")):
                sc, v = from_preds(run); cand.append((v, run, sc))
            v, run, sc = max(cand, key=lambda c: c[0])
            add("RR-ClArC", sc, {"choice": run, "val_clean_auroc": v, "n_candidates": len(cand)})
            for run, name in (("isnet", "ISNet"), ("erm_r50d", "ERM ResNet-50d (RR-ClArC base)")):
                sc, v = from_preds(run); add(name, sc, {"choice": run, "val_clean_auroc": v})
            # arm d
            F = np.load(d / "armd" / "features.npz")
            for e in ENVS:
                assert (F[f"id_{e}"].astype(str) == ids[e]).all()
            sc, info = _head({e: F[f"X_{e}"] for e in ENVS}, y, a, ("pooled", "per_class")); add("arm d (EffNet-B0 + mask_cmc)", sc, info)
            # DINOv2 + mask_cmc, U-MtE
            Xm = {e: zm["X"][[pm[i] for i in ids[e]]] for e in ENVS}
            sc, info = _head(Xm, y, a, ("pooled", "per_class")); add("DINOv2 + mask_cmc", sc, info)
            er = fit_difference_subspace(Xm["train_corr"], zi["X"][[pi[i] for i in ids["train_corr"]]], energy=0.90, max_k=64, seed=0)
            sc, info = _head({e: er(v) for e, v in Xm.items()}, y, a, (None,)); add("U-MtE", sc, info)
    S = pd.concat(scores); OUT.mkdir(parents=True, exist_ok=True)
    S.to_csv(OUT / "scores.csv.gz", index=False); pd.DataFrame(sel).to_csv(OUT / "selection.csv", index=False)
    return S


def compare(S):
    from sklearn.metrics import roc_auc_score
    from wtss.replication_stats import paired_bootstrap, holm, beaten
    au = S.groupby(["trap", "method", "test", "fold"]).apply(lambda g: roc_auc_score(g.y, g.score)).rename("auroc").reset_index()
    tab = au.groupby(["trap", "method", "test"]).auroc.agg(["mean", "std"]).reset_index()
    tab.to_csv(OUT / "auroc_table.csv", index=False)
    rows = []
    for trap in ("trapA", "trapB"):
        for ours in ("arm d (EffNet-B0 + mask_cmc)", "DINOv2 + mask_cmc", "U-MtE"):
            for theirs in ("RR-ClArC", "ISNet"):
                per = {}
                for f in (0, 1, 2):
                    x = S[(S.trap == trap) & (S.fold == f) & (S.test == "test_rev")]
                    o = x[x.method == ours].set_index("image_id"); t = x[x.method == theirs].set_index("image_id").loc[o.index]
                    per[f] = (o.index.tolist(), o.y.to_numpy(), o.score.to_numpy(), t.score.to_numpy())
                b = paired_bootstrap(per, "auroc")
                cl = au[(au.trap == trap) & (au.test == "clean")]
                cd = cl[cl.method == ours].auroc.mean() - cl[cl.method == theirs].auroc.mean()
                rows.append({"trap": trap, "ours": ours, "theirs": theirs, "family": trap == "trapA" and ours != "U-MtE",
                             **{k: b[k] for k in ("ours_mean", "theirs_mean", "delta", "ci95_lo", "ci95_hi", "p_boot_two_sided")},
                             "clean_diff": cd})
    c = pd.DataFrame(rows); m = c.family
    c.loc[m, "p_holm"] = holm(c.loc[m, "p_boot_two_sided"].tolist())
    c["verdict"] = [("beaten" if beaten(r, r.p_holm) and r.clean_diff >= -0.01 else
                     "not beaten (clean drop > 0.01)" if beaten(r, r.p_holm) else "not beaten") if r.family else "descriptive"
                    for _, r in c.iterrows()]
    c.to_csv(OUT / "comparison.csv", index=False)
    md = ["# Phase 5 (thyroid TN3K calipers) — results (PREREGISTRATION_PHASE5.md) — machine 2", "",
          "AUROC, mean ± SD over folds 0–2 (seed 42). Test `test_rev` = marker–label correlation reversed (primary); `clean` = no "
          "marker; `test_corr` = as in training.", "", "| trap | method | test_rev | clean | test_corr |", "|---|---|---|---|---|"]
    for (trap, meth), g in tab.groupby(["trap", "method"]):
        v = {r.test: f"{r['mean']:.3f} ± {r['std']:.3f}" for _, r in g.iterrows()}
        md.append(f"| {trap} | {meth} | {v.get('test_rev', '')} | {v.get('clean', '')} | {v.get('test_corr', '')} |")
    md += ["", "Comparisons on test_rev (crossed fold × image bootstrap; Holm family = 4 on Trap A; 'beaten' also needs clean drop ≤ 0.01):", "",
           "| trap | ours | theirs | ours | theirs | Δ [95 % CI] | p | p Holm | clean Δ | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in c.iterrows():
        md.append(f"| {r.trap} | {r.ours} | {r.theirs} | {r.ours_mean:.3f} | {r.theirs_mean:.3f} | {r.delta:+.3f} [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}] | "
                  f"{r.p_boot_two_sided:.4f} | {'' if pd.isna(r.p_holm) else f'{r.p_holm:.4f}'} | {r.clean_diff:+.3f} | {r.verdict} |")
    (OUT / "results.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["features", "analyze"], required=True)
    a = ap.parse_args()
    if a.stage == "features":
        features()
    else:
        compare(analyze())
