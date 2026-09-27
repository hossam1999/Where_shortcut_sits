"""Theory -> empirical test (docs/PREREGISTRATION_THEORY_PREDICTION.md).

For every cell (run x trap/overlap x arm) fit the linear-Gaussian model's (S, A) to the observed clean and correlated
AUROC, then predict the reversed AUROC (never used in the fit). Zero shared or tuned parameters.
  python scripts/analysis/theory_predict.py   # -> results/theory/prediction_cells.csv, prediction_summary.json,
                                              #    paper/figures/theory_predict.{pdf,png}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm, pearsonr
from sklearn.metrics import roc_auc_score

from wtss import paths

P1, P0 = 0.9, 0.1
V = (P1 * (1 - P1) + P0 * (1 - P0)) / 2
R = paths.RESULTS

TRAPS = {("ISIC hair", "DINOv2"): "spec_e13/dino518_spec", ("ISIC hair", "DermLIP"): "spec_e13/dermlip224_spec",
         ("Thyroid", "DINOv2"): "thyroid/dino518_main", ("Thyroid", "MedSigLIP"): "thyroid/medsiglip448_main",
         ("Thyroid", "ConvNeXt"): "thyroid/convnext384_universal",
         ("Capsule", "DINOv2"): "capsule/dino518_main", ("Capsule", "MedSigLIP"): "capsule/medsiglip448_main",
         ("Capsule", "ConvNeXt"): "capsule/convnext384_universal",
         ("Ovary", "DINOv2"): "ovary/dino518_main", ("Ovary", "MedSigLIP"): "ovary/medsiglip448_text",
         **{(f"CXR {d}", "RAD-DINO"): f"cxr_traps/raddino518/{d}" for d in ("Atelectasis", "Consolidation", "Effusion", "Infiltration")}}
DRAIN = {("Chest drain", "RAD-DINO"): "cxr_drain/raddino518_universal"}
SWEEPS = {("Dermoscopy ruler", "DINOv2"): "synthetic/isic2018/dino518_ruler_fixed_corr_main",
          ("Dermoscopy ruler", "DermLIP"): "synthetic/isic2018/dermlip224_ruler_fixed_corr_main",
          ("Dermoscopy ruler", "DINOv2-224"): "synthetic/isic2018/dino224_ruler_fixed_corr_main",
          ("CXR tube", "RAD-DINO"): "synthetic/nih_ptx/raddino518_tube_corr_main",
          ("CXR tube", "DINOv2"): "synthetic/nih_ptx/dino518_tube_corr_main",
          ("Capsule debris", "DINOv2"): "synthetic/capsule/dino518_debris_corr_main",
          ("Capsule debris", "MedSigLIP"): "synthetic/capsule/medsiglip448_debris_corr_main",
          ("Thyroid caliper", "DINOv2"): "synthetic/thyroid/dino518_caliper_corr_main",
          ("Thyroid caliper", "MedSigLIP"): "synthetic/thyroid/medsiglip448_caliper_corr_main",
          ("Ovary caliper", "DINOv2"): "synthetic/ovary/dino518_caliper_corr_main"}
FINETUNE = {("Thyroid", "ResNet-50 FT"): "finetune/thyroid/resnet50", ("Capsule", "ResNet-50 FT"): "finetune/capsule/resnet50",
            ("Ovary", "ResNet-50 FT"): "finetune/ovary/resnet50",
            ("Thyroid", "ViT-S FT"): "finetune/thyroid/vit_small_patch16_224.augreg_in21k_ft_in1k"}
EXCLUDE = ("balanced", "bal", "dfr", "jtt", "groupdro", "prevcal", "aug", "auto", "select", "u13", "u14")


def model_auc(S, A, q1, q0):
    g = (P1 - P0) * A / (1 + V * A)
    wa2 = A * (P1 - P0) ** 2 / (1 + V * A) ** 2
    d = S + g * (q1 - q0)
    s2 = 2 * S + wa2 * (2 + A * (q1 * (1 - q1) + q0 * (1 - q0)))
    return norm.cdf(d / np.sqrt(s2)) if s2 > 0 else 0.5


def fit_SA(clean, corr, qc, qk):
    zc, zk = norm.ppf(np.clip([clean, corr], 1e-4, 1 - 1e-4))
    f = lambda p: [norm.ppf(np.clip(model_auc(p[0], p[1], *qc), 1e-6, 1 - 1e-6)) - zc,
                   norm.ppf(np.clip(model_auc(p[0], p[1], *qk), 1e-6, 1 - 1e-6)) - zk]
    best = None
    for a0 in (0.0, 0.5, 2.0, 8.0):
        s0 = max(2 * max(zc, 0.01) ** 2, 1e-3)
        r = least_squares(f, [s0, a0], bounds=([0, 0], [100, 400]))
        if best is None or r.cost < best.cost:
            best = r
    return best.x, float(np.sqrt(2 * best.cost))


def auc_table(pred, key):
    rows = []
    seedcol = "seed" if "seed" in pred else "fold"
    for k, q in pred.groupby([key, "method", "env", seedcol]):
        if q.y.nunique() == 2:
            rows.append((*k, roc_auc_score(q.y, q.prob)))
    t = pd.DataFrame(rows, columns=["trap", "method", "env", "seed", "auc"])
    comp = pred.groupby([key, "env", "y"]).artifact_present.mean().unstack()  # q_y per environment
    comp.index = comp.index.set_names(["trap", "env"])
    return t.groupby(["trap", "method", "env"]).auc.mean().unstack(), comp


def cells(kind, runs):
    out = []
    for (cohort, bb), rel in runs.items():
        f = R / rel / "predictions.csv.gz"
        if not f.exists():  # cxr_traps: one predictions file per trap
            fs = sorted((R / rel).glob("trap*/predictions.csv.gz"))
            if not fs:
                print(f"[theory_predict] SKIP {rel}: no saved predictions", flush=True)
                continue
            pred = pd.concat([pd.read_csv(x) for x in fs], ignore_index=True)
        else:
            pred = pd.read_csv(f)
        if "trap" not in pred:
            pred["trap"] = "trapA" if kind == "drain" else pred.get("overlap", "trapA")
        if kind == "sweep":
            pred["trap"] = pred.overlap.map(lambda o: f"r={o:.2f}")
        pred = pred[~pred.method.str.contains("|".join(EXCLUDE))]
        aucs, comp = auc_table(pred, "trap")
        for (trap, m), r in aucs.iterrows():
            if r[["clean", "test_corr", "test_rev"]].isna().any():
                continue
            qc, qk, qr = (tuple(comp.loc[(trap, e), [1, 0]]) for e in ("clean", "test_corr", "test_rev"))
            (S, A), res = fit_SA(r.clean, r.test_corr, qc, qk)
            out.append({"kind": kind, "cohort": cohort, "backbone": bb, "run": rel, "trap": trap, "arm": m,
                        "clean": r.clean, "corr": r.test_corr, "rev": r.test_rev, "S": S, "A": A, "fit_resid": res,
                        "pred_rev": model_auc(S, A, *qr), "ref_clean": r.clean, "ref_sym": 2 * r.clean - r.test_corr})
        print(f"[theory_predict] {kind} {cohort} {bb}: {len(aucs)} arm-traps", flush=True)
    return out


def summarise(d, col="pred_rev"):
    e = d[col] - d.rev
    return {"n": int(len(d)), "mae": float(e.abs().mean()), "r": float(pearsonr(d[col], d.rev)[0]) if len(d) > 2 else None,
            "within_0.05": float((e.abs() <= 0.05).mean())}


def gains(d, key):
    """mask - ERM reversed gain, observed and predicted, per (cohort, backbone, trap)."""
    p = d[d.arm.isin(["erm", "mask"])].pivot_table(index=["cohort", "backbone", key], columns="arm", values=["rev", "pred_rev", "S"])
    g = pd.DataFrame({"obs": p[("rev", "mask")] - p[("rev", "erm")], "pred": p[("pred_rev", "mask")] - p[("pred_rev", "erm")],
                      "dS": p[("S", "mask")] - p[("S", "erm")]}).dropna().reset_index()
    return g


def main():
    d = pd.DataFrame(cells("trap", TRAPS) + cells("drain", DRAIN) + cells("sweep", SWEEPS) + cells("finetune", FINETUNE))
    out = paths.ensure(R / "theory")
    d.to_csv(out / "prediction_cells.csv", index=False)
    frozen = d[d.kind != "finetune"]
    prim = frozen[frozen.arm.isin(["erm", "mask"])]
    S = {"T1_primary": {"theory": summarise(prim), "ref_clean": summarise(prim, "ref_clean"), "ref_sym": summarise(prim, "ref_sym")},
         "T1_all_erm_head_arms": {"theory": summarise(frozen), "ref_clean": summarise(frozen, "ref_clean"),
                                  "ref_sym": summarise(frozen, "ref_sym")},
         "T1_finetune_secondary": summarise(d[(d.kind == "finetune") & d.arm.isin(["erm", "mask"])])}
    # T2 crossover (real traps)
    g = gains(d[d.kind.isin(["trap", "finetune"])], "trap")
    x = g.pivot_table(index=["cohort", "backbone"], columns="trap", values=["obs", "pred"])
    xo = pd.DataFrame({"obs": x[("obs", "trapB")] - x[("obs", "trapA")], "pred": x[("pred", "trapB")] - x[("pred", "trapA")]}).dropna().reset_index()
    xo["finetune"] = xo.backbone.str.contains("FT")
    xo.to_csv(out / "prediction_crossover.csv", index=False)
    xf = xo[~xo.finetune]
    S["T2_crossover"] = {"n": int(len(xf)), "sign_agree": int((np.sign(xf.obs) == np.sign(xf.pred)).sum()),
                         "r": float(pearsonr(xf.pred, xf.obs)[0]), "mae": float((xf.pred - xf.obs).abs().mean()),
                         "finetune_sign_agree": f"{int((np.sign(xo[xo.finetune].obs) == np.sign(xo[xo.finetune].pred)).sum())}/{int(xo.finetune.sum())}"}
    # T3 sweeps
    gs = gains(d[d.kind == "sweep"], "trap")
    gs.to_csv(out / "prediction_sweep_gains.csv", index=False)
    S["T3_sweeps"] = {"n": int(len(gs)), "sign_agree": int((np.sign(gs.obs) == np.sign(gs.pred)).sum()),
                      "r": float(pearsonr(gs.pred, gs.obs)[0]), "mae": float((gs.pred - gs.obs).abs().mean())}
    # T4 in-ROI: sign(S_mask - S_erm) vs sign(observed Trap-A gain)
    ga = g[(g.trap == "trapA") & ~g.backbone.str.contains("FT")]
    S["T4_inroi_sign_from_dS"] = {"n": int(len(ga)), "agree": int((np.sign(ga.dS) == np.sign(ga.obs)).sum())}
    # same endpoints for the two reference predictors (same inputs, no theory)
    for col in ("ref_sym", "ref_clean"):
        dd = d.assign(pred_rev=d[col])
        gg = gains(dd[dd.kind == "trap"], "trap").pivot_table(index=["cohort", "backbone"], columns="trap", values=["obs", "pred"])
        xr = pd.DataFrame({"obs": gg[("obs", "trapB")] - gg[("obs", "trapA")], "pred": gg[("pred", "trapB")] - gg[("pred", "trapA")]}).dropna()
        gr = gains(dd[dd.kind == "sweep"], "trap")
        S[f"reference_{col}"] = {"T2_r": float(pearsonr(xr.pred, xr.obs)[0]), "T2_mae": float((xr.pred - xr.obs).abs().mean()),
                                 "T2_sign_agree": int((np.sign(xr.pred) == np.sign(xr.obs)).sum()),
                                 "T3_r": float(pearsonr(gr.pred, gr.obs)[0]), "T3_mae": float((gr.pred - gr.obs).abs().mean()),
                                 "T3_sign_agree": int((np.sign(gr.pred) == np.sign(gr.obs)).sum())}
    t1, t2 = S["T1_primary"], S["T2_crossover"]
    S["verdict"] = ("quantitatively predictive" if t2["sign_agree"] == t2["n"] and t1["theory"]["r"] >= 0.7 and t2["r"] >= 0.7
                    and t1["theory"]["mae"] < t1["ref_clean"]["mae"] else "qualitative account")
    json.dump(S, open(out / "prediction_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))
    print(xo.round(3).to_string())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for kind, mk in (("trap", "o"), ("drain", "s"), ("sweep", "^")):
        q = prim[prim.kind == kind]
        ax[0].scatter(q.pred_rev, q.rev, s=12, marker=mk, alpha=.7, label={"trap": "real traps", "drain": "chest drain (correlate-carried)", "sweep": "controlled sweeps"}[kind])
    lo = min(prim.pred_rev.min(), prim.rev.min()) - .02
    ax[0].plot([lo, 1], [lo, 1], "k--", lw=.8); ax[0].set_xlabel("theory: reversed AUROC (fit on clean, corr.)"); ax[0].set_ylabel("observed reversed AUROC")
    ax[0].set_title(f"(a) ERM & mask, r={t1['theory']['r']:.2f}, MAE={t1['theory']['mae']:.3f}", fontsize=8); ax[0].legend(fontsize=6)
    fam = lambda c: "Chest X-ray" if c.startswith("CXR") else c
    mk = {"ISIC hair": "o", "Thyroid": "s", "Capsule": "^", "Ovary": "D", "Chest X-ray": "v"}
    for ft, c in ((False, "C0"), (True, "C3")):
        for f_, q in xo[xo.finetune == ft].groupby(xo.cohort.map(fam)):
            ax[1].scatter(q.pred, q.obs, s=20, c=c, marker=mk[f_], label=f"{f_}" + (" (end-to-end)" if ft else ""))
    m = [min(xo.pred.min(), xo.obs.min()) - .03, max(xo.pred.max(), xo.obs.max()) + .03]
    ax[1].plot(m, m, "k--", lw=.8); ax[1].axhline(0, c="grey", lw=.5); ax[1].axvline(0, c="grey", lw=.5)
    ax[1].set_xlabel("theory: crossover (fit on clean, corr.)"); ax[1].set_ylabel("observed crossover")
    ax[1].set_title(f"(b) location crossover, r={t2['r']:.3f}, signs {t2['sign_agree']}/{t2['n']}", fontsize=8); ax[1].legend(fontsize=5.5, loc="upper left")
    for (c, b), q in gs.groupby(["cohort", "backbone"]):
        ax[2].scatter(q.pred, q.obs, s=10, label=f"{c}/{b}")
    m = [min(gs.pred.min(), gs.obs.min()) - .02, max(gs.pred.max(), gs.obs.max()) + .02]
    ax[2].plot(m, m, "k--", lw=.8); ax[2].axhline(0, c="grey", lw=.5); ax[2].axvline(0, c="grey", lw=.5)
    ax[2].set_xlabel("theory: mask gain (fit on clean, corr.)"); ax[2].set_ylabel("observed mask gain")
    ax[2].set_title(f"(c) sweeps, r={S['T3_sweeps']['r']:.2f}, signs {S['T3_sweeps']['sign_agree']}/{S['T3_sweeps']['n']}", fontsize=8)
    ax[2].legend(fontsize=5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    figd = paths.ensure(paths.REPO_ROOT / "paper" / "figures")
    fig.savefig(figd / "theory_predict.pdf"); fig.savefig(figd / "theory_predict.png", dpi=200)


if __name__ == "__main__":
    main()
