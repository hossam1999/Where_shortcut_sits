"""Stage 4 tables, macros and figures: causal checks of the real-artifact crossover (balance, matching, regression
adjustment A2, transplant, neutral paste) and the consequences of the bootstrap correction.
  PYTHONPATH=src python stages/stage4_causal_checks/make_stage4.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import stagelib as L  # noqa: E402

T, F = HERE / "tables", HERE / "figures"
T.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
M = L.Macros("sFour")
COH = ["ISIC hair", "Thyroid", "Ovary", "Capsule"]
KEY = {"ISIC hair": "Hair", "Thyroid": "Thy", "Ovary": "Ov", "Capsule": "Cap"}
CC = None


def cc(claim, cohort):
    global CC
    if CC is None:
        CC = L.csv("review3/crossed_ci.csv")
    if CC is None:
        return None
    q = CC[(CC.claim == claim) & (CC.cohort == cohort)]
    if q.empty:
        return None
    r = q.iloc[0]
    return {"seed_delta_mean": r.estimate, "ci95_lo": r.crossed_lo, "ci95_hi": r.crossed_hi, "p": r.crossed_p}


def balance():
    r = L.csv("review2/R0_R1_crossovers.csv")
    b = L.csv("review2/R1_balance.csv")
    rows = []
    for c in COH:
        q = r[(r.cohort == c) & (r.run == "matched")].iloc[0] if r is not None else None
        before = b[(b.cohort == c) & (b.subset == "pooled") & (b.comparison == "trapA_vs_trapB_before_matching")] if b is not None else None
        worst = before.loc[before.smd.abs().idxmax()] if before is not None and len(before) else None
        rows.append([c, "\\na" if worst is None else L.tex_escape(worst.covariate), L.f3(None if q is None else q.max_abs_smd_before_pooled),
                     L.f3(None if q is None else q.max_abs_smd_after_pooled),
                     "\\na" if q is None else str(int(q.min_pairs_per_label)),
                     "\\na" if q is None else ("yes" if str(q.feasible) == "True" else "\\textbf{no}")])
        if q is not None:
            M.add(f"SmdBefore{KEY[c]}", L.f3(q.max_abs_smd_before_pooled)); M.add(f"SmdAfter{KEY[c]}", L.f3(q.max_abs_smd_after_pooled))
            M.add(f"Pairs{KEY[c]}", int(q.min_pairs_per_label)); M.add(f"Ratio{KEY[c]}", L.f3(q.ratio_matched_to_repro))
        if worst is not None:
            M.add(f"Worst{KEY[c]}", L.tex_escape(worst.covariate).replace("\\_", " "))
    L.table(T / "balance.tex", ["Cohort", "Least balanced covariate", "max $|$SMD$|$ before", "after matching",
                                "min matched pairs per label", "feasible (rule)"], rows,
            "Balance of pre-specified covariates between the in-ROI (Trap~A) and out-of-ROI (Trap~B) artifact images, "
            "before and after 1:1 nearest-neighbour matching within label (calliper 0.2 SD, pooled standardised mean "
            "differences). Registered rule: fewer than 30 matched images per label in either trap makes a cohort "
            "infeasible; target max $|$SMD$|\\le0.1$ after matching.", "tab:bal", size="\\scriptsize", resize=True)
    if b is not None:
        rows = []
        for c in ("Thyroid", "Ovary"):
            for cmp in ("trapA_vs_trapB_before_matching", "trapA_vs_trapB_after_matching"):
                q = b[(b.cohort == c) & (b.subset == "pooled") & (b.comparison == cmp)]
                for x in q.itertuples():
                    if cmp.endswith("before_matching"):
                        after = b[(b.cohort == c) & (b.subset == "pooled") & (b.comparison == "trapA_vs_trapB_after_matching")
                                  & (b.covariate == x.covariate) & (b.level.astype(str) == str(x.level))]
                        rows.append([c, L.tex_escape(x.covariate + ("" if pd.isna(x.level) else f"={x.level}")),
                                     L.s3(x.smd), L.s3(float(after.smd.iloc[0])) if len(after) else "\\na"])
        L.table(T / "balance_detail.tex", ["Cohort", "Covariate", "SMD before", "SMD after"], rows,
                "Per-covariate balance, thyroid and ovary (pooled over labels). The dermoscopy and capsule tables "
                "(17 and 20 covariates) are in \\texttt{results/rerun\\_2026-09-28/review2/R1\\_balance.csv}.",
                "tab:baldet", size="\\scriptsize")


def matched():
    r = L.csv("review2/R0_R1_crossovers.csv")
    rows = []
    for c in COH:
        # all-image and matched crossovers from their registered analyses (R0/R1); the 0.05 SD calliper from R7
        a = r[(r.cohort == c) & (r.run == "repro")].iloc[0].rename({"crossover": "seed_delta_mean"}) if r is not None else None
        q = r[(r.cohort == c) & (r.run == "matched")].iloc[0] if r is not None else None
        m = None if q is None else q.rename({"crossover": "seed_delta_mean"})
        m5 = cc("real-trap crossover (matched05)", c)
        sup = "\\na" if q is None else ("descriptive (infeasible)" if str(q.feasible) != "True" else
                                        ("\\ok" if str(q.M1_supported) == "True" else "\\no"))
        rows.append([c, L.ci_row(a), L.ci_row(m), L.ci_row(m5), sup])
        M.add(f"X{KEY[c]}", L.ci_row(a)); M.add(f"Match{KEY[c]}", L.ci_row(m)); M.add(f"MatchFive{KEY[c]}", L.ci_row(m5))
        if q is not None and str(q.feasible) == "True":
            M.add(f"MOneHolm{KEY[c]}", f"{q.p_holm:.3f}")
    L.table(T / "matched.tex", ["Cohort", "Crossover, all images", "matched (0.2 SD)", "matched (0.05 SD)", "M1 (Holm, 3)"],
            rows, "Crossover before and after covariate matching (reversed-test AUROC, crossed 95\\% CIs). The 0.05 SD "
            "calliper is the stricter sensitivity analysis of the third review (R7).", "tab:match", size="\\scriptsize",
            resize=True)


def a2():
    d = L.csv("final_a2/matched_regression.csv")
    b = L.csv("final_a2/balance_after_matching.csv")
    rows = []
    if d is not None:
        for r in d.itertuples():
            k = KEY.get(r.cohort, r.cohort)
            bal = "\\na"
            if b is not None:
                q = b[b.cohort == r.cohort]
                bal = f"{L.f3(q[q.subset == 'pooled'].smd.abs().max())} / {L.f3(q[q.subset != 'pooled'].smd.abs().max())}"
            rows.append([r.cohort, str(int(r.n_test_images_per_seed)), str(int(r.n_covariate_columns)), bal,
                         L.ci(r.unadjusted, r.unadj_lo, r.unadj_hi), L.ci(r.adjusted, r.adj_lo, r.adj_hi),
                         f"{r.adj_p_holm:.3f}", "\\ok" if bool(r.A2_supported) else "\\no"])
            M.add(f"Unadj{k}", L.ci(r.unadjusted, r.unadj_lo, r.unadj_hi)); M.add(f"Adj{k}", L.ci(r.adjusted, r.adj_lo, r.adj_hi))
            M.add(f"AdjHolm{k}", f"{r.adj_p_holm:.3f}")
        M.add("NATwoOK", int(d.A2_supported.astype(str).eq("True").sum())); M.add("NATwo", len(d))
    L.table(T / "a2.tex", ["Cohort", "Test images per seed", "Covariate columns", "Post-matching max $|$SMD$|$ pooled / per label",
                           "Unadjusted crossover", "Adjusted (doubly robust)", "$p$ (Holm, 3)", "A2"], rows or [["\\na"] * 8],
            "Regression-adjusted crossover on the matched traps (A2): per-image placement values of the masked minus the "
            "ERM score regressed on the Trap-B indicator, the label and all matching covariates, within seed; crossed "
            "bootstrap (2,000 replicates). Capsule is not analysed (matching infeasible; descriptive only).",
            "tab:a2", size="\\scriptsize", resize=True)
    if b is not None:
        rows = []
        for c in ("ISIC hair", "Thyroid", "Ovary"):
            q = b[(b.cohort == c) & (b.subset == "pooled")]
            for x in q.itertuples():
                rows.append([c, L.tex_escape(x.covariate + ("" if pd.isna(x.level) else f"={x.level}")), L.f3(x.mean_1),
                             L.f3(x.mean_0), L.s3(x.smd)])
        L.table(T / "a2_balance.tex", ["Cohort", "Covariate", "Trap A mean", "Trap B mean", "SMD"], rows,
                "Post-matching balance table of the A2 sample (pooled over labels; test images of the reversed "
                "environment).", "tab:a2bal", size="\\scriptsize")


def transplant():
    rows, rows2 = [], []
    for c in COH:
        art, neu = cc("transplant interaction (artifact)", c), cc("transplant interaction (neutral paste)", c)
        n3, bal = cc("artifact - neutral interaction (N3)", c), cc("transplant interaction (balanced)", c)
        mi, mo = cc("transplant mask - ERM (in)", c), cc("transplant mask - ERM (out)", c)
        ni, no = cc("neutral mask - ERM (in)", c), cc("neutral mask - ERM (out)", c)
        rows.append([c, L.ci_row(mo), L.ci_row(mi), L.ci_row(art), L.ci_row(bal)])
        share = None if art is None or neu is None else neu["seed_delta_mean"] / art["seed_delta_mean"]
        rows2.append([c, L.ci_row(no), L.ci_row(ni), L.ci_row(neu), L.ci_row(n3), "\\na" if share is None else f"{100 * share:.0f}\\%"])
        k = KEY[c]
        M.add(f"TI{k}", L.ci_row(art)); M.add(f"TN{k}", L.ci_row(neu)); M.add(f"NThree{k}", L.ci_row(n3))
        M.add(f"TBal{k}", L.ci_row(bal)); M.add(f"Share{k}", "\\na" if share is None else f"{100 * share:.0f}\\%")
    L.table(T / "transplant.tex", ["Cohort", "mask $-$ ERM, pasted outside ROI", "pasted inside ROI", "Location interaction",
                                   "Balanced: interaction"], rows,
            "Real-artifact transplant (T1): the same real artifact instance, cut from a donor image, pasted outside and "
            "inside the ROI of the same artifact-free recipient images. These effects combine the artifact with the paste "
            "itself (seam, occluded tissue); Table~\\ref{tab:neutral} separates them.", "tab:tx", size="\\scriptsize",
            resize=True)
    L.table(T / "neutral.tex", ["Cohort", "mask $-$ ERM, neutral outside", "neutral inside", "Neutral interaction",
                                "Artifact $-$ neutral (N3)", "Neutral / artifact interaction"], rows2,
            "Paste-edge control (R5): the same mask shape and compositing, filled with the donor's own artifact-free "
            "tissue. Only the N3 column --- the interaction with the real artifact minus the interaction with the neutral "
            "paste --- is attributable to the artifact's content.", "tab:neutral", size="\\scriptsize", resize=True)
    shares = [neu["seed_delta_mean"] / art["seed_delta_mean"] for c in COH
              if (art := cc("transplant interaction (artifact)", c)) and (neu := cc("transplant interaction (neutral paste)", c))]
    if shares:
        M.add("ShareMin", f"{100 * min(shares):.0f}\\%"); M.add("ShareMax", f"{100 * max(shares):.0f}\\%")
    n3s = [cc("artifact - neutral interaction (N3)", c) for c in COH]
    M.add("NNThreePos", sum(r is not None and r["ci95_lo"] > 0 for r in n3s))


def correction():
    d = L.csv("bootstrap_correction/sweeps_umte_old_vs_new.csv", root=L.OLD)
    if d is None:
        return
    M.add("CRows", len(d)); M.add("CChanged", int(d.verdict_changed.sum()))
    M.add("CLost", int((d.old_excludes_0 & ~d.new_excludes_0).sum())); M.add("CGained", int((~d.old_excludes_0 & d.new_excludes_0).sum()))
    M.add("CMedian", f"{d.width_ratio.median():.2f}"); M.add("CQlo", f"{d.width_ratio.quantile(.25):.2f}")
    M.add("CQhi", f"{d.width_ratio.quantile(.75):.2f}"); M.add("CEstDiff", int((d.est_diff.abs() > 0.03).sum()))
    d["top"] = d.file.str.split("/").str[0]
    g = d.groupby("top").agg(n=("key", "size"), lost=("verdict_changed", lambda s: int((s & d.loc[s.index, "old_excludes_0"]).sum())),
                             gained=("verdict_changed", lambda s: int((s & ~d.loc[s.index, "old_excludes_0"]).sum())),
                             est=("est_diff", lambda s: int((s.abs() > 0.03).sum())), mx=("est_diff", lambda s: s.abs().max()),
                             wr=("width_ratio", "median"))
    L.table(T / "correction_by_source.tex", ["Result group", "Intervals", "Lost significance", "Gained", "$|\\Delta|$ estimate $>0.03$",
                                             "max $|\\Delta|$ estimate", "median width ratio"],
            [[L.tex_escape(k), str(r.n), str(r.lost), str(r.gained), str(r.est), L.f3(r.mx), f"{r.wr:.2f}"] for k, r in g.iterrows()],
            "Old versus corrected intervals by result group (every regenerated interval with an archived counterpart; "
            "\\texttt{results/bootstrap\\_correction/sweeps\\_umte\\_old\\_vs\\_new.md}).", "tab:corr", size="\\scriptsize",
            resize=True)
    v = d[d.verdict_changed]
    k = v[v.key.str.contains(r"\|mask\||mask-erm|mask - ERM|^mask\|", regex=True) |
          v.file.str.contains("PRIMARY|adhoc|natural|finetune|review2|E14|E12|SUMMARY|T3_cross|X3|location_interaction")]
    brk = lambda x: L.tex_escape(x).replace("/", "/\\allowbreak{}").replace("\\_", "\\_\\allowbreak{}")
    rows = [[brk(r.file.replace("/bootstrap_vs_erm.csv", "").replace("/paired_deltas.csv", "")),
             brk(r.key.replace("|", " | ")), L.ci(r.new_est, r.new_lo, r.new_hi),
             "lost" if r.old_excludes_0 else "gained"] for r in k.itertuples()]
    L.table(T / "correction_claims.tex", ["Result", "Row", "Corrected estimate [95\\% CI]", "Significance"], rows,
            "Verdict changes among intervals that concern masking, the primary family, natural test sets, fine-tuning and "
            "the transplant. Only the corrected interval is shown; the archived interval is in the comparison file.",
            "tab:corrclaims", align="p{4.6cm}p{5.2cm}p{3.4cm}c", size="\\scriptsize")
    M.add("CClaims", len(rows))
    pn, po = L.csv("PRIMARY_CLAIMS.csv"), L.csv("PRIMARY_CLAIMS.csv", root=L.OLD)
    for tag, x in (("PrimNew", pn), ("PrimOld", po)):
        if x is not None:
            M.add(tag, int(((x.p_holm < 0.05) & (x.estimate > 0)).sum()))
    # capsule synthetic cohort re-dealt (reproducibility): archived counterfactual images found in the new test set
    try:
        o = pd.read_csv(L.OLD / "synthetic/capsule/dino518_debris_corr_main/counterfactual_per_image.csv.gz", usecols=["image_id"])
        n = pd.read_csv(L.NEW / "synthetic/capsule/dino518_debris_corr_main/predictions.csv.gz", usecols=["image_id", "env"])
        so, sn = set(o.image_id), set(n[n.env == "test_rev"].image_id)
        M.add("CapOld", len(so)); M.add("CapCommon", len(so & sn)); M.add("CapNewTest", len(sn))
        cap = d[d.file.str.startswith("synthetic/capsule")]
        M.add("CapMaxDiff", L.f3(cap.est_diff.abs().max()))
    except Exception as e:  # noqa: BLE001
        print("capsule overlap:", e)
    width_fig(d)


def width_fig(d):
    plt = L.plot_style()
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ax.scatter((d.old_hi - d.old_lo), (d.new_hi - d.new_lo), s=4, alpha=0.4, color=L.MASK_BLUE)
    m = float(max((d.old_hi - d.old_lo).max(), (d.new_hi - d.new_lo).max()))
    ax.plot([0, m], [0, m], "k--", lw=0.7)
    ax.set_xlabel("width, archived interval"); ax.set_ylabel("width, corrected interval")
    fig.savefig(F / "width_scatter.pdf")


def transplant_figure():
    ex = L.ROOT / "figures" / "examples"
    fs = [("ovary_transplant_in_roi.png", "Ovary: real caliper pasted in ROI"),
          ("ovary_neutral_paste_in_roi.png", "Ovary: neutral paste, same shape"),
          ("capsule_transplant_in_roi.png", "Capsule: real debris pasted in ROI"),
          ("capsule_neutral_paste_in_roi.png", "Capsule: neutral paste, same shape")]
    if not all((ex / f).exists() for f, _ in fs):
        L.MISSING.append("figures/examples transplant (run scripts/make_examples.py)"); return
    from PIL import Image
    plt = L.plot_style()
    fig, ax = plt.subplots(2, 2, figsize=(5.2, 5.4))
    for a, (f, t) in zip(ax.ravel(), fs):
        a.imshow(Image.open(ex / f)); a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=7)
    fig.savefig(F / "transplant_examples.pdf", dpi=150)


def registrations():
    docs = {"Pre": "docs/PRECOMMIT_MATCHED_TRAPS.md", "Rev2": "docs/PREREGISTRATION_REVIEW2.md",
            "Rev3": "docs/PREREGISTRATION_REVIEW3.md", "Final": "docs/PREREGISTRATION_FINAL.md"}
    for k, doc in docs.items():
        M.add(f"Reg{k}", L.reg(doc))
    rows = [["Covariate balance (SMD) and matched traps, M1", "\\ok{} registered", "REVIEW2 (R1)", L.reg(docs["Rev2"]),
             "matched crossover CI $>0$, Holm over 3 feasible cohorts"],
            ["Stricter calliper 0.05 SD (R7)", "\\ok{} registered", "REVIEW3", L.reg(docs["Rev3"]), "sensitivity, descriptive"],
            ["Real-artifact transplant, T1", "\\ok{} registered", "REVIEW2 (R2)", L.reg(docs["Rev2"]),
             "interaction CI $>0$, Holm over 4"],
            ["Neutral paste (paste-edge control), N3", "\\ok{} registered", "REVIEW3 (R5)", L.reg(docs["Rev3"]),
             "artifact $-$ neutral interaction CI $>0$"],
            ["Regression-adjusted crossover, A2", "\\ok{} registered", "FINAL (A2)", L.reg(docs["Final"]),
             "adjusted crossover CI $>0$, Holm over 3"],
            ["Crossed bootstrap everywhere, A1", "\\ok{} registered", "FINAL (A1)", L.reg(docs["Final"]),
             "rewrite claims whose CI now includes 0"],
            ["Capsule matched analysis", "\\no{} descriptive", "REVIEW2 (R1)", L.reg(docs["Rev2"]),
             "infeasible under the registered rule"]]
    rows = [r[:2] + [r[2].replace("\\_", "\\_\\allowbreak{}")] + r[3:] for r in rows]
    L.table(T / "experiments.tex", ["Experiment", "Status", "Registration", "First commit (local time)", "Support criterion"],
            rows, "Experiments of this stage. Commit times come from this machine and are not independent proof of order; "
            "the final registration was additionally pushed to GitHub before the runs.", "tab:exp", align=L.EXP_ALIGN,
            size="\\scriptsize")


def main():
    balance(); matched(); a2(); transplant(); correction(); transplant_figure(); registrations()
    M.write(T / "numbers.tex")
    L.report_missing("stage4")


if __name__ == "__main__":
    main()
