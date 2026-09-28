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
            "calliper is a stricter, registered sensitivity analysis (R7).", "tab:match", size="\\scriptsize",
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


def transplant_details():
    """T2-T4 of the transplant: AUROCs at the in-ROI location, same-head |dp|, balanced interaction (point values)."""
    t = L.csv("review2/R2_transplant.csv")
    if t is None:
        return
    rows = []
    for r in t.itertuples():
        k = KEY[r.cohort]
        rows.append([r.cohort, str(int(r.n_images)), f"{L.f3(r.auc_erm_test_corr_in)} / {L.f3(r.auc_erm_test_rev_in)}",
                     f"{L.f3(r.auc_mask_test_corr_in)} / {L.f3(r.auc_mask_test_rev_in)}",
                     f"{L.f3(r.auc_mask_test_corr_out)} / {L.f3(r.auc_mask_test_rev_out)}",
                     f"{L.f3(r.cf_absdp_erm_in)} / {L.f3(r.cf_absdp_mask_in)}", L.f3(r.cf_absdp_mask_out)])
        M.add(f"DpErmIn{k}", L.f3(r.cf_absdp_erm_in)); M.add(f"DpMaskIn{k}", L.f3(r.cf_absdp_mask_in))
        M.add(f"MaskCleanIn{k}", L.f3(r.auc_mask_clean_in)); M.add(f"ErmCleanIn{k}", L.f3(r.auc_erm_clean_in))
        M.add(f"MaskRevOut{k}", L.f3(r.auc_mask_test_rev_out)); M.add(f"MaskCorrOut{k}", L.f3(r.auc_mask_test_corr_out))
        M.add(f"MaskCleanOut{k}", L.f3(r.auc_mask_clean_out)); M.add(f"NRecip{k}", int(r.n_images))
    L.table(T / "transplant_detail.tex", ["Cohort", "Recipients", "ERM in: corr / rev", "mask in: corr / rev",
                                          "mask out: corr / rev", "$|\\Delta p|$ in: ERM / mask", "$|\\Delta p|$ out: mask"], rows,
            "Transplant, per location: correlated and reversed AUROC of ERM and of the masked model, and the same-head "
            "counterfactual sensitivity to the pasted artifact (T3). Outside the ROI the masked model's correlated and "
            "reversed AUROC coincide and $|\\Delta p|=0$: the pasted artifact is gone.", "tab:txdet",
            size="\\scriptsize", resize=True)


def rebuild():
    """R0: reproduction of the four primary crossovers after re-downloading all data and rebuilding every derived object
    on a second machine (archived point estimate vs rebuilt crossover with crossed CI)."""
    r = L.csv("review2/R0_R1_crossovers.csv")
    o = L.csv("review2/R0_R1_crossovers.csv", root=L.OLD)
    rows = []
    for c in COH:
        q = r[(r.cohort == c) & (r.run == "repro")]
        oq = o[(o.cohort == c) & (o.run == "repro")] if o is not None else None
        arch = None if oq is None or oq.empty else float(oq.archived.iloc[0])
        rows.append([c, L.s3(arch), L.ci_row(q.iloc[0].rename({"crossover": "seed_delta_mean"})) if len(q) else "\\na",
                     L.f3(None if arch is None else abs(float(q.crossover.iloc[0]) - arch))])
    L.table(T / "rebuild.tex", ["Cohort", "First build (point)", "Rebuilt data, second machine", "$|\\Delta|$"], rows,
            "Reproduction of the four crossovers after every dataset was re-downloaded and every derived object (lesion "
            "U-Net, detectors, probe, groups, caches) rebuilt on a second machine (registered criterion: same sign, CI "
            "excluding 0, $|\\Delta|\\le0.03$).", "tab:rebuild")


SENS = [("Thyroid", "main", "thyroid/dino518_main"), ("Thyroid", "large markers ($\\ge50$ px)", "thyroid/dino518_sens_px50"),
        ("Thyroid", "strict location ($r\\ge0.7$ / $r<0.05$)", "thyroid/dino518_sens_strictloc"),
        ("Ovary", "main", "ovary/dino518_main"), ("Ovary", "large markers ($\\ge50$ px)", "ovary/dino518_sens_px50"),
        ("Ovary", "strict location ($r\\ge0.7$ / $r<0.05$)", "ovary/dino518_sens_strictloc"),
        ("Capsule", "main", "capsule/dino518_main"), ("Capsule", "strict Trap B (lesion coverage $<5\\%$)", "capsule/dino518_sens_strictB")]


def robustness():
    rows = []
    for c, lab, d in SENS:
        x = L.js(f"{d}/T3_crossover.json")
        cnt = L.csv(f"{d}/counts.csv")
        n = "\\na" if cnt is None else f"{int(cnt[cnt.trap == 'trapA'][['A1_Y0', 'A1_Y1']].sum(axis=1).iloc[0])} / {int(cnt[cnt.trap == 'trapB'][['A1_Y0', 'A1_Y1']].sum(axis=1).iloc[0])}"
        rows.append([c, lab, n, L.ci_row(x)])
        M.add(f"Sens{L.Macros.clean(d.split('/')[1])}{c}", L.ci_row(x))
    L.table(T / "label_sens.tex", ["Cohort", "Artifact-label definition", "Artifact images A / B", "Crossover"], rows,
            "Sensitivity of the crossover to the definition of the artifact label (registered before running): only "
            "large detections, stricter location thresholds, and for capsule a Trap~B that also requires debris to "
            "cover less than 5\\% of the lesion box.", "tab:labsens", size="\\scriptsize")
    j = L.js("leakage/embedding_groups.json", root=L.OLD)
    rows = []
    for c, key in (("Thyroid", "thyroid"), ("Ovary", "ovary"), ("Capsule", "capsule")):
        x0 = L.js(f"{key}/dino518_main/T3_crossover.json"); x1 = L.js(f"{key}/dino518_emb_groups/T3_crossover.json")
        g = (j or {}).get(key, {})
        rows.append([c, f"{g.get('groups_old', 0):,} $\\to$ {g.get('groups_emb', 0):,}", f"{g.get('tau', float('nan')):.2f}",
                     f"{100 * g.get('largest_group_share', float('nan')):.1f}\\%", L.ci_row(x0), L.ci_row(x1)])
        M.add(f"Emb{c}", L.ci_row(x1))
    L.table(T / "emb_groups.tex", ["Cohort", "Leakage groups", "cosine $\\tau$", "largest group", "Crossover, pHash groups",
                                   "Crossover, embedding groups"], rows,
            "Stricter leakage groups for the cohorts without patient identifiers: images joined whenever their "
            "(mean-centred) DINOv2 embeddings have cosine $\\ge\\tau$, united with the existing groups; $\\tau$ is the most "
            "aggressive grid value that keeps the largest group below 5\\% of the cohort (chosen without labels).",
            "tab:emb", size="\\scriptsize", resize=True)


def estimator():
    r = L.estimator_check(["thyroid/dino518_matched", "ovary/dino518_matched", "capsule/dino518_matched",
                           "spec_e13/dino518_matched", "thyroid/dino518_emb_groups", "ovary/dino518_emb_groups",
                           "capsule/dino518_emb_groups", "thyroid/dino518_sens", "ovary/dino518_sens", "capsule/dino518_sens",
                           "review2/transplant", "thyroid/dino518_repro", "ovary/dino518_repro", "capsule/dino518_repro",
                           "spec_e13/dino518_repro"],
                          T / "estimator_check.tex", "tab:estcheck",
                          "The analyses of this stage under the per-seed and the crossed estimator (same predictions; "
                          "baseline arms only).", changes_path=T / "estimator_changes.tex", changes_label="tab:estchanges",
                          changes_caption="Contrasts of this stage whose verdict depends on the estimator.")
    if r:
        M.add("EstN", r["n"]); M.add("EstLost", r["lost"]); M.add("EstGained", r["gained"]); M.add("EstMed", f"{r['median']:.2f}")
        plt = L.plot_style(); d = r["df"]
        fig, ax = plt.subplots(figsize=(4.8, 2.6))
        ax.scatter((d.old_hi - d.old_lo), (d.new_hi - d.new_lo), s=5, alpha=0.5, color=L.MASK_BLUE)
        m = float(max((d.old_hi - d.old_lo).max(), (d.new_hi - d.new_lo).max()))
        ax.plot([0, m], [0, m], "k--", lw=0.7)
        ax.set_xlabel("width, per-seed interval"); ax.set_ylabel("width, crossed interval")
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
            ["Artifact-label sensitivity", "\\ok{} registered", "SENSITIVITY\\_ARTIFACT\\_LABELS",
             L.reg("docs/SENSITIVITY_ARTIFACT_LABELS.md"), "crossover keeps sign and CI $>0$ in every variant"],
            ["Embedding-based leakage groups", "\\ok{} registered", "EMBEDDING\\_GROUPS", L.reg("docs/PREREGISTRATION_EMBEDDING_GROUPS.md"),
             "same sign and CI verdict as with pHash groups"],
            ["Rebuild on a second machine, R0", "\\ok{} registered", "REVIEW2 (R0)", L.reg(docs["Rev2"]),
             "same sign, CI $>0$, $|\\Delta|\\le0.03$"],
            ["Capsule matched analysis", "\\no{} descriptive", "REVIEW2 (R1)", L.reg(docs["Rev2"]),
             "infeasible under the registered rule"]]
    rows = [r[:2] + [r[2].replace("\\_", "\\_\\allowbreak{}")] + r[3:] for r in rows]
    L.table(T / "experiments.tex", ["Experiment", "Status", "Registration", "First commit (local time)", "Support criterion"],
            rows, "Experiments of this stage and their registration documents (\\texttt{docs/PREREGISTRATION\\_<name>.md} or "
            "\\texttt{docs/<name>.md}). Commit times are set by the committing machine and are not independent proof of order; "
            "the REVIEW2, REVIEW3 and FINAL registrations were also pushed to GitHub before their runs.", "tab:exp", align=L.EXP_ALIGN,
            size="\\scriptsize")


def main():
    balance(); matched(); a2(); transplant(); transplant_details(); rebuild(); robustness(); estimator(); transplant_figure(); registrations()
    M.write(T / "numbers.tex")
    L.report_missing("stage4")


if __name__ == "__main__":
    main()
