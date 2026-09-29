"""Tables and inline sentences after the third review round (docs/PREREGISTRATION_REVIEW3.md).

Runs after scripts/make_review2_tables.py and overrides its headline tables/sentences with crossed (cluster x image)
bootstrap CIs (R6), adds the paste-edge control (R5), the full matched-analysis table (R7), the data-provenance table
and the registration-status table.
  python scripts/make_review3_tables.py   (inputs: results/review3/*.csv, results/review2/*.csv)
"""
from __future__ import annotations

import pandas as pd

from wtss import paths

R2, R3 = paths.RESULTS / "review2", paths.RESULTS / "review3"
T = paths.REPO_ROOT / "paper" / "tables"
COH = ["ISIC hair", "Thyroid", "Ovary", "Capsule"]
NAME = {"ISIC hair": "dermoscopic hair", "Thyroid": "thyroid", "Ovary": "ovary", "Capsule": "capsule"}


def ci(v, lo, hi):
    return f"${v:+.3f}$ [${lo:+.3f}, {hi:+.3f}$]"


def write(name, lines, prefix="review2"):
    (T / f"{prefix}_{name}.tex").write_text("\n".join(lines) + "\n")
    print("wrote", prefix, name)


C = pd.read_csv(R3 / "crossed_ci.csv")


def get(claim, cohort):
    q = C[(C.claim == claim) & (C.cohort == cohort)]
    return None if q.empty else q.iloc[0]


def cx(claim, cohort):
    r = get(claim, cohort)
    return "--" if r is None else ci(r.estimate, r.crossed_lo, r.crossed_hi)


def law_table():
    L = ["\\begin{table}[t]\\centering\\small",
         "\\caption{The location law under four designs (reversed-test AUROC, DINOv2 ViT-B/14; 95\\% crossed "
         "seed$\\times$image bootstrap CIs, Sec.~\\ref{sec:setup}). \\emph{Real traps}: crossover "
         "$[\\text{mask}-\\text{ERM}]_{\\text{out}}-[\\text{mask}-\\text{ERM}]_{\\text{in}}$ between different artifact-bearing "
         "images (rebuilt run). \\emph{Matched}: the same after 1:1 propensity matching of the in-ROI and out-of-ROI "
         "artifact-bearing images (calliper 0.2 SD; Supplement~S11). \\emph{Transplant}: one real artifact instance pasted "
         "inside or outside the ROI of the same artifact-free images (location interaction). \\emph{Neutral paste}: neutral "
         "tissue pasted through the same masks, positions and blending (paste-edge control). Transplant and matching were "
         "designed after the real-trap results and registered before their own results; each design is its own Holm "
         "family.}\\label{tab:law}",
         "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lccccc}\\toprule",
         "Cohort & Real traps & Matched & Transplant (artifact) & Neutral paste & Artifact $-$ neutral \\\\\\midrule"]
    for c in COH:
        real = cx("real-trap crossover (repro)", c) + ("$^\\ddagger$" if c == "Capsule" else "")
        m = get("real-trap crossover (matched)", c)
        mt = "infeasible$^\\dagger$" if c == "Capsule" else ci(m.estimate, m.crossed_lo, m.crossed_hi)
        L.append(" & ".join([c, real, mt, cx("transplant interaction (artifact)", c),
                             cx("transplant interaction (neutral paste)", c), cx("artifact - neutral interaction (N3)", c)]) + " \\\\")
    L += ["\\bottomrule\\end{tabular}}",
          "\\par\\smallskip{\\footnotesize $^\\dagger$19 matched pairs in one label, below the pre-registered minimum of 30. "
          "$^\\ddagger$Descriptive: capsule lesions carrying debris are much larger than those with debris beside them "
          "(standardised mean difference 2.6) and matching is infeasible, so this crossover is not by itself evidence about "
          "location.}", "\\end{table}"]
    write("law", L)


def texts():
    p5 = pd.read_csv(R3 / "R5_paste_edge.csv").set_index("cohort")
    parts = [f"{NAME[c]} {cx('transplant interaction (artifact)', c)}" for c in COH]
    neu = [f"{NAME[c]} {cx('transplant interaction (neutral paste)', c)}" for c in COH]
    n3 = [f"{NAME[c]} {cx('artifact - neutral interaction (N3)', c)}" for c in COH]
    sig = [NAME[c] for c in COH if bool(p5.loc[c, "N3_supported"])]
    rng = f"{100 * p5.ratio_neutral_to_artifact.min():.0f}--{100 * p5.ratio_neutral_to_artifact.max():.0f}\\%"
    cf = "; ".join(f"{NAME[c]} {get('transplant mask - ERM (in)', c).estimate:+.3f}" for c in COH)
    write("transplant_text", [
        f"With the artifact outside the ROI masking removes the shortcut entirely---the masked model's reversed, "
        f"correlated and clean AUROC coincide---whereas with the identical artifact inside the ROI the shortcut survives "
        f"masking (Supplement~S11). The location interaction is positive in all four cohorts: " + ", ".join(parts) +
        " (Table~\\ref{tab:law}). \\emph{Paste-edge control (pre-registered).} Pasting neutral tissue through the same "
        f"masks, at the same positions and with the same blending gives a smaller but still positive interaction ("
        + ", ".join(neu) + f"), {rng} of the artifact's. The neutral patch is itself a local, label-correlated pattern that "
        f"the encoder learns, and masking removes it only outside the ROI. The artifact's content adds significantly to the "
        f"paste in {' and '.join(sig)} (artifact $-$ neutral: " + ", ".join(n3) + "). By the pre-registered rule the "
        "transplant is therefore evidence that \\emph{where} a label-correlated pattern sits---not which images carry it---"
        "decides what masking removes; it is not evidence about the appearance of any particular artifact, and the "
        f"in-ROI masking changes it produces (mask $-$ ERM: {cf}) are properties of pasted patterns, not of real "
        "artifacts in their natural context. For hair this matters most: a transplanted hair carries the shortcut in its "
        "pixels by construction, whereas in the real hair trap the shortcut is carried mostly by correlates of hair "
        "(oracle removal of every hair pixel gains only $+0.015$; Sec.~\\ref{sec:mech}), so the transplant shows the "
        "location effect for pixel-carried hair, not its size for hair in the wild.%"])
    m = pd.read_csv(R3 / "R7_matching_full.csv")
    txt = []
    for c in ["ISIC hair", "Thyroid", "Ovary"]:
        a, b = m[(m.cohort == c) & (m.caliper_sd == 0.2)].iloc[0], m[(m.cohort == c) & (m.caliper_sd == 0.05)].iloc[0]
        txt.append(f"{NAME[c]}: {int(a.kept_per_trap):,} of {int(a.in_roi_images):,} in-ROI and {int(a.out_roi_images):,} "
                   f"out-of-ROI artifact-bearing images kept per trap, pooled max $|$SMD$|$ {a.max_smd_before_pooled:.2f} "
                   f"$\\to$ {a.max_smd_after_pooled:.2f} (within label up to {max(a.max_smd_after_y0, a.max_smd_after_y1):.2f}), "
                   f"crossover {a.crossover_unmatched:+.3f} unmatched, {ci(a.crossover_matched, a.matched_lo, a.matched_hi)} "
                   f"matched, {ci(b.crossover_matched, b.matched_lo, b.matched_hi)} with a calliper of 0.05 SD "
                   f"(${b.change_pct:+.0f}$\\%)")
    write("matched_text", [
        "; ".join(txt) + ". The effect shrinks by up to 16\\% under the stricter calliper but every matched crossover "
        "excludes zero. Balance is not perfect: within a label some covariates keep standardised differences of 0.13--0.19 "
        "after matching (propensity matching balances the score, not every covariate), so residual confounding by these "
        "covariates cannot be excluded; the transplant, which holds the images fixed, does not depend on it. For capsule "
        "matching left only 19 pairs in one label, below the pre-registered minimum of 30; its real-trap crossover is "
        "therefore reported as descriptive, and its location evidence rests on the transplant.%"])
    rep = [f"{NAME[c]} {cx('real-trap crossover (repro)', c)}" for c in COH]
    write("repro_text", [
        "Every primary crossover reproduces the archived point estimate within the pre-registered tolerance of 0.03 "
        "(rebuilt, crossed CIs: " + "; ".join(rep) + ").%"])
    ft = pd.read_csv(R2 / "R4_finetune.csv")
    fx = pd.read_csv(paths.RESULTS / "finetune" / "ft_crossovers.csv").set_index("run")  # regenerated, crossed
    fc = lambda k: ci(fx.loc[k].seed_delta_mean, fx.loc[k].ci95_lo, fx.loc[k].ci95_hi)
    write("ft_text", [
        f"(crossover of a fine-tuned ResNet-50: hair {fc('Dermoscopy hair, ResNet-50')}; capsule {fc('Capsule, ResNet-50')}; "
        f"thyroid ViT-S {fc('Thyroid, ViT-S')}, ResNet-50 {fc('Thyroid, ResNet-50')}). "
        "The fine-tuned ovary network did not learn the task (ERM clean AUROC "
        f"{ft[(ft.cohort == 'ovary') & (ft.arch == 'resnet50_power')].iloc[0].erm_clean_trapA:.2f} and "
        f"{ft[(ft.cohort == 'ovary') & (ft.arch == 'resnet50_power')].iloc[0].erm_clean_trapB:.2f} in Trap~A and B, "
        "15 clusters), so it can neither show nor refute a shortcut effect and ovary is not evaluable for this check; "
        "the replication holds in all three evaluable cohorts%"])
    ab = ", ".join(f"{NAME[c].replace('dermoscopic ', '')} {get('transplant interaction (artifact)', c).estimate:+.2f}" for c in COH)
    write("abstract_transplant", [ab + "%"])


def op_tables():
    o = pd.read_csv(R3 / "operating_points_crossed.csv")
    q = o[o.cohort == "thyroid"]
    ops = {"OP1_maxBA": "max.\\ balanced accuracy", "OP2_spec0.80": "specificity $\\ge0.80$", "OP3_spec0.90": "specificity $\\ge0.90$",
           "OP4_sens0.80": "sensitivity $\\ge0.80$", "OP5_sens0.90": "sensitivity $\\ge0.90$"}
    n = int(q.n_conflicting_positives.iloc[0])
    per = pd.read_csv(R2 / "operating_points_per_seed.csv")
    L = ["\\begin{table}[t]\\centering\\scriptsize",
         "\\caption{Masking versus ERM on the unaltered, patient-disjoint official thyroid test split (614 images, 236 "
         "malignant) at five operating points fixed on validation data (mean over five seeds; $\\Delta$ = mask $-$ ERM with "
         "95\\% crossed seed$\\times$image bootstrap CI, thresholds re-estimated in every replicate). \\emph{Conflicting}: the "
         f"{n} malignant nodules with an in-ROI caliper (the caliper marks benign nodules in this dataset). The subgroup was "
         "defined after the first unaltered-data analysis; the operating-point test in it was registered before these "
         "numbers were computed.}\\label{tab:op}",
         "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lccccc}\\toprule",
         f"Threshold on validation & Sens.\\ ERM $\\to$ mask & $\\Delta$ sensitivity & Sens.\\ conflicting ($n={n}$) & "
         "$\\Delta$ sensitivity, conflicting & $\\Delta$ specificity \\\\\\midrule"]
    for op, lab in ops.items():
        g = q[q.op == op].set_index("metric")
        pe = per[(per.cohort == "thyroid") & (per.op == op)]
        sc = pe.groupby("method").sens_conflict.mean()
        s, c, sp = g.loc["sens"], g.loc["sens_conflict"], g.loc["spec"]
        se = pe.groupby("method").sens.mean()
        L.append(f"{lab} & {se['erm']:.3f} $\\to$ {se['mask']:.3f} & {ci(s.delta, s.crossed_lo, s.crossed_hi)} & "
                 f"{sc['erm']:.3f} $\\to$ {sc['mask']:.3f} & {ci(c.delta, c.crossed_lo, c.crossed_hi)} & "
                 f"{ci(sp.delta, sp.crossed_lo, sp.crossed_hi)} \\\\")
    L += ["\\bottomrule\\end{tabular}}\\end{table}"]
    write("op", L)
    g1, g2 = q[q.op == "OP1_maxBA"].set_index("metric"), q[q.op == "OP2_spec0.80"].set_index("metric")
    pe1 = per[(per.cohort == "thyroid") & (per.op == "OP1_maxBA")].groupby("method").sens.mean()
    pe2 = per[(per.cohort == "thyroid") & (per.op == "OP2_spec0.80")].groupby("method")
    s1, s2, c2 = g1.loc["sens"], g2.loc["sens"], g2.loc["sens_conflict"]
    write("op_inline", [f"{pe1['erm']:.3f} to {pe1['mask']:.3f} at the maximal-balanced-accuracy threshold "
                        f"({ci(s1.delta, s1.crossed_lo, s1.crossed_hi)}) and from {pe2.sens.mean()['erm']:.3f} to "
                        f"{pe2.sens.mean()['mask']:.3f} at a validation specificity of 0.80 "
                        f"({ci(s2.delta, s2.crossed_lo, s2.crossed_hi)})%"])
    write("op_conflict", [f"{ci(c2.delta, c2.crossed_lo, c2.crossed_hi)} ({pe2.sens_conflict.mean()['erm']:.3f} $\\to$ "
                          f"{pe2.sens_conflict.mean()['mask']:.3f}; $n={n}$)%"])


def natural_texts():
    names = {"Thyroid (official split)": "thyroid", "ISIC, BCN held out": "held-out BCN", "ISIC, HAM held out": "held-out HAM",
             "ISIC, MSK held out": "held-out MSK", "Capsule (held-out frames)": "capsule"}
    h = {k: get("natural mask - ERM (hard pairs)", k) for k in names}
    neg = [k for k in names if h[k].crossed_hi < 0]
    pos = [k for k in names if h[k].crossed_lo > 0]
    ns = [k for k in names if k not in neg and k not in pos]
    f = lambda k: ci(h[k].estimate, h[k].crossed_lo, h[k].crossed_hi)
    t = (f"On the shortcut-conflicting pairs masking \\emph{{lowers}} AUROC in {len(neg)} of 5 test sets ("
         + "; ".join(f"{names[k]} {f(k)}" for k in neg) + ")")
    if pos:
        t += ", raises it for " + " and ".join(f"{names[k]} ({f(k)})" for k in pos)
    if ns:
        t += ", and does not change it significantly for " + " and ".join(f"{names[k]} ({f(k)})" for k in ns)
    t += (". Over all test pairs masking changes thyroid AUROC by only "
          f"{cx('natural mask - ERM (all pairs)', 'Thyroid (official split)')}: the harm is confined to the cases in which "
          "the in-ROI artifact points to the wrong diagnosis, which is exactly where a shortcut is dangerous%")
    write("natural_text", [t])
    cap, bcn = "Capsule (held-out frames)", "ISIC, BCN held out"
    write("guide_natural", [
        f"annotation-free U-MtE changes AUROC over masking by {cx('natural mte - mask (all pairs)', cap)} on capsule and "
        f"{cx('natural mte - mask (all pairs)', bcn)} for the held-out BCN hospital, its protected variant by "
        f"{cx('natural mte_protect - mask (all pairs)', cap)} on capsule, and group balancing by "
        f"{cx('natural balanced - mask (all pairs)', cap)} on capsule%"])


def supp_tables():
    m = pd.read_csv(R3 / "R7_matching_full.csv")
    L = ["\\begin{center}\\scriptsize\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccccccc}\\toprule",
         "Cohort & Calliper (SD) & In-ROI / out-of-ROI images & Kept per trap (neg./pos.) & $|$SMD$|_{\\max}$ before & "
         "after (pooled) & after (within label) & Crossover unmatched & Crossover matched \\\\\\midrule"]
    for r in m.itertuples():
        L.append(f"{r.cohort if r.caliper_sd == 0.2 else ''} & {r.caliper_sd:.2f} & {r.in_roi_images:,} / {r.out_roi_images:,} & "
                 f"{r.kept_per_trap:,} ({r.kept_benign}/{r.kept_malignant_or_positive}){'' if r.feasible else '$^\\dagger$'} & "
                 f"{r.max_smd_before_pooled:.2f} & {r.max_smd_after_pooled:.2f} & {max(r.max_smd_after_y0, r.max_smd_after_y1):.2f} & "
                 f"{ci(r.crossover_unmatched, r.unmatched_lo, r.unmatched_hi)} & {ci(r.crossover_matched, r.matched_lo, r.matched_hi)} \\\\")
    L += ["\\bottomrule\\end{tabular}}\\end{center}",
          "{\\footnotesize $^\\dagger$Infeasible (fewer than 30 pairs in one label). Crossed seed$\\times$image CIs.}"]
    write("matched_full", L, "review3")
    p = pd.read_csv(R3 / "R5_paste_edge.csv")
    L = ["\\begin{center}\\small\\begin{tabular}{lcccccc}\\toprule",
         "Cohort & Artifact & Neutral paste & Neutral / artifact & Artifact $-$ neutral & $p_{\\text{Holm}}$ & Attributed to artifact \\\\\\midrule"]
    for r in p.itertuples():
        L.append(f"{r.cohort} & {ci(r.artifact, r.artifact_lo, r.artifact_hi)} & {ci(r.neutral, r.neutral_lo, r.neutral_hi)} & "
                 f"{r.ratio_neutral_to_artifact:.2f} & {ci(r.N3, r.N3_lo, r.N3_hi)} & {r.N3_p_holm:.4f} & "
                 f"{'yes' if r.attributed_to_artifact else 'no'} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{center}",
          "{\\footnotesize Pre-registered rule: attributed to the artifact only if artifact $-$ neutral $>0$ (Holm) and the "
          "neutral interaction is below one quarter of the artifact's.}"]
    write("paste_edge", L, "review3")
    c = C[C.orig_lo.notna()].copy()
    # the original (per-seed) intervals are not printed (docs/PREREGISTRATION_FINAL.md, A1); they are listed in
    # results/bootstrap_correction/sweeps_umte_old_vs_new.md. In the regenerated crossed_ci.csv the orig_* columns were
    # computed with WTSS_BOOTSTRAP=crossed (so they equal the crossed CI); the per-seed reference is the archived file.
    old = pd.read_csv(paths.REPO_ROOT / "results" / "review3" / "crossed_ci.csv")[["claim", "cohort", "orig_lo", "orig_hi"]]
    c = c.drop(columns=["orig_lo", "orig_hi"]).merge(old, on=["claim", "cohort"], how="left", validate="one_to_one")
    if c.orig_lo.isna().any():
        raise SystemExit("crossed table: claim without an archived per-seed interval")
    c["width_ratio"] = (c.crossed_hi - c.crossed_lo) / (c.orig_hi - c.orig_lo)
    c["verdict_changed"] = ((c.orig_lo > 0) | (c.orig_hi < 0)) != c.crossed_excludes_zero.astype(bool)
    L = ["\\begin{center}\\scriptsize\\resizebox{\\linewidth}{!}{\\begin{tabular}{llccc}\\toprule",
         "Claim & Cohort / test set & Crossed seed$\\times$image bootstrap & Width / per-seed width & Verdict changed \\\\\\midrule"]
    for r in c.itertuples():
        L.append(f"{r.claim.replace('_', chr(92) + '_')} & {r.cohort} & {ci(r.estimate, r.crossed_lo, r.crossed_hi)}"
                 f"{'' if r.crossed_excludes_zero else '$^*$'} & {r.width_ratio:.2f} & {'yes' if r.verdict_changed else 'no'} \\\\")
    L += ["\\bottomrule\\end{tabular}}\\end{center}", "{\\footnotesize $^*$Crossed CI includes zero. The original intervals are "
          "listed in \\texttt{results/bootstrap\\_correction/sweeps\\_umte\\_old\\_vs\\_new.md}.}"]
    write("crossed", L, "review3")


def provenance():
    rows = [("ISIC 2019 (hair)", "diagnoses (ISIC); lesion masks for the 10,015 HAM10000 images (manual)",
             "lesion masks for the other 15,316 images (U-Net trained on manual ISIC 2018 masks); hair/ruler masks "
             "(published semi-automatic masks)"),
            ("TN3K/TNCD (thyroid)", "nodule masks; benign/malignant from cytological biopsy",
             "caliper masks for all 3,493 images (rule-based detector)"),
            ("MMOTU (ovary)", "tumour masks; tumour categories", "caliper masks for all 1,202 images (same detector)"),
            ("SEE-AI (capsule)", "lesion boxes and classes; contamination masks for 22 frames (expert)",
             "contamination masks for 5,459 of 5,481 frames (linear probe on DINOv2 patch tokens, IoU 0.62)"),
            ("NIH ChestX-ray14", "drain labels of pneumothorax images (NEATX, non-expert annotators)",
             "disease labels (text-mined from reports); lung masks (TorchXRayVision); drain labels of pneumothorax-negative "
             "images (drain detector)"),
            ("All cohorts", "---", "visual audits of the automatic labels (by the analysis team, which includes an AI agent; "
             "no clinician)")]
    L = ["\\begin{center}\\scriptsize\\begin{tabular}{p{2.6cm}p{5.2cm}p{6.8cm}}\\toprule",
         "Cohort & Human annotation & Automatic annotation (model or rules) \\\\\\midrule"]
    L += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    L += ["\\bottomrule\\end{tabular}\\end{center}"]
    write("provenance", L, "review3")


def registration_status():
    rows = [("Real traps, controlled sweeps, natural data, U-MtE (P1--P4)", "before results", "16-test primary family (Holm)"),
            ("Primary-family grouping", "after results (post hoc)", "---"),
            ("Rebuild on new hardware (R0)", "before results", "tolerance criterion"),
            ("Covariate-matched traps (R1)", "designed after the real-trap results; registered before its own results", "own family (Holm, 4)"),
            ("Real-artifact transplant (R2)", "designed after the real-trap results; registered before its own results", "own family (Holm, 4)"),
            ("Operating points (R3); conflicting subgroup", "test registered before computing; subgroup defined post hoc", "decision rule"),
            ("Fine-tuned hair, ovary power (R4)", "before results", "single tests"),
            ("Paste-edge control (R5)", "designed after the transplant results; registered before its own results", "own family (Holm, 4)"),
            ("Crossed bootstrap (R6), stricter matching (R7)", "designed after the results; registered before computing", "re-estimation / sensitivity")]
    L = ["\\begin{center}\\scriptsize\\begin{tabular}{p{5.3cm}p{6cm}p{3.3cm}}\\toprule",
         "Analysis & Registration status & Multiplicity \\\\\\midrule"] + [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    L += ["\\bottomrule\\end{tabular}\\end{center}"]
    write("registration", L, "review3")


def main():
    law_table(); texts(); op_tables(); natural_texts(); supp_tables(); provenance(); registration_status()


if __name__ == "__main__":
    main()
