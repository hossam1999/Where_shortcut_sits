"""Two summary tables of the main text, generated from the result files (never typed by hand).

  python scripts/make_main_summary_tables.py
    -> paper/tables/main_natural.tex  (masking on the unaltered test data: every test set and model in one table)
    -> paper/tables/main_theory.tex   (prediction error of the linear-Gaussian account in its three tests)
    -> paper/tables/main_remedies.tex, s17_remedies_ci.tex (every remedy minus masking, round-8 confirmation data)
Sources: results/rerun_2026-09-28/review3/{crossed_ci,operating_points_crossed}.csv, results/round5/tier_boot.csv,
results/round6/ft_results.json, results/round6/ft_natural/operating_points.csv, results/round7/tm.json,
results/rerun_2026-09-28/theory{,_with_cxr}/prediction_summary.json, results/round4/theory/summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, NEW = ROOT / "results", ROOT / "results" / "rerun_2026-09-28"
OUT = ROOT / "paper" / "tables"


def ci(e, lo, hi):
    return f"${e:+.3f}$ [${lo:+.3f}, {hi:+.3f}$]"


def natural():
    c = pd.read_csv(NEW / "review3" / "crossed_ci.csv")
    nat = lambda sub, coh: c[(c.claim == f"natural mask - ERM ({sub})") & (c.cohort == coh)].iloc[0]
    op = pd.read_csv(NEW / "review3" / "operating_points_crossed.csv")
    op = op[(op.cohort == "thyroid") & (op.op == "OP2_spec0.80")].set_index("metric")
    ft = json.loads((R / "round6" / "ft_results.json").read_text())
    fop = pd.read_csv(R / "round6" / "ft_natural" / "operating_points.csv")
    fop = fop[(fop.arm == "mask") & (fop.ref == "erm") & (fop.op == "OP2_spec0.80")].set_index("metric")
    tm = json.loads((R / "round7" / "tm.json").read_text())
    tb = pd.read_csv(R / "round5" / "tier_boot.csv")
    tier = lambda t, s, a="mask", r="erm": tb[(tb.tier == t) & (tb.subset == s) & (tb.arm == a) & (tb.ref == r)].iloc[0]
    hard_ft = next(x for x in ft["contrasts"] if x["subset"] == "hard" and x["contrast"] == "mask-erm")
    arrow = lambda a, b: f"{a:.3f} $\\to$ {b:.3f}"
    reg, sub, sens = "registered", "registered test, post hoc subgroup", "sensitivity analysis"
    rows = [("\\emph{Thyroid, official patient-disjoint split: frozen DINOv2}", None, None, None),
            ("AUROC, all pairs", "", ci(*nat("all pairs", "Thyroid (official split)")[["estimate", "crossed_lo", "crossed_hi"]]), reg),
            ("AUROC, conflicting pairs", "", ci(*nat("hard pairs", "Thyroid (official split)")[["estimate", "crossed_lo", "crossed_hi"]]), reg),
            ("Sensitivity at validation specificity 0.80", arrow(op.loc["sens", "erm_value"], op.loc["sens", "mask_value"]),
             ci(*op.loc["sens", ["delta", "ci95_lo", "ci95_hi"]]), reg),
            ("\\quad 78 conflicting malignant nodules", arrow(op.loc["sens_conflict", "erm_value"], op.loc["sens_conflict", "mask_value"]),
             ci(*op.loc["sens_conflict", ["delta", "ci95_lo", "ci95_hi"]]), sub),
            ("\\quad same, threshold re-tuned on the test set", arrow(tm["TM2"]["erm_value"], tm["TM2"]["mask_value"]),
             ci(tm["TM2"]["estimate"], tm["TM2"]["ci95_lo"], tm["TM2"]["ci95_hi"]), reg),
            (f"\\emph{{Same split: fine-tuned ConvNeXt-T (test AUROC {ft['FT0']['estimate']:.3f})}}", None, None, None),
            ("AUROC, conflicting pairs", "", ci(hard_ft["estimate"], hard_ft["ci95_lo"], hard_ft["ci95_hi"]), reg),
            ("Sensitivity at validation specificity 0.80", arrow(fop.loc["sens", "ref_value"], fop.loc["sens", "arm_value"]),
             ci(*fop.loc["sens", ["delta", "ci95_lo", "ci95_hi"]]), reg),
            ("\\quad 78 conflicting malignant nodules", arrow(fop.loc["sens_conflict", "ref_value"], fop.loc["sens_conflict", "arm_value"]),
             ci(*fop.loc["sens_conflict", ["delta", "ci95_lo", "ci95_hi"]]), reg),
            ("\\quad same, threshold re-tuned on the test set", arrow(tm["TM1"]["erm_value"], tm["TM1"]["mask_value"]),
             ci(tm["TM1"]["estimate"], tm["TM1"]["ci95_lo"], tm["TM1"]["ci95_hi"]), reg),
            ("\\emph{Dermoscopy, one ISIC 2019 source held out: frozen DINOv2}", None, None, None)]
    for src in ("BCN", "HAM", "MSK"):
        rows.append((f"AUROC, conflicting pairs, {src} held out", "",
                     ci(*nat("hard pairs", f"ISIC, {src} held out")[["estimate", "crossed_lo", "crossed_hi"]]), reg))
    cal, loose = tier("calibrated", "hard"), tier("d19_le8", "hard")
    rows += [("\\emph{New patients: trained on ISIC 2019, tested on ISIC 2020}", None, None, None),
             ("AUROC, all pairs", "", ci(*tier("calibrated", "all")[["estimate", "ci95_lo", "ci95_hi"]]), sens),
             ("AUROC, conflicting pairs (label-free duplicate rule)", "", ci(cal.estimate, cal.ci95_lo, cal.ci95_hi), sens),
             ("AUROC, conflicting pairs (registered duplicate rule)", "", ci(loose.estimate, loose.ci95_lo, loose.ci95_hi), reg),
             ("\\emph{Capsule endoscopy, held-out frame blocks}", None, None, None),
             ("AUROC, all pairs", "", ci(*nat("all pairs", "Capsule (held-out frames)")[["estimate", "crossed_lo", "crossed_hi"]]), reg),
             ("AUROC, conflicting pairs", "", ci(*nat("hard pairs", "Capsule (held-out frames)")[["estimate", "crossed_lo", "crossed_hi"]]), reg)]
    L = ["\\begin{table}[!tbp]\\centering\\small",
         "\\caption{Masking on the unaltered test data: mask $-$ ERM with crossed 95\\% CIs (mean over five seeds). "
         "\\emph{Conflicting pairs}: positive--negative pairs in which the in-ROI artifact points to the wrong diagnosis. "
         "Sensitivities at thresholds fixed on validation data, except where re-tuned on the test set so that both arms "
         "reach specificity 0.80. The 78 conflicting nodules are the malignant nodules of the test split that carry an "
         "in-ROI caliper; for the frozen model they were defined after its first unaltered-data analysis, for the "
         "fine-tuned model before any of its results. All five operating points: Supplement~S9.}\\label{tab:natural}",
         "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lccl}\\toprule",
         "Measure & ERM $\\to$ mask & mask $-$ ERM [95\\% CI] & Status \\\\\\midrule"]
    for lab, a, d, s in rows:
        L.append(f"\\multicolumn{{4}}{{l}}{{{lab}}} \\\\" if a is None else f"{lab} & {a} & {d} & {s} \\\\")
    L += ["\\bottomrule\\end{tabular}}", "\\end{table}"]
    (OUT / "main_natural.tex").write_text("% generated by scripts/make_main_summary_tables.py\n" + "\n".join(L) + "\n")


def theory():
    a = json.loads((NEW / "theory" / "prediction_summary.json").read_text())
    b = json.loads((NEW / "theory_with_cxr" / "prediction_summary.json").read_text())
    p = json.loads((R / "round4" / "theory" / "summary.json").read_text())
    rows = []
    for lab, s in (("Retrospective: real traps and controlled sweeps", a), ("\\quad with the chest-radiograph cells added", b)):
        t = s["T1_primary"]
        rows.append([lab, str(t["theory"]["n"]), f"{t['theory']['mae']:.3f}", f"{t['ref_sym']['mae']:.3f}",
                     f"{t['ref_clean']['mae']:.3f}", f"{t['theory']['r']:.2f}",
                     f"{s['T2_crossover']['sign_agree']}/{s['T2_crossover']['n']}"])
    t = p["T1"]
    rows.append(["Prospective, registered before the data existed", str(t["theory"]["n"]), f"{t['theory']['mae']:.3f}",
                 f"{t['ref_b_symmetry']['mae']:.3f}", f"{t['ref_a_rev_eq_clean']['mae']:.3f}", f"{t['theory']['r']:.3f}",
                 f"{p['T2']['sign_agree']}/{p['T2']['n']}"])
    L = ["\\begin{table}[t]\\centering\\small",
         "\\caption{Held-out reversed-test AUROC predicted from each model's clean and correlated AUROC only: mean "
         "absolute error of the linear-Gaussian account and of two reference predictors that use the same inputs, the "
         "account's correlation with the observations, and the crossovers whose sign it gets right.}\\label{tab:theory}",
         "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccccc}\\toprule",
         "Test & Models & Account & Symmetry heuristic & Reversed $=$ clean & $r$ & Crossover signs \\\\\\midrule"]
    L += [" & ".join(r) + " \\\\" for r in rows]
    L += ["\\bottomrule\\end{tabular}}", "\\end{table}"]
    (OUT / "main_theory.tex").write_text("% generated by scripts/make_main_summary_tables.py\n" + "\n".join(L) + "\n")


REM_ARMS = [("erm", "ERM (no mask)", "no"), ("balanced", "Group balancing", "yes"), ("umte", "U-MtE", "no"),
            ("umte_balanced", "U-MtE + balancing", "yes"), ("mask_cmc", "Class-conditional constraint", "yes")]
REM_ROWS = [("Trap~A: artifact inside the ROI", [("trap/thyroid", "Thyroid"), ("trap/capsule", "Capsule"),
                                                  ("trap/isic", "Dermoscopy hair"), ("trap/ovary", "Ovary$^*$")], "trapA", "min_rev_corr"),
            ("Trap~B: artifact outside the ROI", [("trap/thyroid", "Thyroid"), ("trap/capsule", "Capsule"),
                                                   ("trap/isic", "Dermoscopy hair"), ("trap/ovary", "Ovary$^*$")], "trapB", "min_rev_corr"),
            ("Unaltered test sets, all pairs", [("natural/thyroid", "Thyroid (official split)"), ("natural/capsule", "Capsule"),
                                                 ("natural/isic_BCN", "ISIC, BCN held out"), ("natural/isic_HAM", "ISIC, HAM held out"),
                                                 ("natural/isic_MSK", "ISIC, MSK held out"), ("natural/isic2020", "ISIC 2019 $\\to$ 2020")],
             "all", "clean"),
            ("Unaltered test sets, conflicting pairs", [("natural/thyroid", "Thyroid (official split)"),
                                                         ("natural/isic_BCN", "ISIC, BCN held out"), ("natural/isic_MSK", "ISIC, MSK held out"),
                                                         ("natural/isic2020", "ISIC 2019 $\\to$ 2020")], "hard", "clean")]


def remedies():
    """Every remedy against masking on the round-8 confirmation data. With results/round8/confirm/absolute_auroc.csv
    (scripts/round8/absolute_scores.py) the main table shows each method's AUROC, masking included; without it, each
    method minus masking. Arrows always come from the registered differences (descriptive_all_cells.csv)."""
    d = pd.read_csv(R / "round8" / "confirm" / "descriptive_all_cells.csv")
    fa = R / "round8" / "confirm" / "absolute_auroc.csv"
    absd = pd.read_csv(fa) if fa.exists() else None
    v = pd.read_csv(R / "round8" / "confirm" / "verdicts.csv")
    v = v[(v.scope == "all cohorts") & (v.candidate == "mask_cmc")].iloc[0]

    def diff(src, c, env, arm):
        q = d[(d.source == src) & (d.cell == c) & (d.env == env) & (d.arm == arm)]
        return None if q.empty else q.iloc[0]

    def cell(src, c, env, arm, mode):
        if mode == "abs":
            q = absd[(absd.source == src) & (absd.cell == c) & (absd.env == env) & (absd.arm == arm)]
            if q.empty:
                return "--"
            if arm == "mask":
                return f"{q.iloc[0].auroc:.3f}"
            r = diff(src, c, env, arm)
            mark = "" if r is None else ("$\\uparrow$" if r.ci95_lo > 0 else ("$\\downarrow$" if r.ci95_hi < 0 else ""))
            return f"{q.iloc[0].auroc:.3f}{mark}"
        r = diff(src, c, env, arm)
        if r is None:
            return "--"
        if mode == "ci":
            return ci(r.arm_minus_mask, r.ci95_lo, r.ci95_hi)
        mark = "$\\uparrow$" if r.ci95_lo > 0 else ("$\\downarrow$" if r.ci95_hi < 0 else "")
        est = f"{r.arm_minus_mask:+.3f}" if round(r.arm_minus_mask, 3) != 0 else "0.000"
        return f"${est}${mark}"

    crit = ("The class-conditional constraint was the registered candidate: the criterion allowed a loss of at most 0.01 "
            "on all pairs and required gains in every Trap~A and on every set of conflicting pairs, and it met "
            f"{int(v.components_met)} of its {int(v.components)} components; the other columns are comparators fitted on "
            "the same data.")
    common = ("on the confirmation data of the first registered round (new trap resamples and the unaltered test sets; "
              "frozen DINOv2). Traps: the worse of reversed and correlated AUROC, so a method cannot score by flipping the "
              "shortcut. $\\uparrow$/$\\downarrow$: better/worse than masking, 95\\% crossed CI of the difference entirely "
              "above/below zero (all intervals in Supplement~S17). $^*$Ovary never influenced a choice. ")
    main_mode = "abs" if absd is not None else "diff"
    arms_main = REM_ARMS[:1] + [("mask", "ROI masking", "no")] + REM_ARMS[1:] if main_mode == "abs" else REM_ARMS
    caption_main = (("AUROC of every method " if main_mode == "abs" else
                     "Every remedy against masking on the same data: AUROC of each method minus that of ROI masking "
                     "(positive = better than masking), ") + common + crit)
    for mode, arms, name, label, size, cap in (
            (main_mode, arms_main, "main_remedies", "tab:remedies", "\\small", caption_main),
            ("ci", REM_ARMS, "s17_remedies_ci", "tab:s17remci", "\\scriptsize",
             "Every remedy against masking on the confirmation data of the first registered round: arm $-$ masking with "
             "crossed 95\\% CIs (the main-text remedies table with its intervals).")):
        L = ["\\begin{table}[!tbp]\\centering" + size, "\\caption{" + cap + "}\\label{" + label + "}",
             "\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "c" * len(arms) + "}\\toprule",
             "Test & " + " & ".join(a[1] for a in arms) + " \\\\",
             "Needs artifact labels & " + " & ".join(a[2] for a in arms) + " \\\\\\midrule"]
        for head, rows, c, env in REM_ROWS:
            L.append(f"\\multicolumn{{{len(arms) + 1}}}{{l}}{{\\emph{{{head}}}}} \\\\")
            for src, lab in rows:
                L.append(f"\\quad {lab} & " + " & ".join(cell(src, c, env, a[0], mode) for a in arms) + " \\\\")
        L += ["\\bottomrule\\end{tabular}}", "\\end{table}"]
        (OUT / f"{name}.tex").write_text("% generated by scripts/make_main_summary_tables.py\n" + "\n".join(L) + "\n")

if __name__ == "__main__":
    natural(); theory(); remedies()
    print("wrote paper/tables/main_{natural,theory,remedies}.tex and s17_remedies_ci.tex")
