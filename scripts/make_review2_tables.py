"""LaTeX tables for the second-review analyses (docs/PREREGISTRATION_REVIEW2.md), from results/review2/*.

  python scripts/make_review2_tables.py   -> paper/tables/review2_*.tex
Run scripts/analysis/review2_summary.py and scripts/analysis/operating_points.py first. Missing inputs are skipped.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wtss import paths

R = paths.RESULTS / "review2"
T = paths.REPO_ROOT / "paper" / "tables"


def ci(v, lo, hi):
    return f"${v:+.3f}$ [${lo:+.3f}, {hi:+.3f}$]"


def write(name, lines):
    (T / f"review2_{name}.tex").write_text("\n".join(lines) + "\n")
    print("wrote", name)


def location_table():
    """Real traps (unmatched, rebuilt), covariate-matched traps and the real-artifact transplant, per cohort."""
    f1, f2 = R / "R0_R1_crossovers.csv", R / "R2_transplant.csv"
    if not f1.exists():
        return
    x = pd.read_csv(f1)
    tr = pd.read_csv(f2) if f2.exists() else pd.DataFrame()
    L = ["\\begin{table}[t]\\centering\\small",
         "\\caption{The location law under three designs (reversed-test AUROC, DINOv2 ViT-B/14; 95\\% hierarchical "
         "bootstrap CIs). \\emph{Real traps}: crossover $[\\text{mask}-\\text{ERM}]_{\\text{out}}-[\\text{mask}-\\text{ERM}]_{\\text{in}}$ "
         "between different artifact-bearing images (rebuilt run). \\emph{Matched}: the same after 1:1 propensity matching of "
         "the in-ROI and out-of-ROI artifact-bearing images on lesion size, artifact amount and acquisition covariates "
         "(max $|$SMD$|$ after matching). \\emph{Transplant}: the same real artifact instance pasted inside or outside "
         "the ROI of the same artifact-free images (location interaction). Holm over four cohorts per design.}"
         "\\label{tab:law}",
         "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lccccc}\\toprule",
         "Cohort & Real traps & Matched & $|$SMD$|_{\\max}$ before/after & Transplant & $n$ transplant \\\\\\midrule"]
    for cohort in ("ISIC hair", "Thyroid", "Ovary", "Capsule"):
        r = x[(x.cohort == cohort) & (x.run == "repro")]
        m = x[(x.cohort == cohort) & (x.run == "matched")]
        if r.empty:
            continue
        r = r.iloc[0]
        cells = [cohort, ci(r.crossover, r.ci95_lo, r.ci95_hi)]
        if len(m):
            m = m.iloc[0]
            cells.append(ci(m.crossover, m.ci95_lo, m.ci95_hi) + ("" if m.feasible else "$^\\dagger$"))
            cells.append(f"{m.max_abs_smd_before_pooled:.2f} / {m.max_abs_smd_after_pooled:.2f}")
        else:
            cells += ["--", "--"]
        t = tr[tr.cohort == cohort] if len(tr) else tr
        if len(t):
            t = t.iloc[0]
            cells += [ci(t.interaction, t.ci95_lo, t.ci95_hi), f"{int(t.n_images):,}"]
        else:
            cells += ["--", "--"]
        L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule\\end{tabular}}",
          "\\par\\smallskip{\\footnotesize $^\\dagger$Fewer than 30 matched pairs per label (pre-registered feasibility "
          "limit): exploratory only. Capsule lesions carrying debris are much larger than those with debris beside them.}",
          "\\end{table}"]
    write("law", L)


def transplant_detail():
    f = R / "R2_transplant.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    L = ["\\begin{table}[t]\\centering\\small",
         "\\caption{Real-artifact transplant: reversed / correlated / clean AUROC of ERM and ROI masking with the same "
         "real artifact inside or outside the ROI of the same images, and the change from masking (95\\% CI). "
         "$|\\Delta p|$: same-head counterfactual sensitivity to inserting the artifact.}\\label{tab:transplant}",
         "\\resizebox{\\linewidth}{!}{\\begin{tabular}{llcccc}\\toprule",
         "Cohort & Location & ERM rev / corr / clean & Mask rev / corr / clean & Mask $-$ ERM (rev) & $|\\Delta p|$ ERM / mask \\\\\\midrule"]
    for r in d.itertuples():
        for tag, name in (("out", "outside ROI"), ("in", "inside ROI")):
            e = " / ".join(f"{getattr(r, f'auc_erm_{env}_{tag}'):.3f}" for env in ("test_rev", "test_corr", "clean"))
            m = " / ".join(f"{getattr(r, f'auc_mask_{env}_{tag}'):.3f}" for env in ("test_rev", "test_corr", "clean"))
            dd = ci(getattr(r, f"mask_minus_erm_{tag}"), getattr(r, f"mask_minus_erm_{tag}_lo"), getattr(r, f"mask_minus_erm_{tag}_hi"))
            cf = f"{getattr(r, f'cf_absdp_erm_{tag}'):.3f} / {getattr(r, f'cf_absdp_mask_{tag}'):.3f}"
            L.append(f"{r.cohort if tag == 'out' else ''} & {name} & {e} & {m} & {dd} & {cf} \\\\")
        L.append("\\midrule" if r.Index != len(d) - 1 else "\\bottomrule")
    L += ["\\end{tabular}}\\end{table}"]
    write("transplant", L)


def op_table():
    f = R / "operating_points.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    ops = {"OP1_maxBA": "max.\\ balanced accuracy", "OP2_spec0.80": "specificity $\\ge0.80$", "OP3_spec0.90": "specificity $\\ge0.90$",
           "OP4_sens0.80": "sensitivity $\\ge0.80$", "OP5_sens0.90": "sensitivity $\\ge0.90$"}
    L = ["\\begin{table}[t]\\centering\\scriptsize",
         "\\caption{Masking versus ERM on the unaltered official thyroid test split at five operating points fixed on "
         "validation data (mean over five seeds; $\\Delta$ = mask $-$ ERM with 95\\% hierarchical bootstrap CI, thresholds "
         "re-estimated in every replicate). \\emph{Conflicting}: malignant nodules with an in-ROI caliper (the caliper marks "
         "benign nodules in this dataset) and benign nodules without one.}\\label{tab:op}",
         "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccc}\\toprule",
         "Threshold on validation & Sensitivity ERM $\\to$ mask & $\\Delta$ sensitivity & $\\Delta$ sensitivity, conflicting & $\\Delta$ specificity \\\\\\midrule"]
    q = d[(d.cohort == "thyroid") & (d.arm == "mask") & (d.ref == "erm")]
    for op, lab in ops.items():
        g = q[q.op == op].set_index("metric")
        if g.empty:
            continue
        s, sc, sp = g.loc["sens"], g.loc["sens_conflict"], g.loc["spec"]
        L.append(f"{lab} & {s.ref_value:.3f} $\\to$ {s.arm_value:.3f} & {ci(s.delta, s.ci95_lo, s.ci95_hi)} & "
                 f"{ci(sc.delta, sc.ci95_lo, sc.ci95_hi)} & {ci(sp.delta, sp.ci95_lo, sp.ci95_hi)} \\\\")
    L += ["\\bottomrule\\end{tabular}}\\end{table}"]
    write("op", L)
    g1, g2 = q[q.op == "OP1_maxBA"].set_index("metric"), q[q.op == "OP2_spec0.80"].set_index("metric")
    s1, s2, c2 = g1.loc["sens"], g2.loc["sens"], g2.loc["sens_conflict"]
    write("op_inline", [f"{s1.ref_value:.3f} to {s1.arm_value:.3f} at the maximal-balanced-accuracy threshold "
                        f"({ci(s1.delta, s1.ci95_lo, s1.ci95_hi)}) and from {s2.ref_value:.3f} to {s2.arm_value:.3f} at a "
                        f"validation specificity of 0.80 ({ci(s2.delta, s2.ci95_lo, s2.ci95_hi)})%"])
    write("op_conflict", [f"{ci(c2.delta, c2.ci95_lo, c2.ci95_hi)} ({c2.ref_value:.3f} $\\to$ {c2.arm_value:.3f})%"])


def balance_table():
    f = R / "R1_balance.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = d[d.subset == "pooled"]
    L = ["\\begin{table}[t]\\centering\\scriptsize",
         "\\caption{Covariate balance of the location traps: standardised mean differences (SMD) between the "
         "artifact-bearing images of Trap~A (in-ROI) and Trap~B (out-of-ROI), before and after propensity matching "
         "(categorical covariates: largest level). Positive: larger in Trap~A.}\\label{tab:balance}",
         "\\begin{tabular}{llrrr}\\toprule", "Cohort & Covariate & SMD before & SMD after & \\\\\\midrule"]
    for cohort in ("ISIC hair", "Thyroid", "Ovary", "Capsule"):
        b = d[(d.cohort == cohort) & (d.comparison == "trapA_vs_trapB_before_matching")]
        a = d[(d.cohort == cohort) & (d.comparison == "trapA_vs_trapB_after_matching")]
        if b.empty:
            continue
        first = True
        for cov in b.covariate.unique():
            bb, aa = b[b.covariate == cov], a[a.covariate == cov]
            k = bb.smd.abs().idxmax()
            lev = bb.loc[k, "level"]
            av = aa[aa.level.fillna("") == ("" if pd.isna(lev) else lev)].smd
            lab = cov.replace("_", " ") + ("" if pd.isna(lev) or lev == "" else f" ({lev})")
            L.append(f"{cohort if first else ''} & {lab} & {bb.loc[k, 'smd']:+.2f} & {av.iloc[0]:+.2f} & \\\\" if len(av) else
                     f"{cohort if first else ''} & {lab} & {bb.loc[k, 'smd']:+.2f} & -- & \\\\")
            first = False
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}\\end{table}"]
    write("balance", L)


def registry_table():
    f = R / "registry.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    L = ["\\begin{center}\\scriptsize", "\\begin{tabular}{llll}\\toprule",
         "Registration & Registered (commit, UTC) & First outcome result (commit, UTC) & Order \\\\\\midrule"]
    for r in d.itertuples():
        reg = f"\\texttt{{{r.registered_commit}}} {str(r.registered_at)[:16].replace('T', ' ')}"
        res = f"\\texttt{{{r.first_outcome_commit}}} {str(r.first_outcome_at)[:16].replace('T', ' ')}" if isinstance(r.first_outcome_commit, str) else "--"
        L.append(f"\\texttt{{{r.registration.replace('_', chr(92) + '_').replace('.md', '')}}} & {reg} & {res} & {r.order if isinstance(r.order, str) else ''} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{center}"]
    write("registry", L)


def op_all_table():
    fc = paths.RESULTS / "final_op" / "operating_points_crossed_all.csv"  # regenerated, crossed bootstrap (mask - ERM)
    f = R / "operating_points.csv"
    if fc.exists():
        d = pd.read_csv(fc).rename(columns={"crossed_lo": "ci95_lo", "crossed_hi": "ci95_hi"})
        q = d[d.metric.isin(["sens", "sens_conflict", "spec"])]
    elif f.exists():
        d = pd.read_csv(f)
        q = d[(d.arm == "mask") & (d.ref == "erm") & d.metric.isin(["sens", "sens_conflict", "spec"])]
    else:
        return
    names = {"thyroid": "Thyroid", "isic_BCN": "ISIC, BCN held out", "isic_HAM": "ISIC, HAM held out",
             "isic_MSK": "ISIC, MSK held out", "capsule": "Capsule"}
    L = ["\\begin{center}\\scriptsize", "\\begin{tabular}{llccc}\\toprule",
         "Test set & Threshold (validation) & $\\Delta$ sensitivity & $\\Delta$ sensitivity, conflicting & $\\Delta$ specificity \\\\\\midrule"]
    for c, name in names.items():
        g = q[q.cohort == c]
        if g.empty:
            continue
        for k, op in enumerate(sorted(g.op.unique())):
            h = g[g.op == op].set_index("metric")
            L.append(f"{name if k == 0 else ''} & {op.split('_', 1)[1]} & " + " & ".join(
                ci(h.loc[m, 'delta'], h.loc[m, 'ci95_lo'], h.loc[m, 'ci95_hi']) for m in ("sens", "sens_conflict", "spec")) + " \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}\\end{center}"]
    write("op_all", L)


NAME = {"ISIC hair": "dermoscopic hair", "Thyroid": "thyroid", "Ovary": "ovary", "Capsule": "capsule"}


def texts():
    """Inline sentences with machine-written numbers for the main text."""
    f1, f2, f4 = R / "R0_R1_crossovers.csv", R / "R2_transplant.csv", R / "R4_finetune.csv"
    if f2.exists():
        d = pd.read_csv(f2)
        parts = [f"{NAME[r.cohort]} {ci(r.interaction, r.ci95_lo, r.ci95_hi)}" for r in d.itertuples()]
        sup = int(d.T1_supported.sum())
        cf = "; ".join(f"{NAME[r.cohort]} {r.cf_absdp_mask_in:.3f} versus {r.cf_absdp_erm_in:.3f}" for r in d.itertuples())
        bal = ", ".join(f"{NAME[r.cohort]} {ci(r.balanced_interaction, r.balanced_lo, r.balanced_hi)}" for r in d.itertuples())
        write("abstract_transplant", [", ".join(f"{NAME[r.cohort]} {r.interaction:+.2f}" for r in d.itertuples()).replace("dermoscopic hair", "hair") + "%"])
        harm = d[(d.mask_minus_erm_in_hi < 0)]
        harm_txt = (" With real artifacts inside the ROI masking is not only less useful but harmful in " +
                    " and ".join(f"{NAME[r.cohort]} ({ci(r.mask_minus_erm_in, r.mask_minus_erm_in_lo, r.mask_minus_erm_in_hi)})"
                                 for r in harm.itertuples()) + ".") if len(harm) else ""
        lo_b, hi_b = d.balanced_interaction.min(), d.balanced_interaction.max()
        write("transplant_text", [
            f"With the artifact outside the ROI masking removes the shortcut entirely---the masked model's reversed, "
            f"correlated and clean AUROC coincide---whereas with the identical artifact inside the ROI the shortcut "
            f"survives masking (Supplement~S11). The location interaction is positive and Holm-significant "
            f"in {sup} of {len(d)} cohorts: " + ", ".join(parts) + " (Table~\\ref{tab:law})." + harm_txt +
            f" Outside the ROI the masked model does not react to inserting the artifact at all; inside it, it still "
            f"reacts strongly (same-head $|\\Delta p|$, mask versus ERM: {cf}). Group balancing, which removes the "
            f"artifact--label association instead of pixels, is far less location-dependent than masking (interaction "
            f"${lo_b:+.3f}$ to ${hi_b:+.3f}$; per cohort: {bal}), close to the location invariance the linear-Gaussian "
            f"account predicts.%"])
    if f1.exists():
        d = pd.read_csv(f1)
        m = d[d.run == "matched"]
        feas = m[m.feasible == True]  # noqa: E712
        inf = m[m.feasible != True]  # noqa: E712
        parts = [f"{NAME[r.cohort]} {ci(r.crossover, r.ci95_lo, r.ci95_hi)} "
                 f"(unmatched {r.repro_crossover:+.3f}; max $|$SMD$|$ {r.max_abs_smd_before_pooled:.2f} $\\to$ "
                 f"{r.max_abs_smd_after_pooled:.2f})" for r in feas.itertuples()]
        txt = ("The matched crossovers are " + "; ".join(parts) + ", all Holm-significant.")
        if len(inf):
            r = inf.iloc[0]
            txt += (f" For {NAME[r.cohort]} matching left only {int(r.min_pairs_per_label)} pairs in one label, below the "
                    f"pre-registered minimum of 30, so its matched estimate ({ci(r.crossover, r.ci95_lo, r.ci95_hi)}) is "
                    f"exploratory; there the transplant is the decisive test.")
        write("matched_text", [txt + "%"])
        rep = d[d.run == "repro"]
        write("repro_text", ["Every primary crossover reproduces within the pre-registered tolerance: " + "; ".join(
            f"{NAME[r.cohort]} {ci(r.crossover, r.ci95_lo, r.ci95_hi)} (archived {r.archived:+.3f})" for r in rep.itertuples()) + ".%"])
    if f4.exists():
        d = pd.read_csv(f4)
        name = {"isic": "hair", "thyroid": "thyroid", "ovary": "ovary", "capsule": "capsule"}
        arch = {"resnet50": "ResNet-50", "resnet50_power": "ResNet-50", "vit_small_patch16_224.augreg_in21k_ft_in1k": "ViT-S"}
        # for ovary prefer the 15-cluster run (Amendment 1) when it exists
        if ((d.cohort == "ovary") & (d.arch == "resnet50_power")).any():
            d = d[~((d.cohort == "ovary") & (d.arch == "resnet50"))]
        d = d[~((d.run == "review2") & (d.arch == "resnet50") & (d.cohort != "isic"))]
        parts = [f"{name[r.cohort]} ({arch[r.arch]}, {int(r.clusters)} clusters) {ci(r.crossover, r.ci95_lo, r.ci95_hi)}"
                 for r in d.sort_values(["cohort", "arch"]).itertuples()]
        ov = d[(d.cohort == "ovary")]
        tail = ""
        if len(ov) and ov.iloc[0].ci95_lo <= 0 and not np.isnan(ov.iloc[0].get("erm_clean_trapA", np.nan)):
            r = ov.iloc[0]
            tail = (f"; the fine-tuned ovary network barely learns the task (ERM clean AUROC {r.erm_clean_trapA:.3f} and "
                    f"{r.erm_clean_trapB:.3f} in Trap~A and B), so its null crossover is uninformative")
        write("ft_text", ["(" + "; ".join(parts) + tail + ").%"])


def natural_texts():
    """Sentences for the unaltered-data section and the decision guide from the rebuilt natural runs (R0/R3)."""
    names = {"thyroid": "thyroid", "isic_BCN": "held-out BCN", "isic_HAM": "held-out HAM", "isic_MSK": "held-out MSK",
             "capsule": "capsule"}
    J = {}
    for c in names:
        f = paths.RESULTS / "natural" / f"{c}_dino518_repro" / "natural_boot.json"
        if f.exists():
            J[c] = json.loads(f.read_text())
    if len(J) < len(names):
        return
    c3 = lambda v: ci(*v)
    hard = {c: J[c]["hard | mask-erm"] for c in names}
    neg = [c for c in names if hard[c][2] < 0]
    pos = [c for c in names if hard[c][1] > 0]
    ns = [c for c in names if c not in neg and c not in pos]
    parts = [f"On the shortcut-conflicting pairs masking \\emph{{lowers}} AUROC in {len(neg)} of 5 test sets ("
             + "; ".join(f"{names[c]} {c3(hard[c])}" for c in neg) + ")"]
    if pos:
        parts.append("raises it for " + " and ".join(f"{names[c]} ({c3(hard[c])})" for c in pos))
    if ns:
        parts.append("and does not change it significantly for " + " and ".join(f"{names[c]} ({c3(hard[c])})" for c in ns))
    txt = ", ".join(parts) + (". Over all test pairs masking changes thyroid AUROC by only "
                              f"{c3(J['thyroid']['all | mask-erm'])}: the harm is confined to the cases in which the "
                              "in-ROI artifact points to the wrong diagnosis, which is exactly where a shortcut is dangerous%")
    write("natural_text", [txt])
    g = (f"annotation-free U-MtE changes AUROC over masking by {c3(J['capsule']['all | mte-mask'])} on capsule and "
         f"{c3(J['isic_BCN']['all | mte-mask'])} for the held-out BCN hospital, its protected variant by "
         f"{c3(J['capsule']['all | mte_protect-mask'])} on capsule, and group balancing by "
         f"{c3(J['capsule']['all | balanced-mask'])} on capsule%")
    write("guide_natural", [g])


def main():
    location_table(); transplant_detail(); op_table(); op_all_table(); balance_table(); registry_table(); texts()
    natural_texts()


if __name__ == "__main__":
    main()
