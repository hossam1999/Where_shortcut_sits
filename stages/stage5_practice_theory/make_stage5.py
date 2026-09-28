"""Stage 5 tables, macros and figures: practice (natural test sets, clinical operating points, the n = 78 thyroid
subgroup), replication breadth (encoders, scale, fine-tuning, external data), the DermLIP exception and the theory
judged by held-out prediction error.
  PYTHONPATH=src python stages/stage5_practice_theory/make_stage5.py"""
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
M = L.Macros("sFive")
NAT = [("Thyroid (official patient-disjoint split)", "thyroid"), ("Dermoscopy, BCN held out", "isic_BCN"),
       ("Dermoscopy, HAM held out", "isic_HAM"), ("Dermoscopy, MSK held out", "isic_MSK"), ("Capsule (held-out frames)", "capsule")]


def ci3(v):
    return None if v is None else {"seed_delta_mean": v[0], "ci95_lo": v[1], "ci95_hi": v[2]}


def natural():
    rows = []
    for lab, k in NAT:
        b = L.js(f"natural/{k}_dino518_repro/natural_boot.json")
        if b is None:
            continue
        rows.append([lab, f"{b['P(a|y=1)']:.3f} / {b['P(a|y=0)']:.3f}", L.ci_row(ci3(b.get("all | mask-erm"))),
                     L.ci_row(ci3(b.get("hard | mask-erm"))), L.ci_row(ci3(b.get("easy | mask-erm")))])
        kk = L.Macros.clean(k)
        for part in ("all", "hard", "easy"):
            M.add(f"Nat{kk}{part}", L.ci_row(ci3(b.get(f"{part} | mask-erm"))))
    L.table(T / "natural.tex", ["Test set (no imposed correlation)", "$P(A{=}1\\mid Y{=}1)$ / $P(A{=}1\\mid Y{=}0)$",
                                "mask $-$ ERM, all pairs", "shortcut-conflicting pairs", "shortcut-aligned pairs"], rows,
            "Natural test sets: AUROC change from masking on all positive--negative pairs and on the pairs whose "
            "artifact status conflicts with (``hard'') or follows (``easy'') the test set's own artifact--label "
            "association (DINOv2, crossed CIs).", "tab:nat", size="\\scriptsize", resize=True)


def clinical():
    d = L.csv("review2/clinical_metrics_repro.csv")
    rows = []
    if d is not None:
        for t in d.test.unique():
            for arm in ("erm", "mask", "balanced"):
                r = d[(d.test == t) & (d.arm == arm)].iloc[0]
                rows.append([L.tex_escape(t) if arm == "erm" else "", {"erm": "ERM", "mask": "mask", "balanced": "balanced"}[arm],
                             L.f3(r.AUROC_mean), L.f3(r.AUPRC_mean), L.f3(r.Brier_mean), L.f3(r.ECE_mean), L.f3(r.Sens_mean),
                             L.f3(r.Spec_mean)])
        th = d[d.test.str.startswith("Thyroid")]
        for arm in ("erm", "mask"):
            r = th[th.arm == arm].iloc[0]
            M.add(f"ThyAuc{arm}", L.f3(r.AUROC_mean)); M.add(f"ThySens{arm}", L.f3(r.Sens_mean)); M.add(f"ThySpec{arm}", L.f3(r.Spec_mean))
    L.table(T / "clinical.tex", ["Test set", "Arm", "AUROC", "AUPRC", "Brier", "ECE", "Sensitivity", "Specificity"], rows,
            "Clinical metrics on the natural test sets (seed means; threshold at maximal balanced accuracy on "
            "validation data).", "tab:clin", size="\\scriptsize")


def op():
    d = L.csv("final_op/operating_points_crossed_all.csv")
    if d is None:
        return
    names = {"OP1_maxBA": "OP1 max.\\ balanced accuracy", "OP2_spec0.80": "OP2 specificity $\\ge0.80$",
             "OP3_spec0.90": "OP3 specificity $\\ge0.90$", "OP4_sens0.80": "OP4 sensitivity $\\ge0.80$",
             "OP5_sens0.90": "OP5 sensitivity $\\ge0.90$"}
    rows = []
    th = d[d.cohort == "thyroid"]
    for o, lab in names.items():
        g = lambda m: th[(th.op == o) & (th.metric == m)].iloc[0]
        s, sp, sc = g("sens"), g("spec"), g("sens_conflict")
        rows.append([lab, f"{L.f3(s.erm_value)} $\\to$ {L.f3(s.mask_value)}", L.ci(s.delta, s.crossed_lo, s.crossed_hi),
                     L.ci(sp.delta, sp.crossed_lo, sp.crossed_hi),
                     f"{L.f3(sc.erm_value)} $\\to$ {L.f3(sc.mask_value)}", L.ci(sc.delta, sc.crossed_lo, sc.crossed_hi)])
        k = L.Macros.clean(o.split("_")[0])
        M.add(f"Sens{k}", L.ci(s.delta, s.crossed_lo, s.crossed_hi)); M.add(f"Conf{k}", L.ci(sc.delta, sc.crossed_lo, sc.crossed_hi))
        M.add(f"ConfErm{k}", L.f3(sc.erm_value)); M.add(f"ConfMask{k}", L.f3(sc.mask_value))
    M.add("NConf", int(th.n_conflicting_positives.iloc[0]))
    n_loss = sum(bool(th[(th.op == o) & (th.metric == "sens")].crossed_hi.iloc[0] < 0) for o in list(names)[1:])
    M.add("NSensLoss", n_loss)
    L.table(T / "op_thyroid.tex", ["Operating point (fixed on validation)", "Sensitivity ERM $\\to$ mask", "$\\Delta$ sensitivity",
                                   "$\\Delta$ specificity", "Sensitivity, conflicting malignant (ERM $\\to$ mask)",
                                   "$\\Delta$ in that subgroup"], rows,
            "Thyroid, official patient-disjoint test split: effect of masking at five validation-fixed operating points "
            "(crossed bootstrap with thresholds re-estimated in every replicate). The subgroup is the malignant nodules "
            "with an in-ROI caliper (shortcut-conflicting).", "tab:opthy", size="\\scriptsize", resize=True)
    rows = []
    for c, lab in (("isic_BCN", "Dermoscopy BCN"), ("isic_HAM", "Dermoscopy HAM"), ("isic_MSK", "Dermoscopy MSK"), ("capsule", "Capsule")):
        q = d[(d.cohort == c) & (d.op == "OP2_spec0.80")]
        s, sp, sc = (q[q.metric == m].iloc[0] for m in ("sens", "spec", "sens_conflict"))
        rows.append([lab, str(int(s.n_conflicting_positives)), L.ci(s.delta, s.crossed_lo, s.crossed_hi),
                     L.ci(sp.delta, sp.crossed_lo, sp.crossed_hi), L.ci(sc.delta, sc.crossed_lo, sc.crossed_hi)])
    L.table(T / "op_other.tex", ["Test set", "Conflicting positives", "$\\Delta$ sensitivity", "$\\Delta$ specificity",
                                 "$\\Delta$ sensitivity, conflicting"], rows,
            "Other natural test sets at OP2 (validation specificity $\\ge0.80$), mask $-$ ERM, crossed CIs.", "tab:opother",
            size="\\scriptsize")


def breadth():
    rows = []
    add = lambda coh, enc, x, kind: rows.append([coh, enc, kind, L.ci_row(x)])
    sc = L.csv("scale/scale_compare.csv")
    pc = L.csv("PRIMARY_CLAIMS.csv")
    for coh, key, d in (("Dermoscopy hair", "isic", None), ("Thyroid", "thyroid", "thyroid"), ("Ovary", "ovary", "ovary"),
                        ("Capsule", "capsule", "capsule")):
        if d is not None and sc is not None:
            for bb in ("ViT-S/14", "ViT-B/14", "ViT-L/14"):
                r = sc[(sc.cohort == d) & (sc.backbone == bb)]
                if len(r):
                    r = r.iloc[0]
                    add(coh, f"DINOv2 {bb}", {"seed_delta_mean": r.cross, "ci95_lo": r.cross_lo, "ci95_hi": r.cross_hi}, "frozen")
        if d is None:
            for bb, rel in (("ViT-S/14", "spec_e13/dinos518_spec_scale"), ("ViT-B/14", "spec_e13/dino518_spec"),
                            ("ViT-L/14", "spec_e13/dinol518_spec_scale")):
                s = L.js(f"{rel}/SUMMARY.json")
                add(coh, f"DINOv2 {bb}", None if s is None else s["crossover_B_minus_A_mask"], "frozen")
            s = L.js("spec_e13/dermlip224_spec/SUMMARY.json")
            add(coh, "DermLIP", None if s is None else s["crossover_B_minus_A_mask"], "frozen")
        for enc, rel in (("MedSigLIP-448", f"{d}/medsiglip448_main" if d != "ovary" else "ovary/medsiglip448_universal"),
                         ("ConvNeXt-384", f"{d}/convnext384_universal")):
            if d is None or not (L.NEW / rel / "T3_crossover.json").exists():
                continue
            add(coh, enc, L.js(f"{rel}/T3_crossover.json"), "frozen")
    ft = L.csv("finetune/ft_crossovers.csv")
    if ft is not None:
        for r in ft.itertuples():
            coh, enc = r.run.split(", ", 1)
            add(coh.replace("Dermoscopy hair", "Dermoscopy hair"), enc, r._asdict(), "fine-tuned")
            M.add(f"Ft{L.Macros.clean(r.run)}", L.ci_row(r._asdict()))
    ex = L.js("external_isic2020/traps_dino518/SUMMARY.json")
    if ex is not None:
        add("Dermoscopy hair, ISIC 2020 (external)", "DINOv2 ViT-B/14", ex["X1_crossover"], "frozen, new patients")
    L.table(T / "breadth.tex", ["Cohort", "Encoder", "Training", "Crossover [95\\% CI]"], rows,
            "Replication breadth of the real-artifact crossover (crossed CIs).", "tab:breadth", size="\\scriptsize")
    xs = [r[3] for r in rows]
    M.add("NBreadth", len(rows))
    vals = []
    for coh, enc, kind, _ in rows:
        pass
    return rows


def breadth_counts():
    """Counts for the prose: how many crossovers have CI above zero."""
    sc = L.csv("scale/scale_compare.csv")
    n, k = 0, 0
    if sc is not None:
        n += len(sc); k += int((sc.cross_lo > 0).sum())
    for rel in ("spec_e13/dinos518_spec_scale", "spec_e13/dino518_spec", "spec_e13/dinol518_spec_scale", "spec_e13/dermlip224_spec"):
        s = L.js(f"{rel}/SUMMARY.json")
        if s is not None:
            n += 1; k += int(s["crossover_B_minus_A_mask"]["ci95_lo"] > 0)
    for rel in ("thyroid/medsiglip448_main", "capsule/medsiglip448_main", "ovary/medsiglip448_universal",
                "thyroid/convnext384_universal", "capsule/convnext384_universal"):
        s = L.js(f"{rel}/T3_crossover.json")
        if s is not None:
            n += 1; k += int(s["ci95_lo"] > 0)
    M.add("NFrozen", n); M.add("NFrozenPos", k)
    ft = L.csv("finetune/ft_crossovers.csv")
    if ft is not None:
        M.add("NFt", len(ft)); M.add("NFtPos", int((ft.ci95_lo > 0).sum()))
    ex = L.js("external_isic2020/traps_dino518/SUMMARY.json")
    if ex is not None:
        M.add("ExtX", L.ci_row(ex["X1_crossover"]))


def external():
    b = L.csv("external_isic2020/traps_dino518/bootstrap_vs_erm.csv")
    g = L.js("external_isic2020/gate.json")
    if b is not None:
        for t in ("trapA", "trapB"):
            r = b[(b.trap == t) & (b.env == "test_rev")].iloc[0]
            M.add(f"Ext{t}", L.ci_row(r))
    if g is not None:
        M.add("ExtPatients", f"{g['n_patients']:,}"); M.add("ExtImages", f"{g['n_images']:,}"); M.add("ExtMel", g["n_melanoma"])


def dermlip():
    d = "spec_e13/dermlip224_spec"
    for t in ("trapA", "trapB"):
        b = L.csv(f"{d}/bootstrap_vs_erm.csv")
        if b is None:
            continue
        for arm in ("mask", "inpaint", "balanced"):
            for env in ("test_rev", "clean"):
                q = b[(b.trap == t) & (b.arm == arm) & (b.env == env) & (b.source == "all")]
                if len(q):
                    M.add(f"Derm{arm}{t}{env}", L.ci_row(q.iloc[0]))
    p = L.csv("theory/prediction_crossover.csv")
    if p is not None:
        r = p[(p.cohort == "ISIC hair") & (p.backbone == "DermLIP")]
        if len(r):
            M.add("DermPred", f"{r.pred.iloc[0]:+.3f}"); M.add("DermObs", f"{r.obs.iloc[0]:+.3f}")


def theory():
    s = L.js("theory/prediction_summary.json")
    if s is None:
        return
    rows = []
    for key, lab in (("T1_primary", "Primary cells (ERM, mask)"), ("T1_all_erm_head_arms", "All ERM-head arms")):
        for m, mlab in (("theory", "linear-Gaussian theory"), ("ref_sym", "heuristic: symmetric reversal"),
                        ("ref_clean", "heuristic: reversed = clean")):
            r = s[key][m]
            rows.append([lab if m == "theory" else "", mlab, str(r["n"]), L.f3(r["mae"]), L.f3(r["r"]), f"{100 * r['within_0.05']:.0f}\\%"])
    f = s["T1_finetune_secondary"]
    rows.append(["Fine-tuned (outside assumptions)", "linear-Gaussian theory", str(f["n"]), L.f3(f["mae"]), L.f3(f["r"]),
                 f"{100 * f['within_0.05']:.0f}\\%"])
    L.table(T / "theory.tex", ["Cells", "Predictor", "$n$", "MAE", "Pearson $r$", "within 0.05"], rows,
            "Held-out prediction of the reversed-test AUROC (never used in fitting): the two-parameter theory fitted to "
            "each cell's clean and correlated AUROC, against two parameter-free heuristics.", "tab:theory", size="\\scriptsize")
    t1, t2, t3 = s["T1_primary"], s["T2_crossover"], s["T3_sweeps"]
    M.add("TMae", L.f3(t1["theory"]["mae"])); M.add("TMaeSym", L.f3(t1["ref_sym"]["mae"])); M.add("TMaeClean", L.f3(t1["ref_clean"]["mae"]))
    M.add("TR", L.f3(t1["theory"]["r"])); M.add("TN", t1["theory"]["n"]); M.add("TWithin", f"{100 * t1['theory']['within_0.05']:.0f}\\%")
    M.add("TTwoN", t2["n"]); M.add("TTwoSign", t2["sign_agree"]); M.add("TTwoMae", L.f3(t2["mae"])); M.add("TTwoR", L.f3(t2["r"]))
    M.add("TTwoFt", t2["finetune_sign_agree"]); M.add("TThreeN", t3["n"]); M.add("TThreeSign", t3["sign_agree"])
    M.add("TThreeMae", L.f3(t3["mae"])); M.add("TThreeSymSign", s["reference_ref_sym"]["T3_sign_agree"])
    M.add("TFourN", s["T4_inroi_sign_from_dS"]["n"]); M.add("TFourAgree", s["T4_inroi_sign_from_dS"]["agree"])
    M.add("TFtMae", L.f3(f["mae"]))
    # figure: predicted vs observed reversed AUROC and crossover
    c = L.csv("theory/prediction_cells.csv"); x = L.csv("theory/prediction_crossover.csv")
    plt = L.plot_style()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    if c is not None:
        q = c[c.arm.isin(["erm", "mask"])]
        ax[0].scatter(q.pred_rev, q.rev, s=8, color=L.MASK_BLUE, label="theory")
        ax[0].scatter(q.ref_sym, q.rev, s=6, color=L.ERM_GREY, alpha=0.5, marker="x", label="symmetric heuristic")
        ax[0].plot([0, 1], [0, 1], "k--", lw=0.7); ax[0].set_xlabel("predicted reversed AUROC"); ax[0].set_ylabel("observed")
        ax[0].legend(frameon=False, fontsize=7)
    if x is not None:
        for ft, mk in ((False, "o"), (True, "s")):
            q = x[x.finetune == ft]
            ax[1].scatter(q.pred, q.obs, marker=mk, color=L.MASK_BLUE if not ft else L.ART_RED, s=14,
                          label="fine-tuned" if ft else "frozen + linear head")
        lim = float(max(x.pred.max(), x.obs.max())) + 0.05
        ax[1].plot([0, lim], [0, lim], "k--", lw=0.7); ax[1].set_xlabel("predicted crossover"); ax[1].set_ylabel("observed")
        ax[1].legend(frameon=False, fontsize=7)
    fig.savefig(F / "theory_prediction.pdf")


def figures():
    plt = L.plot_style()
    d = L.csv("final_op/operating_points_crossed_all.csv")
    if d is not None:
        th = d[d.cohort == "thyroid"]
        ops = ["OP1_maxBA", "OP2_spec0.80", "OP3_spec0.90", "OP4_sens0.80", "OP5_sens0.90"]
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
        for a, m, title in ((ax[0], "sens", "all malignant nodules"), (ax[1], "sens_conflict", "malignant, caliper on the nodule")):
            q = th[th.metric == m].set_index("op").loc[ops]
            x = np.arange(len(ops))
            a.bar(x - 0.18, q.erm_value, 0.36, color=L.ERM_GREY, label="ERM")
            a.bar(x + 0.18, q.mask_value, 0.36, color=L.MASK_BLUE, label="ROI masking")
            a.set_xticks(x, [o.split("_")[0] for o in ops]); a.set_title(title, fontsize=8); a.set_ylim(0, 1)
        ax[0].set_ylabel("sensitivity (official test split)"); ax[0].legend(frameon=False, fontsize=7)
        fig.savefig(F / "op_thyroid.pdf")
    rows = []
    sc = L.csv("scale/scale_compare.csv")
    if sc is not None:
        for r in sc.itertuples():
            rows.append((f"{r.cohort.capitalize()}, DINOv2 {r.backbone}", r.cross, r.cross_lo, r.cross_hi, "frozen"))
    for lab, rel in (("Dermoscopy, DINOv2 ViT-S/14", "spec_e13/dinos518_spec_scale"), ("Dermoscopy, DINOv2 ViT-B/14", "spec_e13/dino518_spec"),
                     ("Dermoscopy, DINOv2 ViT-L/14", "spec_e13/dinol518_spec_scale"), ("Dermoscopy, DermLIP", "spec_e13/dermlip224_spec")):
        s = L.js(f"{rel}/SUMMARY.json")
        if s is not None:
            x = s["crossover_B_minus_A_mask"]; rows.append((lab, x["seed_delta_mean"], x["ci95_lo"], x["ci95_hi"], "frozen"))
    ft = L.csv("finetune/ft_crossovers.csv")
    if ft is not None:
        for r in ft.itertuples():
            rows.append((r.run + " (fine-tuned)", r.seed_delta_mean, r.ci95_lo, r.ci95_hi, "ft"))
    ex = L.js("external_isic2020/traps_dino518/SUMMARY.json")
    if ex is not None:
        x = ex["X1_crossover"]; rows.append(("ISIC 2020, new patients", x["seed_delta_mean"], x["ci95_lo"], x["ci95_hi"], "ext"))
    if rows:
        fig, a = plt.subplots(figsize=(5.6, 0.25 * len(rows) + 0.8))
        col = {"frozen": L.MASK_BLUE, "ft": L.ART_RED, "ext": L.ROI_GREEN}
        for k, (lab, e, lo, hi, kind) in enumerate(rows[::-1]):
            a.plot([lo, hi], [k, k], color=col[kind], lw=2); a.plot(e, k, "o", color=col[kind], ms=4)
        a.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=7); a.axvline(0, color="k", lw=0.7)
        a.set_xlabel("real-artifact crossover (reversed AUROC)")
        fig.savefig(F / "breadth_forest.pdf")


def registrations():
    docs = {"Nat": "docs/PREREGISTRATION_NATURAL.md", "Rev2": "docs/PREREGISTRATION_REVIEW2.md",
            "Rev3": "docs/PREREGISTRATION_REVIEW3.md", "Theory": "docs/PREREGISTRATION_THEORY_PREDICTION.md",
            "Scale": "docs/PREREGISTRATION_SCALE.md", "Ft": "docs/PREREGISTRATION_FINETUNE.md",
            "Final": "docs/PREREGISTRATION_FINAL.md", "Isic": "docs/PREREGISTRATION_ISIC2019_TRAPS.md"}
    for k, doc in docs.items():
        M.add(f"Reg{k}", L.reg(doc))
    rows = [["Zero-shot vision--language scores, Z1", "\\mixed{} registered (descriptive); crossover test post hoc", "TEXT\\_PROMPT",
             L.reg("docs/PREREGISTRATION_TEXT_PROMPT.md"), "correlated $-$ reversed gap reported"],
            ["Natural test sets, N3 (hard pairs)", "\\ok{} registered", "NATURAL", L.reg(docs["Nat"]), "mask $-$ ERM $<0$ on hard pairs"],
            ["Operating points S1--S3", "\\ok{} registered", "REVIEW2 (R3)", L.reg(docs["Rev2"]),
             "``masking lowers sensitivity'' only if S1 and $\\ge3$ of OP2--OP5"],
            ["Thyroid subgroup, $n=$\\sFiveNConf{} (S3)", "\\mixed{} registered test in a post hoc subgroup", "REVIEW2 (R3)",
             L.reg(docs["Rev2"]), "sensitivity loss in the subgroup, CI $<0$"],
            ["Crossed CIs for the subgroup (R6)", "\\ok{} registered", "REVIEW3 (R6)", L.reg(docs["Rev3"]), "report the crossed CI"],
            ["Fine-tuned hair model and literature comparison (R4)", "\\ok{} registered", "REVIEW2 (R4)", L.reg(docs["Rev2"]),
             "crossover CI $>0$; comparison descriptive"],
            ["Encoder scale (S/B/L)", "\\ok{} registered", "SCALE", L.reg(docs["Scale"]), "crossover CI $>0$ at every scale"],
            ["Fine-tuned networks", "\\ok{} registered", "FINETUNE", L.reg(docs["Ft"]), "crossover CI $>0$ (secondary)"],
            ["DermLIP hair traps", "\\ok{} registered", "ISIC2019\\_TRAPS", L.reg(docs["Isic"]), "crossover CI $>0$"],
            ["Theory as held-out predictor, T1--T4", "\\ok{} registered", "THEORY\\_PREDICTION", L.reg(docs["Theory"]),
             "MAE below both heuristics; crossover signs"],
            ["External validation, X1 (ISIC 2020)", "\\ok{} registered", "FINAL (A4)", L.reg(docs["Final"]), "gate, then crossover CI $>0$"]]
    rows = [r[:2] + [r[2].replace("\\_", "\\_\\allowbreak{}")] + r[3:] for r in rows]
    L.table(T / "experiments.tex", ["Experiment", "Status", "Registration", "First commit (local time)", "Support criterion"],
            rows, "Experiments of this stage and their registration documents (\\texttt{docs/PREREGISTRATION\\_<name>.md}). "
            "Commit times are set by the committing machine and are not independent proof of order.",
            "tab:exp", align=L.EXP_ALIGN, size="\\scriptsize")


def zero_shot():
    b = L.js("zero_shot/boot.json"); s_ = L.csv("zero_shot/summary.csv")
    rows = []
    names = {"medsiglip448": "MedSigLIP", "dermlip224": "DermLIP"}
    for bb, coh, lab in (("medsiglip448", "thyroid", "Thyroid"), ("medsiglip448", "ovary", "Ovary"),
                         ("medsiglip448", "capsule", "Capsule"), ("dermlip224", "isic", "Dermoscopy hair")):
        g = lambda t, m: s_[(s_.backbone == bb) & (s_.cohort == coh) & (s_.trap == t) & (s_.method == m)].iloc[0]
        za = g("trapA", "zs")
        x = b.get(f"{bb}|{coh}|crossover (zs_mask-zs)_B-(zs_mask-zs)_A")
        a = b.get(f"{bb}|{coh}|trapA|zs_mask-zs"); bb_ = b.get(f"{bb}|{coh}|trapB|zs_mask-zs")
        rows.append([lab, names[bb], L.f3(za.clean), L.s3(za.corr_minus_rev), L.ci_row(ci3(a)), L.ci_row(ci3(bb_)), L.ci_row(ci3(x))])
        k = L.Macros.clean(coh)
        M.add(f"Zs{k}", L.ci_row(ci3(x))); M.add(f"ZsGap{k}", L.s3(za.corr_minus_rev)); M.add(f"ZsClean{k}", L.f3(za.clean))
    L.table(T / "zero_shot.tex", ["Cohort", "Model", "Zero-shot clean AUROC (Trap A)", "Correlated $-$ reversed gap",
                                  "mask effect, Trap A", "mask effect, Trap B", "Crossover"], rows,
            "Zero-shot vision--language scores (no training: cosine to a positive minus a negative class prompt) on the "
            "trap test environments, unmasked and masked (crossed CIs). The gap shows the untrained score is already swayed "
            "by the artifact; the crossover shows that masking changes it according to the location law.", "tab:zs",
            size="\\scriptsize", resize=True)


def ft_detail():
    rows = []
    for lab, d in (("Dermoscopy hair, ResNet-50", "finetune/isic/resnet50"), ("Thyroid, ResNet-50", "finetune/thyroid/resnet50"),
                   ("Thyroid, ViT-S/16", "finetune/thyroid/vit_small_patch16_224.augreg_in21k_ft_in1k"),
                   ("Capsule, ResNet-50", "finetune/capsule/resnet50"), ("Ovary, ResNet-50 (15 clusters)", "finetune/ovary/resnet50_power")):
        p = L.csv(f"{d}/paired_deltas.csv")
        if p is None:
            continue
        g = lambda t, a, r: p[(p.trap == t) & (p.arm == a) & (p.ref == r)]
        cell = lambda q: L.ci_row(q.iloc[0]) if len(q) else "\\na"
        rows.append([lab, cell(g("trapA", "mask", "erm")), cell(g("trapB", "mask", "erm")),
                     cell(g("trapA", "mask_balanced", "balanced"))])
        k = L.Macros.clean(lab.split(",")[0] + lab.split(",")[1].split("(")[0])
        M.add(f"FtA{k}", cell(g("trapA", "mask", "erm"))); M.add(f"FtB{k}", cell(g("trapB", "mask", "erm")))
    L.table(T / "ft_detail.tex", ["Fine-tuned network", "mask $-$ ERM, Trap A", "mask $-$ ERM, Trap B",
                                  "mask+balanced $-$ balanced, Trap A"], rows,
            "End-to-end fine-tuned networks (ImageNet initialisation, 224\\,px, 8 epochs; folds as bootstrap clusters): "
            "effect of masking per trap (crossed CIs).", "tab:ftdet", size="\\scriptsize")


def scale_gap():
    sc = L.csv("scale/scale_compare.csv")
    if sc is None:
        return
    rows = [[r.cohort.capitalize(), r.backbone, L.f3(r.erm_clean), L.f3(r.erm_gap), L.ci(r.cross, r.cross_lo, r.cross_hi)]
            for r in sc.itertuples()]
    L.table(T / "scale.tex", ["Cohort", "DINOv2", "ERM clean AUROC (Trap A)", "ERM gap corr $-$ rev (Trap A)", "Crossover"],
            rows, "Encoder scale (DINOv2 ViT-S/14 21\\,M, ViT-B/14 86\\,M, ViT-L/14 304\\,M parameters): ERM's shortcut gap in "
            "the in-ROI trap and the crossover.", "tab:scale", size="\\scriptsize")
    for r in sc.itertuples():
        k = L.Macros.clean(r.cohort + r.backbone.split("/")[0])
        M.add(f"Gap{k}", L.f3(r.erm_gap))


def literature():
    d = L.csv("review2/literature.csv")
    if d is None:
        return
    rows = [[L.tex_escape(r.model), f"{r.auroc:.3f}", "\\citet{gong2022acl}" if r.source == "gong2022acl" else "this work"]
            for r in d.itertuples()]
    L.derived("stage5_literature", d.assign(auroc=d.auroc.round(3)))
    L.table(T / "literature.tex", ["Model (thyroid, official TNCD test split)", "AUROC", "Source"], rows,
            "Absolute performance: the frozen encoder with a linear head against published fine-tuned CNNs on the same "
            "patient-disjoint split.", "tab:lit", size="\\scriptsize")
    M.add("LitOurs", f"{d[d.source != 'gong2022acl'].auroc.iloc[0]:.3f}")


def natural_balanced():
    for lab, k in NAT:
        b = L.js(f"natural/{k}_dino518_repro/natural_boot.json")
        if b is None:
            continue
        kk = L.Macros.clean(k)
        for part in ("all", "hard"):
            M.add(f"NatBal{kk}{part}", L.ci_row(ci3(b.get(f"{part} | balanced-mask"))))


def theory_sim():
    d = pd.read_csv(L.OLD / "theory" / "theory_sim.csv")
    md = float((d.sim - d.theory).abs().max())
    L.derived("stage5_theory_sim", pd.DataFrame([{"n_settings": len(d), "max_abs_sim_minus_theory": round(md, 3)}]))
    M.add("SimN", len(d)); M.add("SimMax", f"{md:.3f}")
    r4 = pd.read_csv(L.OLD / "review2" / "R4_finetune.csv")
    q = r4[r4.run.eq("review2") & r4.cohort.eq("ovary")].iloc[0]
    M.add("OvFtCleanA", f"{q.erm_clean_trapA:.2f}"); M.add("OvFtCleanB", f"{q.erm_clean_trapB:.2f}")


def estimator():
    r = L.estimator_check(["natural/", "finetune/", "thyroid/dinos518_scale", "thyroid/dinol518_scale", "ovary/dinos518_scale",
                           "ovary/dinol518_scale", "capsule/dinos518_scale", "capsule/dinol518_scale", "spec_e13/dinos518_spec_scale",
                           "spec_e13/dinol518_spec_scale", "zero_shot/"],
                          T / "estimator_check.tex", "tab:estcheck",
                          "The analyses of this stage under the per-seed and the crossed estimator (same predictions; "
                          "baseline arms only).", changes_path=T / "estimator_changes.tex", changes_label="tab:estchanges",
                          changes_caption="Contrasts of this stage whose verdict depends on the estimator.",
                          extra=lambda d: ~d.key.str.contains("mte|umte|protect", regex=True))
    if r:
        M.add("EstN", r["n"]); M.add("EstLost", r["lost"]); M.add("EstGained", r["gained"]); M.add("EstMed", f"{r['median']:.2f}")


def main():
    natural(); clinical(); op(); breadth(); breadth_counts(); external(); dermlip(); theory(); figures(); registrations()
    zero_shot(); ft_detail(); scale_gap(); literature(); natural_balanced(); theory_sim(); estimator()
    M.write(T / "numbers.tex")
    L.report_missing("stage5")


if __name__ == "__main__":
    main()
