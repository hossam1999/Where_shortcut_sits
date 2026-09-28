"""Stage 3 tables, macros and figures: real artifacts across modalities (crossed bootstrap, regenerated runs).
  PYTHONPATH=src python stages/stage3_real_artifacts/make_stage3.py"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import stagelib as L  # noqa: E402

T, F = HERE / "tables", HERE / "figures"
T.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
M = L.Macros("sThree")
# (label, cohort key, backbone label, results dir, kind) ; kind: spec (ISIC E13 layout) or real (T3 layout)
RUNS = [("Dermoscopy hair", "isic", "DINOv2", "spec_e13/dino518_spec", "spec"),
        ("Dermoscopy hair", "isic", "DermLIP", "spec_e13/dermlip224_spec", "spec"),
        ("Thyroid calipers", "thyroid", "DINOv2", "thyroid/dino518_main", "real"),
        ("Thyroid calipers", "thyroid", "MedSigLIP", "thyroid/medsiglip448_main", "real"),
        ("Ovarian calipers", "ovary", "DINOv2", "ovary/dino518_main", "real"),
        ("Capsule debris", "capsule", "DINOv2", "capsule/dino518_main", "real"),
        ("Capsule debris", "capsule", "MedSigLIP", "capsule/medsiglip448_main", "real")]


def boot(d, kind, trap, arm="mask", env="test_rev"):
    b = L.csv(f"{d}/bootstrap_vs_erm.csv")
    if b is None:
        return None
    q = b[(b.trap == trap) & (b.arm == arm) & (b.env == env)]
    if kind == "spec" and "source" in q:
        q = q[q.source == "all"]
    return None if q.empty else q.iloc[0]


def crossover(d, kind):
    if kind == "spec":
        s = L.js(f"{d}/SUMMARY.json")
        return None if s is None else s["crossover_B_minus_A_mask"]
    return L.js(f"{d}/T3_crossover.json")


def counts(d, kind):
    return L.csv(f"{d}/counts_vs_expected.csv" if kind == "spec" else f"{d}/counts.csv")


def cohort_table():
    desc = {"isic": ("ISIC 2019 dermoscopy (HAM10000, BCN20000, MSK)", "melanoma vs other", "hair (published masks)",
                     "lesion: HAM manual / U-Net"),
            "thyroid": ("TN3K ultrasound, TNCD labels", "malignant vs benign nodule", "calipers (rule detector)",
                        "nodule (manual)"),
            "ovary": ("MMOTU 2D ultrasound", "malignant vs benign tumour", "calipers (rule detector)", "tumour (manual)"),
            "capsule": ("SEE-AI capsule endoscopy", "erosion vs polyp-like lesion", "debris (patch probe)",
                        "lesion box (expert)")}
    rows, seen = [], set()
    for lab, key, bb, d, kind in RUNS:
        if key in seen:
            continue
        seen.add(key)
        c = counts(d, kind)
        cell = lambda t, col: "\\na" if c is None else str(int(c[c.trap == t][col].iloc[0]))
        rows.append([desc[key][0], desc[key][1], desc[key][2], desc[key][3],
                     f"{cell('trapA', 'A0_Y1')}/{cell('trapA', 'A0_Y0')}", f"{cell('trapA', 'A1_Y1')}/{cell('trapA', 'A1_Y0')}",
                     f"{cell('trapB', 'A1_Y1')}/{cell('trapB', 'A1_Y0')}"])
        if c is not None:
            M.add(f"NA{key}", int(c[c.trap == "trapA"][["A1_Y0", "A1_Y1"]].sum(axis=1).iloc[0]))
            M.add(f"NB{key}", int(c[c.trap == "trapB"][["A1_Y0", "A1_Y1"]].sum(axis=1).iloc[0]))
            M.add(f"NZero{key}", int(c[c.trap == "trapA"][["A0_Y0", "A0_Y1"]].sum(axis=1).iloc[0]))
    L.table(T / "cohorts.tex", ["Cohort", "Task", "Artifact (mask source)", "ROI", "Artifact-free pos/neg",
                                "Trap A pos/neg", "Trap B pos/neg"], rows,
            "Real-artifact cohorts and pool sizes (images, positive/negative diagnosis) before the environments are "
            "sampled. Trap A: artifact mostly inside the ROI ($r\\ge0.5$); Trap B: outside ($r<0.1$); both share the "
            "artifact-free group.", "tab:cohorts", align="p{3.4cm}p{2.4cm}p{2.5cm}p{2.2cm}ccc", size="\\scriptsize", resize=True)


def trap_table():
    rows = []
    for lab, key, bb, d, kind in RUNS:
        a, b, x = boot(d, kind, "trapA"), boot(d, kind, "trapB"), crossover(d, kind)
        ea = L.per_seed_auc(d, "erm", "trapA", "test_rev"); eb = L.per_seed_auc(d, "erm", "trapB", "test_rev")
        rows.append([lab, bb, f"{L.f3(ea)} / {L.f3(eb)}", L.ci_row(a), L.ci_row(b), L.ci_row(x)])
        k = L.Macros.clean(f"{key}{bb}")
        M.add(f"A{k}", L.ci_row(a)); M.add(f"B{k}", L.ci_row(b)); M.add(f"X{k}", L.ci_row(x))
        M.add(f"ErmA{k}", L.f3(ea)); M.add(f"ErmB{k}", L.f3(eb))
        ca = L.per_seed_auc(d, "mask", "trapA", "test_corr"); ra = L.per_seed_auc(d, "mask", "trapA", "test_rev")
        M.add(f"MaskCorrA{k}", L.f3(ca)); M.add(f"MaskRevA{k}", L.f3(ra))
    L.table(T / "real_traps.tex", ["Cohort", "Encoder", "ERM reversed AUROC A / B", "mask $-$ ERM, Trap A (in ROI)",
                                   "mask $-$ ERM, Trap B (outside)", "Crossover B $-$ A"], rows,
            "Real-artifact traps: change in reversed-test AUROC from ROI masking, per trap, and the crossover (5 seeds "
            "$\\times$ 5 grouped folds; crossed 95\\% CIs).", "tab:traps", size="\\scriptsize", resize=True)
    xs = [crossover(d, kind) for *_, d, kind in RUNS]
    M.add("NRuns", len(RUNS)); M.add("NXPos", sum(x is not None and float(x["ci95_lo"]) > 0 for x in xs))


def source_strata():
    s = L.js("spec_e13/dino518_spec/SUMMARY.json")
    rows = []
    if s is not None:
        for r in s["source_stratified_trapA_mask"]:
            rows.append([r["source"], L.ci_row(r)])
            M.add(f"Src{r['source']}", L.ci_row(r))
    L.table(T / "source_strata.tex", ["Source", "mask $-$ ERM, Trap A"], rows or [["\\na", "\\na"]],
            "Dermoscopy hair, Trap A, by image source (DINOv2).", "tab:src")


def area_ratio():
    e = L.js("analysis/E14_area_ratio.json")
    rows = []
    if e is not None:
        for k, med in zip(("T1", "T2", "T3"), e["tertile_medians"]):
            rows.append([k.replace("T", "tertile "), L.f3(med), L.ci_row(e[k])])
        M.add("HairRatioMedian", L.f3(e["hair_ratio_median_trapA"]))
    L.table(T / "area_ratio.tex", ["Hair/lesion area tertile", "Median ratio", "mask $-$ ERM, Trap A"],
            rows or [["\\na"] * 3], "Dermoscopy hair, Trap A, by the ratio of in-lesion hair area to lesion area.",
            "tab:ratio")


def e12():
    e = L.js("spec_e13/dino518_e12_contrast/E12.json")
    rows = []
    if e is not None:
        for arm, lab in (("mask", "ROI masking"), ("balanced", "group-balanced"), ("dfr", "DFR")):
            rows.append([lab, L.ci_row(e[arm])]); M.add(f"EOneTwo{arm}", L.ci_row(e[arm]))
    L.table(T / "e12.tex", ["Arm", "change vs ERM (reversed AUROC)"], rows or [["\\na", "\\na"]],
            "E12 overlap-contrast design: hair inside the lesion ($A{=}1$, $r\\ge0.5$) versus hair outside it "
            "($A{=}0$, $r<0.1$), no artifact-free group (DINOv2, 5 seeds $\\times$ 5 folds).", "tab:e12")


def md_table(path: Path):
    """Rows of the first Markdown table in a document (the audit tables are maintained there by hand)."""
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("|") and not re.match(r"^\|[-| ]+\|$", line):
            rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows[1:] if rows else []


def detectors():
    rows = []
    u = L.js("detectors/UNET_SPEC_REPORT.json")
    if u is not None:
        rows.append(["ISIC 2019 lesion masks (BCN, MSK)", "U-Net trained on ISIC 2018 Task 1", "held-out Dice (mean) / "
                     "Dice vs HAM manual masks", f"{L.f3(u['heldout_dice_mean'])} / {L.f3(u['ham_dice_mean'])}"])
        M.add("UnetDice", L.f3(u["heldout_dice_mean"])); M.add("UnetHam", L.f3(u["ham_dice_mean"]))
    e = L.js("analysis/E10/E10.json")
    if e is not None:
        d = e["detector"]
        rows.append(["Hair (DullRazor-type detector, E10)", "morphological", "IoU / precision / recall vs Mendeley masks",
                     f"{L.f3(d['mean_iou'])} / {L.f3(d['precision'])} / {L.f3(d['recall'])}"])
        M.add("HairDetIoU", L.f3(d["mean_iou"]))
    c = L.js("detectors/contam_probe_eval.json")
    if c is not None:
        rows.append(["Capsule debris masks", "patch-token probe (2,160 expert-masked frames)",
                     "held-out pixel accuracy / IoU", f"{L.f3(c['pixel_acc'])} / {L.f3(c['iou_mean'])}"])
        M.add("ProbeIoU", L.f3(c["iou_mean"])); M.add("ProbeAcc", L.f3(c["pixel_acc"]))
    for r in md_table(L.ROOT / "docs" / "ARTIFACT_MASK_AUDIT.md"):
        if len(r) == 4 and r[0].startswith(("Thyroid", "Ovary")):
            rows.append([L.tex_escape(f"{r[0]}, {r[1]}"), "visual audit, 40 images",
                         L.tex_escape(r[2]), L.tex_escape(r[3]).replace("~", "$\\sim$")])
    L.table(T / "detectors.tex", ["Mask", "Source", "Check", "Result"], rows,
            "Validity of the automatic masks that define artifact presence and location. Visual audits: "
            "\\texttt{docs/ARTIFACT\\_MASK\\_AUDIT.md} (by the analysis agent, not a clinician; Stage~6 reports the "
            "blinded image audit prepared for a clinician).", "tab:det", align="p{3.0cm}p{3.0cm}p{3.8cm}p{4.6cm}",
            size="\\scriptsize")


def primary():
    d = L.csv("PRIMARY_CLAIMS.csv")
    rows = []
    if d is not None:
        for r in d.itertuples():
            ok = bool(r._10) if hasattr(r, "_10") else bool(getattr(r, "supported_holm_0_05", False))
            sup = "\\ok" if (r.p_holm < 0.05 and r.estimate > 0) else ("\\no{} (opposite)" if r.p_holm < 0.05 else "\\no")
            rows.append([L.tex_escape(r.family), L.tex_escape(r.cohort), L.ci(r.estimate, r.ci95_lo, r.ci95_hi),
                         f"{r.p:.4f}", f"{r.p_holm:.4f}", sup])
        n_ok = int(((d.p_holm < 0.05) & (d.estimate > 0)).sum())
        M.add("NPrimary", len(d)); M.add("NPrimaryOK", n_ok)
        M.add("NPOneOK", int(((d.p_holm < 0.05) & (d.estimate > 0) & d.family.str.startswith("P1")).sum()))
        for r in d.itertuples():
            M.add(f"P{L.Macros.clean(r.family.split()[0] + r.cohort.split()[0])}", L.ci(r.estimate, r.ci95_lo, r.ci95_hi))
            M.add(f"PHolm{L.Macros.clean(r.family.split()[0] + r.cohort.split()[0])}", f"{r.p_holm:.4f}")
    L.table(T / "primary.tex", ["Family", "Cohort", "Estimate [95\\% CI]", "$p$", "$p$ (Holm, 16)", "Supported"],
            rows or [["\\na"] * 6], "The pre-specified 16-test primary family (Trap~A, reversed test unless stated), "
            "crossed bootstrap, two-sided $p$, Holm over all 16 tests. P1 is this stage's claim; P2--P4 concern the "
            "remedies of Stage~6.", "tab:primary", size="\\scriptsize", resize=True)


def cxr():
    """Chest-radiograph device traps: per-image predictions were not saved, so no interval can be regenerated with
    the crossed bootstrap (docs/PREREGISTRATION_FINAL.md, A1): point estimates only, labelled not re-estimated."""
    rows = []
    for dis in ("Atelectasis", "Consolidation", "Effusion", "Infiltration"):
        b = L.csv(f"cxr_traps/raddino518/{dis}/bootstrap_vs_erm.csv", root=L.OLD)
        x = L.js(f"cxr_traps/raddino518/{dis}/X3_crossover.json", root=L.OLD)
        if b is None:
            continue
        g = lambda t: b[(b.trap == t) & (b.arm == "mask") & (b.env == "test_rev")].seed_delta_mean
        rows.append([dis, L.s3(float(g("trapA").iloc[0])) if len(g("trapA")) else "\\na",
                     L.s3(float(g("trapB").iloc[0])) if len(g("trapB")) else "\\na",
                     L.s3(None if x is None else float(x["seed_delta_mean"]))])
    L.table(T / "cxr.tex", ["Finding (NIH, RANZCR-CLiP devices)", "mask $-$ ERM, Trap A", "Trap B", "Crossover"],
            rows or [["\\na"] * 4], "Chest radiographs with real devices (NIH ChestX-ray14 linked to RANZCR-CLiP; RAD-DINO): Trap A = central "
            "venous catheter inside the lungs, Trap B = endotracheal tube outside them. Point estimates only, \\textbf{not re-estimated} with the corrected bootstrap (per-image predictions were not saved; "
            "the old intervals are withdrawn).", "tab:cxr")
    M.add("NCxr", len(rows))


def registrations():
    docs = {"Isic": "docs/PREREGISTRATION_ISIC2019_TRAPS.md", "Thy": "docs/PREREGISTRATION_THYROID_TRAPS.md",
            "Ov": "docs/PREREGISTRATION_OVARY_TRAPS.md", "Cap": "docs/PREREGISTRATION_CAPSULE_TRAPS.md",
            "Cxr": "docs/PREREGISTRATION_CXR_DEVICE_TRAPS.md", "Stat": "docs/STATISTICAL_PLAN.md",
            "Rep": "docs/REPLICATION_SPEC.md", "Final": "docs/PREREGISTRATION_FINAL.md"}
    for k, doc in docs.items():
        M.add(f"Reg{k}", L.reg(doc))
    rows = [["Dermoscopy hair traps (E13, DINOv2)", "\\ok{} registered", "ISIC2019\\_TRAPS", L.reg(docs["Isic"]),
             "crossover CI $>0$"],
            ["Dermoscopy hair, DermLIP", "\\ok{} registered", "ISIC2019\\_TRAPS", L.reg(docs["Isic"]), "crossover CI $>0$"],
            ["Thyroid caliper traps", "\\ok{} registered", "THYROID\\_TRAPS", L.reg(docs["Thy"]), "crossover CI $>0$"],
            ["Ovarian caliper traps", "\\ok{} registered", "OVARY\\_TRAPS", L.reg(docs["Ov"]), "crossover CI $>0$"],
            ["Capsule debris traps", "\\ok{} registered", "CAPSULE\\_TRAPS", L.reg(docs["Cap"]), "crossover CI $>0$"],
            ["Chest-radiograph device traps", "\\ok{} registered", "CXR\\_DEVICE\\_TRAPS", L.reg(docs["Cxr"]),
             "crossover CI $>0$"],
            ["E12 overlap-contrast trap", "\\mixed{} pilot design, replicated", "REPLICATION\\_SPEC", L.reg(docs["Rep"]),
             "null (pilot result)"],
            ["16-test primary family (Holm)", "\\no{} grouped after the individual results", "STATISTICAL\\_PLAN",
             L.reg(docs["Stat"]), "Holm $p<0.05$, positive estimate"],
            ["Regeneration with the crossed bootstrap", "\\ok{} registered", "FINAL (A1)", L.reg(docs["Final"]),
             "claims whose CI now includes 0 are rewritten"]]
    rows = [r[:2] + [r[2].replace("\\_", "\\_\\allowbreak{}")] + r[3:] for r in rows]
    L.table(T / "experiments.tex", ["Experiment", "Status", "Registration", "First commit (local time)", "Support criterion"],
            rows, "Experiments of this stage with registration status, first commit of the registration and the "
            "support criterion stated in advance. Commit times come from this machine and are not independent proof "
            "of order.", "tab:exp", align=L.EXP_ALIGN, size="\\scriptsize")


def forest():
    plt = L.plot_style()
    rows = []
    for lab, key, bb, d, kind in RUNS:
        x = crossover(d, kind)
        if x is not None:
            rows.append((f"{lab} ({bb})", float(x["seed_delta_mean"]), float(x["ci95_lo"]), float(x["ci95_hi"])))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.4, 0.36 * len(rows) + 0.8))
    for k, (lab, e, lo, hi) in enumerate(rows[::-1]):
        ax.plot([lo, hi], [k, k], color=L.MASK_BLUE, lw=2); ax.plot(e, k, "o", color=L.MASK_BLUE)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=8)
    ax.axvline(0, color="k", lw=0.7); ax.set_xlabel("crossover: [mask $-$ ERM]$_B$ $-$ [mask $-$ ERM]$_A$ (reversed AUROC)")
    fig.savefig(F / "crossover_forest.pdf")


def examples():
    """Real in-ROI / out-of-ROI examples from figures/examples (licence-cleared files only)."""
    ex = L.ROOT / "figures" / "examples"
    panels = [("Dermoscopy hair", "isic_hair_in_roi_roi.png", "isic_hair_out_roi_roi.png"),
              ("Ovarian calipers", "ovary_calipers_in_roi_overlay.png", "ovary_calipers_out_roi_overlay.png"),
              ("Capsule debris", "capsule_debris_in_roi_overlay.png", "capsule_debris_out_roi_overlay.png")]
    if not all((ex / p).exists() for _, a, b in panels for p in (a, b)):
        L.MISSING.append("figures/examples (run scripts/make_examples.py)")
        return
    from PIL import Image
    plt = L.plot_style()
    fig, ax = plt.subplots(len(panels), 2, figsize=(4.8, 2.35 * len(panels)))
    for r, (lab, a, b) in enumerate(panels):
        for c, f in enumerate((a, b)):
            ax[r, c].imshow(Image.open(ex / f)); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(("Trap A: artifact in ROI", "Trap B: artifact outside ROI")[c], fontsize=8)
        ax[r, 0].set_ylabel(lab, fontsize=8)
    fig.savefig(F / "real_examples.pdf", dpi=150)
    info = pd.read_csv(ex / "examples.csv")
    for r in info.itertuples():
        if r.cohort in ("isic_hair", "ovary_calipers", "capsule_debris"):
            M.add(f"ExR{L.Macros.clean(r.cohort + r.cell)}", L.f3(float(r.overlap_r)))


def main():
    cohort_table(); trap_table(); source_strata(); area_ratio(); e12(); detectors(); primary(); cxr(); registrations()
    forest(); examples()
    M.write(T / "numbers.tex")
    L.report_missing("stage3")


if __name__ == "__main__":
    main()
