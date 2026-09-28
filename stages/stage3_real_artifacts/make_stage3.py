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
        ("Capsule debris", "capsule", "MedSigLIP", "capsule/medsiglip448_main", "real"),
        ("Thyroid calipers", "thyroid", "ConvNeXt", "thyroid/convnext384_universal", "real"),
        ("Ovarian calipers", "ovary", "MedSigLIP", "ovary/medsiglip448_universal", "real"),
        ("Capsule debris", "capsule", "ConvNeXt", "capsule/convnext384_universal", "real")]


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
            "ovary": ("MMOTU 2D ultrasound", "solid-component vs cystic tumour", "calipers (rule detector)", "tumour (manual)"),
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
            "\\texttt{docs/ARTIFACT\\_MASK\\_AUDIT.md} and \\texttt{docs/THYROID\\_MARKER\\_AUDIT.md} (visual audits during the "
            "analysis, not by a clinician).", "tab:det", align="p{3.0cm}p{3.0cm}p{3.8cm}p{4.6cm}",
            size="\\scriptsize")


def primary():
    """The four pre-specified location tests (P1), Holm over the four."""
    d = L.csv("PRIMARY_CLAIMS.csv")
    rows = []
    if d is not None:
        d = d[d.family.str.startswith("P1")].copy()
        order = np.argsort(d.p.to_numpy()); m = len(d); adj = np.empty(m); run = 0.0
        for k, i in enumerate(order):
            run = max(run, min(1.0, (m - k) * d.p.to_numpy()[i])); adj[i] = run
        d["p_holm4"] = adj
        L.derived("stage3_p1_holm4", d[["family", "cohort", "estimate", "p", "p_holm4"]])
        for r in d.itertuples():
            sup = "\\ok" if (r.p_holm4 < 0.05 and r.estimate > 0) else "\\no"
            rows.append([L.tex_escape(r.cohort), L.ci(r.estimate, r.ci95_lo, r.ci95_hi), f"{r.p:.4f}", f"{r.p_holm4:.4f}", sup])
        M.add("NPOne", len(d)); M.add("NPOneOK", int(((d.p_holm4 < 0.05) & (d.estimate > 0)).sum()))
        M.add("PMax", f"{d.p.max():.4f}")
    L.table(T / "primary.tex", ["Cohort (DINOv2)", "Crossover [95\\% CI]", "$p$", "$p$ (Holm, 4)", "Supported"],
            rows or [["\\na"] * 5], "The four pre-specified location tests (P1: crossover $>0$), crossed bootstrap, "
            "two-sided bootstrap $p$ (floored at $10^{-4}$), Holm-adjusted over the four cohorts.", "tab:primary")


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
            ["Location tests P1 (four cohorts, Holm)", "\\mixed{} each test registered; Holm grouping post hoc", "STATISTICAL\\_PLAN",
             L.reg(docs["Stat"]), "Holm $p<0.05$, positive estimate"],
            ["Chest drains (NIH)", "\\ok{} registered", "CXR\\_DRAIN", L.reg("docs/PREREGISTRATION_CXR_DRAIN.md"),
             "mask $-$ ERM reported with CI, no direction"],
            ["Crossed-bootstrap intervals (A1)", "\\ok{} registered", "FINAL (A1)", L.reg(docs["Final"]),
             "a claim is stated only if its crossed CI excludes 0"]]
    rows = [r[:2] + [r[2].replace("\\_", "\\_\\allowbreak{}")] + r[3:] for r in rows]
    L.table(T / "experiments.tex", ["Experiment", "Status", "Registration", "First commit (local time)", "Support criterion"],
            rows, "Experiments of this stage with registration status, the registration document "
            "(\\texttt{docs/PREREGISTRATION\\_<name>.md}), its first commit and the support criterion stated in advance. "
            "Commit times are set by the committing machine and are not independent proof of order.", "tab:exp", align=L.EXP_ALIGN, size="\\scriptsize")


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


BASE_ARMS = [("mask", "masking"), ("inpaint", "oracle inpainting"), ("balanced", "group-balanced"), ("dfr", "DFR"),
             ("leace_paired", "paired LEACE"), ("leace_unpaired", "unpaired LEACE"), ("prevcal", "prevalence calibration")]
DINO_RUNS = [("Dermoscopy hair", "spec_e13/dino518_spec", "spec"), ("Thyroid calipers", "thyroid/dino518_main", "real"),
             ("Ovarian calipers", "ovary/dino518_main", "real"), ("Capsule debris", "capsule/dino518_main", "real")]


def all_arms():
    """Every baseline arm in both traps (DINOv2) and the AUROCs of ERM and masking in the three environments."""
    rows, aucrows = [], []
    for lab, d, kind in DINO_RUNS:
        first = True
        for a, alab in BASE_ARMS:
            ra, rb = boot(d, kind, "trapA", a), boot(d, kind, "trapB", a)
            if ra is None and rb is None:
                continue
            rows.append([lab if first else "", alab, L.ci_row(ra), L.ci_row(rb)]); first = False
            k = L.Macros.clean(lab.split()[0] + a)
            M.add(f"ArmA{k}", L.ci_row(ra)); M.add(f"ArmB{k}", L.ci_row(rb))
            M.add(f"ArmCleanA{k}", L.ci_row(boot(d, kind, "trapA", a, "clean")))
        am = L.auc_means(d)
        if am is not None:
            for trap in ("trapA", "trapB"):
                for meth in ("erm", "mask", "balanced", "dfr"):
                    q = am[(am.trap == trap) & (am.method == meth)].set_index("env").auc
                    if q.empty:
                        continue
                    aucrows.append({"cohort": lab, "trap": trap, "method": meth, "clean": q.get("clean"),
                                    "test_corr": q.get("test_corr"), "test_rev": q.get("test_rev")})
    L.table(T / "all_arms.tex", ["Cohort", "Arm $-$ ERM", "Trap A (in ROI)", "Trap B (outside)"], rows,
            "Every baseline arm in the real-artifact traps (DINOv2 @518, 5 seeds $\\times$ 5 folds): change in "
            "reversed-test AUROC versus ERM, crossed 95\\% CIs. Inpainting needs the artifact mask; balancing, DFR, "
            "unpaired LEACE and prevalence calibration need image-level artifact labels (prevalence calibration also at "
            "test time).", "tab:allarms", align="p{2.6cm}p{3.4cm}p{4.0cm}p{4.0cm}", size="\\scriptsize", long=True)
    if aucrows:
        df = L.derived("stage3_auc_means", pd.DataFrame(aucrows))
        t = [[r.cohort if r.method == "erm" and r.trap == "trapA" else "", r.trap.replace("trap", "Trap "), r.method,
              L.f3(r.clean), L.f3(r.test_corr), L.f3(r.test_rev)] for r in df.itertuples()]
        L.table(T / "auc_envs.tex", ["Cohort", "Trap", "Arm", "clean", "correlated", "reversed"], t,
                "AUROC of ERM, masking, balancing and DFR in the three test environments (DINOv2, mean over 5 seeds). A "
                "large correlated$-$reversed gap is reliance on the artifact.", "tab:aucenv", size="\\scriptsize", long=True)
        for r in df.itertuples():
            k = L.Macros.clean(r.cohort.split()[0] + r.trap + r.method)
            M.add(f"Auc{k}Corr", L.f3(r.test_corr)); M.add(f"Auc{k}Rev", L.f3(r.test_rev)); M.add(f"Auc{k}Clean", L.f3(r.clean))


def hair_correlates():
    """What carries the hair shortcut: metadata of the hair groups, the pixel share of the ERM gap, and two archived
    sensitivity designs (author-independent reconstruction, metadata-matched traps; point estimates only)."""
    age = pd.read_csv(L.OLD / "analysis" / "trap_metadata_age.csv")
    sex = pd.read_csv(L.OLD / "analysis" / "trap_metadata_sex.csv")
    site = pd.read_csv(L.OLD / "analysis" / "trap_metadata_site.csv")
    rows = []
    for src in ("HAM", "BCN", "MSK"):
        for g, lab in (("free", "hair-free"), ("trapA", "Trap A (on lesion)"), ("trapB", "Trap B (beside)")):
            a = age[(age.source == src) & (age.group_A == g)].age_approx
            f = sex[(sex.source == src) & (sex.group_A == g)].female
            st = site[(site.source == src) & (site.group_A == g)]
            rows.append([src if g == "free" else "", lab, f"{float(a.iloc[0]):.1f}", f"{float(f.iloc[0]):.2f}",
                         f"{float(st['head/neck'].iloc[0]):.2f}", f"{float(st['lower extremity'].iloc[0]):.2f}"])
    L.table(T / "hair_meta.tex", ["Source", "Group", "Mean age", "Share female", "Share head/neck", "Share lower extremity"],
            rows, "Patient and site metadata of the hair groups (ISIC 2019): hair presence and location go with age, sex "
            "and body site.", "tab:hairmeta", size="\\scriptsize")
    ps = pd.read_csv(L.OLD / "analysis" / "trap_pixel_share_dinov2_b14_518.csv")
    g = ps.groupby("trap").pixel_share_of_gap.mean()
    L.derived("stage3_pixel_share", g.round(3).reset_index())
    M.add("PixShareA", f"{g['trapA'] * 100:.0f}"); M.add("PixShareB", f"{g['trapB'] * 100:.0f}")
    out = []
    for tag, d in (("recon", "real/isic2019/dino518_main"), ("metamatched", "real/isic2019/dino518_matched")):
        b = pd.read_csv(L.OLD / d / "bootstrap_vs_erm.csv"); x = pd.read_csv(L.OLD / d / "crossover_B_minus_A.csv")
        for trap in ("trapA", "trapB"):
            q = b[(b.trap == trap) & (b.arm == "mask") & (b.env == "test_rev") & (b.source == "all")]
            out.append({"design": tag, "quantity": f"mask_minus_erm_{trap}", "estimate": round(float(q.seed_delta_mean.iloc[0]), 3)})
        out.append({"design": tag, "quantity": "crossover", "estimate": round(float(x[x.arm == "mask"].seed_delta_mean.iloc[0]), 3)})
    df = L.derived("stage3_hair_designs", pd.DataFrame(out))
    for r in df.itertuples():
        M.add(f"Hd{L.Macros.clean(r.design + r.quantity)}", L.s3(r.estimate))


def cxr_extra():
    """Chest drains (NIH, NEATX labels) with both backbones and the device-matched follow-up (point estimates)."""
    out = []
    for bb, d in (("RAD-DINO", "cxr_drain/raddino518_universal"), ("DINOv2", "cxr_drain/dino518_universal")):
        m = pd.read_csv(L.OLD / d / "metrics_per_seed.csv")
        g = m.groupby(["method", "env"]).auc.mean().unstack()
        for meth in ("erm", "mask", "balanced", "dfr", "mask_dfr"):
            out.append({"backbone": bb, "method": meth, "clean": round(g.loc[meth, "clean"], 3),
                        "test_corr": round(g.loc[meth, "test_corr"], 3), "test_rev": round(g.loc[meth, "test_rev"], 3)})
    df = L.derived("stage3_drains", pd.DataFrame(out))
    names = {"erm": "ERM", "mask": "lung masking", "balanced": "group-balanced", "dfr": "DFR", "mask_dfr": "masking + DFR"}
    rows = [[r.backbone if r.method == "erm" else "", names[r.method], L.f3(r.clean), L.f3(r.test_corr), L.f3(r.test_rev)]
            for r in df.itertuples()]
    L.table(T / "drains.tex", ["Encoder", "Arm", "clean", "correlated", "reversed"], rows,
            "Real chest drains (NIH ChestX-ray14 pneumothorax, 29,687 images, patient-grouped folds; drains lie inside the "
            "lungs, so this is an in-ROI trap only). AUROC, mean over 5 folds; \\textbf{point estimates, not re-estimated} "
            "with the crossed bootstrap (per-image predictions were not saved).", "tab:drains", size="\\scriptsize")
    for r in df.itertuples():
        k = L.Macros.clean(r.backbone + r.method)
        M.add(f"Dr{k}Rev", L.f3(r.test_rev)); M.add(f"Dr{k}Clean", L.f3(r.clean))
    res = {}
    for trap in ("trapA", "trapB"):
        m = pd.read_csv(L.OLD / "cxr_traps" / "raddino518_devmatched" / "Infiltration" / trap / "metrics_per_seed.csv")
        m = m[m.env == "test_rev"].pivot_table(index="seed", columns="method", values="auc")
        res[trap] = float((m["mask"] - m["erm"]).mean())
    dm = L.derived("stage3_cxr_devmatched", pd.DataFrame([{"trap": k, "mask_minus_erm": round(v, 3)} for k, v in res.items()] +
                                                         [{"trap": "crossover", "mask_minus_erm": round(res["trapB"] - res["trapA"], 3)}]))
    for r in dm.itertuples():
        M.add(f"Dm{L.Macros.clean(r.trap)}", L.s3(r.mask_minus_erm))


def estimator():
    r = L.estimator_check(["thyroid/dino518_main", "thyroid/medsiglip448_main", "ovary/dino518_main", "capsule/dino518_main",
                           "capsule/medsiglip448_main", "thyroid/convnext384_universal", "capsule/convnext384_universal",
                           "ovary/medsiglip448_universal"],
                          T / "estimator_check.tex", "tab:estcheck",
                          "Real-artifact traps of the new cohorts under the per-seed and the crossed estimator (same "
                          "predictions; baseline arms only).", changes_path=T / "estimator_changes.tex",
                          changes_label="tab:estchanges",
                          changes_caption="Contrasts of the real-artifact traps whose verdict depends on the estimator.")
    if r:
        M.add("EstN", r["n"]); M.add("EstLost", r["lost"]); M.add("EstGained", r["gained"]); M.add("EstMed", f"{r['median']:.2f}")


def main():
    cohort_table(); trap_table(); source_strata(); area_ratio(); e12(); detectors(); primary(); cxr(); registrations()
    forest(); examples(); all_arms(); hair_correlates(); cxr_extra(); estimator()
    M.write(T / "numbers.tex")
    L.report_missing("stage3")


if __name__ == "__main__":
    main()
