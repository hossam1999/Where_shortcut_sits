"""Stage 2 tables, macros and figures from the regenerated sweeps (crossed bootstrap).
  python stages/stage2_location_law_mechanism/make_stage2.py"""
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
M = L.Macros("sTwo")
MAIN = L.syn_dir("isic2018", "dino518", "ruler_fixed", "corr", "main")
PROP = L.syn_dir("isic2018", "dino518", "ruler_fixed", "corr", "proposed")
OCC = L.syn_dir("isic2018", "dino518", "ruler_fixed", "occlusion", "occlusion")
DIL = L.syn_dir("isic2018", "dino518", "ruler_fixed", "corr", "dilation")
STR = L.syn_dir("isic2018", "dino518", "ruler_variable", "corr", "stress")
D224 = L.syn_dir("isic2018", "dino224", "ruler_fixed", "corr", "main")
DERM = L.syn_dir("isic2018", "dermlip224", "ruler_fixed", "corr", "main")
OTHER = {"Thyroid caliper, DINOv2": L.syn_dir("thyroid", "dino518", "caliper", "corr", "main"),
         "Thyroid caliper, MedSigLIP": L.syn_dir("thyroid", "medsiglip448", "caliper", "corr", "main"),
         "Ovarian caliper, DINOv2": L.syn_dir("ovary", "dino518", "caliper", "corr", "main"),
         "Capsule debris, DINOv2": L.syn_dir("capsule", "dino518", "debris", "corr", "main"),
         "Capsule debris, MedSigLIP": L.syn_dir("capsule", "medsiglip448", "debris", "corr", "main"),
         "Chest tube, RAD-DINO": L.syn_dir("nih_ptx", "raddino518", "tube", "corr", "main"),
         "Chest tube, DINOv2": L.syn_dir("nih_ptx", "dino518", "tube", "corr", "main")}
OVS = (0.0, 0.25, 0.5, 0.75, 1.0)
ARMS = [("mask", "ROI masking"), ("inpaint", "oracle inpainting"), ("balanced", "group-balanced"), ("dfr", "DFR"),
        ("leace", "paired LEACE"), ("groupdro", "GroupDRO")]


def main_sweep():
    rows = [["ERM (reversed AUROC)"] + [L.f3(L.syn_auc(MAIN, "erm", ov, "test_rev")) for ov in (0.0, 0.5, 1.0)]]
    for a, lab in ARMS:
        rows.append([lab + " $-$ ERM"] + [L.ci_row(L.syn_boot(MAIN, a, ov)) for ov in (0.0, 0.5, 1.0)])
    L.table(T / "sweep_main.tex", ["", "0\\% overlap", "50\\%", "100\\%"], rows,
            "Controlled dermoscopy sweep (ISIC 2018, DINOv2 ViT-B/14 @518, three seeds): change in reversed-test AUROC "
            "versus ERM, crossed CIs.", "tab:sweep", align="lccc", resize=True)
    for ov in OVS:
        M.add(f"Mask{int(ov * 100)}", L.ci_row(L.syn_boot(MAIN, "mask", ov)))
    M.add("Inter", L.ci_row(L.syn_inter(MAIN, "mask")))
    M.add("CorrMaskHundred", L.f3(L.syn_auc(MAIN, "mask", 1.0, "test_corr")))
    M.add("CorrErmHundred", L.f3(L.syn_auc(MAIN, "erm", 1.0, "test_corr")))
    M.add("CorrMaskZero", L.f3(L.syn_auc(MAIN, "mask", 0.0, "test_corr")))
    M.add("RevMaskZero", L.f3(L.syn_auc(MAIN, "mask", 0.0, "test_rev")))
    # harm verdict at 50-100 %
    harm = [L.syn_boot(MAIN, "mask", ov) for ov in (0.5, 0.75, 1.0)]
    ok = [r is not None and float(r["ci95_hi"]) < 0 for r in harm]
    M.add("HarmVerdict", "lowers it significantly at 50, 75 and 100\\% overlap" if all(ok) else
          "lowers it at " + ", ".join(f"{int(ov * 100)}\\%" for ov, k in zip((0.5, 0.75, 1.0), ok) if k) +
          " overlap (other overlaps: CI includes zero)")


def replications():
    rows = []
    for lab, d in (("DINOv2 @224", D224), ("DermLIP @224", DERM), ("DINOv2 @518, randomised ruler", STR)):
        rows.append([lab, L.ci_row(L.syn_boot(d, "mask", 0.0)), L.ci_row(L.syn_boot(d, "mask", 1.0)), L.ci_row(L.syn_inter(d, "mask"))])
    L.table(T / "sweep_replications.tex", ["Backbone / artifact", "mask $-$ ERM at 0\\%", "at 100\\%", "Location interaction"], rows,
            "Replications of the controlled sweep (reversed-test AUROC, crossed CIs).", "tab:rep", resize=True)


def occlusion():
    rows = [[f"{int(ov * 100)}\\%", L.ci_row(L.syn_boot(OCC, "mask", ov, "test_uncorr")), L.ci_row(L.syn_boot(OCC, "inpaint", ov, "test_uncorr"))]
            for ov in OVS]
    L.table(T / "occlusion.tex", ["Overlap", "mask $-$ ERM", "inpaint $-$ ERM"], rows,
            "Occlusion control: ruler present but uncorrelated with the label (50/50); AUROC change on the 50/50 test.",
            "tab:occ")
    rs = [L.syn_boot(OCC, "mask", ov, "test_uncorr") for ov in OVS]
    pos = [r is not None and float(r["ci95_lo"]) > 0 for r in rs]
    neg = [r is not None and float(r["ci95_hi"]) < 0 for r in rs]
    M.add("OccVerdict", "helps at every overlap (every CI above zero)" if all(pos) else
          "changes it significantly at none of the five overlaps (every CI includes zero)" if not any(pos) and not any(neg)
          else f"is significantly positive at {sum(pos)} and significantly negative at {sum(neg)} of the five overlaps")


def dilation():
    ret = L.csv("final_mechanism/dilation_retention.csv")
    rows = []
    for ov in (0.5, 0.75, 1.0):
        for m in (0, 10, 25, 50):
            b = L.syn_boot(DIL, f"dilate{m}", ov)
            rr = L.row(ret, overlap=ov, margin_px=float(m)) if ret is not None else None
            rows.append([f"{int(ov * 100)}\\%" if m == 0 else "", f"{m}", L.f3(None if rr is None else float(rr.retention_mean)),
                         L.f3(None if rr is None else float(rr.ruler_share_of_visible_mean)), L.ci_row(b)])
    L.table(T / "dilation.tex", ["Overlap", "Margin (px)", "Ruler retained", "Ruler share of visible pixels", "mask $-$ ERM (rev)"],
            rows, "Mask dilation: restoring context around the lesion restores ruler pixels (retention) and changes harm.",
            "tab:dil")


def counterfactual():
    c = L.csv("final_mechanism/counterfactual_ci.csv")
    rows = []
    if c is not None:
        for r in c.itertuples():
            rows.append([r.sweep if r.overlap == c[c.sweep == r.sweep].overlap.min() else "", f"{int(r.overlap * 100)}\\%",
                         L.f3(r.absdp_erm), L.f3(r.absdp_mask), L.ci(r.seed_delta_mean, r.ci95_lo, r.ci95_hi)])
        d = c[c.sweep.str.startswith("dermoscopy")]
        full, zero = d[np.isclose(d.overlap, 1.0)], d[np.isclose(d.overlap, 0.0)]
        if len(full):
            M.add("DpMaskFull", L.f3(full.absdp_mask.iloc[0])); M.add("DpErmFull", L.f3(full.absdp_erm.iloc[0]))
            M.add("DpDiffFull", L.ci(full.seed_delta_mean.iloc[0], full.ci95_lo.iloc[0], full.ci95_hi.iloc[0]))
        if len(zero):
            M.add("DpMaskZero", L.f3(zero.absdp_mask.iloc[0])); M.add("DpErmZero", L.f3(zero.absdp_erm.iloc[0]))
    L.table(T / "counterfactual.tex", ["Sweep", "Overlap", "$|\\Delta p|$ ERM", "$|\\Delta p|$ mask", "mask $-$ ERM"],
            rows or [["\\na"] * 5], "Same-head counterfactual sensitivity to inserting the artifact (reversed test, "
            "artifact-bearing images), crossed CIs of the paired difference.", "tab:cf", size="\\scriptsize", resize=True)


def tertiles():
    t = L.csv("final_mechanism/lesion_tertiles.csv")
    rows = []
    if t is not None:
        for tt in ("large", "medium", "small"):
            q = t[t.tertile == tt]
            rows.append([tt, L.f3(q.artifact_to_lesion_ratio_median.iloc[0]), str(int(q.n_melanoma.iloc[0]))] +
                        [L.ci_row(q[np.isclose(q.overlap, ov)].iloc[0]) for ov in (0.0, 0.5, 1.0)])
        s1, l1 = t[(t.tertile == "small") & np.isclose(t.overlap, 1.0)].iloc[0], t[(t.tertile == "large") & np.isclose(t.overlap, 1.0)].iloc[0]
        M.add("SmallFull", L.ci_row(s1)); M.add("LargeFull", L.ci_row(l1))
        M.add("LargeVerdict", "does not differ significantly from zero" if (l1.ci95_lo <= 0 <= l1.ci95_hi) else "differs from zero")
    L.table(T / "tertiles.tex", ["Lesion tertile", "Ruler/lesion area (median)", "Melanomas in test", "mask $-$ ERM at 0\\%", "50\\%", "100\\%"],
            rows or [["\\na"] * 6], "Lesion-size strata of the controlled sweep (reversed-test AUROC, crossed CIs).", "tab:tert",
            size="\\scriptsize", resize=True)


def other_sweeps():
    rows = []
    for lab, d in OTHER.items():
        rows.append([lab, L.ci_row(L.syn_boot(d, "mask", 0.0)), L.ci_row(L.syn_boot(d, "mask", 1.0)), L.ci_row(L.syn_inter(d, "mask"))])
    L.table(T / "other_sweeps.tex", ["Sweep", "mask $-$ ERM at 0\\%", "at 100\\%", "Location interaction"], rows,
            "Controlled sweeps in the other modalities (synthetic caliper, debris, tube pasted at controlled overlap; reversed-test "
            "AUROC, crossed CIs).", "tab:other", size="\\scriptsize", resize=True)


def dose_figure():
    plt = L.plot_style()
    panels = [("Dermoscopy ruler", MAIN)] + [(k, v) for k, v in OTHER.items() if "DINOv2" in k or "RAD-DINO" in k]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels), 2.3), sharey=True)
    for ax, (lab, d) in zip(np.atleast_1d(axes), panels):
        b = L.csv(f"{d}/bootstrap_vs_erm.csv")
        if b is None:
            ax.set_title(lab + "\n(n/a)", fontsize=7); continue
        for arm, col in (("mask", L.MASK_BLUE), ("balanced", "#2ca02c")):
            q = b[(b.method_a == arm) & (b.env == "test_rev")].sort_values("overlap")
            if q.empty:
                continue
            ax.plot(q.overlap, q.seed_delta_mean, color=col, marker="o", ms=3, label=arm)
            ax.fill_between(q.overlap, q.ci95_lo, q.ci95_hi, color=col, alpha=0.2)
        ax.axhline(0, color="k", lw=0.6); ax.set_title(lab, fontsize=7); ax.set_xlabel("overlap $r$")
    np.atleast_1d(axes)[0].set_ylabel("$\\Delta$ reversed AUROC vs ERM"); np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
    fig.savefig(F / "dose_response.pdf")


def cf_figure():
    c = L.csv("final_mechanism/counterfactual_ci.csv")
    if c is None:
        return
    plt = L.plot_style()
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    for s, q in c.groupby("sweep"):
        q = q.sort_values("overlap")
        ax.plot(q.overlap, q.absdp_mask, marker="o", ms=3, label=s)
    d = c[c.sweep.str.startswith("dermoscopy")].sort_values("overlap")
    ax.plot(d.overlap, d.absdp_erm, color=L.ERM_GREY, ls="--", label="dermoscopy, ERM")
    ax.set_xlabel("overlap $r$"); ax.set_ylabel("$|\\Delta p|$ of the masked model"); ax.legend(frameon=False, fontsize=6)
    fig.savefig(F / "counterfactual.pdf")


def text_numbers():
    """Numbers quoted in the prose (all from the same result files as the tables)."""
    for ov, m in ((0.5, 0), (0.5, 25), (1.0, 0), (1.0, 50)):
        M.add(f"Dil{int(ov * 100)}m{m}", L.ci_row(L.syn_boot(DIL, f"dilate{m}", ov)))
    ret = L.csv("final_mechanism/dilation_retention.csv")
    for ov, m in ((0.5, 0), (0.5, 25), (1.0, 0), (1.0, 50)):
        rr = L.row(ret, overlap=ov, margin_px=float(m)) if ret is not None else None
        M.add(f"Ret{int(ov * 100)}m{m}", L.f3(None if rr is None else float(rr.retention_mean)))
        M.add(f"Share{int(ov * 100)}m{m}", L.f3(None if rr is None else float(rr.ruler_share_of_visible_mean)))
    c = L.csv("final_mechanism/counterfactual_ci.csv")
    if c is not None:
        d = c[c.sweep.str.startswith("dermoscopy") & np.isclose(c.overlap, 1.0)]
        if len(d):
            M.add("DpRatioFull", f"{d.absdp_mask.iloc[0] / d.absdp_erm.iloc[0]:.1f}")
        full = c[np.isclose(c.overlap, 1.0)]
        for r in full.itertuples():
            k = L.Macros.clean(r.sweep.split(" (")[0].split()[0] + r.sweep.split("(")[-1])
            M.add(f"Cf{k}", L.ci(r.seed_delta_mean, r.ci95_lo, r.ci95_hi))
            M.add(f"CfMask{k}", L.f3(r.absdp_mask)); M.add(f"CfErm{k}", L.f3(r.absdp_erm))
        M.add("CfNPos", int((full.ci95_lo > 0).sum())); M.add("CfN", len(full))
    t = L.csv("final_mechanism/lesion_tertiles.csv")
    if t is not None:
        for tt in ("large", "medium", "small"):
            for ov in (0.0, 0.5, 1.0):
                M.add(f"Tert{tt.capitalize()}{int(ov * 100)}", L.ci_row(L.row(t, tertile=tt, overlap=ov)))
            M.add(f"Ratio{tt.capitalize()}", L.f3(float(t[t.tertile == tt].artifact_to_lesion_ratio_median.iloc[0])))
    for key, d in (("Cxr", OTHER["Chest tube, RAD-DINO"]), ("CxrDino", OTHER["Chest tube, DINOv2"])):
        for env, lab in (("test_rev", "Rev"), ("clean", "Clean")):
            for ov in (0.0, 1.0):
                M.add(f"{key}Erm{lab}{int(ov * 100)}", L.f3(L.syn_auc(d, "erm", ov, env)))
                M.add(f"{key}Mask{lab}{int(ov * 100)}", L.f3(L.syn_auc(d, "mask", ov, env)))
    for lab, d in OTHER.items():
        k = L.Macros.clean(lab.replace(",", "").replace(" ", ""))
        M.add(f"{k}Zero", L.ci_row(L.syn_boot(d, "mask", 0.0)))
        M.add(f"{k}Full", L.ci_row(L.syn_boot(d, "mask", 1.0)))
        M.add(f"{k}Inter", L.ci_row(L.syn_inter(d, "mask")))
    for lab, d in (("DinoTwoTwoFour", D224), ("Dermlip", DERM), ("Stress", STR)):
        M.add(f"{lab}Inter", L.ci_row(L.syn_inter(d, "mask")))
    allsw = {"dermoscopy": MAIN, "d224": D224, "dermlip": DERM, "stress": STR, **OTHER}
    inter = [L.syn_inter(d, "mask") for d in allsw.values()]
    full = [L.syn_boot(d, "mask", 1.0) for d in allsw.values()]
    M.add("NSweeps", len(allsw))
    M.add("NInterPos", sum(r is not None and float(r["ci95_lo"]) > 0 for r in inter))
    M.add("NInterNeg", sum(r is not None and float(r["ci95_hi"]) < 0 for r in inter))
    M.add("NHarmFull", sum(r is not None and float(r["ci95_hi"]) < 0 for r in full))
    M.add("NHelpFull", sum(r is not None and float(r["ci95_lo"]) > 0 for r in full))
    M.add("NNullFull", sum(r is not None and float(r["ci95_lo"]) <= 0 <= float(r["ci95_hi"]) for r in full))


def registrations():
    docs = {"Rep": "docs/REPLICATION_SPEC.md", "Final": "docs/PREREGISTRATION_FINAL.md",
            "Thy": "docs/PREREGISTRATION_THYROID_TRAPS.md", "Ov": "docs/PREREGISTRATION_OVARY_TRAPS.md",
            "Cap": "docs/PREREGISTRATION_CAPSULE_TRAPS.md", "Cxr": "docs/PREREGISTRATION_CXR_DEVICE_TRAPS.md",
            "Rev": "docs/PREREGISTRATION_REVIEW2.md"}
    for k, doc in docs.items():
        M.add(f"Reg{k}", L.reg(doc))
    rows = [
        ["Controlled dermoscopy sweep (E1)", "\\ok{} registered", "REPLICATION\\_SPEC", L.reg(docs["Rep"]),
         "interaction CI $>0$; mask$-$ERM CI $>0$ at 0\\% and $<0$ at 100\\%"],
        ["Replications: DINOv2@224, DermLIP, randomised ruler", "\\ok{} registered", "REPLICATION\\_SPEC",
         L.reg(docs["Rep"]), "same signs and verdicts as E1"],
        ["Occlusion control (E4)", "\\ok{} registered", "REPLICATION\\_SPEC", L.reg(docs["Rep"]),
         "no harm on the 50/50 test at any overlap"],
        ["Mask dilation (E5)", "\\ok{} registered", "REPLICATION\\_SPEC", L.reg(docs["Rep"]),
         "harm grows with retention at 50\\%; flat at 100\\%"],
        ["Same-head counterfactual $|\\Delta p|$ (E6, T3)", "\\ok{} registered", "REVIEW2 (T3)", L.reg(docs["Rev"]),
         "mask $>$ ERM at in-ROI overlap, CI $>0$"],
        ["Lesion-size tertiles (E7)", "\\mixed{} pilot analysis, replicated", "REPLICATION\\_SPEC", L.reg(docs["Rep"]),
         "harm largest in small lesions"],
        ["Thyroid caliper sweep", "\\ok{} registered", "THYROID\\_TRAPS", L.reg(docs["Thy"]), "interaction CI $>0$"],
        ["Ovarian caliper sweep", "\\ok{} registered", "OVARY\\_TRAPS", L.reg(docs["Ov"]), "interaction CI $>0$"],
        ["Capsule debris sweep", "\\ok{} registered", "CAPSULE\\_TRAPS", L.reg(docs["Cap"]), "interaction CI $>0$"],
        ["Chest tube sweep (RAD-DINO)", "\\ok{} registered", "CXR\\_DEVICE\\_TRAPS", L.reg(docs["Cxr"]),
         "interaction CI $>0$"],
        ["Chest tube sweep (DINOv2)", "\\no{} post hoc", "---", "---", "descriptive only"],
        ["Crossed-bootstrap intervals for every sweep (A1)", "\\ok{} registered", "FINAL (A1)", L.reg(docs["Final"]),
         "a claim is stated only if its crossed CI excludes 0"],
    ]
    rows = [r[:2] + [r[2].replace("\\_", "\\_\\allowbreak{}")] + r[3:] for r in rows]
    L.table(T / "experiments.tex", ["Experiment", "Status", "Registration", "First commit (local time)", "Support criterion"],
            rows, "Experiments of this stage: registration status, the registration document "
            "(\\texttt{docs/PREREGISTRATION\\_<name>.md} or the replication specification), the commit that first contains it, and "
            "what was stated in advance to count as support. Commit times are set by the committing machine and are not "
            "independent proof of order.", "tab:exp", align=L.EXP_ALIGN, size="\\scriptsize")


def examples():
    """Real images with the synthetic artifact at 0/50/100 % overlap (licence-cleared cohorts only: ISIC 2018 CC BY-NC,
    MMOTU CC BY 4.0, SEE-AI CC BY 4.0). Thyroid (TN3K, licence unclear) and NIH are not shown."""
    import json
    import cv2
    from PIL import Image
    sys.path.insert(0, str(L.ROOT / "src"))
    from wtss.synthetic import draw_caliper, draw_debris, draw_ruler
    from wtss.data.isic2018 import isic2018_loaders
    from wtss.data.thyroid import load_ovary_synthetic_cohort
    from wtss.data.capsule import load_capsule_synthetic_cohort
    pl_i = json.loads((L.ROOT / "frozen" / "placements_518_65x19.json").read_text())
    rows = [("Dermoscopy (ISIC 2018)", sorted(pl_i)[7], pl_i, isic2018_loaders(518),
             lambda im, x, y, i, ov: draw_ruler(im, x, y, 65, 19))]
    co, pl, ld, geo = load_ovary_synthetic_cohort(518)
    w, h = geo[518]
    rows.append(("Ovarian tumour (MMOTU)", co.df[co.df.split == "test"].image_id.iloc[3], pl, ld,
                 lambda im, x, y, i, ov, w=w, h=h: draw_caliper(im, x, y, w, h, image_id=f"{i}|{ov:.2f}")))
    co, pl, ld, geo = load_capsule_synthetic_cohort(518)
    w2, h2 = geo[518]
    rows.append(("Capsule lesion (SEE-AI)", co.df[co.df.split == "test"].image_id.iloc[3], pl, ld,
                 lambda im, x, y, i, ov, w=w2, h=h2: draw_debris(im, x, y, w, h, image_id=f"{i}|{ov:.2f}")))
    plt = L.plot_style()
    fig, ax = plt.subplots(len(rows), 3, figsize=(6.6, 2.25 * len(rows)))
    for r, (lab, iid, pls, (lr, lm), draw) in enumerate(rows):
        img, roi = np.asarray(lr(iid)), np.asarray(lm(iid))
        for k, ov in enumerate(("0.00", "0.50", "1.00")):
            p = pls[iid][ov]
            im, _ = draw(Image.fromarray(img), int(p["x"]), int(p["y"]), iid, float(ov))
            a = np.asarray(im).copy()
            cnt, _ = cv2.findContours((roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(a, cnt, -1, (0, 160, 0), 2)
            ax[r, k].imshow(a); ax[r, k].set_xticks([]); ax[r, k].set_yticks([])
            if r == 0:
                ax[r, k].set_title(f"overlap {float(ov):.0%}")
        ax[r, 0].set_ylabel(lab, fontsize=8)
    fig.savefig(F / "sweep_examples.pdf", dpi=150)


OTHER_ARMS = [("mask", "masking"), ("inpaint", "oracle inpainting"), ("balanced", "group-balanced"), ("dfr", "DFR"),
              ("leace", "paired LEACE")]


def other_arms():
    """Every baseline arm in every other-modality sweep, at no and at full overlap, plus balancing's interaction."""
    rows = []
    for lab, d in OTHER.items():
        first = True
        e0, e1 = L.syn_auc(d, "erm", 0.0, "test_rev"), L.syn_auc(d, "erm", 1.0, "test_rev")
        c0 = L.syn_auc(d, "erm", 0.0, "clean")
        rows.append([lab, "ERM (AUROC: clean; reversed)", f"{L.f3(c0)}; {L.f3(e0)}", f"{L.f3(L.syn_auc(d, 'erm', 1.0, 'clean'))}; {L.f3(e1)}", "---"])
        for a, alab in OTHER_ARMS:
            r0, r1 = L.syn_boot(d, a, 0.0), L.syn_boot(d, a, 1.0)
            if r0 is None and r1 is None:
                continue
            rows.append(["", alab + " $-$ ERM", L.ci_row(r0), L.ci_row(r1), L.ci_row(L.syn_inter(d, a))])
    L.table(T / "other_arms.tex", ["Sweep", "Arm", "$r=0$", "$r=1$", "Location interaction"], rows,
            "All baseline arms in the controlled sweeps of the other modalities: ERM's clean and reversed AUROC and each "
            "arm's change in reversed-test AUROC versus ERM at no and at full overlap (crossed CIs). The last column is the "
            "arm's own location interaction ($[\\cdot]_{r=0}-[\\cdot]_{r=1}$); an arm that does not depend on where the "
            "artifact sits has an interaction near zero.", "tab:otherarms", align="p{2.6cm}p{3.2cm}ccc",
            size="\\scriptsize", long=True)
    for lab, d in OTHER.items():
        k = L.Macros.clean(lab.replace(",", "").replace(" ", ""))
        M.add(f"{k}BalInter", L.ci_row(L.syn_inter(d, "balanced")))
    M.add("DermBalInter", L.ci_row(L.syn_inter(MAIN, "balanced")))
    M.add("DermInpInter", L.ci_row(L.syn_inter(MAIN, "inpaint")))


def dose_figure_all():
    plt = L.plot_style()
    panels = [("Dermoscopy ruler, DINOv2", MAIN), ("Dermoscopy ruler, DermLIP", DERM)] + list(OTHER.items())
    fig, axes = plt.subplots(2, 5, figsize=(10.5, 4.6), sharey=True)
    axes = axes.ravel()
    for ax, (lab, d) in zip(axes, panels):
        b = L.csv(f"{d}/bootstrap_vs_erm.csv")
        if b is None:
            ax.set_title(lab + " (n/a)", fontsize=7); continue
        for arm, col in (("mask", L.MASK_BLUE), ("balanced", "#2ca02c"), ("inpaint", "#9467bd")):
            q = b[(b.method_a == arm) & (b.env == "test_rev")].sort_values("overlap")
            if q.empty:
                continue
            ax.plot(q.overlap, q.seed_delta_mean, color=col, marker="o", ms=2.5, label=arm)
            ax.fill_between(q.overlap, q.ci95_lo, q.ci95_hi, color=col, alpha=0.18)
        ax.axhline(0, color="k", lw=0.6); ax.set_title(lab, fontsize=7); ax.set_xlabel("overlap $r$", fontsize=7)
    for ax in axes[len(panels):]:
        ax.axis("off")
    axes[0].set_ylabel("$\\Delta$ reversed AUROC vs ERM"); axes[5].set_ylabel("$\\Delta$ reversed AUROC vs ERM")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(F / "dose_response_all.pdf")


def estimator():
    r = L.estimator_check(["synthetic/thyroid", "synthetic/ovary", "synthetic/capsule", "synthetic/nih_ptx"],
                          T / "estimator_check.tex", "tab:estcheck",
                          "The sweeps of the other modalities under the per-seed and the crossed estimator (same predictions; "
                          "baseline arms only).", changes_path=T / "estimator_changes.tex", changes_label="tab:estchanges",
                          changes_caption="Contrasts of the other-modality sweeps whose verdict depends on the estimator.")
    if r:
        M.add("EstN", r["n"]); M.add("EstLost", r["lost"]); M.add("EstGained", r["gained"]); M.add("EstMed", f"{r['median']:.2f}")


def main():
    main_sweep(); replications(); occlusion(); dilation(); counterfactual(); tertiles(); other_sweeps(); dose_figure(); cf_figure()
    text_numbers(); registrations(); other_arms(); dose_figure_all(); estimator()
    try:
        examples()
    except Exception as e:  # data not on this machine -> figure missing, reported
        print("examples failed:", repr(e))
        L.MISSING.append("stage2 figures/sweep_examples.pdf")
    M.write(T / "numbers.tex")
    L.report_missing("stage2")


if __name__ == "__main__":
    main()
