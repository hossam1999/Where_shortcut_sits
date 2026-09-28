"""Stage 1 tables, macros and figures, regenerated from result files (no manual steps).
  python stages/stage1_problem_pilot_protocol/make_stage1.py"""
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
M = L.Macros("sOne")
SYN = L.syn_dir("isic2018", "dino518", "ruler_fixed", "corr", "main")

PILOT = [("E1", "controlled ruler sweep, DINOv2@518: mask $-$ ERM at 0--100\\% overlap; location interaction"),
         ("E2", "the same at 224\\,px (DINOv2) and with the dermatology model DermLIP"),
         ("E3", "appearance randomisation (colour, opacity, ticks, scale of the ruler)"),
         ("E4", "occlusion control: ruler present but uncorrelated with the label (50/50)"),
         ("E5", "mask dilation: restoring ruler pixels around the lesion (retention)"),
         ("E6", "same-head counterfactual sensitivity $|\\Delta p|$ to inserting the ruler"),
         ("E7", "lesion-size strata (artifact-to-lesion area ratio)"),
         ("E8", "oracle inpainting and a prediction-consistency repair"),
         ("E9", "bridge: group-balanced training, DFR, GroupDRO, paired LEACE at 100\\% overlap"),
         ("E10", "hair decodability from the embedding; paired LEACE on real hair"),
         ("E11", "atlas of real artifacts (hair, ink, vignetting) in ISIC 2019 / HAM10000; vignetting trap"),
         ("E12", "overlap-contrast trap: in-lesion versus out-of-lesion hair, no artifact-free group"),
         ("E13", "real hair traps (Trap A in-lesion, Trap B out-of-lesion, shared hair-free group), ISIC 2019"),
         ("E14", "area-ratio strata of the real hair trap"),
         ("E15", "E13 with DermLIP")]


def pilot_table():
    L.table(T / "pilot_overview.tex", ["Exp.", "What it tests"], PILOT, "The pilot experiments (specification: "
            "\\texttt{docs/REPLICATION\\_SPEC.md}).", "tab:pilot", align="lp{13cm}")


def counts(md_path: Path):
    if not md_path.exists():
        return None
    m = re.search(r"Verdict counts: (\{.*\})", md_path.read_text())
    return None if not m else eval(m.group(1))  # noqa: S307 (own generated file)


def ledger():
    old = counts(L.OLD / "SUMMARY.md")
    if old:
        M.add("OrigMatch", old.get("MATCH")); M.add("OrigSignCI", old.get("SIGN+CI")); M.add("OrigMismatch", old.get("MISMATCH"))
        M.add("OrigN", sum(old.values()))
    new_md = L.NEW / "SUMMARY.md"
    new = counts(new_md)
    for k, v in (("NewMatch", "MATCH"), ("NewSignCI", "SIGN+CI"), ("NewMismatch", "MISMATCH")):
        M.add(k, None if not new else new.get(v, 0))
    M.add("NewN", None if not new else sum(new.values()))
    rows = []
    if new_md.exists():
        for line in new_md.read_text().splitlines():
            if not line.startswith("| E") or "quantity" in line:
                continue
            c = [x.strip() for x in re.split(r"\s\|\s", " " + line.strip().strip("|") + " ")]  # |Δp| contains pipes
            e, q, exp, got, v = c[0], c[1], c[2], c[3], c[4]
            pilot_point = exp.split("[")[0].strip()  # pilot interval (old estimator) is not reproduced
            rows.append([e, L.tex_escape(q).replace("−", "$-$").replace("≥", "$\\ge$").replace("|Δp|", "$|\\Delta p|$"),
                         pilot_point.replace("−", "-"), got.replace("−", "-"), v])
    if not rows:
        L.MISSING.append("results/rerun_2026-09-28/SUMMARY.md")
        rows = [["\\na"] * 5]
    L.table(T / "ledger.tex", ["Exp.", "Quantity", "Pilot (point)", "Replication (crossed CI)", "Verdict"], rows,
            "Replication ledger: every number the pilot reported, its re-derived value with the crossed 95\\% interval, and "
            "the verdict. The pilot's own intervals came from the per-seed estimator (Sec.~\\ref{sec:bootstrap}) and are not "
            "reproduced; only its point values are shown.", "tab:ledger", align="llp{1.7cm}p{3.3cm}l", size="\\scriptsize",
            long=True)


def sweep_numbers():
    for ov, name in ((0.0, "EOneZero"), (1.0, "EOneHundred")):
        M.add(name, L.ci_row(L.syn_boot(SYN, "mask", ov)))
    M.add("EOneInter", L.ci_row(L.syn_inter(SYN, "mask")))
    sc = L.ROOT / "frozen" / "multires_common_support_ids.csv"
    M.add("NPilot", f"{len(pd.read_csv(sc)):,}" if sc.exists() else None)


def leakage():
    d = L.csv("leakage/LEAKAGE_SUMMARY.csv")
    if d is None:
        (T / "leakage_text.tex").write_text("\\na\n"); (T / "leakage.tex").write_text("")
        return
    g = d[d.split == "grouped"].set_index("overlap")
    r = d[d.split != "grouped"].groupby("overlap")[["erm_gap", "mask_minus_erm_rev", "cross_split_near_dup_pairs"]].mean()
    rows = []
    for ov in sorted(d.overlap.unique()):
        rows.append([f"{ov:.0%}".replace("%", "\\%"), "grouped", "0", L.f3(g.loc[ov, "erm_gap"]), L.s3(g.loc[ov, "mask_minus_erm_rev"])])
        rows.append(["", "random (mean of 5)", f"{r.loc[ov, 'cross_split_near_dup_pairs']:.0f}", L.f3(r.loc[ov, "erm_gap"]),
                     L.s3(r.loc[ov, "mask_minus_erm_rev"])])
    L.table(T / "leakage.tex", ["Overlap", "Split", "Near-duplicate pairs across split", "ERM gap (corr $-$ rev)", "mask $-$ ERM (rev)"],
            rows, "Leakage: grouped versus random image-level splits (controlled sweep, DINOv2@518).", "tab:leak", resize=True)
    infl = [(r.loc[ov, "erm_gap"] / g.loc[ov, "erm_gap"] - 1) * 100 for ov in (0.0, 1.0) if ov in g.index]
    (T / "leakage_text.tex").write_text(
        f"Random splits leave near-duplicate images on both sides of the split and inflate the measured ERM shortcut gap by "
        f"{infl[0]:.0f}\\% (no overlap) and {infl[1]:.0f}\\% (full overlap), and shift both masking effects "
        f"(Table~\\ref{{tab:leak}}); every analysis therefore uses grouped splits.%\n")


PILOT_GROUPS = ["synthetic/isic2018", "spec_e13/dino518_spec/", "spec_e13/dermlip224_spec/", "spec_e13/dino518_e12",
                "analysis/E14"]


def correction():
    """Per-seed (pilot) vs crossed bootstrap on the pilot's own analyses only."""
    r = L.estimator_check(PILOT_GROUPS, T / "estimator_check.tex", "tab:estcheck",
                          "The pilot's analyses under both interval estimators (same predictions, same point estimates; "
                          "10,000 replicates each). Width ratio $>1$: the crossed interval is wider.",
                          changes_path=T / "estimator_changes.tex", changes_label="tab:estchanges",
                          changes_caption="The pilot contrasts whose verdict (interval excludes 0 or not) depends on the "
                          "estimator. Every other contrast keeps its verdict.")
    if r is None:
        for k in ("CorrRows", "CorrMedian", "CorrQlo", "CorrQhi", "CorrLost", "CorrGained"):
            M.add(k, None)
        return
    d = r["df"]
    M.add("CorrRows", f"{r['n']:,}"); M.add("CorrMedian", f"{r['median']:.2f}")
    M.add("CorrQlo", f"{r['q25']:.2f}"); M.add("CorrQhi", f"{r['q75']:.2f}")
    M.add("CorrLost", r["lost"]); M.add("CorrGained", r["gained"])
    M.add("CorrMinRatio", f"{d.width_ratio.min():.2f}"); M.add("CorrMaxRatio", f"{d.width_ratio.max():.2f}")
    plt = L.plot_style()
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.hist(d.width_ratio.dropna().clip(0, 4), bins=30, color=L.MASK_BLUE)
    ax.axvline(1, color="k", lw=0.8, ls="--")
    ax.set_xlabel("crossed width / per-seed width"); ax.set_ylabel("pilot intervals")
    fig.savefig(F / "width_ratio.pdf")


def registry():
    M.add("RegRep", L.reg("docs/REPLICATION_SPEC.md"))
    M.add("RegCrossed", L.reg("docs/PREREGISTRATION_REVIEW3.md"))
    M.add("RegFinal", L.reg("docs/PREREGISTRATION_FINAL.md"))


def environments():
    plt = L.plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 2.3))
    envs = [("training", .9, .1), ("correlated test", .9, .1), ("reversed test", .1, .9), ("clean test", .5, .5)]
    x = np.arange(len(envs))
    ax.bar(x - 0.18, [e[1] for e in envs], 0.34, color=L.ART_RED, label="$P(A{=}1\\mid Y{=}1)$")
    ax.bar(x + 0.18, [e[2] for e in envs], 0.34, color=L.ERM_GREY, label="$P(A{=}1\\mid Y{=}0)$")
    ax.set_xticks(x, [e[0] for e in envs]); ax.set_ylim(0, 1); ax.set_ylabel("artifact rate")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.savefig(F / "environments.pdf")


def ruler_examples():
    sys.path.insert(0, str(L.ROOT / "src"))
    from PIL import Image
    import cv2
    from wtss.synthetic import draw_ruler
    from wtss.data.isic2018 import isic2018_loaders
    pl = json.loads((L.ROOT / "frozen" / "placements_518_65x19.json").read_text())
    lr, lm = isic2018_loaders(518)
    iid = sorted(pl)[7]
    img, roi = lr(iid), lm(iid)
    plt = L.plot_style()
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for k, ov in enumerate(("0.00", "0.50", "1.00")):
        p = pl[iid][ov]
        im, _ = draw_ruler(Image.fromarray(img), p["x"], p["y"], 65, 19)
        a = np.asarray(im).copy()
        cnt, _ = cv2.findContours((roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(a, cnt, -1, (0, 160, 0), 2)
        ax[k].imshow(a); ax[k].set_xticks([]); ax[k].set_yticks([]); ax[k].set_title(f"overlap {float(ov):.0%}")
    fig.savefig(F / "ruler_examples.pdf")


def pilot_sweep():
    """Reversed-test AUROC by overlap for the main arms of the regenerated controlled sweep (the pilot's question)."""
    d = L.csv("synthetic/isic2018/dino518_ruler_fixed_corr_main/summary_auc.csv")
    if d is None:
        return
    plt = L.plot_style()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for arm, lab in (("erm", "ERM"), ("mask", "ROI masking"), ("inpaint", "oracle inpainting"), ("balanced", "group-balanced")):
        for a, env in zip(ax, ("test_rev", "test_corr")):
            q = d[(d.method == arm) & (d.env == env)].sort_values("overlap")
            a.plot(q.overlap, q.auc, marker="o", ms=3, color=L.ARM_COLOURS.get(arm, "k"), label=lab)
    for a, t in zip(ax, ("reversed test (0.1/0.9)", "correlated test (0.9/0.1)")):
        a.set_xlabel("ruler overlap with the lesion $r$"); a.set_title(t, fontsize=8)
    ax[0].set_ylabel("AUROC (mean of 3 seeds)"); ax[0].legend(frameon=False, fontsize=7)
    fig.savefig(F / "pilot_sweep.pdf")


PROTOCOL = [
    ("Environments", "training and correlated test $P(A{=}1\\mid Y{=}1)=0.9$, $P(A{=}1\\mid Y{=}0)=0.1$; reversed test 0.1/0.9; clean test: no artifact (synthetic) or artifact independent of the label (real)", "a strong, known shortcut whose use is exposed when the association is reversed"),
    ("Primary metric", "reversed-test AUROC; clean and correlated AUROC reported beside it", "threshold-free; a model that uses the artifact must fall on the reversed test"),
    ("Encoders", "frozen DINOv2 ViT-B/14 at 518\\,px (CLS token), DINOv2 at 224\\,px, DermLIP (dermatology CLIP model) at 224\\,px", "how foundation models are used on small medical data; every arm sees the same features"),
    ("Heads", "logistic regression (liblinear, class-balanced); $C\\in\\{0.01,0.1,1,10\\}$ and the decision threshold (maximal balanced accuracy) chosen on clean validation data only", "no test or artifact information enters model selection"),
    ("Arms", "ERM; ROI masking; oracle inpainting of the artifact; group-balanced training; DFR; paired and unpaired LEACE; GroupDRO; prevalence calibration", "masking is compared with methods that target the artifact itself"),
    ("Splits", "grouped by lesion identifier $\\cup$ perceptual hash (Hamming $\\le 8$); no group crosses a split", "near-duplicates across a split inflate shortcut measurements (Sec.~\\ref{sec:leak})"),
    ("Seeds", "3 seeds (42, 123, 456) for the synthetic sweeps; 5 seeds $\\times$ 5 grouped folds for the real hair traps", "training and split variation are both part of the uncertainty"),
    ("Uncertainty", "crossed seed $\\times$ image bootstrap, 10,000 replicates, percentile 95\\% intervals", "seeds share test images (Sec.~\\ref{sec:bootstrap})"),
    ("Registration", "design, claims and support criteria committed to the repository before each verdict-producing run", "separates confirmation from exploration"),
]


def protocol():
    L.table(T / "protocol.tex", ["Component", "Choice", "Why"], [[a, b, c] for a, b, c in PROTOCOL],
            "The protocol fixed in this stage.", "tab:protocol", align="p{2.2cm}p{7.8cm}p{5.0cm}",
            size="\\scriptsize")


def deviations():
    """Deviations of the re-implementation from the written pilot specification (CHANGES.md), minus added arms."""
    rows = []
    for line in (L.ROOT / "CHANGES.md").read_text().splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) != 4 or not c[0].isdigit():
            continue
        if "added arms" in c[2]:
            continue
        esc = lambda t: L.tex_escape(re.sub(r"\*\*|`", "", t)).replace("≤", "$\\le$").replace("×", "$\\times$") \
            .replace("~", "$\\sim$").replace("–", "--").replace("−", "$-$").replace("∪", "$\\cup$").replace("→", "$\\to$")
        rows.append([c[0], esc(c[1]), esc(c[2]), esc(c[3])])
    L.table(T / "deviations.tex", ["\\#", "Specification", "Re-implementation", "Reason and effect"], rows,
            "Where the re-implementation deviates from the pilot's written specification, and why.", "tab:dev",
            align="lp{3.0cm}p{4.6cm}p{7.0cm}", size="\\scriptsize")


E13_ARMS = [("erm", "ERM"), ("mask", "ROI masking"), ("inpaint", "oracle inpainting of hair"), ("balanced", "group-balanced"),
            ("dfr", "DFR"), ("leace_paired", "paired LEACE"), ("leace_unpaired", "unpaired LEACE (image-level $A$)")]


def e13():
    """E13 (DINOv2@518) and E15 (DermLIP@224): per-arm AUROC and change vs ERM, both traps."""
    for tag, d, lab in (("dino", "spec_e13/dino518_spec", "DINOv2@518"), ("dermlip", "spec_e13/dermlip224_spec", "DermLIP@224")):
        comp = L.csv(f"{d}/COMPARISON.csv")
        rows = []
        for arm, name in E13_ARMS:
            r = []
            for trap in ("trapA", "trapB"):
                c = L.row(comp, trap=trap, arm=arm)
                b = L.trap_boot(d, arm, trap) if arm != "erm" else None
                if c is None:
                    r += ["\\na", "\\na"]
                    continue
                aucs = f"{L.f3(c.clean_got)} / {L.f3(c.corr_got)} / {L.f3(c.rev_got)}"
                r += [aucs, "---" if arm == "erm" else L.ci_row(b)]
            rows.append([name] + r)
        L.table(T / f"e13_{tag}.tex", ["Arm", "Trap A: clean / corr / rev", "Trap A: $\\Delta$ rev vs ERM",
                                       "Trap B: clean / corr / rev", "Trap B: $\\Delta$ rev vs ERM"], rows,
                f"Real hair traps ({lab}; E13{'' if tag == 'dino' else '/E15'}): AUROC in the three test environments "
                "(mean over 5 seeds) and change in reversed-test AUROC versus ERM (crossed 95\\% CI). Trap A: hair on the "
                "lesion; Trap B: hair beside it; both share the hair-free group.", f"tab:e13{tag}",
                align="p{3.2cm}cccc", size="\\scriptsize", resize=True)
        cnt = L.csv(f"{d}/counts_vs_expected.csv")
        if tag == "dino" and cnt is not None:
            for trap in ("trapA", "trapB"):
                c = L.row(cnt, trap=trap)
                for k in ("A0_Y0", "A0_Y1", "A1_Y0", "A1_Y1"):
                    M.add(f"Cnt{trap}{k}", f"{int(c[k]):,}"); M.add(f"Exp{trap}{k}", f"{int(c['exp_' + k]):,}")
                M.add(f"RevMel{trap}", int(c.rev_mel_seed42))
        s = L.js(f"{d}/SUMMARY.json")
        if s:
            M.add(f"X{tag}", L.ci_row(s["crossover_B_minus_A_mask"]))
            for st in s.get("source_stratified_trapA_mask", []):
                M.add(f"Src{tag}{st['source']}", L.ci_row(st))
        M.add(f"A{tag}", L.ci_row(L.trap_boot(d, "mask", "trapA")))
        M.add(f"B{tag}", L.ci_row(L.trap_boot(d, "mask", "trapB")))
        M.add(f"AClean{tag}", L.ci_row(L.trap_boot(d, "mask", "trapA", "clean")))


def e10():
    j = L.js("analysis/E10/E10.json")
    if not j:
        return
    h1, det, h2 = j["H1"], j["detector"], j["H2"]
    M.add("HairDec", f"{h1['auroc']:.3f} [{h1['ci'][0]:.3f}, {h1['ci'][1]:.3f}]")
    M.add("HairDecN", h1["n_test"]); M.add("HairPMel", f"{h1['P_hair_mel']:.3f}"); M.add("HairPBen", f"{h1['P_hair_ben']:.3f}")
    M.add("DetIoU", f"{det['mean_iou']:.3f}"); M.add("DetIoUMed", f"{det['median_iou']:.3f}")
    M.add("DetP", f"{det['precision']:.3f}"); M.add("DetR", f"{det['recall']:.3f}")
    M.add("LeaceN", h2["n_join"]); M.add("LeaceTrain", h2["n_train_pairs"])
    M.add("LeaceDecRaw", f"{h2['A_decoder_raw']:.3f}"); M.add("LeaceDecAfter", f"{h2['A_decoder_leace']:.3f}")
    M.add("LeaceCleanRaw", f"{h2['clean_auroc_raw']:.3f}"); M.add("LeaceCleanAfter", f"{h2['clean_auroc_leace']:.3f}")
    M.add("LeaceDpRaw", f"{h2['abs_dp_raw']:.3f}"); M.add("LeaceDpAfter", f"{h2['abs_dp_leace']:.3f}")


def e12_e14():
    j = L.js("spec_e13/dino518_e12_contrast/E12.json")
    if j:
        for k in ("mask", "balanced", "dfr"):
            v = j.get(k) or {}
            if isinstance(v, dict) and "ci95_lo" in v:
                M.add(f"EOneTwo{k}", L.ci_row(v, "seed_delta_mean" if "seed_delta_mean" in v else "delta"))
    j = L.js("analysis/E14_area_ratio.json")
    if j:
        M.add("EFourteenMed", f"{j['hair_ratio_median_trapA']:.3f}")
        for t in ("T1", "T2", "T3"):
            v = j.get(t)
            if isinstance(v, dict) and "ci95_lo" in v:
                M.add(f"EFourteen{t}", L.ci_row(v, "seed_delta_mean" if "seed_delta_mean" in v else "delta"))


def sweep_extra():
    """Numbers for the text: arms of the controlled sweep at 100% overlap and correlated AUROC of the masked model."""
    for arm in ("inpaint", "balanced", "dfr", "leace", "groupdro", "inpaint_consistency", "inpaint_consistency_lam0"):
        for ov in (0.5, 1.0):
            M.add(f"Sw{arm}{ov}", L.ci_row(L.syn_boot(SYN, arm, ov)))
    for arm in ("erm", "mask"):
        for ov in (0.0, 1.0):
            for env in ("test_corr", "test_rev", "clean"):
                M.add(f"Auc{arm}{env}{ov}", L.f3(L.syn_auc(SYN, arm, ov, env)))
    stress = L.syn_dir("isic2018", "dino518", "ruler_variable", "corr", "stress")
    M.add("StressInter", L.ci_row(L.syn_inter(stress, "mask")))
    for d, k in ((L.syn_dir("isic2018", "dino224", "ruler_fixed", "corr", "main"), "DinoTTF"),
                 (L.syn_dir("isic2018", "dermlip224", "ruler_fixed", "corr", "main"), "Dermlip")):
        M.add(f"{k}Inter", L.ci_row(L.syn_inter(d, "mask")))
        M.add(f"{k}Zero", L.ci_row(L.syn_boot(d, "mask", 0.0))); M.add(f"{k}Full", L.ci_row(L.syn_boot(d, "mask", 1.0)))


def hair_examples():
    """Two ISIC 2019 images of the real hair traps (lesion outline only; the hair masks carry no redistribution licence)."""
    from PIL import Image
    ex = L.ROOT / "figures" / "examples"
    fs = [ex / "isic_hair_in_roi_roi.png", ex / "isic_hair_out_roi_roi.png"]
    if not all(f.exists() for f in fs):
        L.MISSING.append("figures/examples/isic_hair_*"); return
    plt = L.plot_style()
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 3.1))
    for a, f, t in zip(ax, fs, ("Trap A: hair on the lesion", "Trap B: hair beside the lesion")):
        a.imshow(Image.open(f)); a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=8)
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.savefig(F / "hair_examples.pdf")


def main():
    pilot_table(); ledger(); sweep_numbers(); leakage(); correction(); registry(); environments(); pilot_sweep(); protocol()
    deviations(); e13(); e10(); e12_e14(); sweep_extra(); hair_examples()
    try:
        ruler_examples()
    except Exception as e:  # data not available on this machine -> figure missing, reported
        print("ruler_examples failed:", e)
    M.write(T / "numbers.tex")
    L.report_missing("stage1")


if __name__ == "__main__":
    main()
