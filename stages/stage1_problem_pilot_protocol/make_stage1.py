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
            "Pilot replication ledger, re-checked with the regenerated runs and the corrected estimator. The pilot's own "
            "intervals (old estimator) are not reproduced.", "tab:ledger", align="llp{1.7cm}p{3.3cm}l", size="\\scriptsize",
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


def correction():
    f = L.ROOT / "results" / "bootstrap_correction" / "sweeps_umte_old_vs_new.csv"
    if not f.exists():
        for k in ("CorrRows", "CorrMedian", "CorrQlo", "CorrQhi", "CorrLost", "CorrMaxDiff"):
            M.add(k, None)
        return
    d = pd.read_csv(f)
    M.add("CorrRows", f"{len(d):,}"); M.add("CorrMedian", f"{d.width_ratio.median():.2f}")
    M.add("CorrQlo", f"{d.width_ratio.quantile(.25):.2f}"); M.add("CorrQhi", f"{d.width_ratio.quantile(.75):.2f}")
    M.add("CorrLost", int((d.old_excludes_0 & ~d.new_excludes_0).sum()))
    M.add("CorrMaxDiff", f"{d.est_diff.abs().max():.3f}")
    M.add("CorrEstDiff", int((d.est_diff.abs() > 0.03).sum()))
    M.add("CorrGained", int((~d.old_excludes_0 & d.new_excludes_0).sum()))
    plt = L.plot_style()
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.hist(d.width_ratio.dropna().clip(0, 4), bins=40, color=L.MASK_BLUE)
    ax.axvline(1, color="k", lw=0.8, ls="--")
    ax.set_xlabel("corrected width / original width"); ax.set_ylabel("intervals")
    fig.savefig(F / "width_ratio.pdf")


def registry():
    f = L.NEW / "registry" / "registry.csv"
    d = pd.read_csv(f) if f.exists() else pd.read_csv(L.OLD / "review2" / "registry.csv")
    M.add("NRegistered", len(d))
    M.add("RegRep", L.reg("docs/REPLICATION_SPEC.md"))
    M.add("RegStat", L.reg("docs/STATISTICAL_PLAN.md"))
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
    ("Environments", "training and correlated test $P(A{=}1\\mid Y{=}1)=0.9$, $P(A{=}1\\mid Y{=}0)=0.1$; reversed test swapped; clean test artifact-free or independent", "a strong, known shortcut whose use is measurable by its reversal"),
    ("Primary metric", "reversed-test AUROC; clean and correlated AUROC as secondary", "threshold-free; reversal isolates shortcut use"),
    ("Encoders", "frozen DINOv2 ViT-B/14 at 518 px (all cohorts); DermLIP, MedSigLIP, RAD-DINO, ConvNeXt, DINOv2 S/L as replications", "typical deployment on small medical data; identical features across arms"),
    ("Heads", "logistic regression; C and threshold chosen on clean validation data only", "no test information in model selection"),
    ("Arms", "ERM; ROI masking; oracle inpainting; group balancing; DFR; LEACE; GroupDRO; later remedies", "masking is compared with remedies that target the artifact"),
    ("Splits", "grouped by lesion identifier, perceptual hash, patient where available", "no near-duplicate leakage (Sec.~\\ref{sec:leak})"),
    ("Seeds", "3 seeds (sweeps); 5 seeds $\\times$ 5 grouped folds (real traps)", "variation of training and split"),
    ("Uncertainty", "crossed seed $\\times$ image bootstrap, 10,000 replicates", "seeds share test images (Sec.~\\ref{sec:bootstrap})"),
    ("Registration", "design, claims and support criteria committed before each experiment", "separates confirmation from exploration"),
]


def protocol():
    L.table(T / "protocol.tex", ["Component", "Choice", "Why"], [[a, b, c] for a, b, c in PROTOCOL],
            "The protocol used unchanged in every later stage.", "tab:protocol", align="p{2.2cm}p{7.8cm}p{5.0cm}",
            size="\\scriptsize")


def main():
    pilot_table(); ledger(); sweep_numbers(); leakage(); correction(); registry(); environments(); pilot_sweep(); protocol()
    try:
        ruler_examples()
    except Exception as e:  # data not available on this machine -> figure missing, reported
        print("ruler_examples failed:", e)
    M.write(T / "numbers.tex")
    L.report_missing("stage1")


if __name__ == "__main__":
    main()
