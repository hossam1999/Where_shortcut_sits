"""Stage 6 tables, macros and figures: remedies (decision guide, P2-P4), the blinded image audit (placeholder until the
review is done; filled by audit/analyse_audit.py), external validation, the paper (page counts, audit), CLAIM and the
submission statements.
  PYTHONPATH=src python stages/stage6_remedies_audit_paper/make_stage6.py"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import stagelib as L  # noqa: E402

T, F = HERE / "tables", HERE / "figures"
T.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
M = L.Macros("sSix")
UNIV = [("Dermoscopy hair", "spec_e13/dino518_spec_universal", True), ("Thyroid calipers", "thyroid/dino518_universal", False),
        ("Ovarian calipers", "ovary/dino518_universal", False), ("Capsule debris", "capsule/dino518_universal", False)]
ARMS = [("mask", "ROI masking"), ("mte", "U-MtE"), ("mte_protect", "U-MtE protected"), ("balanced", "group balancing"),
        ("dfr", "DFR"), ("mte_balanced", "U-MtE + balancing")]


def remedies():
    rows = []
    for lab, rel, spec in UNIV:
        b = L.csv(f"{rel}/bootstrap_vs_erm.csv")
        r = [lab]
        for arm, _ in ARMS:
            q = None if b is None else b[(b.trap == "trapA") & (b.arm == arm) & (b.env == "test_rev")]
            if q is not None and spec and "source" in q:
                q = q[q.source == "all"]
            r.append("\\na" if q is None or q.empty else L.ci_row(q.iloc[0]))
            if q is not None and len(q):
                M.add(f"Rem{L.Macros.clean(lab.split()[0] + arm)}", L.ci_row(q.iloc[0]))
        rows.append(r)
    L.table(T / "remedies.tex", ["Cohort (Trap A, in-ROI)"] + [a[1] for a in ARMS], rows,
            "Remedies in the in-ROI trap: change in reversed-test AUROC versus ERM (DINOv2, 5 seeds $\\times$ 5 folds, "
            "crossed CIs). U-MtE needs no artifact annotation; balancing and DFR need image-level artifact labels.",
            "tab:rem", size="\\scriptsize", resize=True)


def primary_remedies():
    d = L.csv("PRIMARY_CLAIMS.csv")
    if d is None:
        return
    q = d[~d.family.str.startswith("P1")]
    rows = [[L.tex_escape(r.family), L.tex_escape(r.cohort), L.ci(r.estimate, r.ci95_lo, r.ci95_hi), f"{r.p_holm:.4f}",
             "\\ok" if (r.p_holm < 0.05 and r.estimate > 0) else ("\\no{} (opposite)" if r.p_holm < 0.05 else "\\no")]
            for r in q.itertuples()]
    L.table(T / "primary_remedies.tex", ["Family", "Cohort", "Estimate [95\\% CI]", "$p$ (Holm, 16)", "Supported"], rows,
            "Remedy tests of the 16-test primary family (Trap~A, reversed test).", "tab:primrem", size="\\scriptsize")
    M.add("NRemOK", int(((q.p_holm < 0.05) & (q.estimate > 0)).sum())); M.add("NRem", len(q))


def natural_remedies():
    c = L.csv("review3/crossed_ci.csv")
    if c is None:
        return
    rows = []
    for coh in c[c.claim.str.startswith("natural")].cohort.unique():
        g = lambda k: c[(c.claim == k) & (c.cohort == coh)]
        r = [L.tex_escape(coh)]
        for k in ("natural mte - mask (all pairs)", "natural mte_protect - mask (all pairs)",
                  "natural balanced - mask (all pairs)", "natural mte_balanced - mask (all pairs)"):
            x = g(k)
            r.append("\\na" if x.empty else L.ci(x.estimate.iloc[0], x.crossed_lo.iloc[0], x.crossed_hi.iloc[0]))
        rows.append(r)
    L.table(T / "natural_remedies.tex", ["Natural test set", "U-MtE $-$ mask", "U-MtE protected $-$ mask",
                                         "balancing $-$ mask", "U-MtE + balancing $-$ mask"], rows,
            "Remedies on the natural test sets (AUROC, all pairs, crossed CIs): no remedy improves on masking uniformly.",
            "tab:natrem", size="\\scriptsize", resize=True)
    q = c[c.claim.isin(["natural mte - mask (all pairs)", "natural mte_protect - mask (all pairs)",
                        "natural balanced - mask (all pairs)", "natural mte_balanced - mask (all pairs)"])]
    pos, neg = int((q.crossed_lo > 0).sum()), int((q.crossed_hi < 0).sum())
    M.add("NatRemPos", pos); M.add("NatRemNeg", neg); M.add("NatRemN", len(q))


def audit():
    s = L.ROOT / "audit" / "review_sheet.csv"
    if s.exists():
        d = pd.read_csv(s)
        M.add("AuditN", len(d)); M.add("AuditCohorts", d.cohort.nunique())
        rated = d.artifact_present.astype(str).str.strip().str.lower().isin(["yes", "no"]).sum()
        M.add("AuditRated", int(rated))
    h = L.ROOT / "audit" / "KEY_SHA256.txt"
    M.add("AuditHash", "\\texttt{" + (h.read_text().split()[0][:16] if h.exists() else "n/a") + "\\ldots}")
    f = T / "audit_results.tex"
    if not f.exists() or "PLACEHOLDER" in f.read_text():
        f.write_text("% PLACEHOLDER: replaced by `python audit/analyse_audit.py <filled sheet> [second sheet]` after the review\n"
                     "\\begin{table}[H]\\centering\\caption{Image audit of the automatic labels (to be filled).}\\label{tab:audit}\n"
                     "\\fbox{\\parbox{0.9\\linewidth}{\\centering\\textbf{Audit not yet performed.} The table of precision, recall, "
                     "location agreement, Cohen's $\\kappa$ and ROI-mask correctness per cohort is written here by "
                     "\\texttt{audit/analyse\\_audit.py} once the review sheet has been filled.}}\n\\end{table}\n")
    lic = []
    for line in (L.ROOT / "audit" / "LICENCES.md").read_text().splitlines():
        if line.startswith("|") and not re.match(r"^\|[-| :]+\|$", line):
            lic.append([L.tex_escape(c.strip()).replace("/", "/\\allowbreak{}").replace("**", "") for c in line.strip().strip("|").split("|")])
    if lic:
        head, body = lic[0], lic[1:]
        L.table(T / "licences.tex", head, body, "Licence decisions for publishing images (\\texttt{audit/LICENCES.md}); "
                "unclear means not published.", "tab:lic", align="p{3.6cm}p{4.0cm}p{2.8cm}p{4.2cm}", size="\\scriptsize")


def external():
    g, s = L.js("external_isic2020/gate.json"), L.js("external_isic2020/traps_dino518/SUMMARY.json")
    seg = L.js("external_isic2020/segmenter.json")
    b = L.csv("external_isic2020/traps_dino518/bootstrap_vs_erm.csv")
    if g:
        for k in ("A0_pos", "A0_neg", "trapA_A1_pos", "trapA_A1_neg", "trapB_A1_pos", "trapB_A1_neg"):
            M.add(f"Ext{k}", f"{g[k]:,}")
        M.add("ExtPatients", f"{g['n_patients']:,}"); M.add("ExtImages", f"{g['n_images']:,}")
    if seg:
        M.add("SegBA", L.f3(seg["hair_free_balanced_accuracy"])); M.add("SegIoU", L.f3(seg["pixel_iou_mean"]))
        M.add("SegTau", seg["tau_pred_hair_px"])
    if s:
        M.add("ExtX", L.ci_row(s["X1_crossover"]))
    if b is not None:
        rows = []
        for t, lab in (("trapA", "Trap A (hair on the lesion)"), ("trapB", "Trap B (hair beside it)")):
            r = [lab] + [L.ci_row(b[(b.trap == t) & (b.env == e)].iloc[0]) for e in ("test_rev", "test_corr", "clean")]
            rows.append(r)
            M.add(f"Ext{t}", r[1])
        L.table(T / "external.tex", ["ISIC 2020, mask $-$ ERM", "reversed test", "correlated test", "clean test"], rows,
                "External validation on new patients (ISIC 2020; patient-disjoint folds; DINOv2; crossed CIs).", "tab:ext",
                size="\\small", resize=True)


def paper():
    for name in ("main", "supplement"):
        f = L.ROOT / "paper" / f"{name}.pdf"
        if f.exists():
            out = subprocess.run(["pdfinfo", str(f)], capture_output=True, text=True).stdout
            m = re.search(r"Pages:\s+(\d+)", out)
            M.add(f"Pages{name.capitalize()}", m.group(1) if m else "\\na")
    s = (L.ROOT / "paper" / "main.tex").read_text()
    a = s[s.index("\\begin{abstract}") + 16:s.index("\\end{abstract}")]
    a = re.sub(r"\\input\{[^}]*\}", "X", a); a = re.sub(r"\$[^$]*\$", "N", a)
    M.add("AbstractWords", len(a.split()))
    t = re.search(r"\\title\{([^}]*)\}", s)
    M.add("Title", t.group(1) if t else "\\na")
    r = subprocess.run([sys.executable, str(L.ROOT / "scripts" / "verify" / "audit_numbers_final.py"), "--docs", "paper"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    last = r[-1] if r else ""
    m = re.match(r"(\d+) intervals and (\d+) three-decimal numbers checked in paper; (\d+) failed", last)
    if m:
        M.add("AuditIntervals", m.group(1)); M.add("AuditNumbers", m.group(2)); M.add("AuditFailed", m.group(3))
    c = (L.ROOT / "paper" / "claim_checklist.tex").read_text()
    items = re.findall(r"(?m)^(\d+) & .*? & (Yes|No|NA|Partial) &", c)
    M.add("ClaimN", len(items))
    for k in ("Yes", "No", "NA", "Partial"):
        M.add(f"Claim{k}", sum(1 for _, v in items if v == k))
    part = [n for n, v in items if v in ("Partial", "No")]
    M.add("ClaimPartialItems", ", ".join(part) if part else "none")


def registrations():
    docs = {"Final": "docs/PREREGISTRATION_FINAL.md", "Umte": "docs/PREREGISTRATION_UNIVERSAL_I2E.md",
            "Abl": "docs/PREREGISTRATION_UMTE_ABLATION.md", "Rev2": "docs/PREREGISTRATION_REVIEW2.md",
            "Stat": "docs/STATISTICAL_PLAN.md"}
    for k, doc in docs.items():
        M.add(f"Reg{k}", L.reg(doc))
    rows = [["U-MtE and variants (P2--P4)", "\\ok{} registered", "UNIVERSAL\\_I2E", L.reg(docs["Umte"]), "Holm $p<0.05$, estimate $>0$"],
            ["Erasure-rank ablation", "\\ok{} registered", "UMTE\\_ABLATION", L.reg(docs["Abl"]), "descriptive"],
            ["16-test primary family", "\\no{} grouped after the individual results", "STATISTICAL\\_PLAN", L.reg(docs["Stat"]), "Holm over 16"],
            ["Natural-data remedies", "\\ok{} registered", "REVIEW2", L.reg(docs["Rev2"]), "descriptive (decision guide)"],
            ["Image audit (prepared, not run)", "\\ok{} registered", "FINAL (Part B)", L.reg(docs["Final"]),
             "precision, recall, location agreement, $\\kappa$"],
            ["External validation X1", "\\ok{} registered", "FINAL (A4)", L.reg(docs["Final"]), "gate, then crossover CI $>0$"]]
    rows = [r[:2] + [r[2].replace("\\_", "\\_\\allowbreak{}")] + r[3:] for r in rows]
    L.table(T / "experiments.tex", ["Experiment", "Status", "Registration", "First commit (local time)", "Support criterion"],
            rows, "Experiments of this stage. Commit times come from this machine and are not independent proof of order.",
            "tab:exp", align=L.EXP_ALIGN, size="\\scriptsize")


def remedies_figure():
    plt = L.plot_style()
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    width = 0.13
    for k, (arm, lab) in enumerate(ARMS):
        vals, lo, hi = [], [], []
        for _, rel, spec in UNIV:
            b = L.csv(f"{rel}/bootstrap_vs_erm.csv")
            q = b[(b.trap == "trapA") & (b.arm == arm) & (b.env == "test_rev")] if b is not None else None
            if q is not None and spec and "source" in q:
                q = q[q.source == "all"]
            r = None if q is None or q.empty else q.iloc[0]
            vals.append(np.nan if r is None else r.seed_delta_mean)
            lo.append(0 if r is None else r.seed_delta_mean - r.ci95_lo); hi.append(0 if r is None else r.ci95_hi - r.seed_delta_mean)
        x = np.arange(len(UNIV)) + (k - len(ARMS) / 2) * width
        ax.bar(x, vals, width, yerr=[lo, hi], label=lab, color=L.ARM_COLOURS.get(arm, None), capsize=1.5, error_kw={"lw": 0.6})
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(len(UNIV)), [u[0] for u in UNIV], fontsize=8)
    ax.set_ylabel("$\\Delta$ reversed AUROC vs ERM (Trap A)"); ax.legend(frameon=False, fontsize=6.5, ncol=3)
    fig.savefig(F / "remedies.pdf")


def external_examples():
    """ISIC 2020 images (CC BY-NC 4.0) from the two external traps, with the U-Net lesion outline only (hair masks are
    derived from unlicensed ISIC 2019 masks and are not drawn). Fixed seed; any image of the cohort (not only test)."""
    import cv2
    from PIL import Image
    c = L.csv("external_isic2020/cohort.csv")
    seg = L.js("external_isic2020/segmenter.json")
    W = Path("/root/data/external/isic2020_work/cache_518")
    if c is None or seg is None or not (W / "rgb.npy").exists():
        L.MISSING.append("ISIC 2020 cache (external examples)"); return
    ids = (W / "ids.txt").read_text().split(); idx = {k: j for j, k in enumerate(ids)}
    rgb = np.load(W / "rgb.npy", mmap_mode="r"); roi = np.load(W / "roi.npy", mmap_mode="r")
    c = c[c.lesion_frac > 0]
    hair = c.hair_px > seg["tau_pred_hair_px"]
    cells = [("hair/ruler on the lesion (Trap A)", c[hair & (c.r >= 0.5)]), ("hair/ruler beside it (Trap B)", c[hair & (c.r < 0.1)]),
             ("hair-free", c[~hair])]
    rng = np.random.default_rng(20260928)
    plt = L.plot_style()
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for a, (lab, d) in zip(ax, cells):
        if lab != "hair-free":  # stated rule: random image among the top quartile of predicted hair/ruler area
            d = d[d.hair_px >= d.hair_px.quantile(0.75)]
        i = sorted(d.image_name)[int(rng.integers(len(d)))]
        im = np.ascontiguousarray(rgb[idx[i]]).copy()
        cnt, _ = cv2.findContours((np.asarray(roi[idx[i]]) > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cnt, -1, (0, 200, 0), 2)
        a.imshow(im); a.set_xticks([]); a.set_yticks([]); a.set_title(lab, fontsize=8)
    fig.savefig(F / "external_examples.pdf", dpi=150)


def main():
    remedies(); primary_remedies(); natural_remedies(); audit(); external(); paper(); registrations(); remedies_figure(); external_examples()
    M.write(T / "numbers.tex")
    L.report_missing("stage6")


if __name__ == "__main__":
    main()
