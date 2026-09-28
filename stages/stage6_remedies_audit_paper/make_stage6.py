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


# ------------------------------------------------------------------------------------------------ added content
UNIV_ALL = [("Dermoscopy hair", "DINOv2", "spec_e13/dino518_spec_universal"), ("Thyroid calipers", "DINOv2", "thyroid/dino518_universal"),
            ("Thyroid calipers", "MedSigLIP", "thyroid/medsiglip448_universal"), ("Thyroid calipers", "ConvNeXt", "thyroid/convnext384_universal"),
            ("Ovarian calipers", "DINOv2", "ovary/dino518_universal"), ("Ovarian calipers", "MedSigLIP", "ovary/medsiglip448_universal"),
            ("Capsule debris", "DINOv2", "capsule/dino518_universal"), ("Capsule debris", "MedSigLIP", "capsule/medsiglip448_universal"),
            ("Capsule debris", "ConvNeXt", "capsule/convnext384_universal")]
ROB_ARMS = [("erm", "ERM"), ("mask", "mask"), ("mte", "U-MtE"), ("mte_protect", "U-MtE prot."), ("mte_aug", "overlay aug."),
            ("jtt", "JTT"), ("balanced", "balanced"), ("dfr", "DFR"), ("mask_dfr", "mask+DFR"), ("mte_balanced", "U-MtE+bal.")]


def tbool(v, *keys):
    for k in keys:
        if k in v:
            return v[k]
    return None


def remedies_all():
    """Every remedy arm, every cohort and encoder: reversed AUROC change vs ERM in Trap A and Trap B."""
    rows = []
    for coh, bb, rel in UNIV_ALL:
        b = L.csv(f"{rel}/bootstrap_vs_erm.csv")
        if b is None:
            continue
        first = True
        for arm, lab in (("mask", "masking"), ("mte", "U-MtE"), ("mte_protect", "U-MtE protected"), ("mte_aug", "same overlays as augmentation"),
                         ("balanced", "group balancing"), ("dfr", "DFR"), ("mask_dfr", "masking + DFR"), ("mte_balanced", "U-MtE + balancing")):
            g = lambda t: b[(b.trap == t) & (b.arm == arm) & (b.env == "test_rev") & ((b.source == "all") if "source" in b else True)]
            a_, b_ = g("trapA"), g("trapB")
            if a_.empty and b_.empty:
                continue
            rows.append([f"{coh}, {bb}" if first else "", lab, L.ci_row(a_.iloc[0]) if len(a_) else "\\na",
                         L.ci_row(b_.iloc[0]) if len(b_) else "\\na"]); first = False
            k = L.Macros.clean(coh.split()[0] + bb + arm)
            if len(a_):
                M.add(f"RA{k}", L.ci_row(a_.iloc[0]))
            if len(b_):
                M.add(f"RB{k}", L.ci_row(b_.iloc[0]))
    L.table(T / "remedies_all.tex", ["Cohort, encoder", "Arm $-$ ERM", "Trap A (in ROI)", "Trap B (outside)"], rows,
            "Every remedy in every real-artifact trap and encoder: change in reversed-test AUROC versus ERM (5 seeds $\\times$ "
            "5 grouped folds, crossed CIs). U-MtE and the augmentation arm use only the ROI mask; balancing, DFR and their "
            "combinations need image-level artifact labels.", "tab:remall", align="p{3.4cm}p{4.2cm}p{3.5cm}p{3.5cm}",
            size="\\scriptsize", long=True)


def robustness():
    """min(reversed, correlated) AUROC per arm (robustness summary) and shortcut-flip flags, Trap A."""
    out = []
    for coh, bb, rel in UNIV_ALL + [("Chest drains", "RAD-DINO", "cxr_drain/raddino518_universal")]:
        root = L.OLD if rel.startswith("cxr") else L.NEW
        m = L.csv(f"{rel}/metrics_per_seed.csv", root)
        if m is None:
            continue
        if "trap" in m:
            m = m[m.trap == "trapA"]
        g = m.groupby(["method", "env"]).auc.mean().unstack()
        for arm, _ in ROB_ARMS:
            if arm in g.index:
                r = g.loc[arm]
                out.append({"cohort": coh, "backbone": bb, "arm": arm, "clean": round(r.get("clean"), 3),
                            "corr": round(r.get("test_corr"), 3), "rev": round(r.get("test_rev"), 3),
                            "min_rev_corr": round(min(r.get("test_rev"), r.get("test_corr")), 3),
                            "flip": bool(r.get("test_corr") < r.get("test_rev") - 0.02)})
    df = L.derived("stage6_robustness", pd.DataFrame(out))
    rows = []
    for (coh, bb), q in df.groupby(["cohort", "backbone"], sort=False):
        best = q.min_rev_corr.max()
        cells = []
        for arm, _ in ROB_ARMS:
            r = q[q.arm == arm]
            if r.empty:
                cells.append("--"); continue
            v = L.f3(float(r.min_rev_corr.iloc[0])) + ("$^\\dagger$" if bool(r.flip.iloc[0]) else "")
            cells.append("\\textbf{" + v + "}" if float(r.min_rev_corr.iloc[0]) == best else v)
        rows.append([f"{coh}, {bb}"] + cells)
        for r in q.itertuples():
            M.add(f"Rob{L.Macros.clean(coh.split()[0] + bb + r.arm)}", L.f3(r.min_rev_corr))
    L.table(T / "robustness.tex", ["Trap A"] + [a[1] for a in ROB_ARMS], rows,
            "Robustness summary in the in-ROI trap: $\\min$(reversed, correlated) AUROC per arm (mean over seeds; best per "
            "row in bold; $^\\dagger$ = the arm flips the shortcut, correlated $<$ reversed $-0.02$). Chest drains: point "
            "estimates, not re-estimated.", "tab:rob", size="\\scriptsize", resize=True)


def primary_all():
    d = L.csv("PRIMARY_CLAIMS.csv")
    if d is None:
        return
    rows = [[L.tex_escape(r.family), L.tex_escape(r.cohort), L.ci(r.estimate, r.ci95_lo, r.ci95_hi), f"{r.p:.4f}", f"{r.p_holm:.4f}",
             "\\ok" if (r.p_holm < 0.05 and r.estimate > 0) else ("\\no{} (opposite)" if r.p_holm < 0.05 else "\\no")]
            for r in d.itertuples()]
    L.table(T / "primary_all.tex", ["Family", "Cohort", "Estimate [95\\% CI]", "$p$", "$p$ (Holm, 16)", "Supported"], rows,
            "The 16-test primary family (DINOv2, Trap~A reversed test unless stated): the four location tests and the twelve "
            "remedy tests, Holm-corrected together.", "tab:primall", size="\\scriptsize", resize=True)
    M.add("NPrimOK", int(((d.p_holm < 0.05) & (d.estimate > 0)).sum())); M.add("NPrim", len(d))


def supp_contrasts():
    c = L.csv("analysis/supp_contrasts.csv")
    e = L.csv("analysis/emb_groups_primary.csv")
    rows = []
    if c is not None:
        for r in c.itertuples():
            rows.append([L.tex_escape(r.cohort), L.tex_escape(r.contrast).replace(" - ", " $-$ "), L.ci_row(r._asdict())])
            M.add(f"Sc{L.Macros.clean(r.cohort + r.contrast)}", L.ci_row(r._asdict()))
    L.table(T / "supp_contrasts.tex", ["Cohort", "Contrast (Trap A)", "Estimate [95\\% CI]"], rows,
            "Secondary remedy contrasts (reversed test, crossed CIs).", "tab:supp", size="\\scriptsize")
    rows = []
    if e is not None:
        for r in e[~e.family.str.startswith("P1")].itertuples():
            rows.append([r.cohort.capitalize(), L.tex_escape(r.family), L.ci_row(r._asdict()),
                         "yes" if bool(r.excludes_zero_positive) else "\\textbf{no}"])
            M.add(f"Emb{L.Macros.clean(r.cohort + r.family.split()[0])}", L.ci_row(r._asdict()))
    L.table(T / "emb_remedies.tex", ["Cohort", "Test", "With embedding-based leakage groups", "CI $>0$"], rows,
            "The remedy tests re-run with the stricter embedding-based leakage groups (cohorts without patient identifiers).",
            "tab:embrem", size="\\scriptsize")


def sweeps_remedies():
    b = L.js("adhoc_bootstraps.json")
    rows = []
    names = {"thyroid_dino518": "Thyroid caliper, DINOv2", "thyroid_medsiglip": "Thyroid caliper, MedSigLIP",
             "ovary_dino518": "Ovarian caliper, DINOv2", "capsule_dino518": "Capsule debris, DINOv2",
             "capsule_medsiglip": "Capsule debris, MedSigLIP", "cxr_raddino518": "Chest tube, RAD-DINO"}
    val = lambda k: None if b is None or k not in b else b[k]
    for key, lab in names.items():
        cells = []
        for c in ("umte-mask", "umte_protect-mask", "umte_balanced-mask", "umte_balanced-balanced"):
            v = val(f"sweep|{key}|r=1.0|{c}")
            if isinstance(v, dict):
                cells.append(L.ci_row(v))
            elif isinstance(v, list):
                cells.append(L.ci(*v))
            else:
                cells.append("\\na")
        rows.append([lab] + cells)
        for c, x in zip(("Umte", "Prot", "Bal", "BalBal"), cells):
            M.add(f"Sw{L.Macros.clean(key)}{c}", x)
    prop = L.syn_dir("isic2018", "dino518", "ruler_fixed", "corr", "proposed")
    cells = [L.ci_row(L.syn_boot(prop, a, 1.0)) for a in ("ui2e", "umte", "umte_protect", "umte_balanced")]
    L.table(T / "sweeps_isic.tex", ["Dermoscopy ruler sweep, $r=1$ (DINOv2)", "U-I2E $-$ ERM", "U-MtE $-$ ERM",
                                     "U-MtE protected $-$ ERM", "U-MtE+bal.\\ $-$ ERM"], [["change in reversed AUROC"] + cells],
            "The dermoscopy ruler sweep at full overlap: the insert-then-erase precursor (U-I2E, erasing on unmasked "
            "features) and the mask-then-erase variants, versus ERM (crossed CIs).", "tab:swisic", size="\\scriptsize", resize=True)
    M.add("SwIsicUitwoe", cells[0]); M.add("SwIsicUmte", cells[1]); M.add("SwIsicBal", cells[3])
    M.add("SwIsicBalInter", L.ci_row(L.syn_inter(prop, "umte_balanced")))
    L.table(T / "sweeps_remedies.tex", ["Controlled sweep, $r=1$", "U-MtE $-$ mask", "U-MtE protected $-$ mask",
                                        "U-MtE+bal.\\ $-$ mask", "U-MtE+bal.\\ $-$ balanced"], rows,
            "Remedies in the controlled sweeps at full overlap (reversed-test AUROC, crossed CIs).", "tab:swrem", size="\\scriptsize",
            resize=True)


def baselines():
    """JTT and SPLINCE (regenerated) with min(rev, corr)."""
    out = []
    for coh, rels in (("Thyroid", ("thyroid/dino518_jtt", "thyroid/dino518_splice_v2")), ("Capsule", ("capsule/dino518_jtt", "capsule/dino518_splice_v2")),
                      ("Dermoscopy hair", ("spec_e13/dino518_spec_jtt", "spec_e13/dino518_spec_splice_v2"))):
        for rel in rels:
            m = L.csv(f"{rel}/metrics_per_seed.csv")
            if m is None:
                m = L.csv(f"{rel}/metrics_per_seed.csv", L.OLD)
            if m is None:
                continue
            m = m[m.trap == "trapA"] if "trap" in m else m
            g = m.groupby(["method", "env"]).auc.mean().unstack()
            for arm in g.index:
                r = g.loc[arm]
                out.append({"cohort": coh, "run": rel, "arm": arm, "clean": round(r.clean, 3), "corr": round(r.test_corr, 3),
                            "rev": round(r.test_rev, 3), "min_rev_corr": round(min(r.test_rev, r.test_corr), 3)})
    df = L.derived("stage6_jtt_splice", pd.DataFrame(out).drop_duplicates(["cohort", "arm"]))
    names = {"jtt": "JTT", "mask_jtt": "mask+JTT", "mte": "U-MtE", "umte_jtt": "U-MtE+JTT", "splice": "SPLINCE",
             "mask_splice": "mask+SPLINCE", "mte_balanced": "U-MtE+bal.", "mte_protect": "U-MtE prot.", "balanced": "balanced", "mask": "mask"}
    rows = [[r.cohort, names.get(r.arm, L.tex_escape(r.arm)), L.f3(r.clean), L.f3(r.corr), L.f3(r.rev), L.f3(r.min_rev_corr)]
            for r in df.itertuples() if r.arm in names]
    L.table(T / "jtt_splice.tex", ["Cohort (Trap A)", "Arm", "clean", "correlated", "reversed", "min(rev, corr)"], rows,
            "Published baselines: JTT (label-free error upweighting) and SPLINCE (faithful whitened implementation of the "
            "covariance-preserving concept projection, needs artifact labels), beside U-MtE (mean over seeds).", "tab:jtt",
            size="\\scriptsize", long=True)
    for r in df.itertuples():
        M.add(f"Bl{L.Macros.clean(r.cohort.split()[0] + r.arm)}", L.f3(r.min_rev_corr))
        M.add(f"BlRev{L.Macros.clean(r.cohort.split()[0] + r.arm)}", L.f3(r.rev)); M.add(f"BlCorr{L.Macros.clean(r.cohort.split()[0] + r.arm)}", L.f3(r.corr))
        M.add(f"BlClean{L.Macros.clean(r.cohort.split()[0] + r.arm)}", L.f3(r.clean))


def rank_ablation():
    out = []
    for d in sorted((L.NEW / "ablation_umte").glob("*_*")):
        coh, setting = d.name.split("_", 1)
        m = pd.read_csv(d / "metrics_per_seed.csv")
        m = m[m.trap == "trapA"] if "trap" in m else m
        g = m[m.env == "test_rev"].groupby("method").auc.mean()
        for arm in ("mte", "mte_protect"):
            if arm in g and "mask" in g:
                out.append({"cohort": coh, "setting": setting, "arm": arm, "minus_mask": round(float(g[arm] - g["mask"]), 3)})
    df = L.derived("stage6_rank_ablation", pd.DataFrame(out))
    order = ["e0.50", "e0.70", "e0.80", "e0.90", "e0.95", "e0.99", "k1", "k4", "k16", "k64"]
    lab = lambda s_: ("energy " + s_[1:] + (" (main)" if s_ == "e0.90" else "")) if s_.startswith("e") else f"$k={s_[1:]}$"
    rows = []
    for st in order:
        q = df[df.setting == st]
        if q.empty:
            continue
        cell = lambda c, a: (lambda x: L.s3(float(x.iloc[0])) if len(x) else "--")(q[(q.cohort == c) & (q.arm == a)].minus_mask)
        rows.append([lab(st), cell("thyroid", "mte"), cell("ovary", "mte"), cell("ovary", "mte_protect"), cell("capsule", "mte"),
                     cell("capsule", "mte_protect")])
    L.table(T / "rank_ablation.tex", ["Erased rank", "Thyroid U-MtE", "Ovary U-MtE", "Ovary protected", "Capsule U-MtE",
                                      "Capsule protected"], rows,
            "Erasure-rank ablation (DINOv2, Trap A): reversed-AUROC difference to masking (mean over seeds) for the "
            "energy rule (share of held-out overlay-shift energy removed; cap 64) and for fixed ranks $k$.", "tab:rank",
            size="\\scriptsize")


def finetune_remedies():
    rows = []
    for lab, rel in (("Thyroid, ResNet-50", "finetune/thyroid/resnet50"), ("Thyroid, ViT-S/16", "finetune/thyroid/vit_small_patch16_224.augreg_in21k_ft_in1k"),
                     ("Capsule, ResNet-50", "finetune/capsule/resnet50"), ("Ovary, ResNet-50 (15 clusters)", "finetune/ovary/resnet50_power_g4")):
        p = L.csv(f"{rel}/paired_deltas.csv")
        if p is None:
            continue
        g = lambda a, r_: p[(p.trap == "trapA") & (p.arm == a) & (p.ref == r_)]
        cell = lambda q: L.ci_row(q.iloc[0]) if len(q) else "--"
        row = [lab] + [cell(g(a, r_)) for a, r_ in (("mte_ft", "mask"), ("umte_ft", "mask"), ("cons_ft", "mask"), ("umte_cons_ft", "mask"),
                                                    ("mte_post", "mask_post"))]
        rows.append(row)
        k = L.Macros.clean(lab.split(",")[0] + lab.split(",")[1].split("(")[0])
        for a, x in zip(("MteFt", "UmteFt", "ConsFt", "UmteConsFt", "MtePost"), row[1:]):
            M.add(f"Ft{k}{a}", x)
    L.table(T / "ft_remedies.tex", ["Fine-tuned network (Trap A)", "invariance (mte\\_ft)", "erase-while-fine-tuning (umte\\_ft)",
                                    "prediction consistency (cons\\_ft)", "all three (umte\\_cons\\_ft)", "fine-tune, then erase"], rows,
            "Remedies for end-to-end fine-tuned networks: change in reversed-test AUROC versus fine-tuned masking (last "
            "column: versus the same fine-tuned body without erasure). Folds as bootstrap clusters; crossed CIs.",
            "tab:ftrem", size="\\scriptsize", resize=True)


def archived_points():
    """Remedy analyses whose per-image predictions were not saved: point estimates only (not re-estimated)."""
    out = []
    lj = json.loads((L.OLD / "lama_comparison.json").read_text())
    for k, v in lj.items():
        out.append({"analysis": "LaMa inpainting (oracle masks)", "cohort": k.split("|")[0], "contrast": k.split("|")[1], "estimate": round(v[0], 3)})
    for coh, rel in (("thyroid", "thyroid/medsiglip448_text"), ("ovary", "ovary/medsiglip448_text"), ("capsule", "capsule/medsiglip448_text"),
                     ("hair (DermLIP)", "spec_e13/dermlip224_spec_text")):
        p = pd.read_csv(L.OLD / rel / "paired_deltas.csv")
        for a, r_ in (("mask_text_erase", "mask"), ("mte", "mask"), ("mask_text_erase_balanced", "balanced")):
            q = p[(p.trap == "trapA") & (p.arm == a) & (p.ref == r_) & (p.env == "test_rev")]
            if len(q):
                out.append({"analysis": "text-prompted erasure", "cohort": coh, "contrast": f"{a}-{r_}", "estimate": round(float(q.seed_delta_mean.iloc[0]), 3)})
    for coh, rel in (("hair", "slas/isic_k50"), ("thyroid", "slas/thyroid_k50")):
        p = pd.read_csv(L.OLD / rel / "bootstrap.csv")
        for a, r_ in (("tok_mask", "tok_erm"), ("mts", "tok_mask"), ("slas", "tok_erm"), ("mts_oracle", "tok_mask"), ("mts_balanced", "tok_balanced")):
            q = p[(p.trap == "trapA") & (p.arm == a) & (p.ref == r_) & (p.env == "test_rev")]
            if len(q):
                out.append({"analysis": "SLAS", "cohort": coh, "contrast": f"{a}-{r_}", "estimate": round(float(q.seed_delta_mean.iloc[0]), 3)})
    for coh, rel in (("thyroid", "thyroid/dino518_u10"), ("capsule", "capsule/dino518_u10"), ("hair", "spec_e13/dino518_spec_u10")):
        p = pd.read_csv(L.OLD / rel / "paired_deltas.csv")
        for a, r_ in (("mte_dfr", "dfr"), ("mte_dfr", "mask_dfr"), ("mask_dfr", "dfr")):
            q = p[(p.trap == "trapA") & (p.arm == a) & (p.ref == r_) & (p.env == "test_rev")]
            if len(q):
                out.append({"analysis": "erasure + DFR (U10)", "cohort": coh, "contrast": f"{a}-{r_}", "estimate": round(float(q.seed_delta_mean.iloc[0]), 3)})
    for coh, rel in (("thyroid", "thyroid/dino518_pbal"), ("hair", "spec_e13/dino518_spec_pbal")):
        f = L.OLD / rel / "metrics_per_seed.csv"
        if f.exists():
            m = pd.read_csv(f); m = m[(m.trap == "trapA") & (m.env == "test_rev")].groupby("method").auc.mean()
            for a in ("umte_pbal", "mte", "pbal", "erm"):
                if a in m:
                    out.append({"analysis": "pseudo-group balancing (U8)", "cohort": coh, "contrast": f"{a} reversed AUROC", "estimate": round(float(m[a]), 3)})
    p = pd.read_csv(L.OLD / "cxr_drain/raddino518_universal/paired_deltas.csv")
    for a, r_ in (("mte", "mask"), ("mte_protect", "mask"), ("mte", "mte_aug"), ("mte", "jtt"), ("mte_balanced", "balanced")):
        q = p[(p.arm == a) & (p.ref == r_) & (p.env == "test_rev")]
        if len(q):
            out.append({"analysis": "chest drains, RAD-DINO", "cohort": "drains", "contrast": f"{a}-{r_}", "estimate": round(float(q.seed_delta_mean.iloc[0]), 3)})
    for tag, f in (("location-adaptive selection (U13)", "adaptive_select_summary.csv"), ("split-validation selection (U14)", "adaptive_select_split_summary.csv")):
        d = pd.read_csv(L.OLD / f)
        # the summary files concatenate the runs in this order; dino518 runs are thyroid, capsule (, ovary)
        seen = {}
        order = {"dino518_auto": ["thyroid", "capsule", "ovary"], "dino518_auto2": ["thyroid", "capsule", "ovary"],
                 "dino518_spec_auto": ["hair"], "dino518_spec_auto2": ["hair"], "raddino518_auto": ["chest drains"],
                 "raddino518_auto2": ["chest drains"]}
        blocks = (d.run != d.run.shift()) | (d.trap.eq("trapA") & d.trap.shift().eq("trapB"))
        d = d.assign(block=blocks.cumsum())
        for bl, q in d.groupby("block", sort=True):
            run = q.run.iloc[0]; k = seen.get(run, 0); seen[run] = k + 1
            coh = order.get(run, [run])[min(k, len(order.get(run, [run])) - 1)]
            for r in q[(q.trap == "trapA") & q.ref.isin(["mask", "dfr"])].itertuples():
                out.append({"analysis": tag, "cohort": coh, "contrast": f"auto-{r.ref}", "estimate": round(float(r.seed_delta_mean), 3)})
    df = L.derived("stage6_archived_points", pd.DataFrame(out))
    for r in df.itertuples():
        M.add(f"Ar{L.Macros.clean(r.analysis.split()[0] + r.cohort + r.contrast)}", L.s3(r.estimate) if "AUROC" not in r.contrast else L.f3(r.estimate))
    pretty = lambda t: L.tex_escape(t).replace("-", " $-$ ")
    rows = [[r.analysis if (k == 0 or df.analysis.iloc[k - 1] != r.analysis) else "", L.tex_escape(r.cohort), pretty(r.contrast),
             L.f3(r.estimate) if "AUROC" in r.contrast else L.s3(r.estimate)] for k, r in enumerate(df.itertuples())]
    L.table(T / "archived_points.tex", ["Analysis", "Cohort / run", "Contrast (Trap A, reversed)", "Point estimate"], rows,
            "Remedy analyses whose per-image predictions were not saved, so their intervals could not be re-estimated with "
            "the crossed bootstrap: \\textbf{point estimates only}.", "tab:archived", align="p{4.0cm}p{3.6cm}p{5.2cm}c",
            size="\\scriptsize", long=True)


def scale_remedy():
    sc = L.csv("scale/scale_compare.csv")
    if sc is None:
        return
    rows = [[r.cohort.capitalize(), r.backbone, L.ci(r.mte_mask, r.mte_mask_lo, r.mte_mask_hi),
             L.ci(r.mte_protect_mask, r.mte_protect_mask_lo, r.mte_protect_mask_hi)] for r in sc.itertuples()]
    L.table(T / "scale_remedy.tex", ["Cohort", "DINOv2", "U-MtE $-$ mask (Trap A)", "U-MtE protected $-$ mask"], rows,
            "U-MtE across encoder sizes (crossed CIs).", "tab:scalerem", size="\\scriptsize")


def overlay_figure():
    """The generic overlay library applied to an openly licensed image (MMOTU, CC BY 4.0)."""
    from PIL import Image
    sys.path.insert(0, str(L.ROOT / "src"))
    try:
        from wtss.synthetic import draw_generic_artifact
    except Exception as e:  # noqa: BLE001
        print("overlay figure:", e); return
    import inspect
    base = Image.open(L.ROOT / "figures" / "examples" / "ovary_calipers_out_roi_roi.png").convert("RGB").resize((518, 518))
    sig = inspect.signature(draw_generic_artifact)
    plt = L.plot_style()
    fig, ax = plt.subplots(1, 4, figsize=(8.0, 2.2))
    ax[0].imshow(base); ax[0].set_title("ovarian ultrasound (MMOTU)", fontsize=7)
    arr0 = np.asarray(base, dtype=float)
    cands = []
    for j in range(40):  # show the three draws of a fixed key list that change the image most (visibility only)
        im = draw_generic_artifact(base.copy(), None, f"stage6_example_{j}")
        im = im[0] if isinstance(im, tuple) else im
        cands.append((float(np.abs(np.asarray(im, dtype=float) - arr0).mean()), j, im))
    for k, (_, j, im) in enumerate(sorted(cands, reverse=True)[:3], start=1):
        ax[k].imshow(im); ax[k].set_title(f"generic overlay (draw {j})", fontsize=7)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.savefig(F / "overlays.pdf", dpi=150)


def registrations():
    docs = {"Final": "docs/PREREGISTRATION_FINAL.md", "Umte": "docs/PREREGISTRATION_UNIVERSAL_I2E.md",
            "Abl": "docs/PREREGISTRATION_UMTE_ABLATION.md", "Rev2": "docs/PREREGISTRATION_REVIEW2.md",
            "Stat": "docs/STATISTICAL_PLAN.md"}
    for k, doc in docs.items():
        M.add(f"Reg{k}", L.reg(doc))
    R = lambda d: L.reg(f"docs/{d}.md")
    rows = [["U-I2E / U-MtE, protection, balancing, U7--U14", "\\ok{} registered (with amendments before each run)", "UNIVERSAL\\_I2E",
             L.reg(docs["Umte"]), "per claim, CI; no gaming (clean loss $\\le0.02$, corr $\\ge$ rev $-0.02$)"],
            ["Remedy tests P2--P4 in the 16-test family", "\\mixed{} tests registered; family grouped post hoc", "STATISTICAL\\_PLAN",
             L.reg(docs["Stat"]), "Holm $p<0.05$, estimate $>0$"],
            ["Erasure-rank ablation", "\\ok{} registered", "UMTE\\_ABLATION", L.reg(docs["Abl"]), "descriptive"],
            ["Encoder size (S2)", "\\ok{} registered", "SCALE", R("PREREGISTRATION_SCALE"), "protected U-MtE $>$ mask, CI"],
            ["LaMa inpainting", "\\ok{} registered", "INPAINT\\_LAMA", R("PREREGISTRATION_INPAINT_LAMA"), "reported with CI, no direction"],
            ["Text-prompted erasure (V1--V3)", "\\ok{} registered", "TEXT\\_PROMPT", R("PREREGISTRATION_TEXT_PROMPT"), "V1: text erasure $>$ mask"],
            ["SLAS (S0--S6)", "\\mixed{} registration and localisation result share a commit", "SLAS", R("PREREGISTRATION_SLAS"),
             "S1: mask-then-suppress $>$ token masking"],
            ["Chest drains (D1--D5)", "\\ok{} registered", "CXR\\_DRAIN", R("PREREGISTRATION_CXR_DRAIN"), "U-MtE $>$ mask, CI"],
            ["Fine-tuning F1--F3", "\\ok{} registered", "FINETUNE", R("PREREGISTRATION_FINETUNE"), "mte\\_ft $>$ mask, CI"],
            ["Fine-tune, then erase", "\\ok{} registered", "FT\\_ERASE", R("PREREGISTRATION_FT_ERASE"), "mte\\_post $>$ mask\\_post"],
            ["Erase while fine-tuning (G1--G4)", "\\ok{} registered (amendments before running)", "FT\\_UMTE", R("PREREGISTRATION_FT_UMTE"),
             "umte(\\_cons)\\_ft $>$ mask for both architectures"],
            ["Leakage: embedding groups (P2--P4)", "\\ok{} registered", "EMBEDDING\\_GROUPS", R("PREREGISTRATION_EMBEDDING_GROUPS"),
             "same sign and CI verdict"],
            ["Natural data (N1, N2, N4)", "\\ok{} registered", "NATURAL", R("PREREGISTRATION_NATURAL"), "protected U-MtE $\\ge$ mask"],
            ["Clinician image audit (prepared)", "\\ok{} registered", "FINAL (Part B)", L.reg(docs["Final"]),
             "precision, recall, location agreement, $\\kappa$"]]
    rows = [r[:2] + [r[2].replace("\\_", "\\_\\allowbreak{}")] + r[3:] for r in rows]
    L.table(T / "experiments.tex", ["Experiment", "Status", "Registration", "First commit (local time)", "Support criterion"],
            rows, "Experiments of this stage and their registration documents (\\texttt{docs/PREREGISTRATION\\_<name>.md} or "
            "\\texttt{docs/<name>.md}). Commit times are set by the committing machine and are not independent proof of order.",
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
    remedies(); primary_remedies(); natural_remedies(); audit(); external(); paper(); registrations(); remedies_figure()
    remedies_all(); robustness(); primary_all(); supp_contrasts(); sweeps_remedies(); baselines(); rank_ablation()
    finetune_remedies(); archived_points(); scale_remedy(); overlay_figure()
    try:
        external_examples()
    except Exception as e:  # noqa: BLE001
        print('external examples:', e)
    M.write(T / "numbers.tex")
    L.report_missing("stage6")


if __name__ == "__main__":
    main()
