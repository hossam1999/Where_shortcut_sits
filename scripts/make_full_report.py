"""Build the complete standalone research record (report/full/full_report.pdf).

Narrative: report/full/narrative.tex (hand-written, stage by stage). Everything else is generated here from the
repository so the record cannot drift from the results: tables from result CSVs, every design/pre-registration/audit
document converted from Markdown, every generated paper table and every figure.
  python scripts/make_full_report.py && (cd report/full && tectonic -X compile full_report.tex)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pypandoc

from wtss import paths

ROOT = paths.REPO_ROOT
R = paths.RESULTS
OUT = ROOT / "report" / "full"
GEN = OUT / "gen"

# experiment registry, in the order the work was done (stage order of the narrative)
DOCS = [("Replication specification of the pilot (author protocol)", "REPLICATION_SPEC.md"),
        ("Audit of the original thesis claims", "THESIS_CLAIMS_AUDIT.md"),
        ("Data sources and licences", "DATA.md"),
        ("Pre-registration: ISIC 2019 real-hair traps", "PREREGISTRATION_ISIC2019_TRAPS.md"),
        ("Pre-commit: matched traps", "PRECOMMIT_MATCHED_TRAPS.md"),
        ("Pre-registration: chest-radiograph device traps", "PREREGISTRATION_CXR_DEVICE_TRAPS.md"),
        ("Pre-registration: real chest drains", "PREREGISTRATION_CXR_DRAIN.md"),
        ("Pre-registration: thyroid ultrasound traps", "PREREGISTRATION_THYROID_TRAPS.md"),
        ("Thyroid caliper detector audit", "THYROID_MARKER_AUDIT.md"),
        ("Pre-registration: capsule-endoscopy traps", "PREREGISTRATION_CAPSULE_TRAPS.md"),
        ("Pre-registration: ovarian ultrasound traps", "PREREGISTRATION_OVARY_TRAPS.md"),
        ("Breast ultrasound feasibility audit (not used)", "BREAST_MARKER_AUDIT.md"),
        ("Pre-registration: universal insertion-erasure (U-MtE) and ablations U1–U14", "PREREGISTRATION_UNIVERSAL_I2E.md"),
        ("Pre-registration: SLAS (few-shot patch suppression)", "PREREGISTRATION_SLAS.md"),
        ("Pre-registration: end-to-end fine-tuning", "PREREGISTRATION_FINETUNE.md"),
        ("Pre-registration: natural (unaltered) test distributions", "PREREGISTRATION_NATURAL.md"),
        ("Sensitivity to artifact-label definitions", "SENSITIVITY_ARTIFACT_LABELS.md"),
        ("Artifact-mask visual audit", "ARTIFACT_MASK_AUDIT.md"),
        ("Pre-registration: LaMa inpainting comparison", "PREREGISTRATION_INPAINT_LAMA.md"),
        ("Pre-registration: text-prompted erasure and zero-shot reliance", "PREREGISTRATION_TEXT_PROMPT.md"),
        ("Statistical plan and multiplicity", "STATISTICAL_PLAN.md"),
        ("Theory (linear-Gaussian account)", "THEORY.md"),
        ("Pre-registration: theory versus real encoders", "PREREGISTRATION_THEORY_PREDICTION.md"),
        ("Pre-registration: embedding-based leakage groups", "PREREGISTRATION_EMBEDDING_GROUPS.md"),
        ("Pre-registration: backbone scale", "PREREGISTRATION_SCALE.md"),
        ("Pre-registration: U-MtE rank ablation", "PREREGISTRATION_UMTE_ABLATION.md"),
        ("Pre-registration: fine-tune, then erase", "PREREGISTRATION_FT_ERASE.md"),
        ("Pre-registration: erase-while-fine-tuning", "PREREGISTRATION_FT_UMTE.md"),
        ("Related work and novelty assessment", "RELATED_WORK_NOVELTY.md"),
        ("Second external review: verification of the claims and responses", "REVIEW2_RESPONSE.md"),
        ("Pre-registration: second-review analyses (transplant, matched traps, operating points, fine-tuned hair)", "PREREGISTRATION_REVIEW2.md"),
        ("Findings overview (living document)", "FINDINGS_OVERVIEW.md")]
RESULT_MD = [("Pilot replication: all 88 numbers", "SUMMARY.md"), ("Holm-corrected primary claims", "PRIMARY_CLAIMS.md"),
             ("Every arm × cohort × backbone", "CROSS_COHORT.md")]


def md2tex(md: str) -> str:
    md = re.sub(r"^# .*\n", "", md, count=1)  # the section title comes from the registry
    tex = pypandoc.convert_text(md, "latex", format="gfm", extra_args=["--wrap=preserve", "--shift-heading-level-by=1", "--syntax-highlighting=none"])
    tex = tex.replace("\\tightlist", "").replace("\\def\\LTcaptype{none}", "")
    return "{\\small\n" + tex + "\n}\n"


def tabular(df: pd.DataFrame, cols: list[str], heads: list[str], align: str, caption: str, label: str, size="\\small") -> str:
    L = [f"\\begin{{table}}[H]\\centering{size}", f"\\caption{{{caption}}}\\label{{{label}}}",
         f"\\begin{{tabular}}{{{align}}}\\toprule", " & ".join(heads) + " \\\\\\midrule"]
    for _, r in df.iterrows():
        L.append(" & ".join(re.sub(r"(?<!\\)_", r"\\_", str(r[c])) for c in cols) + " \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(L) + "\n"


def ci(e, lo, hi):
    return f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def gen_tables():
    T = {}
    p = pd.read_csv(R / "PRIMARY_CLAIMS.csv")
    p["ci"] = [ci(*v) for v in p[["estimate", "ci95_lo", "ci95_hi"]].to_numpy()]
    p["ph"] = p.p_holm.map(lambda x: f"{x:.4f}"); p["ok"] = p["supported_holm_0.05"].map({True: "yes", False: "no"})
    T["primary"] = tabular(p, ["family", "cohort", "ci", "ph", "ok"], ["Family", "Cohort", "Estimate [95\\% CI]", "$p_{\\text{Holm}}$", "Supported"],
                           "llccc", "The 16 pre-registered primary tests (DINOv2 ViT-B/14 @518; reversed-test AUROC; Holm over the family).", "tab:primary")
    x = pd.read_csv(R / "theory" / "prediction_crossover.csv")
    x["o"] = x.obs.map(lambda v: f"{v:+.3f}"); x["p"] = x.pred.map(lambda v: f"{v:+.3f}")
    x["ft"] = x.finetune.map({True: "end-to-end", False: "frozen"})
    T["theory_x"] = tabular(x, ["cohort", "backbone", "ft", "o", "p"], ["Cohort", "Backbone", "Regime", "Observed crossover", "Theory (fit on clean+corr)"],
                            "lllcc", "Theory versus real encoders: location crossover implied by $(S,A)$ fitted to clean and correlated AUROC only.", "tab:theoryx")
    s = json.load(open(R / "theory" / "prediction_summary.json"))
    T["theory_s"] = ("\\begin{table}[H]\\centering\\small\\caption{Theory-vs-real summary (prediction\\_summary.json).}\\label{tab:theorys}"
                     "\\begin{tabular}{lccc}\\toprule Endpoint & Theory & Ref.\\ rev$=$clean & Ref.\\ rev$=2\\cdot$clean$-$corr\\\\\\midrule"
                     f"T1 reversed AUROC, ERM+mask ({s['T1_primary']['theory']['n']} models): MAE & {s['T1_primary']['theory']['mae']:.3f} & {s['T1_primary']['ref_clean']['mae']:.3f} & {s['T1_primary']['ref_sym']['mae']:.3f}\\\\"
                     f"T1 Pearson $r$ & {s['T1_primary']['theory']['r']:.3f} & {s['T1_primary']['ref_clean']['r']:.3f} & {s['T1_primary']['ref_sym']['r']:.3f}\\\\"
                     f"T1 all ERM-head arms ({s['T1_all_erm_head_arms']['theory']['n']}): MAE & {s['T1_all_erm_head_arms']['theory']['mae']:.3f} & {s['T1_all_erm_head_arms']['ref_clean']['mae']:.3f} & {s['T1_all_erm_head_arms']['ref_sym']['mae']:.3f}\\\\"
                     f"T2 crossover ({s['T2_crossover']['n']} pairs): $r$ / MAE / signs & {s['T2_crossover']['r']:.3f} / {s['T2_crossover']['mae']:.3f} / {s['T2_crossover']['sign_agree']} & {s['reference_ref_clean']['T2_r']:.3f} / {s['reference_ref_clean']['T2_mae']:.3f} / {s['reference_ref_clean']['T2_sign_agree']} & {s['reference_ref_sym']['T2_r']:.3f} / {s['reference_ref_sym']['T2_mae']:.3f} / {s['reference_ref_sym']['T2_sign_agree']}\\\\"
                     f"T3 sweeps ({s['T3_sweeps']['n']} points): $r$ / MAE / signs & {s['T3_sweeps']['r']:.3f} / {s['T3_sweeps']['mae']:.3f} / {s['T3_sweeps']['sign_agree']} & {s['reference_ref_clean']['T3_r']:.3f} / {s['reference_ref_clean']['T3_mae']:.3f} / {s['reference_ref_clean']['T3_sign_agree']} & {s['reference_ref_sym']['T3_r']:.3f} / {s['reference_ref_sym']['T3_mae']:.3f} / {s['reference_ref_sym']['T3_sign_agree']}\\\\"
                     f"T4 in-ROI sign from $S_m-S$ & {s['T4_inroi_sign_from_dS']['agree']}/{s['T4_inroi_sign_from_dS']['n']} & & \\\\"
                     f"Fine-tuned (outside model): rev MAE & {s['T1_finetune_secondary']['mae']:.3f} & & \\\\"
                     f"\\bottomrule\\end{{tabular}}\\par\\smallskip Pre-registered verdict: \\textbf{{{s['verdict']}}}.\\end{{table}}\n")
    T["theory_s"] = T["theory_s"].replace("\\midrule", "\\midrule\n").replace("\\\\", "\\\\\n")
    d = pd.read_csv(R / "scale" / "scale_compare.csv")
    d["x"] = [ci(*v) for v in d[["cross", "cross_lo", "cross_hi"]].to_numpy()]
    d["p"] = [ci(*v) for v in d[["mte_protect_mask", "mte_protect_mask_lo", "mte_protect_mask_hi"]].to_numpy()]
    d["m"] = [ci(*v) for v in d[["mte_mask", "mte_mask_lo", "mte_mask_hi"]].to_numpy()]
    d["g"] = d.erm_gap.map(lambda v: f"{v:.3f}")
    T["scale"] = tabular(d, ["cohort", "backbone", "x", "p", "m", "g"], ["Cohort", "DINOv2", "Crossover", "U-MtE$_{\\text{prot}}-$mask", "U-MtE$-$mask", "ERM gap"],
                         "llcccc", "Backbone scale (Trap A for the U-MtE contrasts; ERM gap = correlated $-$ reversed AUROC, Trap A).", "tab:scale", "\\scriptsize")
    e = pd.read_csv(R / "leakage" / "embedding_groups_compare.csv")
    e["m"] = [ci(*v) for v in e[["main", "main_lo", "main_hi"]].to_numpy()]
    e["n"] = [ci(*v) for v in e[["emb", "emb_lo", "emb_hi"]].to_numpy()]
    e["r"] = e.robust.map({True: "yes", False: "\\textbf{no}"})
    T["emb"] = tabular(e, ["cohort", "test", "m", "n", "r"], ["Cohort", "Test", "Main groups", "Embedding groups", "Robust"],
                       "llccc", "Primary contrasts under stricter embedding-based leakage groups.", "tab:emb", "\\scriptsize")
    k = pd.read_csv(R / "leakage" / "LEAKAGE_SUMMARY.csv")
    for c in ("erm_clean", "erm_corr", "erm_rev", "erm_gap", "mask_minus_erm_rev"):
        k[c] = k[c].map(lambda v: f"{v:.3f}")
    T["leak"] = tabular(k, ["split", "overlap", "cross_split_near_dup_pairs", "erm_clean", "erm_corr", "erm_rev", "erm_gap", "mask_minus_erm_rev"],
                        ["Split", "Overlap", "Near-dup pairs across split", "ERM clean", "ERM corr", "ERM rev", "ERM gap", "mask$-$ERM rev"],
                        "lccccccc", "Leakage assessment (controlled dermoscopy sweep): grouped (lesion ID + pHash) versus five random splits.", "tab:leak", "\\scriptsize")
    a = pd.read_csv(R / "ablation_umte" / "summary.csv")
    a = a.pivot_table(index=["cohort", "setting"], columns="arm", values="test_rev").reset_index()
    a["d1"] = (a.mte - a["mask"]).map(lambda v: f"{v:+.3f}"); a["d2"] = (a.mte_protect - a["mask"]).map(lambda v: f"{v:+.3f}")
    for c in ("mask", "mte", "mte_protect"):
        a[c] = a[c].map(lambda v: f"{v:.3f}")
    T["abl"] = tabular(a, ["cohort", "setting", "mask", "mte", "mte_protect", "d1", "d2"],
                       ["Cohort", "Rank rule", "mask", "U-MtE", "U-MtE$_{\\text{prot}}$", "U-MtE$-$mask", "prot$-$mask"],
                       "llccccc", "U-MtE erasure-rank ablation (Trap A reversed AUROC, mean over seeds; e = held-out energy, k = fixed rank).", "tab:abl", "\\scriptsize")
    rows = []
    for f in sorted((R / "finetune").glob("*/*/paired_deltas.csv")):
        q = pd.read_csv(f)
        for _, r in q.iterrows():
            rows.append({"cohort": f.parent.parent.name, "arch": f.parent.name.split("_patch")[0].replace("vit_small", "ViT-S/16").replace("resnet50", "ResNet-50"),
                         "trap": r.trap, "c": f"{r.arm} $-$ {r.ref}".replace("_", "\\_"), "d": ci(r.seed_delta_mean, r.ci95_lo, r.ci95_hi)})
    T["ft"] = tabular(pd.DataFrame(rows), ["cohort", "arch", "trap", "c", "d"], ["Cohort", "Architecture", "Trap", "Contrast", "$\\Delta$ reversed AUROC [95\\% CI]"],
                      "lllll", "Every end-to-end fine-tuning contrast (5 folds of seed 42 as clusters).", "tab:ft", "\\scriptsize")
    m = []
    for f in sorted((R / "finetune").glob("*/*/predictions.csv.gz")):
        q = pd.read_csv(f)
        from wtss.stats import safe_auc
        g = q.groupby(["trap", "method", "env", "seed"]).apply(lambda z: safe_auc(z.y, z.prob), include_groups=False).groupby(["trap", "method", "env"]).mean().unstack()
        for (tr, meth), r in g.iterrows():
            m.append({"cohort": f.parent.parent.name, "arch": f.parent.name.split("_patch")[0].replace("vit_small", "ViT-S/16").replace("resnet50", "ResNet-50"),
                      "trap": tr, "arm": meth.replace("_", "\\_"), **{k2: f"{r.get(k2, float('nan')):.3f}" for k2 in ("clean", "test_corr", "test_rev")}})
    T["ft_auc"] = tabular(pd.DataFrame(m), ["cohort", "arch", "trap", "arm", "clean", "test_corr", "test_rev"],
                          ["Cohort", "Arch.", "Trap", "Arm", "Clean", "Correlated", "Reversed"], "lllllccc",
                          "End-to-end fine-tuning: AUROC per arm (mean over folds).", "tab:ftauc", "\\scriptsize")
    nat = []
    for f in sorted((R / "natural").glob("*/natural_boot.json")):
        j = json.load(open(f))
        for key, v in j.items():
            if isinstance(v, list):
                nat.append({"test": f.parent.name.replace("_dino518", "").replace("_", " "), "c": key.replace("_", "\\_").replace("|", "$|$"), "d": ci(*v)})
    T["natural"] = tabular(pd.DataFrame(nat), ["test", "c", "d"], ["Test set", "Subset $|$ contrast", "$\\Delta$ AUROC [95\\% CI]"], "lll",
                           "Natural (unaltered) test distributions: every bootstrap contrast (hard = shortcut-conflicting pairs).", "tab:natural", "\\scriptsize")
    g = json.load(open(R / "leakage" / "embedding_groups.json"))
    gg = pd.DataFrame([{"c": c, **v} for c, v in g.items()])
    T["embg"] = tabular(gg, ["c", "n_images", "groups_old", "groups_emb", "tau", "largest_group_share"],
                        ["Cohort", "Images", "Groups (pHash/frames)", "Groups (+embedding)", "$\\tau$", "Largest group share"],
                        "lccccc", "Embedding-based near-duplicate groups (mean-centred DINOv2 CLS, cosine $\\ge\\tau$).", "tab:embg", "\\small")
    for name, tex in T.items():
        (GEN / f"tab_{name}.tex").write_text(tex)


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    gen_tables()
    reg = []
    for title, f in DOCS:
        reg.append(f"\\section{{{title}}}\\label{{doc:{f[:-3]}}}\n\\noindent\\textit{{Source: docs/{f.replace('_', chr(92) + '_')}}}\n\n" + md2tex((ROOT / "docs" / f).read_text()))
    (GEN / "registry.tex").write_text("\n".join(reg))
    res = []
    for title, f in RESULT_MD:
        res.append(f"\\section{{{title}}}\n\\noindent\\textit{{Source: results/{f.replace('_', chr(92) + '_')}}}\n\n" + md2tex((R / f).read_text()))
    (GEN / "results_md.tex").write_text("\n".join(res))
    tabs = sorted(t for t in (ROOT / "paper" / "tables").glob("*.tex")
                  if not t.stem.endswith(("_text", "_inline", "_conflict", "_abstract_transplant")))  # inline sentence snippets
    (GEN / "paper_tables.tex").write_text("\n".join(f"\\subsection*{{{t.stem.replace('_', ' ')}}}\n\\input{{../../paper/tables/{t.name}}}\n\\clearpage" for t in tabs))
    print("generated", len(DOCS), "documents,", len(tabs), "paper tables")


if __name__ == "__main__":
    main()
