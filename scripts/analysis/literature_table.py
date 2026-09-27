"""Absolute performance next to published results on the same public data (docs/PREREGISTRATION_REVIEW2.md, R4 F2).

Published values are copied from the cited papers (verified against the PDFs); ours are read from result files.
  python scripts/analysis/literature_table.py   -> results/review2/literature.csv, paper/tables/review2_literature.tex
"""
from __future__ import annotations

import pandas as pd

from wtss import paths
from wtss.stats import safe_auc

R = paths.RESULTS
# Gong et al., MICCAI 2022 (TNCD introduced; official patient-disjoint split, 614 test images), Table 2, AUC in %
PUBLISHED = [
    ("Thyroid (TNCD official split)", "ResNet-18, cross-entropy (fine-tuned)", 0.7731, "gong2022acl"),
    ("Thyroid (TNCD official split)", "DenseNet-121, cross-entropy (fine-tuned)", 0.7805, "gong2022acl"),
    ("Thyroid (TNCD official split)", "ConvNeXt-T, cross-entropy (fine-tuned)", 0.7838, "gong2022acl"),
    ("Thyroid (TNCD official split)", "ResNet-18, adaptive curriculum (fine-tuned)", 0.7989, "gong2022acl"),
]


def ours():
    rows = []
    for tag, label in (("thyroid_dino518", "archived run"), ("thyroid_dino518_repro", "rebuilt run")):
        f = R / "natural" / tag / "predictions.csv.gz"
        if f.exists():
            p = pd.read_csv(f)
            p = p[(p.env == "clean") & (p.method == "erm")]
            auc = p.groupby("seed").apply(lambda q: safe_auc(q.y, q.prob), include_groups=False)
            rows.append(("Thyroid (TNCD official split)", f"DINOv2 ViT-B/14 frozen + linear head, ERM ({label})",
                         float(auc.mean()), "this work"))
        else:  # archived predictions are not committed; use the archived metrics table
            m = R / "natural" / tag / "natural_metrics.csv"
            if m.exists():
                d = pd.read_csv(m, index_col=0)
                rows.append(("Thyroid (TNCD official split)", f"DINOv2 ViT-B/14 frozen + linear head, ERM ({label})",
                             float(d.loc["erm", "auc"]), "this work"))
    return rows


def main():
    d = pd.DataFrame(PUBLISHED + ours(), columns=["cohort", "model", "auroc", "source"])
    out = paths.ensure(R / "review2")
    d.to_csv(out / "literature.csv", index=False)
    print(d.round(3).to_string())
    L = ["\\begin{center}\\small", "\\begin{tabular}{llc}\\toprule", "Model & Source & Test AUROC \\\\\\midrule"]
    for r in d.itertuples():
        src = f"\\citet{{{r.source}}}" if r.source != "this work" else "this work"
        L.append(f"{r.model} & {src} & {r.auroc:.3f} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{center}"]
    (paths.REPO_ROOT / "paper" / "tables" / "review2_literature.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
