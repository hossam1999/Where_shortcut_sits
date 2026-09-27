"""Clinical secondary metrics on the unaltered (natural) test sets: AUROC, AUPRC, Brier score, ECE (10 equal-width
bins) and sensitivity / specificity at the operating point fixed on clean validation data (the saved `pred`).
Mean (SD) over training seeds.
  python scripts/analysis/clinical_metrics.py   # -> results/natural/clinical_metrics.csv, paper/tables/natural_clinical.tex
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from wtss import paths

RUNS = {"Thyroid (official split)": "thyroid_dino518", "ISIC, BCN held out": "isic_BCN_dino518",
        "ISIC, MSK held out": "isic_MSK_dino518", "ISIC, HAM held out": "isic_HAM_dino518",
        "Capsule (held-out frames)": "capsule_dino518"}
ARMS = {"erm": "ERM", "mask": "mask", "mte": "U-MtE", "balanced": "balanced", "mte_balanced": "U-MtE$_{\\text{bal}}$"}


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))


def main():
    rows = []
    for name, rel in RUNS.items():
        p = pd.read_csv(paths.RESULTS / "natural" / rel / "predictions.csv.gz")
        p = p[(p.env == "clean") & p.method.isin(ARMS)]
        for (m, s), q in p.groupby(["method", "seed"]):
            y, pr, pd_ = q.y.to_numpy(), q.prob.to_numpy(), q.pred.to_numpy()
            rows.append({"test": name, "arm": m, "seed": s, "AUROC": roc_auc_score(y, pr), "AUPRC": average_precision_score(y, pr),
                         "Brier": float(np.mean((pr - y) ** 2)), "ECE": ece(y, pr),
                         "Sens": float(pd_[y == 1].mean()), "Spec": float(1 - pd_[y == 0].mean()), "prev": float(y.mean())})
    d = pd.DataFrame(rows)
    out = d.groupby(["test", "arm"]).agg(["mean", "std"]).drop(columns="seed")
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    out.to_csv(paths.RESULTS / "natural" / "clinical_metrics.csv", index=False)
    print(out.round(3).to_string())
    cols = ["AUROC", "AUPRC", "Brier", "ECE", "Sens", "Spec"]
    L = ["\\begin{table}[t]\\centering\\scriptsize",
         "\\caption{Clinical secondary metrics on the unaltered test sets (Sec.~\\ref{sec:robust}): mean (SD) over "
         "training seeds. Sensitivity and specificity at the operating point fixed on clean validation data; ECE with 10 "
         "equal-width bins; AUPRC baseline = prevalence.}\\label{tab:natural_clinical}",
         "\\begin{tabular}{ll" + "c" * len(cols) + "}\\toprule", "Test set & Arm & " + " & ".join(cols) + " \\\\\\midrule"]
    for name in RUNS:
        q = out[out.test == name]
        prev = d[d.test == name].prev.mean()
        for i, (arm, lab) in enumerate(ARMS.items()):
            r = q[q.arm == arm]
            if r.empty:
                continue
            r = r.iloc[0]
            first = f"{name} ($\\pi={prev:.2f}$)" if i == 0 else ""
            L.append(f"{first} & {lab} & " + " & ".join(f"{r[c + '_mean']:.3f} ({r[c + '_std']:.3f})" for c in cols) + " \\\\")
        L.append("\\midrule" if name != list(RUNS)[-1] else "\\bottomrule")
    L += ["\\end{tabular}\\end{table}"]
    f = paths.REPO_ROOT / "paper" / "tables" / "natural_clinical.tex"
    f.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
