"""Mechanism figure: same-head counterfactual sensitivity to the artifact, |p(x + artifact) - p(x)|, versus overlap with
the ROI, from the controlled sweeps' saved counterfactual pairs (per seed mean; band = min-max over seeds).
  python scripts/analysis/mechanism_figure.py   # -> paper/figures/mechanism.{pdf,png}, results/analysis/mechanism.csv
"""
from __future__ import annotations

import matplotlib
import pandas as pd

from wtss import paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S = paths.RESULTS / "synthetic"
RUNS = [("Dermoscopy ruler (DINOv2)", "isic2018/dino518_ruler_fixed_corr_universal"),
        ("Thyroid caliper (DINOv2)", "thyroid/dino518_caliper_corr_main"),
        ("Ovary caliper (DINOv2)", "ovary/dino518_caliper_corr_main"),
        ("Capsule debris (DINOv2)", "capsule/dino518_debris_corr_main"),
        ("Chest tube (RAD-DINO)", "nih_ptx/raddino518_tube_corr_main")]
ARMS = [("erm", "ERM", "k", "-"), ("mask", "ROI masking", "C0", "-"), ("umte", "U-MtE", "C1", "--"),
        ("umte_balanced", "U-MtE + balanced", "C2", ":")]


def main():
    fig, axes = plt.subplots(1, len(RUNS), figsize=(13, 2.7), sharey=False)
    rows = []
    for ax, (title, rel) in zip(axes, RUNS):
        p = pd.read_csv(S / rel / "counterfactual_per_image.csv.gz")
        s = p.groupby(["method", "overlap", "seed"]).abs_delta_p.mean().reset_index()
        for arm, lab, c, ls in ARMS:
            q = s[s.method == arm]
            if q.empty:
                continue
            g = q.groupby("overlap").abs_delta_p.agg(["mean", "min", "max"])
            ax.plot(g.index, g["mean"], ls, color=c, label=lab, marker="o", ms=3)
            ax.fill_between(g.index, g["min"], g["max"], color=c, alpha=.15, lw=0)
            rows += [{"sweep": title, "arm": arm, "overlap": o, **r} for o, r in g.iterrows()]
        ax.set_title(title, fontsize=8); ax.set_xlabel("artifact–ROI overlap r", fontsize=8); ax.tick_params(labelsize=7)
    axes[0].set_ylabel("|Δp| when the artifact\nis inserted (same head)", fontsize=8)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    out = paths.ensure(paths.REPO_ROOT / "paper" / "figures")
    fig.savefig(out / "mechanism.pdf"); fig.savefig(out / "mechanism.png", dpi=200)
    pd.DataFrame(rows).to_csv(paths.ensure(paths.RESULTS / "analysis") / "mechanism.csv", index=False)


if __name__ == "__main__":
    main()
