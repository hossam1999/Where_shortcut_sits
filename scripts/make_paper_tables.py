"""Generate every paper table (LaTeX + Markdown) from results/. Numbers are never typed by hand.

Writes paper/tables/*.tex and results/PAPER_TABLES.md. Tables whose inputs are missing are skipped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wtss import paths

R = paths.RESULTS
T = paths.ensure(paths.REPO_ROOT / "paper" / "tables")
MD: list[str] = []

ARM_NAMES = {"erm": "ERM", "mask": "ROI masking", "inpaint": "Inpainting (oracle)", "balanced": "Group-balanced",
             "groupdro": "GroupDRO", "dfr": "DFR", "leace": "LEACE (paired, oracle renders)",
             "leace_paired": "LEACE (paired, removal)", "leace_unpaired": "LEACE (unpaired labels)",
             "inpaint_consistency": "Repair consistency", "inpaint_consistency_lam0": "Repair consistency, λ=0",
             "i2e": "Insert-then-erase (artifact templates)", "i2e_balanced": "Insert-then-erase + balanced (artifact templates)",
             "i2e_rank1": "Insert-then-erase rank-1 (ablation)",
             "insert_aug": "Insertion augmentation", "prevcal": "Prevalence calibration",
             "dilate0": "Mask +0 px", "dilate10": "Mask +10 px", "dilate25": "Mask +25 px", "dilate50": "Mask +50 px"}
SUPERVISION = {"erm": "—", "mask": "ROI mask (train+test)", "inpaint": "artifact pixels (train+test)",
               "balanced": "image-level A (train)", "groupdro": "image-level A (train)", "dfr": "image-level A (val)",
               "leace": "paired renders (train)", "leace_paired": "artifact pixels (train)",
               "leace_unpaired": "image-level A (train)", "i2e": "artifact templates", "i2e_balanced": "artifact templates + A (train)",
               "i2e_rank1": "artifact templates", "insert_aug": "artifact templates", "prevcal": "image-level A (train+test)",
               "inpaint_consistency": "artifact pixels (train)", "inpaint_consistency_lam0": "artifact pixels (train)"}


def ci(p, lo, hi):
    return f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def emit(name: str, df: pd.DataFrame, caption: str):
    tex = df.to_latex(index=False, escape=True, caption=caption.replace("%", r"\%").replace("_", r"\_"),
                      label=f"tab:{name}")
    # wide tables: full width, small font, scaled to the text width
    tex = (tex.replace("\\begin{table}", "\\begin{table*}\\scriptsize").replace("\\end{table}", "\\end{table*}")
              .replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}").replace("\\end{tabular}", "\\end{tabular}}"))
    (T / f"{name}.tex").write_text(tex)
    MD.append(f"### {caption}\n\n{df.to_markdown(index=False)}\n")


def synthetic_table(tag_dir: Path, name: str, caption: str, arms, overlaps=(0.0, 0.5, 1.0)):
    b = pd.read_csv(tag_dir / "bootstrap_vs_erm.csv")
    s = pd.read_csv(tag_dir / "summary_auc.csv")
    rows = []
    for arm in ["erm", *arms]:
        r = {"Method": ARM_NAMES.get(arm, arm), "Needs": SUPERVISION.get(arm, "")}
        for ov in overlaps:
            q = s[(s.method == arm) & np.isclose(s.overlap, ov)].set_index("env").auc
            if q.empty:
                continue
            r[f"clean@{int(ov * 100)}"] = f"{q.get('clean', np.nan):.3f}"
            r[f"rev@{int(ov * 100)}"] = f"{q.get('test_rev', np.nan):.3f}"
            if arm != "erm":
                x = b[(b.method_a == arm) & np.isclose(b.overlap, ov)]
                r[f"Δrev@{int(ov * 100)}"] = ci(x.seed_delta_mean.iloc[0], x.ci95_lo.iloc[0], x.ci95_hi.iloc[0]) if len(x) else ""
        rows.append(r)
    emit(name, pd.DataFrame(rows).fillna(""), caption)


def merged_synthetic(dirs):
    """Concatenate main + proposed runs of the same cohort/backbone (same ERM by construction)."""
    out = Path("/tmp") / ("merged_" + "_".join(d.name for d in dirs))
    out.mkdir(exist_ok=True)
    b = pd.concat([pd.read_csv(d / "bootstrap_vs_erm.csv") for d in dirs if (d / "bootstrap_vs_erm.csv").exists()])
    s = pd.concat([pd.read_csv(d / "summary_auc.csv") for d in dirs if (d / "summary_auc.csv").exists()])
    b.drop_duplicates(["method_a", "overlap"]).to_csv(out / "bootstrap_vs_erm.csv", index=False)
    s.drop_duplicates(["method", "overlap", "env"]).to_csv(out / "summary_auc.csv", index=False)
    return out


def trap_table(d: Path, name: str, caption: str, traps=("trapA", "trapB")):
    b = pd.read_csv(d / "bootstrap_vs_erm.csv")
    s = pd.read_csv(d / "summary_auc.csv")
    rows = []
    arms = ["erm"] + [a for a in ARM_NAMES if a in set(s.method) and a != "erm"]
    for arm in arms:
        r = {"Method": ARM_NAMES.get(arm, arm), "Needs": SUPERVISION.get(arm, "")}
        for t in traps:
            q = s[(s.trap == t) & (s.method == arm)].set_index("env").auc
            lab = {"trapA": "A", "trapB": "B", "drain": "D"}[t]
            r[f"{lab} clean"] = f"{q.get('clean', np.nan):.3f}"
            r[f"{lab} corr"] = f"{q.get('test_corr', np.nan):.3f}"
            r[f"{lab} rev"] = f"{q.get('test_rev', np.nan):.3f}"
            if arm != "erm":
                x = b[(b.trap == t) & (b.arm == arm) & (b.env == "test_rev") & (b.source == "all")]
                r[f"{lab} Δrev"] = ci(x.seed_delta_mean.iloc[0], x.ci95_lo.iloc[0], x.ci95_hi.iloc[0]) if len(x) else ""
                x = b[(b.trap == t) & (b.arm == arm) & (b.env == "clean") & (b.source == "all")]
                r[f"{lab} Δclean"] = f"{x.seed_delta_mean.iloc[0]:+.3f}" if len(x) else ""
        rows.append(r)
    emit(name, pd.DataFrame(rows).fillna(""), caption)
    if (d / "crossover_B_minus_A.csv").exists():
        c = pd.read_csv(d / "crossover_B_minus_A.csv")
        emit(name + "_crossover", pd.DataFrame({"Method": [ARM_NAMES.get(a, a) for a in c.arm],
                                                "Crossover (B − A)": [ci(*v) for v in c[["seed_delta_mean", "ci95_lo", "ci95_hi"]].to_numpy()]}),
             caption + " — location crossover [Δ vs ERM]_TrapB − [Δ vs ERM]_TrapA")


def main():
    S = R / "synthetic"
    d = S / "isic2018"
    main_dirs = [d / "dino518_ruler_fixed_corr_main", d / "dino518_ruler_fixed_corr_proposed"]
    if all((x / "bootstrap_vs_erm.csv").exists() for x in main_dirs):
        m = merged_synthetic(main_dirs)
        synthetic_table(m, "synthetic_isic_dino518", "Synthetic ruler, ISIC 2018, DINOv2 ViT-B/14 @518: AUROC and reversed-test Δ vs ERM [95% CI]",
                        ["mask", "inpaint", "balanced", "dfr", "leace", "inpaint_consistency", "i2e", "i2e_balanced", "i2e_rank1", "insert_aug"])
        li = pd.read_csv(main_dirs[0] / "location_interaction.csv")
        emit("synthetic_interaction", pd.DataFrame({"Method": [ARM_NAMES.get(a, a) for a in li.method],
             "Interaction (0% − 100%)": [ci(*v) for v in li[["seed_delta_mean", "ci95_lo", "ci95_hi"]].to_numpy()]}),
             "Location interaction [Δ vs ERM]_0% − [Δ vs ERM]_100% (DINOv2 @518)")
    for bb, lab in [("dino224", "DINOv2 @224"), ("dermlip224", "DermLIP/PanDerm @224")]:
        x = d / f"{bb}_ruler_fixed_corr_main"
        if (x / "bootstrap_vs_erm.csv").exists():
            synthetic_table(x, f"synthetic_isic_{bb}", f"Synthetic ruler, ISIC 2018, {lab}", ["mask", "balanced", "dfr", "leace", "i2e", "i2e_balanced"])
    for bb, lab in [("dino518", "DINOv2 @518"), ("raddino518", "RAD-DINO @518")]:
        x = S / "nih_ptx" / f"{bb}_tube_corr_main"
        if (x / "bootstrap_vs_erm.csv").exists():
            synthetic_table(x, f"synthetic_cxr_{bb}", f"Synthetic tube, NIH ChestX-ray14 pneumothorax, {lab}",
                            ["mask", "inpaint", "balanced", "dfr", "leace", "i2e", "i2e_balanced", "i2e_rank1", "insert_aug"])
    for bb, lab in [("dino518", "DINOv2 @518"), ("dermlip224", "DermLIP/PanDerm @224")]:
        x = R / "real" / "isic2019" / f"{bb}_main"
        if (x / "bootstrap_vs_erm.csv").exists():
            trap_table(x, f"traps_isic2019_{bb}", f"ISIC 2019 real hair, {lab}: Trap A (in-lesion) and Trap B (extra-lesional)")
    for bb, lab in [("raddino518", "RAD-DINO @518"), ("dino518", "DINOv2 @518")]:
        x = R / "real" / "nih_drain" / f"{bb}_main"
        if (x / "bootstrap_vs_erm.csv").exists():
            trap_table(x, f"traps_drain_{bb}", f"NIH ChestX-ray14 real chest drains (pneumothorax), {lab}", traps=("drain",))
    (R / "PAPER_TABLES.md").write_text("# Paper tables (generated)\n\n" + "\n".join(MD))
    print(f"wrote {len(MD)} tables")


if __name__ == "__main__":
    main()
