"""Cross-cohort table of reversed / clean AUROC (seed means) for every arm found in the result directories.

  python scripts/make_cross_cohort_table.py      # -> results/CROSS_COHORT.md
"""
from __future__ import annotations

import pandas as pd

from wtss import paths

RUNS = {  # (cohort label, backbone): result dirs merged (first occurrence of an arm wins)
    ("ISIC 2019 hair", "DINOv2"): ["spec_e13/dino518_spec", "spec_e13/dino518_spec_universal", "spec_e13/dino518_spec_ablation",
                                  "spec_e13/dino518_spec_protect_generic", "spec_e13/dino518_spec_pbal", "spec_e13/dino518_spec_u10", "spec_e13/dino518_spec_jtt"],
    ("ISIC 2019 hair", "DermLIP"): ["spec_e13/dermlip224_spec", "spec_e13/dermlip224_spec_universal",
                                   "spec_e13/dermlip224_spec_protect"],
    ("Thyroid US markers", "DINOv2"): ["thyroid/dino518_main", "thyroid/dino518_universal", "thyroid/dino518_ablation",
                                       "thyroid/dino518_protect_generic", "thyroid/dino518_pbal", "thyroid/dino518_u10"],
    ("Thyroid US markers", "MedSigLIP"): ["thyroid/medsiglip448_main", "thyroid/medsiglip448_universal"],
    ("Thyroid US markers", "ConvNeXt"): ["thyroid/convnext384_universal"],
    ("Capsule debris", "ConvNeXt"): ["capsule/convnext384_universal"],
    ("NIH chest drains (in-ROI only)", "RAD-DINO"): ["cxr_drain/raddino518_universal"],
    ("NIH chest drains (in-ROI only)", "DINOv2"): ["cxr_drain/dino518_universal"],
    ("Ovary US markers", "DINOv2"): ["ovary/dino518_main", "ovary/dino518_universal"],
    ("Ovary US markers", "MedSigLIP"): ["ovary/medsiglip448_text"],
    ("Capsule debris", "DINOv2"): ["capsule/dino518_main", "capsule/dino518_universal", "capsule/dino518_protect_generic",
                                   "capsule/dino518_u10"],
    ("Capsule debris", "MedSigLIP"): ["capsule/medsiglip448_main", "capsule/medsiglip448_universal"],
}
# arms fitted with the generic library in *_universal / later runs get a "U-" prefix where the name is shared
GENERIC_DIRS = ("text", "universal", "ablation", "protect_generic", "pbal", "u10", "spec_protect", "jtt")
ORDER = ["erm", "mask", "inpaint", "balanced", "dfr", "leace_paired", "leace_unpaired", "mte", "mte_balanced",
         "U-mte", "U-mte_balanced", "U-mte_aug", "U-mte_protect", "U-mte_protect_balanced", "U-mte_dfr", "U-mask_dfr",
         "U-umte_pbal"]
SHARED = {"erm", "mask", "inpaint", "balanced", "dfr", "leace_paired", "leace_unpaired", "prevcal", "mask_dfr", "jtt",
          "mask_jtt"}


def load(label):
    rows = {}
    for rel in RUNS[label]:
        f = paths.RESULTS / rel / "metrics_per_seed.csv"
        if not f.exists():
            continue
        m = pd.read_csv(f)
        if "trap" not in m:  # single-trap cohorts (e.g. chest drains: in-ROI only)
            m["trap"] = "trapA"
        gen = any(rel.endswith(g) or f"_{g}" in rel for g in GENERIC_DIRS)
        for (trap, method, env), q in m.groupby(["trap", "method", "env"]):
            name = method if (method in SHARED or not gen) else f"U-{method}"
            if method == "mask_dfr":
                name = "U-mask_dfr"
            rows.setdefault((trap, name, env), q.auc.mean())
    return rows


def main():
    out = ["# Cross-cohort results (seed-mean AUROC; reversed test = shortcut-reversed; auto-generated)\n",
           "Arms prefixed U- use the generic, artifact-agnostic overlay library. `rev (clean)`. † = gaming "
           "(correlated-test AUROC < reversed − 0.02: the shortcut is flipped, not removed; excluded by pre-registration).\n",
           "min(rev, corr) is a post-hoc secondary summary that penalises both shortcut use and shortcut flipping.\n",
           "Label needs: erm/mask/U-mte/U-mte_protect: none; balanced/dfr/leace/*_balanced/*_dfr: image-level artifact labels.\n"]
    for label in RUNS:
        rows = load(label)
        if not rows:
            continue
        out.append(f"\n## {label[0]} — {label[1]}\n\n| arm | Trap A rev / corr (clean) | Trap A min(rev,corr) | "
                   f"Trap B rev / corr (clean) | Trap B min(rev,corr) |\n|---|---|---|---|---|")
        arms = [a for a in ORDER if any(k[1] == a for k in rows)]
        for a in arms:
            cell = []
            for t in ("trapA", "trapB"):
                r, c, k = rows.get((t, a, "test_rev")), rows.get((t, a, "clean")), rows.get((t, a, "test_corr"))
                flag = " †" if (r is not None and k is not None and k < r - 0.02) else ""
                cell.append(f"{r:.3f} / {k:.3f} ({c:.3f}){flag}" if r is not None else "–")
                cell.append(f"{min(r, k):.3f}" if r is not None else "–")
            out.append(f"| {a} | " + " | ".join(cell) + " |")
    (paths.RESULTS / "CROSS_COHORT.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
