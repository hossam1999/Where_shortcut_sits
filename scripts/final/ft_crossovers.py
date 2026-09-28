"""Crossover (Trap B - Trap A of mask - ERM, reversed test) of every fine-tuned run, crossed bootstrap.
  -> <WTSS_RESULTS>/finetune/ft_crossovers.csv"""
from __future__ import annotations

import os

import pandas as pd

os.environ["WTSS_BOOTSTRAP"] = "crossed"
from wtss import paths  # noqa: E402
from wtss.stats import difference_of_deltas  # noqa: E402

RUNS = {"Dermoscopy hair, ResNet-50": "finetune/isic/resnet50", "Thyroid, ResNet-50": "finetune/thyroid/resnet50",
        "Thyroid, ViT-S": "finetune/thyroid/vit_small_patch16_224.augreg_in21k_ft_in1k",
        "Capsule, ResNet-50": "finetune/capsule/resnet50", "Ovary, ResNet-50 (15 clusters)": "finetune/ovary/resnet50_power"}
rows = []
for lab, rel in RUNS.items():
    p = pd.read_csv(paths.RESULTS / rel / "predictions.csv.gz")
    x = difference_of_deltas(p[p.trap == "trapB"], p[p.trap == "trapA"], "mask", "erm", "test_rev", 10000, 11)
    auc = p[(p.env == "clean")].groupby(["trap", "method"]).apply(lambda q: None, include_groups=False)
    rows.append({"run": lab, "dir": rel, "n_clusters": int(p.seed.nunique() * (p.fold.nunique() if "fold" in p else 1)),
                 **{k: x[k] for k in ("seed_delta_mean", "ci95_lo", "ci95_hi", "p_boot_two_sided") if k in x}})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(paths.RESULTS / "finetune" / "ft_crossovers.csv", index=False)
