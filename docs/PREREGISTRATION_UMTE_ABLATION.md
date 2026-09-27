# Pre-registration — U-MtE design ablation (erasure rank)

Committed before running. The main U-MtE fixes the rank rule a priori (smallest k whose held-out paired-difference
energy falls below 10 %, i.e. energy 0.90, max 64). Reviewers will ask how sensitive the results are to that choice.

## Design
Cached DINOv2 ViT-B/14 @518 features, generic overlays, Trap A (in-ROI) of thyroid, ovary and capsule; same seeds,
folds and heads as the main runs. Settings: energy ∈ {0.50, 0.70, 0.80, 0.90 (main), 0.95, 0.99} and fixed
k ∈ {1, 4, 16, 64}. Arms: mask, U-MtE, U-MtE_protect (`scripts/analysis/umte_ablation.py`). The energy-0.90 setting
must reproduce the main U-MtE numbers exactly (built-in check).

**Amendment (before any ovary/capsule result):** to fit the time budget the grid was trimmed to energy ∈ {0.50,
0.90, 0.99} and k ∈ {1, 4, 16, 64}; thyroid e0.70/e0.80/e0.95 (already run) are kept in the results folder.

## Reported
Reversed, correlated and clean AUROC per setting. Descriptive; the question is whether U-MtE − mask keeps its sign
over a wide range of ranks (robust) or only near the chosen one (fragile). No setting will replace the main one.
