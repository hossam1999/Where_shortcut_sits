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

## Results (results/ablation_umte/summary.csv; Trap A reversed AUROC, mean over seeds)
| setting | thyroid U-MtE − mask | ovary U-MtE − mask | ovary protect − mask | capsule U-MtE − mask | capsule protect − mask |
|---|---|---|---|---|---|
| energy 0.50 | +0.211 | −0.002 | +0.033 | −0.054 | −0.008 |
| energy 0.90 (main) | +0.222 | +0.019 | +0.043 | −0.115 | −0.010 |
| energy 0.99 | +0.222 | +0.019 | +0.043 | −0.115 | −0.010 |
| k = 1 | +0.071 | +0.012 | +0.007 | +0.001 | +0.001 |
| k = 4 | +0.063 | −0.041 | +0.015 | −0.008 | −0.002 |
| k = 16 | +0.180 | −0.010 | +0.020 | −0.050 | −0.009 |
| k = 64 | +0.222 | +0.019 | +0.043 | −0.115 | −0.010 |
- Built-in check passed: energy 0.90 reproduces the main U-MtE numbers exactly. In these cohorts the 90 % rule reaches
  the rank cap (identical to k = 64).
- **Robust where U-MtE works**: thyroid +0.18 to +0.22 for every setting that erases ≥ 16 directions (or energy ≥ 0.5);
  one to four directions recover only a third of it (+0.06/+0.07) — the overlay shift is multi-dimensional.
- The protected variant never harms capsule by more than 0.010 and helps ovary in every setting (+0.007 to +0.043).
  Unprotected erasure harms capsule more the more directions it removes (debris ≈ fibrin), as the theory's ρ predicts.
