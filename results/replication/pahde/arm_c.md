# P2 — arm c: U-MtE (ROI = lesion, k = 64), label-free (ISIC 2019, seed 0)

One seed (A3): 95% CIs are image-only bootstrap intervals (10,000 replicates); training-seed variance is not included. Δ vs Vanilla is paired on the same test images. Paper numbers (Table 1, ResNet-50, ISIC 2019) are descriptive.

## Selection (validation: mean of clean and biased accuracy)

| C | val clean | val biased | mean | chosen |
|---|---|---|---|---|
| 0.01 | 0.699 | 0.689 | 0.694 | **yes** |
| 0.1 | 0.692 | 0.687 | 0.690 |  |
| 1 | 0.686 | 0.679 | 0.682 |  |
| 10 | 0.683 | 0.675 | 0.679 |  |

## Test (scored once)

| test | n | arm [95% CI] | our Vanilla | Δ vs Vanilla [95% CI] | paper Vanilla | paper RRR | paper RR-ClArC |
|---|---|---|---|---|---|---|---|
| clean | 2533 | 0.680 [0.662, 0.698] | 0.839 | -0.159 [-0.179, -0.139] | 0.87 | 0.84 | 0.86 |
| biased | 2533 | 0.678 [0.660, 0.697] | 0.311 | +0.367 [+0.341, +0.394] | 0.28 | 0.51 | 0.76 |
