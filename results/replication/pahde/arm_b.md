# P2 — arm b: DINOv2 ViT-B/14 @ 518 on lesion-masked images + mask_cmc (ISIC 2019, seed 0)

One seed (A3): 95% CIs are image-only bootstrap intervals (10,000 replicates); training-seed variance is not included. Δ vs Vanilla is paired on the same test images. Paper numbers (Table 1, ResNet-50, ISIC 2019) are descriptive.

## Selection (validation: mean of clean and biased accuracy)

| C | val clean | val biased | mean | chosen |
|---|---|---|---|---|
| 0.01 | 0.699 | 0.696 | 0.697 | **yes** |
| 0.1 | 0.693 | 0.689 | 0.691 |  |
| 1 | 0.686 | 0.679 | 0.683 |  |
| 10 | 0.682 | 0.676 | 0.679 |  |

## Test (scored once)

| test | n | arm [95% CI] | our Vanilla | Δ vs Vanilla [95% CI] | paper Vanilla | paper RRR | paper RR-ClArC |
|---|---|---|---|---|---|---|---|
| clean | 2533 | 0.685 [0.666, 0.703] | 0.839 | -0.155 [-0.174, -0.135] | 0.87 | 0.84 | 0.86 |
| biased | 2533 | 0.681 [0.663, 0.700] | 0.311 | +0.370 [+0.345, +0.397] | 0.28 | 0.51 | 0.76 |
