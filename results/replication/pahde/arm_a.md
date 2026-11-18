# P2 — arm a: mask_cmc on our Vanilla ResNet-50d (ISIC 2019, seed 0)

One seed (A3): 95% CIs are image-only bootstrap intervals (10,000 replicates); training-seed variance is not included. Δ vs Vanilla is paired on the same test images. Paper numbers (Table 1, ResNet-50, ISIC 2019) are descriptive; the head-to-head against RR-ClArC waits for the RR-ClArC decision (STATUS 2026-11-18).

## Selection (validation: mean of clean and biased accuracy)

| C | val clean | val biased | mean | chosen |
|---|---|---|---|---|
| 0.01 | 0.835 | 0.710 | 0.772 | **yes** |
| 0.1 | 0.850 | 0.655 | 0.752 |  |
| 1 | 0.851 | 0.603 | 0.727 |  |
| 10 | 0.849 | 0.593 | 0.721 |  |

## Test (scored once)

| test | n | arm a [95% CI] | our Vanilla | Δ vs Vanilla [95% CI] | paper Vanilla | paper RRR | paper RR-ClArC |
|---|---|---|---|---|---|---|---|
| clean | 2533 | 0.822 [0.807, 0.836] | 0.839 | -0.017 [-0.029, -0.005] | 0.87 | 0.84 | 0.86 |
| biased | 2533 | 0.691 [0.673, 0.709] | 0.311 | +0.380 [+0.360, +0.401] | 0.28 | 0.51 | 0.76 |
