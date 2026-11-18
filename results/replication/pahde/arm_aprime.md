# P2 — arm a' (A19): mask_cmc on the selected A18 RR-ClArC network (resnet50d_RRClarc_lamb5e9_adam_lr0.0001_identity_2), ISIC 2019, seed 0

Identical to arm a except for the network (unmasked input, penultimate features, one δ, class-balanced multinomial refit, C on validation). One seed (A3): 95% CIs are image-only bootstrap intervals (10,000 replicates), paired on the same test images; training-seed variance is not included.

## Selection (validation: mean of clean and biased accuracy)

| C | val clean | val biased | mean | chosen |
|---|---|---|---|---|
| 0.01 | 0.833 | 0.669 | 0.751 | **yes** |
| 0.1 | 0.849 | 0.618 | 0.733 |  |
| 1 | 0.851 | 0.573 | 0.712 |  |
| 10 | 0.849 | 0.557 | 0.703 |  |

## Test (scored once): head-to-head vs RR-ClArC

| test | n | arm a' [95% CI] | RR-ClArC (A18) | Δ vs RR-ClArC [95% CI] | our Vanilla | Δ vs Vanilla [95% CI] |
|---|---|---|---|---|---|---|
| clean | 2533 | 0.818 [0.803, 0.833] | 0.839 | -0.021 [-0.033, -0.009] | 0.839 | -0.021 [-0.033, -0.009] |
| biased | 2533 | 0.667 [0.648, 0.685] | 0.524 | +0.143 [+0.126, +0.161] | 0.311 | +0.356 [+0.335, +0.377] |

Note: as for arm a, a selected C at the edge of the registered grid {0.01, 0.1, 1, 10} is reported as is; any wider grid is post hoc only (sensitivity, no verdict).
