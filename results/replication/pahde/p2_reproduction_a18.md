# P2 (Pahde et al. 2025) — RR-ClArC reproduction, ISIC 2019, seed 0 — A18 PRIMARY: the paper's settings, λ = 5·10⁹ (Tab. 12), lr 1e-4 (p. 31), Vanilla weights only, fresh Adam

One seed (A3): 95% CIs are image-only bootstrap intervals (10,000 replicates); training-seed variance is not included.

## Selection on P2's validation split (section 3: mean of clean and biased accuracy)

| λ | layer | val clean | val biased | mean | chosen |
|---|---|---|---|---|---|
| 5e+09 | identity_2 | 0.843 | 0.530 | 0.687 | **yes** |
| 5e+09 | last_conv | 0.842 | 0.376 | 0.609 |  |
| 5e+09 | identity_1 | 0.847 | 0.353 | 0.600 |  |

## Test (scored once, chosen model and Vanilla)

| method | test | n | accuracy [95% CI] |
|---|---|---|---|
| Vanilla (ours, seed 0) | clean | 2533 | 0.839 [0.825, 0.853] |
| Vanilla (ours, seed 0) | biased | 2533 | 0.311 [0.293, 0.329] |
| RR-ClArC (chosen: resnet50d_RRClarc_lamb5e9_adam_lr0.0001_identity_2) | clean | 2533 | 0.839 [0.825, 0.853] |
| RR-ClArC (chosen: resnet50d_RRClarc_lamb5e9_adam_lr0.0001_identity_2) | biased | 2533 | 0.524 [0.505, 0.543] |

Target (section 1): RR-ClArC biased-test accuracy 0.76 (paper Table 1), reproduced if |ours − 0.76| ≤ 0.03. Ours: 0.524 → **not reproduced**. Paper clean accuracy 0.86 (descriptive).
