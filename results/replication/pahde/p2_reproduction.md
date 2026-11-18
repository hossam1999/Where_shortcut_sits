# P2 (Pahde et al. 2025) — RR-ClArC reproduction, ISIC 2019, seed 0 — section-3 runs, their code path: λ ∈ {10, 50}, effective lr 1e-5 from the loaded optimizer state

One seed (A3): 95% CIs are image-only bootstrap intervals (10,000 replicates); training-seed variance is not included.

## Selection on P2's validation split (section 3: mean of clean and biased accuracy)

| λ | layer | val clean | val biased | mean | chosen |
|---|---|---|---|---|---|
| 10 | last_conv | 0.844 | 0.351 | 0.597 | **yes** |
| 10 | identity_1 | 0.850 | 0.337 | 0.593 |  |
| 50 | identity_1 | 0.850 | 0.332 | 0.591 |  |
| 50 | last_conv | 0.844 | 0.324 | 0.584 |  |
| 10 | identity_2 | 0.852 | 0.296 | 0.574 |  |
| 50 | identity_2 | 0.850 | 0.273 | 0.562 |  |

## Test (scored once, chosen model and Vanilla)

| method | test | n | accuracy [95% CI] |
|---|---|---|---|
| Vanilla (ours, seed 0) | clean | 2533 | 0.839 [0.825, 0.853] |
| Vanilla (ours, seed 0) | biased | 2533 | 0.311 [0.293, 0.329] |
| RR-ClArC (chosen: resnet50d_RRClarc_lamb10_adam_lr0.001_last_conv) | clean | 2533 | 0.840 [0.826, 0.854] |
| RR-ClArC (chosen: resnet50d_RRClarc_lamb10_adam_lr0.001_last_conv) | biased | 2533 | 0.338 [0.320, 0.356] |

Target (section 1): RR-ClArC biased-test accuracy 0.76 (paper Table 1), reproduced if |ours − 0.76| ≤ 0.03. Ours: 0.338 → **not reproduced**. Paper clean accuracy 0.86 (descriptive).
