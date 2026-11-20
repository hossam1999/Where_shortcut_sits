# Arm d (A20/A21) — P3 90/10 inside variant (A6), EfficientNet-B0 on lung-masked images + mask_cmc, seed 1234

One seed: test values are point estimates on the rebuilt P3 test (A9/A10; both variants); the paired comparison with the ISNet 90/10 inside run is added when that run is scored.

Encoder training: 19 epochs, best validation macro-F1 0.982 at epoch 13. Head: mode pooled, C = 0.03 (edge of the A21 grid: **no**).

| test | variant | n | macro-F1 |
|---|---|---|---|
| biased | primary | 4190 | 0.993 |
| biased | size-matched | 3180 | 0.996 |
| standard | primary | 4190 | 0.406 |
| standard | size-matched | 3180 | 0.405 |
| deceiving | primary | 4190 | 0.002 |
| deceiving | size-matched | 3180 | 0.001 |
| A24 biased | cohort validation | 3483 | 0.982 |
| A24 standard | cohort validation | 3483 | 0.813 |
| A24 deceiving | cohort validation | 3483 | 0.033 |
