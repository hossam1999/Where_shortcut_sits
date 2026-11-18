# Arm d (A20/A21) — P3 90/10 inside variant (A6), EfficientNet-B0 on lung-masked images + mask_cmc, seed 1234

One seed: test values are point estimates on the rebuilt P3 test (A9/A10; both variants); the paired comparison with the ISNet 90/10 inside run is added when that run is scored.

Encoder training: 20 epochs, best validation macro-F1 0.873 at epoch 14. Head: mode pooled, C = 0.03 (edge of the A21 grid: **no**).

| test | variant | n | macro-F1 |
|---|---|---|---|
| biased | primary | 4190 | 0.997 |
| biased | size-matched | 3180 | 0.996 |
| standard | primary | 4190 | 0.410 |
| standard | size-matched | 3180 | 0.404 |
| deceiving | primary | 4190 | 0.010 |
| deceiving | size-matched | 3180 | 0.008 |
