# Arm d (A20/A21) — P1 location test, tube inside (A15/A16 design), seed 1234

One seed: 95% CIs are image-only bootstrap intervals (10,000 replicates), paired on the same test images; training-seed variance is not included.

Encoder (EfficientNet-B0) training: 16 epochs, best validation AUROC 0.685 at epoch 10. Head: mode pooled, C = 0.001 (edge of the A21 grid: **lowest**).

| test | n | arm d AUROC | Standard | Δ vs Standard [95% CI] | Masked | Δ vs Masked [95% CI] |
|---|---|---|---|---|---|---|
| corr | 2772 | 0.961 | 0.943 | +0.018 [+0.010, +0.026] | 0.935 | +0.026 [+0.018, +0.035] |
| rev | 2772 | 0.735 | 0.514 | +0.222 [+0.202, +0.241] | 0.469 | +0.266 [+0.245, +0.287] |
| clean | 2772 | 0.906 | 0.866 | +0.040 [+0.028, +0.052] | 0.837 | +0.069 [+0.055, +0.083] |
