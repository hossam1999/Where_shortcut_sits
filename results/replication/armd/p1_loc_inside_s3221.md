# Arm d (A20/A21) — P1 location test, tube inside (A15/A16 design), seed 1234

One seed: 95% CIs are image-only bootstrap intervals (10,000 replicates), paired on the same test images; training-seed variance is not included.

Encoder (EfficientNet-B0) training: 9 epochs, best validation AUROC 0.596 at epoch 3. Head: mode pooled, C = 0.001 (edge of the A21 grid: **lowest**).

| test | n | arm d AUROC | Standard | Δ vs Standard [95% CI] | Masked | Δ vs Masked [95% CI] |
|---|---|---|---|---|---|---|
| corr | 2772 | 0.966 | 0.971 | -0.005 [-0.010, +0.000] | 0.957 | +0.009 [+0.002, +0.016] |
| rev | 2772 | 0.780 | 0.624 | +0.157 [+0.141, +0.172] | 0.540 | +0.240 [+0.222, +0.259] |
| clean | 2772 | 0.914 | 0.903 | +0.010 [+0.000, +0.020] | 0.853 | +0.060 [+0.047, +0.074] |
