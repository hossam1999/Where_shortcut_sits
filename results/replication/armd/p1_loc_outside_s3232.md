# Arm d (A20/A21) — P1 location test, tube outside (A15/A16 design), seed 1234

One seed: 95% CIs are image-only bootstrap intervals (10,000 replicates), paired on the same test images; training-seed variance is not included.

Encoder (EfficientNet-B0) training: 8 epochs, best validation AUROC 0.917 at epoch 2. Head: mode pooled, C = 0.003 (edge of the A21 grid: **no**).

| test | n | arm d AUROC | Standard | Δ vs Standard [95% CI] | Masked | Δ vs Masked [95% CI] |
|---|---|---|---|---|---|---|
| corr | 2772 | 0.922 | 0.930 | -0.008 [-0.016, -0.001] | 0.900 | +0.022 [+0.012, +0.032] |
| rev | 2772 | 0.922 | 0.923 | -0.001 [-0.009, +0.006] | 0.900 | +0.022 [+0.012, +0.032] |
| clean | 2772 | 0.922 | 0.927 | -0.005 [-0.012, +0.002] | 0.900 | +0.022 [+0.012, +0.032] |
