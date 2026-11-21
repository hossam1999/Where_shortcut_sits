# Arm d (A20/A21) — P1 location test, tube outside (A15/A16 design), seed 1234

One seed: 95% CIs are image-only bootstrap intervals (10,000 replicates), paired on the same test images; training-seed variance is not included.

Encoder (EfficientNet-B0) training: 7 epochs, best validation AUROC 0.924 at epoch 1. Head: mode pooled, C = 0.001 (edge of the A21 grid: **lowest**).

| test | n | arm d AUROC | Standard | Δ vs Standard [95% CI] | Masked | Δ vs Masked [95% CI] |
|---|---|---|---|---|---|---|
| corr | 2772 | 0.928 | 0.923 | +0.004 [-0.003, +0.012] | 0.907 | +0.020 [+0.011, +0.030] |
| rev | 2772 | 0.928 | 0.923 | +0.005 [-0.002, +0.013] | 0.907 | +0.020 [+0.011, +0.030] |
| clean | 2772 | 0.928 | 0.923 | +0.005 [-0.002, +0.013] | 0.907 | +0.020 [+0.011, +0.030] |
