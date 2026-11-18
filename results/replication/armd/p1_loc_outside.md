# Arm d (A20/A21) — P1 location test, tube outside (A15/A16 design), seed 1234

One seed: 95% CIs are image-only bootstrap intervals (10,000 replicates), paired on the same test images; training-seed variance is not included.

Encoder (EfficientNet-B0) training: 8 epochs, best validation AUROC 0.919 at epoch 2. Head: mode pooled, C = 0.001 (edge of the A21 grid: **lowest**).

| test | n | arm d AUROC | Standard | Δ vs Standard [95% CI] | Masked | Δ vs Masked [95% CI] |
|---|---|---|---|---|---|---|
| corr | 2772 | 0.921 | 0.901 | +0.020 [+0.011, +0.029] | 0.881 | +0.039 [+0.029, +0.050] |
| rev | 2772 | 0.921 | 0.894 | +0.027 [+0.017, +0.036] | 0.881 | +0.039 [+0.029, +0.050] |
| clean | 2772 | 0.921 | 0.898 | +0.023 [+0.014, +0.032] | 0.881 | +0.039 [+0.029, +0.050] |
