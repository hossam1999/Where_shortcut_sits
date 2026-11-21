# Arm d (A20/A21) — P1 location test, tube inside (A15/A16 design), seed 1234

One seed: 95% CIs are image-only bootstrap intervals (10,000 replicates), paired on the same test images; training-seed variance is not included.

Encoder (EfficientNet-B0) training: 14 epochs, best validation AUROC 0.638 at epoch 8. Head: mode per_class, C = 0.001 (edge of the A21 grid: **lowest**).

| test | n | arm d AUROC | Standard | Δ vs Standard [95% CI] | Masked | Δ vs Masked [95% CI] |
|---|---|---|---|---|---|---|
| corr | 2772 | 0.966 | 0.887 | +0.080 [+0.069, +0.091] | 0.846 | +0.121 [+0.107, +0.134] |
| rev | 2772 | 0.746 | 0.879 | -0.133 [-0.148, -0.117] | 0.836 | -0.090 [-0.107, -0.073] |
| clean | 2772 | 0.911 | 0.884 | +0.026 [+0.016, +0.037] | 0.844 | +0.067 [+0.054, +0.080] |
