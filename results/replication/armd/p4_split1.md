# Arm d (A20/A21) — P4 split 1, trap test (flags used: dark_corner, hair, gel_border, gel_bubble, ruler, ink, vasc), seed 1234

One seed: 95% CIs are image-only bootstrap intervals (10,000 replicates), paired on the same test images; training-seed variance is not included.

Encoder (EfficientNet-B0) training: 29 epochs, best validation AUROC 1.000 at epoch 23. Head: mode pooled, C = 0.001 (edge of the A21 grid: **lowest**).

| test | n | arm d AUROC | Binary-ERM | Δ vs Binary-ERM [95% CI] |
|---|---|---|---|---|
| test | 772 | 0.666 | 0.355 | +0.311 [+0.258, +0.363] |

Note: in Bissoto's trap sets all 259 validation images are also in training (A13), so validation AUROC is near 1 and early stopping says little.
