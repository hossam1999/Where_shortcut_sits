# Arm d (A20/A21) — P1 rsna (their original test; secondary, no verdict, A14), seed 1234

One seed: 95% CIs are image-only bootstrap intervals (10,000 replicates), paired on the same test images; training-seed variance is not included.

Encoder (EfficientNet-B0) training: 10 epochs, best validation AUROC 0.322 at epoch 4. Head: mode per_class, C = 0.001 (edge of the A21 grid: **lowest**).

| test | n | arm d AUROC | ActDiff (theirs) | Δ vs ActDiff (theirs) [95% CI] | Standard (theirs) | Δ vs Standard (theirs) [95% CI] |
|---|---|---|---|---|---|---|
| test | 1352 | 0.498 | 0.708 | -0.210 [-0.253, -0.166] | 0.186 | +0.313 [+0.281, +0.344] |
