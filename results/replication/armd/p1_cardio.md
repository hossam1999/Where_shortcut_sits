# Arm d (A20/A21) — P1 xray (their original test; secondary, no verdict, A14), seed 1234

One seed: 95% CIs are image-only bootstrap intervals (10,000 replicates), paired on the same test images; training-seed variance is not included.

Encoder (EfficientNet-B0) training: 17 epochs, best validation AUROC 0.659 at epoch 11. Head: mode pooled, C = 0.001 (edge of the A21 grid: **lowest**).

| test | n | arm d AUROC | Masked (theirs, A5) | Δ vs Masked (theirs, A5) [95% CI] | Standard (theirs) | Δ vs Standard (theirs) [95% CI] |
|---|---|---|---|---|---|---|
| test | 2776 | 0.718 | 0.829 | -0.111 [-0.128, -0.095] | 0.627 | +0.091 [+0.075, +0.107] |
