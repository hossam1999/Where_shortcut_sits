# Primary claims (Holm family-wise over all rows; BH for reference)

Supported after Holm (correct direction): 12/16

| family | cohort | estimate [95 % CI] | p | p (Holm) | p (BH) |
|---|---|---|---|---|---|
| P1 location crossover | ISIC hair (DINOv2) | +0.145 [+0.105, +0.181] | 0.0001 | 0.0016 ✓ | 0.0001 |
| P1 location crossover | Thyroid (DINOv2) | +0.230 [+0.204, +0.255] | 0.0001 | 0.0016 ✓ | 0.0001 |
| P1 location crossover | Capsule (DINOv2) | +0.368 [+0.340, +0.397] | 0.0001 | 0.0016 ✓ | 0.0001 |
| P1 location crossover | Ovary (DINOv2) | +0.170 [+0.116, +0.224] | 0.0001 | 0.0016 ✓ | 0.0001 |
| P2 U-MtE (protected) > mask | ISIC hair | +0.041 [+0.030, +0.054] | 0.0001 | 0.0016 ✓ | 0.0001 |
| P2 U-MtE (protected) > mask | Thyroid | +0.217 [+0.174, +0.257] | 0.0001 | 0.0016 ✓ | 0.0001 |
| P2 U-MtE (protected) > mask | Capsule | -0.010 [-0.021, -0.001] | 0.0300 | 0.0900 | 0.0343 |
| P2 U-MtE (protected) > mask | Ovary | +0.043 [+0.026, +0.061] | 0.0001 | 0.0016 ✓ | 0.0001 |
| P3 erase > augment | ISIC hair | +0.045 [+0.037, +0.054] | 0.0001 | 0.0016 ✓ | 0.0001 |
| P3 erase > augment | Thyroid | +0.207 [+0.182, +0.235] | 0.0001 | 0.0016 ✓ | 0.0001 |
| P3 erase > augment | Capsule | -0.130 [-0.147, -0.110] | 0.0001 | 0.0016 ✗ (opposite direction) | 0.0001 |
| P3 erase > augment | Ovary | +0.035 [+0.010, +0.059] | 0.0056 | 0.0224 ✓ | 0.0069 |
| P4 U-MtE+balanced > balanced | ISIC hair | -0.013 [-0.041, +0.018] | 0.4230 | 0.4230 | 0.4230 |
| P4 U-MtE+balanced > balanced | Thyroid | +0.112 [+0.090, +0.134] | 0.0001 | 0.0016 ✓ | 0.0001 |
| P4 U-MtE+balanced > balanced | Capsule | +0.082 [+0.064, +0.100] | 0.0001 | 0.0016 ✓ | 0.0001 |
| P4 U-MtE+balanced > balanced | Ovary | +0.028 [+0.000, +0.057] | 0.0494 | 0.0988 | 0.0527 |
