# R9 — dose-response over the real overlap r (DINOv2 ViT-B/14 @518)

Environment: {"commit": "ac90660", "host": "074495c7c1a5", "WTSS_BOOTSTRAP": "crossed", "gpu": "NVIDIA RTX PRO 4000 Blackwell"}

Slope of g_b = rev(mask) - rev(ERM) on the bin mean r; crossed bootstrap; Holm over tested cohorts. H9: slope < 0.

| cohort | status | bins | slope_ci | p_holm | H9_supported | spearman_x_gain | zero_crossing_r |
|---|---|---|---|---|---|---|---|
| ISIC hair | tested | b1, b2, b3, b4, b5 | -0.235 [-0.276, -0.194] | 0.000 | True | -1.000 | 0.401 |
| Thyroid calipers | tested | b1, b2, b3, b4, b5 | -0.250 [-0.303, -0.197] | 0.000 | True | -1.000 | nan |
| Capsule debris | tested | b1, b2, b3, b4, b5 | -0.489 [-0.539, -0.438] | 0.000 | True | -1.000 | nan |
| Ovary calipers | tested | b1, b2b3b4, b5 | -0.205 [-0.300, -0.109] | 0.000 | True | -1.000 | nan |

Per-bin gains:

| cohort | bin | x_mean_r | gain_ci |
|---|---|---|---|
| ISIC hair | b1 | 0.026 | +0.124 [+0.085, +0.162] |
| ISIC hair | b2 | 0.194 | +0.065 [+0.029, +0.098] |
| ISIC hair | b3 | 0.393 | +0.001 [-0.034, +0.035] |
| ISIC hair | b4 | 0.614 | -0.025 [-0.057, +0.006] |
| ISIC hair | b5 | 0.888 | -0.086 [-0.119, -0.056] |
| Thyroid calipers | b1 | 0.008 | +0.336 [+0.295, +0.377] |
| Thyroid calipers | b2 | 0.204 | +0.264 [+0.180, +0.349] |
| Thyroid calipers | b3 | 0.409 | +0.208 [+0.136, +0.283] |
| Thyroid calipers | b4 | 0.637 | +0.136 [+0.099, +0.172] |
| Thyroid calipers | b5 | 0.913 | +0.116 [+0.091, +0.141] |
| Capsule debris | b1 | 0.041 | +0.465 [+0.422, +0.508] |
| Capsule debris | b2 | 0.188 | +0.300 [+0.259, +0.343] |
| Capsule debris | b3 | 0.390 | +0.147 [+0.111, +0.182] |
| Capsule debris | b4 | 0.621 | +0.096 [+0.061, +0.130] |
| Capsule debris | b5 | 0.881 | +0.032 [-0.004, +0.069] |
| Ovary calipers | b1 | 0.004 | +0.293 [+0.216, +0.369] |
| Ovary calipers | b2b3b4 | 0.531 | +0.179 [+0.091, +0.266] |
| Ovary calipers | b5 | 0.909 | +0.109 [+0.049, +0.170] |

- ISIC hair: merges: none
- Thyroid calipers: merges: none
- Capsule debris: merges: none
- Ovary calipers: merges: b2 failed the gate -> merged with b3 into b2b3; b2b3 failed the gate -> merged with b4 into b2b3b4
