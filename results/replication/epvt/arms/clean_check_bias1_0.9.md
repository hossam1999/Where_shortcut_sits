# P6 reported clean check (A42; NOT part of the A39 verdict) — AUROC on test_bias_0_1

| bias (training) | arm | arm clean AUROC | EPVT clean AUROC (5 seeds) | Δ | within EPVT − 0.01 |
|---|---|---|---|---|---|
| 1 | b_dino518_mask_cmc | 0.889 | 0.910 | -0.021 | no |
| 1 | c_umte | 0.870 | 0.910 | -0.040 | no |
| 1 | a_epvt_mask_cmc | 0.885 | 0.910 | -0.025 | no |
| 1 | d_effb0_mask_cmc | 0.924 | 0.910 | +0.014 | yes |
| 0.9 | b_dino518_mask_cmc | 0.889 | 0.929 | -0.040 | no |
| 0.9 | c_umte | 0.877 | 0.929 | -0.051 | no |
| 0.9 | a_epvt_mask_cmc | 0.886 | 0.929 | -0.043 | no |
| 0.9 | d_effb0_mask_cmc | 0.932 | 0.929 | +0.004 | yes |
