# P6: our arms vs our EPVT (A39) — bias 1 — machine 2

Test AUROC on `test_bias_<b>_1`, each arm scored once; crossed seed × image bootstrap vs EPVT's 5 seeds (A36 mixup).
**Interim (bias 1 only): unadjusted; the A39 verdict uses both bias levels with Holm over 8.**

| bias | arm | arm mean | EPVT mean | Δ [95 % CI] | p (unadj.) |
|---|---|---|---|---|---|
| 1 | b_dino518_mask_cmc | 0.782 | 0.696 | +0.086 [+0.071, +0.101] | 0.0001 |
| 1 | c_umte | 0.625 | 0.696 | -0.072 [-0.088, -0.055] | 0.0001 |
| 1 | a_epvt_mask_cmc | 0.758 | 0.696 | +0.061 [+0.044, +0.079] | 0.0001 |
| 1 | d_effb0_mask_cmc | 0.718 | 0.696 | +0.022 [+0.006, +0.038] | 0.0096 |

ERM (5 seeds, reference): bias 1 0.668
