# P6: our arms vs our EPVT (A39) — bias 1, 0.9 — machine 2

Test AUROC on `test_bias_<b>_1`, each arm scored once; crossed seed × image bootstrap vs EPVT's 5 seeds (A36 mixup).
Holm over the 8 A39 comparisons; 'beaten' = section 2.

| bias | arm | arm mean | EPVT mean | Δ [95 % CI] | p (unadj.) | p Holm | beaten |
|---|---|---|---|---|---|---|---|
| 1 | b_dino518_mask_cmc | 0.782 | 0.696 | +0.086 [+0.071, +0.101] | 0.0001 | 0.0008 | yes |
| 1 | c_umte | 0.625 | 0.696 | -0.072 [-0.088, -0.055] | 0.0001 | 0.0008 | no |
| 1 | a_epvt_mask_cmc | 0.758 | 0.696 | +0.061 [+0.044, +0.079] | 0.0001 | 0.0008 | yes |
| 1 | d_effb0_mask_cmc | 0.718 | 0.696 | +0.022 [+0.006, +0.038] | 0.0096 | 0.0192 | yes |
| 0.9 | b_dino518_mask_cmc | 0.806 | 0.775 | +0.031 [+0.014, +0.048] | 0.0001 | 0.0008 | yes |
| 0.9 | c_umte | 0.687 | 0.775 | -0.088 [-0.106, -0.071] | 0.0001 | 0.0008 | no |
| 0.9 | a_epvt_mask_cmc | 0.795 | 0.775 | +0.021 [+0.006, +0.036] | 0.0040 | 0.0120 | yes |
| 0.9 | d_effb0_mask_cmc | 0.754 | 0.775 | -0.021 [-0.039, -0.003] | 0.0260 | 0.0260 | no |

ERM (5 seeds, reference): bias 1 0.668, bias 0.9 0.762
