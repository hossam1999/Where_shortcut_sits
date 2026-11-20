# P6: our arms vs our EPVT (A39) — bias 1 — machine 2

Test AUROC on `test_bias_<b>_1`, each arm scored once; crossed seed × image bootstrap vs EPVT's 5 seeds (A36 mixup).
**Interim (bias 1 only): unadjusted; the A39 verdict uses both bias levels with Holm over 8.**

| bias | arm | arm mean | EPVT mean | Δ [95 % CI] | p (unadj.) |
|---|---|---|---|---|---|
| 1 | b_dino518_mask_cmc | 0.782 | 0.497 | +0.285 [+0.265, +0.306] | 0.0001 |
| 1 | c_umte | 0.625 | 0.497 | +0.128 [+0.105, +0.151] | 0.0001 |
| 1 | a_epvt_mask_cmc | 0.758 | 0.497 | +0.261 [+0.236, +0.286] | 0.0001 |
| 1 | d_effb0_mask_cmc | 0.718 | 0.497 | +0.222 [+0.199, +0.244] | 0.0001 |

ERM (5 seeds, reference): bias 1 0.505
