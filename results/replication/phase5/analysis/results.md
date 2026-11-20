# Phase 5 (thyroid TN3K calipers) — results (PREREGISTRATION_PHASE5.md) — machine 2

AUROC, mean ± SD over folds 0–2 (seed 42). Test `test_rev` = marker–label correlation reversed (primary); `clean` = no marker; `test_corr` = as in training.

| trap | method | test_rev | clean | test_corr |
|---|---|---|---|---|
| trapA | DINOv2 + mask_cmc | 0.716 ± 0.027 | 0.797 ± 0.011 | 0.851 ± 0.012 |
| trapA | ERM ResNet-50d (RR-ClArC base) | 0.440 ± 0.022 | 0.625 ± 0.006 | 0.795 ± 0.022 |
| trapA | ISNet | 0.561 ± 0.172 | 0.633 ± 0.033 | 0.696 ± 0.145 |
| trapA | RR-ClArC | 0.467 ± 0.049 | 0.643 ± 0.022 | 0.805 ± 0.021 |
| trapA | U-MtE | 0.504 ± 0.044 | 0.727 ± 0.014 | 0.891 ± 0.017 |
| trapA | arm d (EffNet-B0 + mask_cmc) | 0.690 ± 0.076 | 0.733 ± 0.057 | 0.773 ± 0.048 |
| trapB | DINOv2 + mask_cmc | 0.789 ± 0.062 | 0.787 ± 0.018 | 0.834 ± 0.038 |
| trapB | ERM ResNet-50d (RR-ClArC base) | 0.546 ± 0.019 | 0.564 ± 0.101 | 0.589 ± 0.133 |
| trapB | ISNet | 0.673 ± 0.006 | 0.706 ± 0.044 | 0.728 ± 0.089 |
| trapB | RR-ClArC | 0.595 ± 0.045 | 0.615 ± 0.027 | 0.604 ± 0.075 |
| trapB | U-MtE | 0.782 ± 0.022 | 0.781 ± 0.048 | 0.823 ± 0.054 |
| trapB | arm d (EffNet-B0 + mask_cmc) | 0.746 ± 0.009 | 0.754 ± 0.021 | 0.797 ± 0.054 |

Comparisons on test_rev (crossed fold × image bootstrap; Holm family = 4 on Trap A; 'beaten' also needs clean drop ≤ 0.01):

| trap | ours | theirs | ours | theirs | Δ [95 % CI] | p | p Holm | clean Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| trapA | arm d (EffNet-B0 + mask_cmc) | RR-ClArC | 0.690 | 0.467 | +0.224 [+0.159, +0.292] | 0.0001 | 0.0004 | +0.090 | beaten |
| trapA | arm d (EffNet-B0 + mask_cmc) | ISNet | 0.690 | 0.561 | +0.129 [+0.035, +0.239] | 0.0024 | 0.0048 | +0.100 | beaten |
| trapA | DINOv2 + mask_cmc | RR-ClArC | 0.716 | 0.467 | +0.249 [+0.192, +0.304] | 0.0001 | 0.0004 | +0.155 | beaten |
| trapA | DINOv2 + mask_cmc | ISNet | 0.716 | 0.561 | +0.155 [+0.019, +0.316] | 0.0182 | 0.0182 | +0.164 | beaten |
| trapA | U-MtE | RR-ClArC | 0.504 | 0.467 | +0.038 [-0.016, +0.091] | 0.1532 |  | +0.084 | descriptive |
| trapA | U-MtE | ISNet | 0.504 | 0.561 | -0.057 [-0.195, +0.096] | 0.4880 |  | +0.093 | descriptive |
| trapB | arm d (EffNet-B0 + mask_cmc) | RR-ClArC | 0.746 | 0.595 | +0.151 [+0.067, +0.235] | 0.0008 |  | +0.138 | descriptive |
| trapB | arm d (EffNet-B0 + mask_cmc) | ISNet | 0.746 | 0.673 | +0.073 [-0.007, +0.149] | 0.0698 |  | +0.048 | descriptive |
| trapB | DINOv2 + mask_cmc | RR-ClArC | 0.789 | 0.595 | +0.194 [+0.107, +0.280] | 0.0002 |  | +0.172 | descriptive |
| trapB | DINOv2 + mask_cmc | ISNet | 0.789 | 0.673 | +0.116 [+0.028, +0.207] | 0.0120 |  | +0.082 | descriptive |
| trapB | U-MtE | RR-ClArC | 0.782 | 0.595 | +0.188 [+0.102, +0.273] | 0.0002 |  | +0.166 | descriptive |
| trapB | U-MtE | ISNet | 0.782 | 0.673 | +0.110 [+0.035, +0.186] | 0.0042 |  | +0.075 | descriptive |
