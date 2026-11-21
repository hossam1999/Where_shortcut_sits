# P1 location test — formal 3-seed analysis (A43; section 5) — machine 2

Seeds 1234 (machine 1, committed per-image scores; different GPU and software build), 3232 and 3221 (machine 2). Their code and Standard config; Masked = the Standard model with test masking (A5); our arms a–d. Crossed seed × image bootstrap (10,000; one Poisson weight per image shared by seeds, methods and positions). Epoch selection: their rule (registered table; A17's epoch-100 sensitivity analysis is not repeated).

| method | position | reversed AUROC (mean ± SD) | correlated | clean | gain over Standard [95 % CI] | vs Masked [95 % CI] | crossover [95 % CI] |
|---|---|---|---|---|---|---|---|
| Standard (theirs) | inside | 0.672 ± 0.187 | 0.933 | 0.885 | — | — | — |
| Standard (theirs) | outside | 0.913 ± 0.017 | 0.918 | 0.916 | — | — | — |
| Masked (theirs, A5) | inside | 0.615 ± 0.195 | 0.912 | 0.845 | -0.057 [-0.082, -0.039] | — | +0.040 [+0.018, +0.067] |
| Masked (theirs, A5) | outside | 0.896 ± 0.013 | 0.896 | 0.896 | -0.017 [-0.025, -0.010] | — | +0.040 [+0.018, +0.067] |
| a: mask_cmc on their net | inside | 0.867 ± 0.010 | 0.899 | 0.889 | +0.195 [-0.006, +0.338] | +0.252 [+0.037, +0.384] | -0.195 [-0.324, -0.007] |
| a: mask_cmc on their net | outside | 0.913 ± 0.007 | 0.913 | 0.913 | +0.000 [-0.012, +0.013] | +0.017 [+0.009, +0.026] | -0.195 [-0.324, -0.007] |
| b: DINOv2 + mask_cmc | inside | 0.876 ± 0.005 | 0.912 | 0.897 | +0.204 [-0.006, +0.359] | +0.261 [+0.038, +0.404] | -0.215 [-0.354, -0.015] |
| b: DINOv2 + mask_cmc | outside | 0.902 ± 0.002 | 0.902 | 0.902 | -0.011 [-0.025, +0.005] | +0.006 [-0.007, +0.020] | -0.215 [-0.354, -0.015] |
| c: U-MtE | inside | 0.618 ± 0.021 | 0.963 | 0.890 | -0.054 [-0.257, +0.080] | +0.004 [-0.214, +0.131] | +0.055 [-0.068, +0.251] |
| c: U-MtE | outside | 0.915 ± 0.006 | 0.915 | 0.915 | +0.002 [-0.010, +0.014] | +0.019 [+0.009, +0.029] | +0.055 [-0.068, +0.251] |
| d: EfficientNet-B0 + mask_cmc (A20) | inside | 0.754 ± 0.023 | 0.964 | 0.910 | +0.082 [-0.129, +0.218] | +0.139 [-0.086, +0.269] | -0.072 [-0.194, +0.127] |
| d: EfficientNet-B0 + mask_cmc (A20) | outside | 0.924 ± 0.004 | 0.924 | 0.924 | +0.010 [-0.003, +0.026] | +0.027 [+0.016, +0.040] | -0.072 [-0.194, +0.127] |

**Registered expectations (section 5), Holm over 5:**

| expectation | Δ [95 % CI] | p | p Holm | verdict |
|---|---|---|---|---|
| E-L1 Masked crossover > 0 | +0.040 [+0.018, +0.067] | 0.0001 | 0.0005 | supported |
| E-L2 arm a gain inside > 0 | +0.195 [-0.006, +0.338] | 0.0742 | 0.2760 | not supported |
| E-L2 arm b gain inside > 0 | +0.204 [-0.006, +0.359] | 0.0690 | 0.2760 | not supported |
| E-L2 arm d gain inside > 0 (A20) | +0.082 [-0.129, +0.218] | 0.5162 | 1.0000 | not supported |
| E-L3 arm c gain inside > 0 | -0.054 [-0.257, +0.080] | 0.6054 | 1.0000 | not supported |

**Per seed, reversed-test AUROC** (seed 1234 = machine 1):

| method | position | seed 1234 | seed 3232 | seed 3221 |
|---|---|---|---|---|
| Standard (theirs) | inside | 0.514 | 0.879 | 0.624 |
| Standard (theirs) | outside | 0.894 | 0.923 | 0.923 |
| Masked (theirs, A5) | inside | 0.469 | 0.836 | 0.540 |
| Masked (theirs, A5) | outside | 0.881 | 0.900 | 0.907 |
| a: mask_cmc on their net | inside | 0.856 | 0.871 | 0.874 |
| a: mask_cmc on their net | outside | 0.908 | 0.911 | 0.921 |
| b: DINOv2 + mask_cmc | inside | 0.877 | 0.870 | 0.880 |
| b: DINOv2 + mask_cmc | outside | 0.899 | 0.903 | 0.904 |
| c: U-MtE | inside | 0.598 | 0.618 | 0.640 |
| c: U-MtE | outside | 0.908 | 0.916 | 0.920 |
| d: EfficientNet-B0 + mask_cmc (A20) | inside | 0.735 | 0.746 | 0.780 |
| d: EfficientNet-B0 + mask_cmc (A20) | outside | 0.921 | 0.922 | 0.928 |
