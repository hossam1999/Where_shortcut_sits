# P3 corner benchmark: our ISNet (isnet_corner_all_seed0) on the rebuilt test — DESCRIPTIVE

Supporting evidence of test equivalence only, **not a gate** (author, 2026-11-18): P3 stays descriptive (A9 check 0.223 / 0.260). P3 Table 1 (COVID-19 with synthetic background bias), ISNet: biased / standard / deceiving 0.775 ± 0.008 each. One seed.

| test | primary (n = 4,190) | size-matched (n = 3,180) | P3 Table 1 |
|---|---|---|---|
| biased | 0.427 | 0.403 | 0.775 ± 0.008 |
| standard | 0.385 | 0.357 | 0.775 ± 0.008 |
| deceiving | 0.387 | 0.362 | 0.775 ± 0.008 |

Kept epoch (lowest validation loss, their rule): 46 of 48.
