# P3 90/10 inside variant (A6): our arms b and c, seed 0 data (= the ISNet 90/10 inside run's data)

Macro-F1 on the rebuilt P3 test (both A9 variants). P3 is descriptive only (A9 check failed, 2026-11-18). One data seed; point estimates; the paired comparison with ISNet is added when that run is scored. Validation = P3's validation split with its marks (registered); registered C grid.

- **b_dino518_mask_cmc**: selected mode **pooled**, C = 0.1 (validation macro-F1 0.980; edge of the registered grid: no)
- **c_umte**: selected mode **k = 64**, C = 0.1 (validation macro-F1 0.973; edge of the registered grid: no)

| arm | test | primary | size-matched |
|---|---|---|---|
| b_dino518_mask_cmc | biased | 0.947 | 0.946 |
| b_dino518_mask_cmc | standard | 0.398 | 0.394 |
| b_dino518_mask_cmc | deceiving | 0.136 | 0.114 |
| c_umte | biased | 0.837 | 0.837 |
| c_umte | standard | 0.389 | 0.381 |
| c_umte | deceiving | 0.168 | 0.144 |

A24 SECONDARY (cohort validation, used for selection): | arm | condition | macro-F1 |
|---|---|---|
| b_dino518_mask_cmc | biased | 0.980 |
| b_dino518_mask_cmc | standard | 0.798 |
| b_dino518_mask_cmc | deceiving | 0.197 |
| c_umte | biased | 0.973 |
| c_umte | standard | 0.780 |
| c_umte | deceiving | 0.359 |
