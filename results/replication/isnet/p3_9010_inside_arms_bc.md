# P3 90/10 inside variant (A6): our arms b and c, seed 0 data (= the ISNet 90/10 inside run's data)

Macro-F1 on the rebuilt P3 test (both A9 variants). P3 is descriptive only (A9 check failed, 2026-11-18). One data seed; point estimates; the paired comparison with ISNet is added when that run is scored. Validation = P3's validation split with its marks (registered); registered C grid.

- **b_dino518_mask_cmc**: selected mode **pooled**, C = 0.1 (validation macro-F1 0.855; edge of the registered grid: no)
- **c_umte**: selected mode **k = 64**, C = 1 (validation macro-F1 0.810; edge of the registered grid: no)

| arm | test | primary | size-matched |
|---|---|---|---|
| b_dino518_mask_cmc | biased | 0.996 | 0.997 |
| b_dino518_mask_cmc | standard | 0.419 | 0.416 |
| b_dino518_mask_cmc | deceiving | 0.032 | 0.026 |
| c_umte | biased | 0.965 | 0.974 |
| c_umte | standard | 0.413 | 0.419 |
| c_umte | deceiving | 0.073 | 0.064 |
