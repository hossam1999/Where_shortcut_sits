# P3 original corner benchmark (every image marked, A12): our arm c (U-MtE), the corner ISNet run's data

Macro-F1 on the rebuilt P3 test (both A9 variants). P3 is descriptive only (A9 check failed, 2026-11-18). One data seed; point estimates; the paired comparison with ISNet is added when that run is scored. Validation = P3's validation split with its marks (registered); registered C grid.

- **c_umte**: selected mode **k = 64**, C = 10 (validation macro-F1 0.284; edge of the registered grid: highest)

| arm | test | primary | size-matched |
|---|---|---|---|
| c_umte | biased | 0.383 | 0.383 |
| c_umte | standard | 0.382 | 0.383 |
| c_umte | deceiving | 0.382 | 0.383 |
