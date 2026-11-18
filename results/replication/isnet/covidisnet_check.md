# A9 equivalence check: P3's released COVIDIsNet.pt on our rebuilt o.o.d. test

Target: P3 Table 2, ISNet macro-F1 0.773 (within 0.03 → P3 verdicts allowed on that variant, A9). Rows = true class (normal, pneumonia, COVID-19), columns = predicted.

## primary (370 / 1,319 / 2,501): n = 4190, macro-F1 **0.223** (per class 0.139 / 0.492 / 0.038) → not within 0.03: descriptive only

| true \ predicted | normal | pneumonia | COVID-19 | P3 Table 4 (ISNet) |
|---|---|---|---|---|
| normal | 36 | 334 | 0 | 210 / 157 / 3 |
| pneumonia | 11 | 1308 | 0 | 81 / 1210 / 4 |
| COVID-19 | 101 | 2352 | 48 | 93 / 156 / 1266 |

## size-matched (370 / 1,295 / 1,515): n = 3180, macro-F1 **0.260** (per class 0.152 / 0.592 / 0.036) → not within 0.03: descriptive only

| true \ predicted | normal | pneumonia | COVID-19 | P3 Table 4 (ISNet) |
|---|---|---|---|---|
| normal | 36 | 334 | 0 | 210 / 157 / 3 |
| pneumonia | 10 | 1285 | 0 | 81 / 1210 / 4 |
| COVID-19 | 58 | 1429 | 28 | 93 / 156 / 1266 |

