# P3 location test (A5/A6, P3's design: every image marked): ISNet with marks in the corner vs inside the lungs — DESCRIPTIVE

P3 is descriptive only (A9 check failed, 2026-11-18). One seed. Primary = rebuilt external test (both A9 variants); secondary (A24) = P3's cohort-validation images rendered in the same conditions (used for epoch selection).

| marks | test | rebuilt test, primary | size-matched | A24 secondary (cohort validation) |
|---|---|---|---|---|
| corner (outside the lungs) | biased | 0.427 | 0.403 | 0.841 |
| corner (outside the lungs) | standard | 0.385 | 0.357 | 0.834 |
| corner (outside the lungs) | deceiving | 0.387 | 0.362 | 0.831 |
| inside the lungs | biased | 0.999 | 0.998 | 0.998 |
| inside the lungs | standard | 0.160 | 0.193 | 0.167 |
| inside the lungs | deceiving | 0.000 | 0.000 | 0.000 |

Reading: ISNet's LRP penalty acts only outside the lung masks. Marks in the corner are ignored (the three tests agree); marks inside the lungs drive the decision completely (deceiving 0.000 on both the external test and in distribution).
