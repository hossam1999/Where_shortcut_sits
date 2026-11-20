# P3 (Bassi et al. 2024) — final table, DESCRIPTIVE (machine 1, 2026-11-20)

**A30 (2026-11-20): every row of OUR arms below is INVALID** (image–label misalignment in our P3 arm adapter; ISNet rows are unaffected). Corrected arm rows replace them after re-scoring.

**Caveats.** (1) P3 is not reproduced and carries no verdict (A9: P3's released COVIDIsNet.pt scores 0.223 / 0.260 on our rebuilt test and fails on P3's own training sources; no pipeline difference found). (2) The A24 secondary evaluation uses P3's cohort-validation images, which were used for epoch / model selection. (3) One seed. (4) By the author's scope change (2026-11-19) the A24 feature passes for our arms were not run; our arms on P3 are reported as a limitation.

Macro-F1. Rebuilt external test (primary): full / size-matched (A9). A24 secondary: cohort validation.

| design (marks) | method | biased | standard | deceiving | A24 biased | A24 standard | A24 deceiving |
|---|---|---|---|---|---|---|---|
| corner (outside the lungs), every image marked | ISNet (theirs) | 0.427 / 0.403 | 0.385 / 0.357 | 0.387 / 0.362 | 0.841 | 0.834 | 0.831 |
| inside the lungs, every image marked | ISNet (theirs) | 0.999 / 0.998 | 0.160 / 0.193 | 0.000 / 0.000 | 0.998 | 0.167 | 0.000 |
| inside the lungs, 90/10 | ISNet (theirs) | 0.989 / 0.990 | 0.435 / 0.418 | 0.034 / 0.028 | 0.974 | 0.745 | 0.033 |
| corner (outside the lungs), every image marked | c: U-MtE (A12) | 0.383 / 0.383 | 0.382 / 0.383 | 0.382 / 0.383 | not run | not run | not run |
| inside the lungs, 90/10 | b: DINOv2 + mask_cmc | 0.996 / 0.997 | 0.419 / 0.416 | 0.032 / 0.026 | not run | not run | not run |
| inside the lungs, 90/10 | c: U-MtE | 0.965 / 0.974 | 0.413 / 0.419 | 0.073 / 0.064 | not run | not run | not run |
| inside the lungs, 90/10 | d: EfficientNet-B0 + mask_cmc (A20) | 0.997 / 0.996 | 0.410 / 0.404 | 0.010 / 0.008 | not run | not run | not run |

**Reading.** ISNet ignores marks outside its lung masks (corner: the three tests agree, 0.83 in distribution) but is driven by marks inside the lungs (deceiving 0.000 with every image marked; 0.034 with 90/10 marks). In the all-marked inside design mark and class never disagree, so no method can separate them (not identifiable). Our arms on the 90/10 inside variant are also driven by the class-specific marks (deceiving 0.010–0.073), because the shortcut is which shape is present, not whether a mark is present. Arm c on the corner benchmark is at chance already on validation (macro-F1 0.284; U-MtE erased k = 64 directions of lung-only features); reported as registered, not diagnosed further (scope change).
