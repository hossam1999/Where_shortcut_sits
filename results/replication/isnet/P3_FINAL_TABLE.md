# P3 (Bassi et al. 2024) — final table, DESCRIPTIVE (machine 1, 2026-11-20; corrected after A30)

**Caveats.** (1) P3 is not reproduced and carries no verdict (A9: P3's released COVIDIsNet.pt scores 0.223 / 0.260 on our rebuilt test and fails on P3's own training sources; no pipeline difference found). (2) A24 secondary = P3's cohort-validation images, used for epoch / model selection. (3) One seed. (4) Our arms' rows were re-run once after the A30 fix (image–label misalignment in our P3 arm adapter); the earlier arm rows are kept, labelled INVALID. ISNet rows were never affected.

Macro-F1. Rebuilt external test (primary): full / size-matched (A9). A24 secondary: cohort validation.

| design (marks) | method | biased | standard | deceiving | A24 biased | A24 standard | A24 deceiving |
|---|---|---|---|---|---|---|---|
| corner (outside the lungs), every image marked | ISNet (theirs) | 0.427 / 0.403 | 0.385 / 0.357 | 0.387 / 0.362 | 0.841 | 0.834 | 0.831 |
| corner (outside the lungs), every image marked | c: U-MtE (A12) | 0.383 / 0.376 | 0.383 / 0.376 | 0.383 / 0.376 | 0.848 | 0.848 | 0.848 |
| inside the lungs, every image marked | ISNet (theirs) | 0.999 / 0.998 | 0.160 / 0.193 | 0.000 / 0.000 | 0.998 | 0.167 | 0.000 |
| inside the lungs, 90/10 | ISNet (theirs) | 0.989 / 0.990 | 0.435 / 0.418 | 0.034 / 0.028 | 0.974 | 0.745 | 0.033 |
| inside the lungs, 90/10 | b: DINOv2 + mask_cmc | 0.947 / 0.946 | 0.398 / 0.394 | 0.136 / 0.114 | 0.980 | 0.798 | 0.197 |
| inside the lungs, 90/10 | c: U-MtE | 0.837 / 0.837 | 0.389 / 0.381 | 0.168 / 0.144 | 0.973 | 0.780 | 0.359 |
| inside the lungs, 90/10 | d: EfficientNet-B0 + mask_cmc (A20) | 0.993 / 0.996 | 0.406 / 0.405 | 0.002 / 0.001 | 0.982 | 0.813 | 0.033 |

**Reading.** ISNet ignores marks outside its lung masks (corner: the three tests agree; 0.83 in distribution) but is driven by marks inside the lungs (deceiving 0.000 with every image marked, 0.034 with 90/10 marks; in distribution as well). On the 90/10 inside variant our arms are also largely driven by the class-specific marks (deceiving 0.002–0.168 on the external test, 0.033–0.359 in distribution): P3's shortcut is which shape is present, which a within-class presence label cannot remove; this is reported as a limitation. U-MtE (label-free) keeps the most signal under swapped marks (A24 deceiving 0.359). With every image marked inside the lungs, mark and class never disagree, so no method can separate them (not identifiable). Arm c on the corner benchmark: the lung mask removes the corner marks, so its three conditions are identical (0.848 in distribution, 0.383 on the external test).
