# Pre-registration — natural-distribution evaluation (no trap resampling)

Committed before any model is fitted for this analysis.
- Thyroid (TN3K + TNCD): train on the official trainval split with its natural association between in-ROI calipers
  and the label (in-ROI calipers: 37.6 % of benign vs 24.1 % of malignant images); test on the untouched official
  test split (614 images). 5 seeds of a group-safe 80/20 train/validation split = bootstrap clusters.
- A (for group-using arms) = in-ROI caliper (>= 15 px, r >= 0.5).
- Metrics on the natural test set: AUROC; cross-group AUROC (min over the two counter-/pro-shortcut pairings of
  groups; the natural analogue of min(reversed, correlated)); AUROC within the in-ROI-caliper subgroup.
- Arms: erm, mask, balanced, mask_balanced, U-MtE, U-MtE_protect, U-MtE_balanced, U-MtE_protect_balanced, DFR,
  mask+DFR, JTT (DINOv2@518, generic library).
- Claims: **N1** cross-group AUROC(U-MtE_protect) > cross-group AUROC(mask) and **N2** overall AUROC(U-MtE_protect)
  ≥ AUROC(mask) − 0.01 (no cost on the natural distribution). Reported whichever way they go.

## Results — thyroid, DINOv2@518 (results/natural/thyroid_dino518; 5 seeds; official TN3K test, 614 images)
| arm | AUROC | cross-group AUROC | AUROC within in-ROI-caliper images |
|---|---|---|---|
| ERM | 0.736 | 0.679 | 0.781 |
| mask | 0.731 | **0.576** | 0.738 |
| U-MtE | **0.742** | 0.627 | 0.760 |
| U-MtE_protect | 0.715 | 0.518 | 0.723 |
| balanced | 0.716 | **0.713** | 0.768 |
| U-MtE_balanced | 0.731 | 0.664 | 0.755 |
| DFR | 0.697 | 0.625 | 0.749 |
- Under the natural distribution, masking lowers the cross-group AUROC by 0.10 (0.679 → 0.576): the in-ROI caliper is
  retained while other evidence is removed — the thesis effect without any resampling.
- **N1 not supported** (U-MtE_protect 0.518 < mask 0.576); unprotected U-MtE recovers half the loss (0.627).
- **N2 not supported** (U-MtE_protect AUROC − mask = −0.017 [−0.027, −0.007]); unprotected U-MtE has the best overall
  AUROC (0.742). Group balancing gives the best cross-group AUROC (0.713).
Subset bootstraps (results/natural/thyroid_dino518/subset_boot.json; post hoc, secondary):
- Hard pairs (malignant WITH in-ROI caliper vs benign WITHOUT): mask − ERM = **−0.102 [−0.139, −0.066]**;
  U-MtE − mask = +0.051 [+0.033, +0.072]; balanced − mask = +0.142 [+0.107, +0.178].
- Easy pairs (malignant without vs benign with): mask − ERM = +0.048 [+0.019, +0.079].
Masking helps where the natural shortcut agrees with the label and harms where it disagrees — the in-ROI
shortcut is amplified by masking in unaltered data.

## Amendment (2026-09-27, before fitting) — more natural-distribution tests
Same design and arms as the thyroid analysis (scripts/run_natural.py):
- **ISIC 2019 leave-one-hospital-out** (isic_BCN, isic_HAM, isic_MSK): train on the other two sources (natural
  association: in-lesion hair 31 % of melanomas vs 20 % of benign lesions, similar in every source), test on the held-out
  source; A = in-lesion hair (> 30 native px, r >= 0.5); all QC-passed images.
- **Capsule**: SEE-AI, per seed a group-safe 80/20 split of frame-block groups; A = debris >= 10 % with r >= 0.5
  (natural association: 11.6 % of erosions vs 20.9 % of polyp-like lesions).
The "hard" (shortcut-conflicting) pairing is determined from the test-set association direction.
Claims: **N3** mask − ERM < 0 on hard pairs (thesis effect in natural data) and **N4** U-MtE − mask > 0 on hard pairs,
per cohort. Reported either way.

## Results — amendment cohorts (DINOv2@518, 5 seeds; results/natural/*_dino518/natural_boot.json)
Hard = shortcut-conflicting pairs (direction from the test-set association). Δ reversed-free AUROC on hard pairs:
| cohort | P(A\|Y=1) / P(A\|Y=0) | mask − ERM (N3: < 0) | U-MtE − mask (N4: > 0) | U-MtE_bal − mask | best cross-group AUROC |
|---|---|---|---|---|---|
| thyroid (official split) | 0.241 / 0.376 | **−0.102 [−0.139, −0.066]** | **+0.051 [+0.033, +0.072]** | — | balanced 0.713 |
| ISIC, HAM held out | 0.274 / 0.184 | +0.053 [+0.037, +0.068] | **+0.023 [+0.017, +0.032]** | +0.065 [+0.056, +0.075] | **U-MtE_bal 0.786** |
| ISIC, BCN held out | 0.362 / 0.247 | **−0.016 [−0.027, −0.005]** | +0.003 [+0.000, +0.006] | +0.034 [+0.029, +0.039] | balanced 0.735 ≈ U-MtE_bal 0.734 |
| ISIC, MSK held out | 0.114 / 0.076 | **−0.056 [−0.084, −0.029]** | −0.008 [−0.015, −0.002] | +0.041 [+0.029, +0.053] | balanced 0.734 |
| capsule | 0.101 / 0.232 | +0.021 [+0.000, +0.044] | +0.000 [−0.010, +0.009] | +0.014 [+0.003, +0.025] | **U-MtE_bal 0.947** |
**N3 supported in 3/5** (thyroid, BCN, MSK): on unaltered data masking harms the shortcut-conflicting cases whenever
it does. **N4 supported in 2/5** (thyroid, HAM). U-MtE_balanced improves on masking for hard pairs in every cohort
where measured (4/4) and gives the best or tied-best cross-group AUROC in 3/5; plain balancing is best in 2/5.
