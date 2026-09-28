# R10 — external natural test: ISIC 2019 -> ISIC 2020 (DINOv2 ViT-B/14 @518)

Environment: {"commit": "ac90660", "host": "074495c7c1a5", "WTSS_BOOTSTRAP": "crossed", "gpu": "NVIDIA RTX PRO 4000 Blackwell"}

Counts:

```
{
 "tau_pred_hair_px": 200,
 "isic2020_with_lesion_mask": 32997,
 "dropped_near_duplicates": 7657,
 "dropped_same_name": 0,
 "test_images": 25340,
 "test_melanomas": 483,
 "test_patients": 2044,
 "isic2019_train_val_images": 25179,
 "isic2019_melanomas": 4509,
 "isic2019_in_lesion_hair": {
  "P(a|y=1)": 0.30538922155688625,
  "P(a|y=0)": 0.19646831156265118
 },
 "isic2020_in_lesion_hair": {
  "P(a|y=1)": 0.16149068322981366,
  "P(a|y=0)": 0.07647745101983344
 }
}
```

Hard pairs: melanomas without in-lesion hair vs benign lesions with in-lesion hair (conflict with the training association).

| method | auc | cross_group | auc_in_lesion_hair | auc_hard | auc_easy |
|---|---|---|---|---|---|
| balanced | 0.779 | 0.749 | 0.776 | 0.805 | 0.749 |
| dfr | 0.765 | 0.730 | 0.746 | 0.782 | 0.731 |
| erm | 0.791 | 0.772 | 0.779 | 0.772 | 0.798 |
| jtt | 0.706 | 0.632 | 0.722 | 0.796 | 0.632 |
| mask | 0.811 | 0.746 | 0.781 | 0.746 | 0.843 |
| mask_balanced | 0.792 | 0.776 | 0.777 | 0.786 | 0.783 |
| mask_dfr | 0.767 | 0.717 | 0.730 | 0.718 | 0.781 |
| mte | 0.799 | 0.752 | 0.783 | 0.752 | 0.827 |
| mte_balanced | 0.778 | 0.764 | 0.780 | 0.787 | 0.764 |
| mte_protect | 0.804 | 0.715 | 0.752 | 0.715 | 0.839 |
| mte_protect_balanced | 0.790 | 0.753 | 0.747 | 0.753 | 0.789 |

Crossed bootstrap (5 seeds; one Poisson weight per ISIC 2020 image):

| subset | arm | ref | value |
|---|---|---|---|
| hard | mask | erm | -0.026 [-0.052, -0.000] |
| hard | balanced | mask | +0.059 [+0.035, +0.084] |
| hard | mte_balanced | mask | +0.041 [+0.030, +0.051] |
| hard | mte | mask | +0.005 [-0.002, +0.013] |
| easy | mask | erm | +0.045 [+0.005, +0.087] |
| easy | balanced | mask | -0.094 [-0.138, -0.051] |
| easy | mte_balanced | mask | -0.079 [-0.099, -0.059] |
| easy | mte | mask | -0.016 [-0.029, -0.003] |
| all | mask | erm | +0.020 [+0.003, +0.038] |
| all | balanced | mask | -0.032 [-0.051, -0.014] |
| all | mte_balanced | mask | -0.033 [-0.040, -0.026] |
| all | mte | mask | -0.012 [-0.018, -0.007] |

- **H10a** hard-pair AUROC(mask) - AUROC(ERM) < 0: -0.026 [-0.052, -0.000] -> **SUPPORTED**
- **H10b** OP5 sensitivity among melanomas without in-lesion hair, mask - ERM < 0: -0.007 [-0.060, +0.047] -> **NOT SUPPORTED**
- **H10c** hard-pair AUROC(balanced) - AUROC(mask) > 0: +0.059 [+0.035, +0.084] -> **SUPPORTED**

Operating points (thresholds from ISIC 2019 validation only):

| arm | ref | op | metric | value |
|---|---|---|---|---|
| mask | erm | OP1_maxBA | sens | 0.516 vs 0.434: +0.082 [-0.014, +0.167] |
| mask | erm | OP1_maxBA | spec | 0.897 vs 0.899: -0.003 [-0.046, +0.035] |
| mask | erm | OP1_maxBA | sens_conflict | 0.519 vs 0.432: +0.087 [-0.011, +0.168] |
| mask | erm | OP2_spec0.80 | sens | 0.477 vs 0.472: +0.005 [-0.033, +0.052] |
| mask | erm | OP2_spec0.80 | spec | 0.914 vs 0.885: +0.029 [+0.017, +0.040] |
| mask | erm | OP2_spec0.80 | sens_conflict | 0.482 vs 0.470: +0.012 [-0.031, +0.063] |
| mask | erm | OP3_spec0.90 | sens | 0.324 vs 0.309: +0.015 [-0.026, +0.062] |
| mask | erm | OP3_spec0.90 | spec | 0.963 vs 0.947: +0.016 [+0.009, +0.023] |
| mask | erm | OP3_spec0.90 | sens_conflict | 0.332 vs 0.309: +0.023 [-0.024, +0.072] |
| mask | erm | OP4_sens0.80 | sens | 0.545 vs 0.523: +0.022 [-0.033, +0.077] |
| mask | erm | OP4_sens0.80 | spec | 0.880 vs 0.860: +0.021 [+0.002, +0.039] |
| mask | erm | OP4_sens0.80 | sens_conflict | 0.547 vs 0.524: +0.022 [-0.031, +0.079] |
| mask | erm | OP5_sens0.90 | sens | 0.699 vs 0.702: -0.003 [-0.055, +0.052] |
| mask | erm | OP5_sens0.90 | spec | 0.760 vs 0.733: +0.028 [-0.011, +0.060] |
| mask | erm | OP5_sens0.90 | sens_conflict | 0.693 vs 0.700: -0.007 [-0.060, +0.047] |
| balanced | mask | OP1_maxBA | sens | 0.450 vs 0.516: -0.065 [-0.144, +0.015] |
| balanced | mask | OP1_maxBA | spec | 0.888 vs 0.897: -0.009 [-0.049, +0.030] |
| balanced | mask | OP1_maxBA | sens_conflict | 0.462 vs 0.519: -0.057 [-0.136, +0.029] |
| balanced | mask | OP2_spec0.80 | sens | 0.484 vs 0.477: +0.007 [-0.045, +0.050] |
| balanced | mask | OP2_spec0.80 | spec | 0.872 vs 0.914: -0.042 [-0.055, -0.030] |
| balanced | mask | OP2_spec0.80 | sens_conflict | 0.497 vs 0.482: +0.015 [-0.041, +0.058] |
| balanced | mask | OP3_spec0.90 | sens | 0.330 vs 0.324: +0.005 [-0.042, +0.050] |
| balanced | mask | OP3_spec0.90 | spec | 0.939 vs 0.963: -0.023 [-0.031, -0.016] |
| balanced | mask | OP3_spec0.90 | sens_conflict | 0.334 vs 0.332: +0.002 [-0.050, +0.048] |
| balanced | mask | OP4_sens0.80 | sens | 0.544 vs 0.545: -0.001 [-0.057, +0.050] |
| balanced | mask | OP4_sens0.80 | spec | 0.834 vs 0.880: -0.047 [-0.061, -0.027] |
| balanced | mask | OP4_sens0.80 | sens_conflict | 0.556 vs 0.547: +0.009 [-0.048, +0.063] |
| balanced | mask | OP5_sens0.90 | sens | 0.700 vs 0.699: +0.001 [-0.049, +0.052] |
| balanced | mask | OP5_sens0.90 | spec | 0.711 vs 0.760: -0.050 [-0.082, -0.016] |
| balanced | mask | OP5_sens0.90 | sens_conflict | 0.714 vs 0.693: +0.021 [-0.032, +0.072] |
