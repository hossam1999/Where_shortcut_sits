# Round 5 — robustness of the ISIC 2020 result (R12, R13)

Environment: {"commit": "a02db23", "host": "074495c7c1a5", "WTSS_BOOTSTRAP": "crossed", "gpu": "NVIDIA RTX PRO 4000 Blackwell"}

R10 is unchanged. R12 and R13 are sensitivity analyses of that one set of predictions (every ISIC 2020 image with a lesion mask).

Calibration (label-free, committed before fitting): t* = 1, c* = 0.971. F(t*) = 0.0011, G(c*) = 0.0095. Duplicate if d19 ≤ t* or s19 ≥ c* or the image name is in ISIC 2019.

Kept images by tier:

| tier | dropped | kept |
|---|---|---|
| exact | 26 | 32971 |
| d19_le2 | 138 | 32859 |
| calibrated | 45 | 32952 |
| d19_le8 | 7657 | 25340 |

Counts of kept images by label and hair group:

| tier | y | n | H0 | Hin | Hout | ambiguous | in_lesion_hair |
|---|---|---|---|---|---|---|---|
| exact | 0 | 32393 | 1953 | 1868 | 17509 | 11063 | 2056 |
| exact | 1 | 578 | 53 | 74 | 212 | 239 | 79 |
| d19_le2 | 0 | 32284 | 1944 | 1868 | 17433 | 11039 | 2056 |
| d19_le2 | 1 | 575 | 53 | 74 | 209 | 239 | 79 |
| calibrated | 0 | 32377 | 1953 | 1868 | 17504 | 11052 | 2056 |
| calibrated | 1 | 575 | 53 | 74 | 209 | 239 | 79 |
| d19_le8 | 0 | 24857 | 1434 | 1744 | 12693 | 8986 | 1901 |
| d19_le8 | 1 | 483 | 44 | 74 | 158 | 207 | 78 |

Prediction match against R10 on the 25,340 test images: pass = True, max |Δp| = 2.1812327277714871e-07, unmatched rows = 0.

Crossed bootstrap by tier (hard / easy / all; mask−ERM, balanced−mask, mte_balanced−mask):

| tier | subset | arm | ref | value |
|---|---|---|---|---|
| exact | hard | mask | erm | -0.037 [-0.061, -0.012] |
| exact | hard | balanced | mask | +0.072 [+0.048, +0.096] |
| exact | hard | mte_balanced | mask | +0.045 [+0.035, +0.056] |
| exact | easy | mask | erm | +0.048 [+0.009, +0.090] |
| exact | easy | balanced | mask | -0.099 [-0.143, -0.057] |
| exact | easy | mte_balanced | mask | -0.082 [-0.103, -0.061] |
| exact | all | mask | erm | +0.017 [-0.000, +0.033] |
| exact | all | balanced | mask | -0.028 [-0.045, -0.010] |
| exact | all | mte_balanced | mask | -0.033 [-0.040, -0.026] |
| d19_le2 | hard | mask | erm | -0.037 [-0.062, -0.013] |
| d19_le2 | hard | balanced | mask | +0.072 [+0.048, +0.097] |
| d19_le2 | hard | mte_balanced | mask | +0.045 [+0.035, +0.055] |
| d19_le2 | easy | mask | erm | +0.048 [+0.008, +0.090] |
| d19_le2 | easy | balanced | mask | -0.099 [-0.144, -0.056] |
| d19_le2 | easy | mte_balanced | mask | -0.082 [-0.103, -0.062] |
| d19_le2 | all | mask | erm | +0.016 [-0.000, +0.032] |
| d19_le2 | all | balanced | mask | -0.027 [-0.044, -0.010] |
| d19_le2 | all | mte_balanced | mask | -0.033 [-0.040, -0.026] |
| calibrated | hard | mask | erm | -0.037 [-0.061, -0.012] |
| calibrated | hard | balanced | mask | +0.072 [+0.048, +0.097] |
| calibrated | hard | mte_balanced | mask | +0.045 [+0.035, +0.055] |
| calibrated | easy | mask | erm | +0.048 [+0.009, +0.090] |
| calibrated | easy | balanced | mask | -0.099 [-0.143, -0.057] |
| calibrated | easy | mte_balanced | mask | -0.082 [-0.103, -0.062] |
| calibrated | all | mask | erm | +0.016 [-0.000, +0.033] |
| calibrated | all | balanced | mask | -0.027 [-0.045, -0.010] |
| calibrated | all | mte_balanced | mask | -0.033 [-0.040, -0.026] |
| d19_le8 | hard | mask | erm | -0.026 [-0.052, -0.000] |
| d19_le8 | hard | balanced | mask | +0.059 [+0.035, +0.084] |
| d19_le8 | hard | mte_balanced | mask | +0.041 [+0.030, +0.051] |
| d19_le8 | easy | mask | erm | +0.045 [+0.005, +0.087] |
| d19_le8 | easy | balanced | mask | -0.094 [-0.138, -0.051] |
| d19_le8 | easy | mte_balanced | mask | -0.079 [-0.099, -0.059] |
| d19_le8 | all | mask | erm | +0.020 [+0.003, +0.038] |
| d19_le8 | all | balanced | mask | -0.032 [-0.051, -0.014] |
| d19_le8 | all | mte_balanced | mask | -0.033 [-0.040, -0.026] |

OP1–OP5 sensitivity among melanomas without in-lesion hair (thresholds from ISIC 2019 validation only):

| tier | arm | ref | op | value |
|---|---|---|---|---|
| exact | mask | erm | OP1_maxBA | +0.069 [-0.020, +0.153] |
| exact | mask | erm | OP2_spec0.80 | -0.006 [-0.044, +0.046] |
| exact | mask | erm | OP3_spec0.90 | +0.000 [-0.041, +0.043] |
| exact | mask | erm | OP4_sens0.80 | +0.012 [-0.041, +0.066] |
| exact | mask | erm | OP5_sens0.90 | -0.022 [-0.075, +0.036] |
| exact | balanced | mask | OP1_maxBA | -0.036 [-0.113, +0.050] |
| exact | balanced | mask | OP2_spec0.80 | +0.036 [-0.016, +0.080] |
| exact | balanced | mask | OP3_spec0.90 | +0.026 [-0.016, +0.072] |
| exact | balanced | mask | OP4_sens0.80 | +0.025 [-0.029, +0.078] |
| exact | balanced | mask | OP5_sens0.90 | +0.039 [-0.013, +0.089] |
| d19_le2 | mask | erm | OP1_maxBA | +0.069 [-0.029, +0.149] |
| d19_le2 | mask | erm | OP2_spec0.80 | -0.006 [-0.044, +0.044] |
| d19_le2 | mask | erm | OP3_spec0.90 | +0.002 [-0.039, +0.043] |
| d19_le2 | mask | erm | OP4_sens0.80 | +0.012 [-0.039, +0.065] |
| d19_le2 | mask | erm | OP5_sens0.90 | -0.022 [-0.075, +0.035] |
| d19_le2 | balanced | mask | OP1_maxBA | -0.037 [-0.116, +0.047] |
| d19_le2 | balanced | mask | OP2_spec0.80 | +0.036 [-0.017, +0.080] |
| d19_le2 | balanced | mask | OP3_spec0.90 | +0.025 [-0.020, +0.070] |
| d19_le2 | balanced | mask | OP4_sens0.80 | +0.025 [-0.031, +0.078] |
| d19_le2 | balanced | mask | OP5_sens0.90 | +0.040 [-0.013, +0.087] |
| calibrated | mask | erm | OP1_maxBA | +0.069 [-0.025, +0.146] |
| calibrated | mask | erm | OP2_spec0.80 | -0.006 [-0.044, +0.044] |
| calibrated | mask | erm | OP3_spec0.90 | +0.002 [-0.040, +0.045] |
| calibrated | mask | erm | OP4_sens0.80 | +0.012 [-0.041, +0.067] |
| calibrated | mask | erm | OP5_sens0.90 | -0.022 [-0.076, +0.039] |
| calibrated | balanced | mask | OP1_maxBA | -0.037 [-0.118, +0.048] |
| calibrated | balanced | mask | OP2_spec0.80 | +0.036 [-0.018, +0.082] |
| calibrated | balanced | mask | OP3_spec0.90 | +0.025 [-0.024, +0.070] |
| calibrated | balanced | mask | OP4_sens0.80 | +0.025 [-0.026, +0.079] |
| calibrated | balanced | mask | OP5_sens0.90 | +0.040 [-0.015, +0.089] |
| d19_le8 | mask | erm | OP1_maxBA | +0.087 [-0.011, +0.168] |
| d19_le8 | mask | erm | OP2_spec0.80 | +0.012 [-0.031, +0.063] |
| d19_le8 | mask | erm | OP3_spec0.90 | +0.023 [-0.024, +0.072] |
| d19_le8 | mask | erm | OP4_sens0.80 | +0.022 [-0.031, +0.079] |
| d19_le8 | mask | erm | OP5_sens0.90 | -0.007 [-0.060, +0.047] |
| d19_le8 | balanced | mask | OP1_maxBA | -0.057 [-0.136, +0.029] |
| d19_le8 | balanced | mask | OP2_spec0.80 | +0.015 [-0.041, +0.058] |
| d19_le8 | balanced | mask | OP3_spec0.90 | +0.002 [-0.050, +0.048] |
| d19_le8 | balanced | mask | OP4_sens0.80 | +0.009 [-0.048, +0.063] |
| d19_le8 | balanced | mask | OP5_sens0.90 | +0.021 [-0.032, +0.072] |

**H12** calibrated hard-pair AUROC mask − ERM < 0: -0.037 [-0.061, -0.012] -> **robust**.

**H13** strict hard-pair AUROC mask − ERM < 0: -0.005 [-0.057, +0.048].
Gate met (53 H0 melanomas, 1868 Hin benign lesions).
Without Holm: **NOT SUPPORTED**.
With Holm over {H12, H13} (one-sided bootstrap p, H12 p=0.0010 -> 0.0020, H13 p=0.4379 -> 0.4379): H12 rejects 0 at 0.05; H13 **NOT SUPPORTED**.

R13 descriptive pairs on the calibrated test set (strict easy; balanced−mask on strict hard; Hout melanomas or Hout benign lesions against H0):

| subset | arm | ref | value |
|---|---|---|---|
| strict_hard | mask | erm | -0.005 [-0.057, +0.048] |
| strict_easy | mask | erm | +0.022 [-0.019, +0.064] |
| strict_hard | balanced | mask | +0.016 [-0.039, +0.075] |
| hout_mel_vs_h0_ben | mask | erm | -0.028 [-0.060, +0.004] |
| h0_mel_vs_hout_ben | mask | erm | +0.059 [+0.010, +0.112] |

Runtime: calibration 119.2 s; fit and analysis 644.7 s. GPU: NVIDIA RTX PRO 4000 Blackwell. Commit: a02db23. d19≤8 vs R10 bootstrap max |Δ| = 5.19544619118073e-07, operating-point max |Δ| = 9.71445146547012e-17.
