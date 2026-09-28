# Round 7 — thresholds matched on the test set

| hypothesis | model | conflicting positives | sensitivity ERM → mask | mask − ERM [95% CI] | p | Holm p | verdict |
|---|---|---|---|---|---|---|---|
| TM1 | finetuned_round6 | 78 | 0.544 → 0.390 | -0.154 [-0.302, -0.023] | 0.0096 | 0.0192 | SUPPORTED |
| TM2 | frozen_stage5 | 78 | 0.505 → 0.331 | -0.174 [-0.313, -0.012] | 0.0179 | 0.0192 | SUPPORTED |

## All matched thresholds (descriptive except TM1–TM2)

| model | cohort | match | metric | ERM | mask | mask − ERM [95% CI] |
|---|---|---|---|---|---|---|
| finetuned_round6 | thyroid | M-spec80 | sens | 0.557 | 0.540 | -0.017 [-0.124, +0.070] |
| finetuned_round6 | thyroid | M-spec80 | sens_conflict | 0.544 | 0.390 | -0.154 [-0.302, -0.023] |
| finetuned_round6 | thyroid | M-spec80 | sens_aligned | 0.563 | 0.614 | +0.051 [-0.070, +0.150] |
| finetuned_round6 | thyroid | M-spec80 | spec | 0.806 | 0.804 | -0.002 [-0.009, +0.013] |
| finetuned_round6 | thyroid | M-spec90 | sens | 0.336 | 0.364 | +0.028 [-0.070, +0.119] |
| finetuned_round6 | thyroid | M-spec90 | sens_conflict | 0.323 | 0.218 | -0.105 [-0.221, +0.019] |
| finetuned_round6 | thyroid | M-spec90 | sens_aligned | 0.343 | 0.437 | +0.094 [-0.028, +0.196] |
| finetuned_round6 | thyroid | M-spec90 | spec | 0.903 | 0.904 | +0.001 [-0.008, +0.010] |
| finetuned_round6 | thyroid | M-sens80 | sens | 0.801 | 0.801 | +0.000 [-0.005, +0.006] |
| finetuned_round6 | thyroid | M-sens80 | sens_conflict | 0.803 | 0.715 | -0.087 [-0.172, +0.018] |
| finetuned_round6 | thyroid | M-sens80 | sens_aligned | 0.800 | 0.843 | +0.043 [-0.009, +0.083] |
| finetuned_round6 | thyroid | M-sens80 | spec | 0.611 | 0.542 | -0.069 [-0.163, +0.025] |
| frozen_stage5 | thyroid | M-spec80 | sens | 0.511 | 0.536 | +0.025 [-0.083, +0.132] |
| frozen_stage5 | thyroid | M-spec80 | sens_conflict | 0.505 | 0.331 | -0.174 [-0.313, -0.012] |
| frozen_stage5 | thyroid | M-spec80 | sens_aligned | 0.514 | 0.638 | +0.124 [+0.001, +0.231] |
| frozen_stage5 | thyroid | M-spec80 | spec | 0.803 | 0.803 | -0.001 [-0.010, +0.009] |
| frozen_stage5 | thyroid | M-spec90 | sens | 0.286 | 0.311 | +0.025 [-0.085, +0.141] |
| frozen_stage5 | thyroid | M-spec90 | sens_conflict | 0.231 | 0.151 | -0.079 [-0.244, +0.086] |
| frozen_stage5 | thyroid | M-spec90 | sens_aligned | 0.313 | 0.390 | +0.077 [-0.040, +0.200] |
| frozen_stage5 | thyroid | M-spec90 | spec | 0.903 | 0.904 | +0.001 [-0.008, +0.009] |
| frozen_stage5 | thyroid | M-sens80 | sens | 0.801 | 0.801 | +0.000 [-0.006, +0.006] |
| frozen_stage5 | thyroid | M-sens80 | sens_conflict | 0.779 | 0.713 | -0.067 [-0.155, +0.026] |
| frozen_stage5 | thyroid | M-sens80 | sens_aligned | 0.811 | 0.844 | +0.033 [-0.013, +0.075] |
| frozen_stage5 | thyroid | M-sens80 | spec | 0.538 | 0.515 | -0.022 [-0.127, +0.084] |
| frozen_stage5 | isic_BCN | M-spec80 | sens | 0.635 | 0.597 | -0.038 [-0.057, -0.017] |
| frozen_stage5 | isic_BCN | M-spec80 | sens_conflict | 0.597 | 0.557 | -0.040 [-0.065, -0.015] |
| frozen_stage5 | isic_BCN | M-spec80 | sens_aligned | 0.703 | 0.669 | -0.034 [-0.062, -0.003] |
| frozen_stage5 | isic_BCN | M-spec80 | spec | 0.800 | 0.800 | -0.000 [-0.001, +0.001] |
| frozen_stage5 | isic_BCN | M-spec90 | sens | 0.476 | 0.461 | -0.015 [-0.036, +0.007] |
| frozen_stage5 | isic_BCN | M-spec90 | sens_conflict | 0.436 | 0.422 | -0.015 [-0.040, +0.010] |
| frozen_stage5 | isic_BCN | M-spec90 | sens_aligned | 0.547 | 0.531 | -0.016 [-0.049, +0.016] |
| frozen_stage5 | isic_BCN | M-spec90 | spec | 0.900 | 0.900 | +0.000 [-0.001, +0.001] |
| frozen_stage5 | isic_BCN | M-sens80 | sens | 0.800 | 0.800 | +0.000 [-0.000, +0.000] |
| frozen_stage5 | isic_BCN | M-sens80 | sens_conflict | 0.770 | 0.774 | +0.004 [-0.008, +0.014] |
| frozen_stage5 | isic_BCN | M-sens80 | sens_aligned | 0.855 | 0.848 | -0.007 [-0.025, +0.014] |
| frozen_stage5 | isic_BCN | M-sens80 | spec | 0.606 | 0.526 | -0.080 [-0.108, -0.051] |
| frozen_stage5 | isic_HAM | M-spec80 | sens | 0.567 | 0.626 | +0.059 [+0.027, +0.093] |
| frozen_stage5 | isic_HAM | M-spec80 | sens_conflict | 0.528 | 0.595 | +0.067 [+0.032, +0.106] |
| frozen_stage5 | isic_HAM | M-spec80 | sens_aligned | 0.671 | 0.711 | +0.039 [-0.019, +0.095] |
| frozen_stage5 | isic_HAM | M-spec80 | spec | 0.800 | 0.800 | +0.000 [-0.002, +0.002] |
| frozen_stage5 | isic_HAM | M-spec90 | sens | 0.411 | 0.452 | +0.041 [+0.010, +0.072] |
| frozen_stage5 | isic_HAM | M-spec90 | sens_conflict | 0.365 | 0.415 | +0.050 [+0.014, +0.085] |
| frozen_stage5 | isic_HAM | M-spec90 | sens_aligned | 0.532 | 0.551 | +0.020 [-0.038, +0.075] |
| frozen_stage5 | isic_HAM | M-spec90 | spec | 0.900 | 0.901 | +0.001 [-0.001, +0.001] |
| frozen_stage5 | isic_HAM | M-sens80 | sens | 0.801 | 0.801 | +0.000 [-0.001, +0.001] |
| frozen_stage5 | isic_HAM | M-sens80 | sens_conflict | 0.767 | 0.781 | +0.014 [+0.000, +0.027] |
| frozen_stage5 | isic_HAM | M-sens80 | sens_aligned | 0.890 | 0.852 | -0.038 [-0.071, -0.001] |
| frozen_stage5 | isic_HAM | M-sens80 | spec | 0.524 | 0.596 | +0.072 [+0.033, +0.112] |
| frozen_stage5 | isic_MSK | M-spec80 | sens | 0.593 | 0.614 | +0.020 [-0.030, +0.070] |
| frozen_stage5 | isic_MSK | M-spec80 | sens_conflict | 0.586 | 0.585 | -0.001 [-0.054, +0.053] |
| frozen_stage5 | isic_MSK | M-spec80 | sens_aligned | 0.653 | 0.847 | +0.193 [+0.067, +0.311] |
| frozen_stage5 | isic_MSK | M-spec80 | spec | 0.802 | 0.801 | -0.001 [-0.004, +0.006] |
| frozen_stage5 | isic_MSK | M-spec90 | sens | 0.435 | 0.470 | +0.036 [-0.018, +0.085] |
| frozen_stage5 | isic_MSK | M-spec90 | sens_conflict | 0.426 | 0.443 | +0.017 [-0.040, +0.068] |
| frozen_stage5 | isic_MSK | M-spec90 | sens_aligned | 0.507 | 0.697 | +0.190 [+0.074, +0.307] |
| frozen_stage5 | isic_MSK | M-spec90 | spec | 0.901 | 0.901 | -0.000 [-0.003, +0.003] |
| frozen_stage5 | isic_MSK | M-sens80 | sens | 0.801 | 0.801 | +0.000 [-0.002, +0.002] |
| frozen_stage5 | isic_MSK | M-sens80 | sens_conflict | 0.791 | 0.780 | -0.011 [-0.022, -0.002] |
| frozen_stage5 | isic_MSK | M-sens80 | sens_aligned | 0.877 | 0.970 | +0.093 [+0.017, +0.175] |
| frozen_stage5 | isic_MSK | M-sens80 | spec | 0.562 | 0.593 | +0.032 [-0.026, +0.097] |
| frozen_stage5 | capsule | M-spec80 | sens | 0.832 | 0.963 | +0.131 [+0.098, +0.179] |
| frozen_stage5 | capsule | M-spec80 | sens_conflict | 0.887 | 0.924 | +0.037 [-0.013, +0.110] |
| frozen_stage5 | capsule | M-spec80 | sens_aligned | 0.824 | 0.968 | +0.144 [+0.107, +0.196] |
| frozen_stage5 | capsule | M-spec80 | spec | 0.804 | 0.807 | +0.003 [-0.005, +0.022] |
| frozen_stage5 | capsule | M-spec90 | sens | 0.676 | 0.909 | +0.233 [+0.167, +0.304] |
| frozen_stage5 | capsule | M-spec90 | sens_conflict | 0.736 | 0.846 | +0.111 [+0.011, +0.225] |
| frozen_stage5 | capsule | M-spec90 | sens_aligned | 0.667 | 0.916 | +0.249 [+0.184, +0.322] |
| frozen_stage5 | capsule | M-spec90 | spec | 0.901 | 0.901 | +0.000 [-0.005, +0.009] |
| frozen_stage5 | capsule | M-sens80 | sens | 0.801 | 0.801 | +0.000 [-0.003, +0.003] |
| frozen_stage5 | capsule | M-sens80 | sens_conflict | 0.861 | 0.716 | -0.145 [-0.233, -0.065] |
| frozen_stage5 | capsule | M-sens80 | sens_aligned | 0.792 | 0.810 | +0.018 [+0.008, +0.032] |
| frozen_stage5 | capsule | M-sens80 | spec | 0.825 | 0.957 | +0.132 [+0.096, +0.174] |
