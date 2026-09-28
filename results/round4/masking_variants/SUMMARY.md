# R8 — masking implementations (DINOv2 ViT-B/14 @518, real traps)

Environment: {"commit": "f381bfc", "host": "074495c7c1a5", "WTSS_BOOTSTRAP": "crossed", "gpu": "NVIDIA RTX PRO 4000 Blackwell"}

Primary: crossover C_v = [rev(v) - rev(ERM)]_TrapB - [rev(v) - rev(ERM)]_TrapA, crossed bootstrap, Holm over 16.

| cohort | view | C_v | p_holm16 | supported |
|---|---|---|---|---|
| ISIC hair | mask_black | +0.170 [+0.133, +0.207] | 0.002 | True |
| ISIC hair | mask_blur | +0.154 [+0.117, +0.191] | 0.002 | True |
| ISIC hair | crop_box | +0.091 [+0.064, +0.117] | 0.002 | True |
| ISIC hair | crop_mask | +0.154 [+0.121, +0.187] | 0.002 | True |
| Thyroid calipers | mask_black | +0.223 [+0.181, +0.265] | 0.002 | True |
| Thyroid calipers | mask_blur | +0.218 [+0.177, +0.258] | 0.002 | True |
| Thyroid calipers | crop_box | +0.246 [+0.206, +0.285] | 0.002 | True |
| Thyroid calipers | crop_mask | +0.250 [+0.210, +0.290] | 0.002 | True |
| Capsule debris | mask_black | +0.367 [+0.318, +0.416] | 0.002 | True |
| Capsule debris | mask_blur | +0.278 [+0.221, +0.330] | 0.002 | True |
| Capsule debris | crop_box | +0.317 [+0.267, +0.367] | 0.002 | True |
| Capsule debris | crop_mask | +0.358 [+0.305, +0.409] | 0.002 | True |
| Ovary calipers | mask_black | +0.153 [+0.063, +0.242] | 0.002 | True |
| Ovary calipers | mask_blur | +0.112 [+0.027, +0.195] | 0.010 | True |
| Ovary calipers | crop_box | +0.145 [+0.083, +0.207] | 0.002 | True |
| Ovary calipers | crop_mask | +0.160 [+0.080, +0.240] | 0.002 | True |

**H8a** (full-removal views, 8 tests): 8/8 positive and Holm-significant -> decision: **implementation-independent**.

Reference (mean-colour fill, re-fitted here):

| cohort | C_mask |
|---|---|
| ISIC hair | +0.147 [+0.111, +0.185] |
| Thyroid calipers | +0.239 [+0.196, +0.280] |
| Capsule debris | +0.377 [+0.328, +0.426] |
| Ovary calipers | +0.170 [+0.088, +0.250] |

**H8b** secondary contrast C_mask - C_v (partial-removal views expected > 0):

| cohort | view | contrast |
|---|---|---|
| ISIC hair | mask_black | -0.023 [-0.040, -0.007] |
| ISIC hair | mask_blur | -0.006 [-0.042, +0.024] |
| ISIC hair | crop_box | +0.056 [+0.012, +0.099] |
| ISIC hair | crop_mask | -0.007 [-0.033, +0.019] |
| Thyroid calipers | mask_black | +0.016 [-0.004, +0.036] |
| Thyroid calipers | mask_blur | +0.020 [-0.008, +0.049] |
| Thyroid calipers | crop_box | -0.007 [-0.049, +0.032] |
| Thyroid calipers | crop_mask | -0.011 [-0.037, +0.014] |
| Capsule debris | mask_black | +0.010 [-0.001, +0.021] |
| Capsule debris | mask_blur | +0.099 [+0.067, +0.132] |
| Capsule debris | crop_box | +0.059 [+0.032, +0.087] |
| Capsule debris | crop_mask | +0.019 [+0.001, +0.037] |
| Ovary calipers | mask_black | +0.017 [-0.034, +0.079] |
| Ovary calipers | mask_blur | +0.058 [+0.004, +0.113] |
| Ovary calipers | crop_box | +0.024 [-0.045, +0.096] |
| Ovary calipers | crop_mask | +0.010 [-0.046, +0.068] |

H8b crossover part: 8/8 partial-removal crossovers positive and Holm-significant.

Secondary (descriptive): view minus ERM per trap, reversed (delta_rev) and clean (delta_clean) AUROC.

| cohort | view | trapA_delta_clean | trapA_delta_rev | trapB_delta_clean | trapB_delta_rev |
|---|---|---|---|---|---|
| Capsule debris | crop_box | +0.046 [+0.024, +0.068] | +0.063 [+0.037, +0.090] | +0.174 [+0.145, +0.206] | +0.381 [+0.336, +0.425] |
| Capsule debris | crop_mask | +0.056 [+0.033, +0.079] | +0.091 [+0.060, +0.120] | +0.205 [+0.173, +0.238] | +0.448 [+0.403, +0.494] |
| Capsule debris | mask | +0.064 [+0.044, +0.086] | +0.084 [+0.055, +0.113] | +0.217 [+0.186, +0.251] | +0.461 [+0.416, +0.506] |
| Capsule debris | mask_black | +0.063 [+0.044, +0.084] | +0.086 [+0.056, +0.116] | +0.215 [+0.183, +0.248] | +0.453 [+0.410, +0.497] |
| Capsule debris | mask_blur | +0.047 [+0.027, +0.068] | +0.051 [+0.024, +0.077] | +0.156 [+0.129, +0.184] | +0.329 [+0.282, +0.374] |
| ISIC hair | crop_box | -0.005 [-0.019, +0.009] | -0.017 [-0.038, +0.003] | +0.010 [-0.007, +0.026] | +0.074 [+0.049, +0.098] |
| ISIC hair | crop_mask | -0.010 [-0.030, +0.011] | -0.030 [-0.062, +0.002] | +0.011 [-0.014, +0.035] | +0.124 [+0.089, +0.160] |
| ISIC hair | mask | -0.023 [-0.045, -0.001] | -0.049 [-0.083, -0.017] | -0.014 [-0.041, +0.014] | +0.098 [+0.060, +0.136] |
| ISIC hair | mask_black | -0.028 [-0.052, -0.005] | -0.061 [-0.097, -0.028] | -0.006 [-0.034, +0.022] | +0.109 [+0.071, +0.148] |
| ISIC hair | mask_blur | -0.004 [-0.022, +0.014] | -0.023 [-0.051, +0.005] | +0.020 [-0.006, +0.047] | +0.131 [+0.096, +0.166] |
| Ovary calipers | crop_box | +0.041 [+0.014, +0.068] | +0.037 [+0.000, +0.072] | +0.078 [+0.038, +0.117] | +0.182 [+0.122, +0.241] |
| Ovary calipers | crop_mask | +0.050 [+0.007, +0.092] | +0.135 [+0.078, +0.194] | +0.107 [+0.046, +0.167] | +0.296 [+0.219, +0.371] |
| Ovary calipers | mask | +0.055 [+0.014, +0.095] | +0.132 [+0.071, +0.195] | +0.097 [+0.032, +0.159] | +0.302 [+0.225, +0.378] |
| Ovary calipers | mask_black | +0.028 [-0.013, +0.069] | +0.113 [+0.053, +0.175] | +0.093 [+0.037, +0.149] | +0.265 [+0.188, +0.342] |
| Ovary calipers | mask_blur | +0.050 [+0.012, +0.086] | +0.074 [+0.021, +0.130] | +0.065 [+0.014, +0.117] | +0.186 [+0.116, +0.257] |
| Thyroid calipers | crop_box | +0.015 [-0.003, +0.032] | -0.006 [-0.023, +0.011] | +0.087 [+0.040, +0.135] | +0.239 [+0.203, +0.277] |
| Thyroid calipers | crop_mask | +0.083 [+0.063, +0.101] | +0.092 [+0.070, +0.115] | +0.174 [+0.129, +0.219] | +0.342 [+0.303, +0.381] |
| Thyroid calipers | mask | +0.081 [+0.063, +0.100] | +0.109 [+0.086, +0.133] | +0.166 [+0.118, +0.213] | +0.348 [+0.307, +0.388] |
| Thyroid calipers | mask_black | +0.067 [+0.044, +0.089] | +0.116 [+0.093, +0.140] | +0.171 [+0.125, +0.218] | +0.339 [+0.297, +0.379] |
| Thyroid calipers | mask_blur | +0.065 [+0.044, +0.085] | +0.107 [+0.086, +0.130] | +0.154 [+0.109, +0.199] | +0.326 [+0.286, +0.365] |
