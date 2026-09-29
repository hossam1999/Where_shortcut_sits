# Archived analyses regenerated (A1 addendum, docs/PREREGISTRATION_FINAL.md)

Old: archived runs without per-image predictions (per-seed bootstrap or point estimate). New: regenerated runs in results/rerun_2026-09-28/ with saved predictions, crossed seed x image bootstrap (10,000). Verdict = CI excludes zero.

## capsule template U-MtE

- rows: 32; regenerated: 32; verdict changed: 1; point estimates differing by more than 0.03: 4

| file | key | old | new | verdict changed | Δ > 0.03 | note |
|---|---|---|---|---|---|---|
| capsule/dino518_protect_tmpl/paired_deltas.csv | trapA|mte_protect|mte|test_rev | +0.383 [+0.357, +0.414] | +0.395 [+0.361, +0.429] | no |  |  |
| capsule/dino518_protect_tmpl/paired_deltas.csv | trapA|mte_protect|mte|clean | +0.159 [+0.143, +0.175] | +0.183 [+0.158, +0.208] | no |  |  |
| capsule/dino518_protect_tmpl/paired_deltas.csv | trapA|mte_protect|mask|test_rev | -0.012 [-0.028, +0.003] | +0.003 [-0.014, +0.021] | no |  |  |
| capsule/dino518_protect_tmpl/paired_deltas.csv | trapA|mte_protect|mask|clean | -0.028 [-0.036, -0.020] | -0.016 [-0.028, -0.006] | no |  |  |
| capsule/dino518_protect_tmpl/paired_deltas.csv | trapA|mte_protect_balanced|mte_balanced|test_rev | +0.125 [+0.097, +0.154] | +0.176 [+0.140, +0.213] | no | **yes** |  |
| capsule/dino518_protect_tmpl/paired_deltas.csv | trapA|mte_protect_balanced|mte_balanced|clean | +0.085 [+0.058, +0.113] | +0.135 [+0.108, +0.163] | no | **yes** |  |
| capsule/dino518_protect_tmpl/paired_deltas.csv | trapB|mte_protect|mte|test_rev | +0.095 [+0.067, +0.120] | +0.103 [+0.075, +0.132] | no |  |  |
| capsule/dino518_protect_tmpl/paired_deltas.csv | trapB|mte_protect|mte|clean | +0.066 [+0.049, +0.083] | +0.074 [+0.054, +0.094] | no |  |  |
| capsule/dino518_protect_tmpl/paired_deltas.csv | trapB|mte_protect|mask|test_rev | -0.029 [-0.040, -0.018] | -0.019 [-0.032, -0.007] | no |  |  |
| capsule/dino518_protect_tmpl/paired_deltas.csv | trapB|mte_protect|mask|clean | -0.029 [-0.040, -0.018] | -0.020 [-0.029, -0.011] | no |  |  |
| capsule/dino518_protect_tmpl/paired_deltas.csv | trapB|mte_protect_balanced|mte_balanced|test_rev | +0.083 [+0.052, +0.113] | +0.084 [+0.056, +0.112] | no |  |  |
| capsule/dino518_protect_tmpl/paired_deltas.csv | trapB|mte_protect_balanced|mte_balanced|clean | +0.073 [+0.053, +0.091] | +0.068 [+0.047, +0.089] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapA|mask|test_rev|-erm | +0.096 [+0.081, +0.113] | +0.084 [+0.055, +0.112] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapA|mask|clean|-erm | +0.075 [+0.061, +0.088] | +0.064 [+0.044, +0.086] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapA|mte|test_rev|-erm | -0.298 [-0.321, -0.276] | -0.308 [-0.348, -0.269] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapA|mte|clean|-erm | -0.113 [-0.135, -0.091] | -0.135 [-0.165, -0.104] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapA|mte_balanced|test_rev|-erm | +0.123 [+0.098, +0.148] | +0.088 [+0.047, +0.129] | no | **yes** |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapA|mte_balanced|clean|-erm | +0.007 [-0.018, +0.034] | -0.032 [-0.066, -0.002] | **yes** | **yes** |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapA|mte_protect|test_rev|-erm | +0.084 [+0.060, +0.108] | +0.087 [+0.054, +0.119] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapA|mte_protect|clean|-erm | +0.046 [+0.031, +0.061] | +0.048 [+0.022, +0.075] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapA|mte_protect_balanced|test_rev|-erm | +0.249 [+0.226, +0.271] | +0.264 [+0.231, +0.297] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapA|mte_protect_balanced|clean|-erm | +0.092 [+0.075, +0.108] | +0.103 [+0.079, +0.128] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapB|mask|test_rev|-erm | +0.465 [+0.440, +0.490] | +0.461 [+0.415, +0.506] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapB|mask|clean|-erm | +0.206 [+0.188, +0.224] | +0.217 [+0.186, +0.251] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapB|mte|test_rev|-erm | +0.342 [+0.311, +0.370] | +0.339 [+0.287, +0.389] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapB|mte|clean|-erm | +0.111 [+0.088, +0.134] | +0.124 [+0.089, +0.158] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapB|mte_balanced|test_rev|-erm | +0.370 [+0.334, +0.406] | +0.378 [+0.326, +0.428] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapB|mte_balanced|clean|-erm | +0.106 [+0.083, +0.129] | +0.133 [+0.098, +0.169] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapB|mte_protect|test_rev|-erm | +0.436 [+0.410, +0.462] | +0.442 [+0.395, +0.486] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapB|mte_protect|clean|-erm | +0.178 [+0.161, +0.195] | +0.198 [+0.166, +0.230] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapB|mte_protect_balanced|test_rev|-erm | +0.453 [+0.426, +0.482] | +0.462 [+0.414, +0.509] | no |  |  |
| capsule/dino518_protect_tmpl/bootstrap_vs_erm.csv | trapB|mte_protect_balanced|clean|-erm | +0.179 [+0.162, +0.195] | +0.201 [+0.169, +0.234] | no |  |  |

## LaMa

- rows: 19; regenerated: 19; verdict changed: 2; point estimates differing by more than 0.03: 2

| file | key | old | new | verdict changed | Δ > 0.03 | note |
|---|---|---|---|---|---|---|
| lama_comparison.json | thyroid|lama-mask | +0.025 [+0.009, +0.040] | +0.027 [+0.000, +0.054] | no |  |  |
| lama_comparison.json | thyroid|mask_lama-mask | +0.226 [+0.213, +0.238] | +0.223 [+0.206, +0.242] | no |  |  |
| lama_comparison.json | thyroid|mte_protect-mask_lama | -0.009 [-0.045, +0.023] | +0.060 [+0.013, +0.105] | **yes** | **yes** |  |
| lama_comparison.json | thyroid|mte_balanced-mask_lama | +0.162 [+0.152, +0.173] | +0.165 [+0.146, +0.185] | no |  |  |
| lama_comparison.json | thyroid|lama-erm | +0.136 [+0.130, +0.142] | +0.136 [+0.124, +0.149] | no |  |  |
| lama_comparison.json | thyroid|mte_protect-mask_lama (universal run) | -0.009 [-0.045, +0.023] | +0.060 [+0.013, +0.105] | **yes** | **yes** | extra: protected arm of the regenerated universal run instead of protect_generic |
| thyroid/dino518_lama/bootstrap_vs_erm.csv | trapA|mask|test_rev|-erm | +0.201 [+0.187, +0.215] | +0.197 [+0.167, +0.226] | no |  |  |
| thyroid/dino518_lama/bootstrap_vs_erm.csv | trapA|mask|clean|-erm | +0.090 [+0.076, +0.104] | +0.083 [+0.058, +0.109] | no |  |  |
| thyroid/dino518_lama/bootstrap_vs_erm.csv | trapB|mask|test_rev|-erm | +0.271 [+0.246, +0.296] | +0.281 [+0.242, +0.320] | no |  |  |
| thyroid/dino518_lama/bootstrap_vs_erm.csv | trapB|mask|clean|-erm | +0.135 [+0.100, +0.168] | +0.157 [+0.112, +0.201] | no |  |  |
| lama_comparison.json | ovary|lama-mask | +0.027 [-0.017, +0.069] | +0.027 [-0.039, +0.091] | no |  |  |
| lama_comparison.json | ovary|mask_lama-mask | +0.123 [+0.089, +0.155] | +0.128 [+0.084, +0.169] | no |  |  |
| lama_comparison.json | ovary|mte_protect-mask_lama | -0.080 [-0.108, -0.053] | -0.085 [-0.120, -0.050] | no |  |  |
| lama_comparison.json | ovary|mte_balanced-mask_lama | +0.067 [+0.037, +0.095] | +0.061 [+0.026, +0.095] | no |  |  |
| lama_comparison.json | ovary|lama-erm | +0.158 [+0.141, +0.176] | +0.159 [+0.130, +0.190] | no |  |  |
| ovary/dino518_lama/bootstrap_vs_erm.csv | trapA|mask|test_rev|-erm | +0.096 [+0.062, +0.132] | +0.101 [+0.046, +0.157] | no |  |  |
| ovary/dino518_lama/bootstrap_vs_erm.csv | trapA|mask|clean|-erm | +0.067 [+0.045, +0.089] | +0.068 [+0.028, +0.108] | no |  |  |
| ovary/dino518_lama/bootstrap_vs_erm.csv | trapB|mask|test_rev|-erm | +0.253 [+0.212, +0.294] | +0.253 [+0.177, +0.329] | no |  |  |
| ovary/dino518_lama/bootstrap_vs_erm.csv | trapB|mask|clean|-erm | +0.087 [+0.047, +0.125] | +0.087 [+0.027, +0.144] | no |  |  |

