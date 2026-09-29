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

