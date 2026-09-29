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

**Provenance of the thyroid U-MtE-protect change** (checked 2026-09-29, no GPU run): the archived thyroid protect_generic run was committed in 419b252 (2026-09-27 01:56), before 001f743 (04:40, protected arm added to the drivers) and before later runner changes. The regenerated protect_generic run gives U-MtE-protect − ERM +0.392 on Trap A reversed, equal to the protected arm of the regenerated universal run (+0.392), against +0.328 archived; every other arm of the run reproduces within 0.002. `git diff 419b252 HEAD -- scripts/run_thyroid_traps.py src/wtss/experiments/ src/wtss/methods/insertion.py`, restricted to the protected arm, shows no change in its computation: `disease_directions` and `protect` are identical, the subspace fit keeps energy 0.9 and at most 64 directions, and the call site is the same. The diff changes only the cache-coverage check (`_covered`), the BLAS thread limit (`threadpool_limits(2)` removed), the process-pool conditions and adds new arms. The cached feature files the arm reads were rewritten after the archived run (erm/mask 2026-09-27 19:05, mask_insert_generic 2026-09-28 00:56), so the two runs used different extractions of the same views. The cause is still unclear; nothing was adjusted.

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

## chest drains

- rows: 32; regenerated: 32; verdict changed: 10; point estimates differing by more than 0.03: 0

| file | key | old | new | verdict changed | Δ > 0.03 | note |
|---|---|---|---|---|---|---|
| cxr_drain/raddino518_universal/paired_deltas.csv | mask|erm|test_rev | +0.049 [+0.035, +0.061] | +0.039 [+0.019, +0.058] | no |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | mask|erm|clean | -0.016 [-0.024, -0.008] | -0.015 [-0.023, -0.007] | no |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | mte|mask|test_rev | +0.029 [+0.021, +0.036] | +0.040 [+0.023, +0.057] | no |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | mte|mask|clean | -0.024 [-0.030, -0.018] | -0.021 [-0.028, -0.014] | no |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | mte|mte_aug|test_rev | +0.031 [+0.022, +0.042] | +0.036 [+0.018, +0.054] | no |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | mte|mte_aug|clean | -0.020 [-0.026, -0.014] | -0.020 [-0.027, -0.014] | no |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | mte_protect|mask|test_rev | +0.030 [+0.022, +0.037] | +0.042 [+0.028, +0.058] | no |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | mte_protect|mask|clean | -0.003 [-0.009, +0.002] | -0.004 [-0.009, -0.001] | **yes** |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | mte_balanced|balanced|test_rev | -0.079 [-0.097, -0.061] | -0.079 [-0.106, -0.044] | no |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | mte_balanced|balanced|clean | -0.027 [-0.035, -0.020] | -0.029 [-0.039, -0.018] | no |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | mask_dfr|dfr|test_rev | -0.036 [-0.055, -0.014] | -0.031 [-0.060, -0.002] | no |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | mask_dfr|dfr|clean | +0.001 [-0.011, +0.013] | +0.000 [-0.022, +0.021] | no |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | jtt|erm|test_rev | -0.016 [-0.020, -0.011] | -0.015 [-0.023, -0.006] | no |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | jtt|erm|clean | -0.012 [-0.019, -0.005] | -0.012 [-0.024, +0.003] | **yes** |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | mte|jtt|test_rev | +0.094 [+0.077, +0.109] | +0.094 [+0.077, +0.111] | no |  |  |
| cxr_drain/raddino518_universal/paired_deltas.csv | mte|jtt|clean | -0.028 [-0.041, -0.014] | -0.024 [-0.039, -0.009] | no |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | mask|erm|test_rev | +0.012 [+0.002, +0.023] | +0.005 [-0.013, +0.023] | **yes** |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | mask|erm|clean | +0.012 [-0.000, +0.023] | +0.011 [-0.004, +0.029] | no |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | mte|mask|test_rev | -0.008 [-0.015, +0.000] | -0.001 [-0.009, +0.008] | no |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | mte|mask|clean | -0.000 [-0.006, +0.006] | -0.001 [-0.008, +0.007] | no |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | mte|mte_aug|test_rev | -0.010 [-0.016, -0.002] | -0.002 [-0.010, +0.008] | **yes** |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | mte|mte_aug|clean | -0.001 [-0.007, +0.005] | -0.001 [-0.007, +0.006] | no |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | mte_protect|mask|test_rev | -0.011 [-0.019, -0.003] | -0.003 [-0.013, +0.006] | **yes** |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | mte_protect|mask|clean | -0.016 [-0.023, -0.009] | -0.013 [-0.019, -0.007] | no |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | mte_balanced|balanced|test_rev | +0.044 [+0.025, +0.064] | +0.031 [-0.023, +0.079] | **yes** |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | mte_balanced|balanced|clean | +0.023 [+0.010, +0.037] | +0.020 [+0.001, +0.040] | no |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | mask_dfr|dfr|test_rev | +0.018 [+0.001, +0.034] | +0.023 [-0.004, +0.050] | **yes** |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | mask_dfr|dfr|clean | +0.017 [-0.003, +0.039] | +0.017 [+0.000, +0.033] | **yes** |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | jtt|erm|test_rev | +0.014 [+0.009, +0.019] | +0.009 [+0.002, +0.016] | no |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | jtt|erm|clean | -0.004 [-0.011, +0.002] | -0.001 [-0.010, +0.006] | no |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | mte|jtt|test_rev | -0.010 [-0.019, -0.000] | -0.005 [-0.021, +0.011] | **yes** |  |  |
| cxr_drain/dino518_universal/paired_deltas.csv | mte|jtt|clean | +0.016 [+0.004, +0.028] | +0.012 [-0.005, +0.029] | **yes** |  |  |

## chest devices

- rows: 180; regenerated: 180; verdict changed: 19; point estimates differing by more than 0.03: 9

| file | key | old | new | verdict changed | Δ > 0.03 | note |
|---|---|---|---|---|---|---|
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|prevcal|test_rev|-erm | +0.565 [+0.539, +0.592] | +0.572 [+0.533, +0.610] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|prevcal|clean|-erm | -0.018 [-0.044, +0.008] | -0.026 [-0.076, +0.022] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|mask|test_rev|-erm | +0.045 [+0.013, +0.077] | +0.059 [+0.019, +0.099] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|mask|clean|-erm | +0.016 [+0.001, +0.030] | +0.020 [-0.002, +0.040] | **yes** |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|inpaint|test_rev|-erm | +0.100 [+0.083, +0.117] | +0.107 [+0.086, +0.131] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|inpaint|clean|-erm | +0.019 [+0.006, +0.032] | +0.018 [+0.002, +0.036] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|balanced|test_rev|-erm | +0.266 [+0.229, +0.311] | +0.300 [+0.246, +0.358] | no | **yes** |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|balanced|clean|-erm | +0.054 [+0.037, +0.072] | +0.068 [+0.037, +0.100] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|dfr|test_rev|-erm | +0.306 [+0.261, +0.353] | +0.287 [+0.231, +0.340] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|dfr|clean|-erm | +0.016 [-0.020, +0.048] | +0.005 [-0.033, +0.044] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|leace_paired|test_rev|-erm | +0.092 [+0.075, +0.109] | +0.107 [+0.086, +0.131] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|leace_paired|clean|-erm | +0.028 [+0.014, +0.041] | +0.027 [+0.012, +0.043] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|leace_unpaired|test_rev|-erm | +0.375 [+0.343, +0.408] | +0.394 [+0.340, +0.449] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|leace_unpaired|clean|-erm | +0.042 [+0.016, +0.067] | +0.062 [+0.025, +0.100] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|i2e|test_rev|-erm | +0.002 [-0.007, +0.012] | +0.008 [-0.023, +0.042] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|i2e|clean|-erm | -0.002 [-0.010, +0.007] | +0.006 [-0.013, +0.025] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|i2e_balanced|test_rev|-erm | +0.256 [+0.216, +0.293] | +0.322 [+0.277, +0.366] | no | **yes** |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|i2e_balanced|clean|-erm | +0.049 [+0.027, +0.071] | +0.063 [+0.031, +0.096] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|i2e_rank1|test_rev|-erm | -0.007 [-0.023, +0.005] | +0.004 [-0.010, +0.018] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|i2e_rank1|clean|-erm | -0.003 [-0.011, +0.004] | +0.003 [-0.005, +0.012] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|insert_aug|test_rev|-erm | -0.005 [-0.017, +0.006] | +0.004 [-0.012, +0.022] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapA|insert_aug|clean|-erm | -0.001 [-0.008, +0.006] | +0.002 [-0.009, +0.013] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|prevcal|test_rev|-erm | +0.569 [+0.514, +0.611] | +0.569 [+0.512, +0.614] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|prevcal|clean|-erm | -0.022 [-0.041, -0.004] | -0.022 [-0.053, +0.007] | **yes** |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|mask|test_rev|-erm | +0.105 [+0.094, +0.114] | +0.105 [+0.092, +0.117] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|mask|clean|-erm | +0.030 [+0.021, +0.039] | +0.030 [+0.017, +0.042] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|inpaint|test_rev|-erm | +0.007 [-0.001, +0.011] | +0.007 [-0.002, +0.011] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|inpaint|clean|-erm | +0.003 [+0.001, +0.006] | +0.003 [+0.001, +0.007] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|balanced|test_rev|-erm | +0.302 [+0.287, +0.314] | +0.302 [+0.284, +0.319] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|balanced|clean|-erm | +0.066 [+0.051, +0.082] | +0.066 [+0.047, +0.086] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|dfr|test_rev|-erm | +0.328 [+0.301, +0.353] | +0.328 [+0.299, +0.356] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|dfr|clean|-erm | +0.040 [+0.025, +0.056] | +0.040 [+0.019, +0.061] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|leace_paired|test_rev|-erm | +0.019 [+0.011, +0.026] | +0.019 [+0.011, +0.027] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|leace_paired|clean|-erm | +0.007 [+0.005, +0.011] | +0.007 [+0.004, +0.012] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|leace_unpaired|test_rev|-erm | +0.357 [+0.341, +0.371] | +0.357 [+0.339, +0.376] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|leace_unpaired|clean|-erm | +0.074 [+0.060, +0.090] | +0.074 [+0.054, +0.095] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|i2e|test_rev|-erm | +0.003 [-0.006, +0.010] | +0.003 [-0.006, +0.011] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|i2e|clean|-erm | -0.005 [-0.010, +0.002] | -0.005 [-0.012, +0.003] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|i2e_balanced|test_rev|-erm | +0.285 [+0.274, +0.296] | +0.285 [+0.270, +0.300] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|i2e_balanced|clean|-erm | +0.054 [+0.038, +0.071] | +0.054 [+0.035, +0.075] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|i2e_rank1|test_rev|-erm | +0.002 [+0.001, +0.004] | +0.002 [+0.001, +0.004] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|i2e_rank1|clean|-erm | +0.001 [+0.000, +0.003] | +0.001 [+0.000, +0.003] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|insert_aug|test_rev|-erm | +0.003 [+0.001, +0.006] | +0.003 [+0.001, +0.006] | no |  |  |
| cxr_traps/raddino518/Atelectasis/bootstrap_vs_erm.csv | trapB|insert_aug|clean|-erm | +0.001 [-0.001, +0.003] | +0.001 [-0.002, +0.004] | no |  |  |
| cxr_traps/raddino518/Atelectasis/X3_crossover.json | raddino518|Atelectasis|crossover | +0.060 [+0.028, +0.089] | +0.046 [+0.011, +0.082] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|prevcal|test_rev|-erm | +0.578 [+0.543, +0.612] | +0.564 [+0.510, +0.614] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|prevcal|clean|-erm | -0.036 [-0.070, -0.002] | -0.025 [-0.090, +0.034] | **yes** |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|mask|test_rev|-erm | +0.070 [+0.044, +0.096] | +0.074 [+0.025, +0.125] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|mask|clean|-erm | +0.007 [-0.013, +0.026] | +0.010 [-0.024, +0.045] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|inpaint|test_rev|-erm | +0.120 [+0.089, +0.152] | +0.121 [+0.084, +0.166] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|inpaint|clean|-erm | +0.016 [-0.004, +0.037] | +0.013 [-0.012, +0.039] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|balanced|test_rev|-erm | +0.286 [+0.252, +0.320] | +0.252 [+0.184, +0.325] | no | **yes** |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|balanced|clean|-erm | +0.060 [+0.034, +0.086] | +0.029 [-0.010, +0.070] | **yes** | **yes** |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|dfr|test_rev|-erm | +0.279 [+0.230, +0.328] | +0.237 [+0.166, +0.306] | no | **yes** |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|dfr|clean|-erm | -0.004 [-0.044, +0.038] | -0.016 [-0.064, +0.032] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|leace_paired|test_rev|-erm | +0.126 [+0.099, +0.154] | +0.120 [+0.075, +0.169] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|leace_paired|clean|-erm | +0.029 [+0.012, +0.047] | +0.023 [-0.003, +0.049] | **yes** |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|leace_unpaired|test_rev|-erm | +0.396 [+0.354, +0.438] | +0.357 [+0.295, +0.417] | no | **yes** |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|leace_unpaired|clean|-erm | +0.063 [+0.033, +0.094] | +0.016 [-0.037, +0.066] | **yes** | **yes** |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|i2e|test_rev|-erm | +0.016 [-0.000, +0.032] | -0.009 [-0.039, +0.017] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|i2e|clean|-erm | -0.004 [-0.016, +0.009] | -0.026 [-0.057, -0.004] | **yes** |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|i2e_balanced|test_rev|-erm | +0.298 [+0.260, +0.337] | +0.286 [+0.221, +0.351] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|i2e_balanced|clean|-erm | +0.058 [+0.030, +0.085] | +0.034 [-0.009, +0.078] | **yes** |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|i2e_rank1|test_rev|-erm | +0.001 [-0.011, +0.012] | +0.003 [-0.005, +0.012] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|i2e_rank1|clean|-erm | +0.001 [-0.009, +0.010] | +0.001 [-0.004, +0.006] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|insert_aug|test_rev|-erm | +0.010 [-0.010, +0.028] | +0.008 [-0.016, +0.029] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapA|insert_aug|clean|-erm | +0.008 [-0.005, +0.021] | -0.007 [-0.023, +0.007] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|prevcal|test_rev|-erm | +0.474 [+0.443, +0.509] | +0.474 [+0.441, +0.510] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|prevcal|clean|-erm | +0.003 [-0.010, +0.016] | +0.003 [-0.020, +0.026] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|mask|test_rev|-erm | +0.068 [+0.049, +0.086] | +0.070 [+0.047, +0.092] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|mask|clean|-erm | +0.042 [+0.030, +0.054] | +0.042 [+0.026, +0.058] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|inpaint|test_rev|-erm | +0.003 [-0.011, +0.011] | +0.003 [-0.011, +0.011] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|inpaint|clean|-erm | +0.003 [+0.000, +0.006] | +0.003 [-0.000, +0.007] | **yes** |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|balanced|test_rev|-erm | +0.190 [+0.177, +0.203] | +0.190 [+0.173, +0.207] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|balanced|clean|-erm | +0.042 [+0.028, +0.055] | +0.042 [+0.024, +0.060] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|dfr|test_rev|-erm | +0.241 [+0.218, +0.269] | +0.241 [+0.213, +0.272] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|dfr|clean|-erm | +0.031 [+0.009, +0.054] | +0.031 [+0.003, +0.059] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|leace_paired|test_rev|-erm | +0.015 [-0.001, +0.027] | +0.015 [-0.001, +0.028] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|leace_paired|clean|-erm | +0.004 [+0.000, +0.008] | +0.004 [-0.001, +0.009] | **yes** |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|leace_unpaired|test_rev|-erm | +0.300 [+0.274, +0.320] | +0.300 [+0.270, +0.326] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|leace_unpaired|clean|-erm | +0.009 [-0.011, +0.029] | +0.009 [-0.018, +0.036] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|i2e|test_rev|-erm | +0.010 [-0.003, +0.022] | +0.010 [-0.004, +0.022] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|i2e|clean|-erm | +0.001 [-0.005, +0.008] | +0.001 [-0.007, +0.009] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|i2e_balanced|test_rev|-erm | +0.192 [+0.180, +0.204] | +0.192 [+0.175, +0.208] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|i2e_balanced|clean|-erm | +0.037 [+0.020, +0.052] | +0.037 [+0.016, +0.057] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|i2e_rank1|test_rev|-erm | -0.004 [-0.012, +0.001] | -0.004 [-0.013, +0.001] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|i2e_rank1|clean|-erm | +0.003 [-0.000, +0.009] | +0.003 [-0.000, +0.009] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|insert_aug|test_rev|-erm | +0.001 [-0.012, +0.012] | -0.003 [-0.016, +0.008] | no |  |  |
| cxr_traps/raddino518/Consolidation/bootstrap_vs_erm.csv | trapB|insert_aug|clean|-erm | +0.001 [-0.004, +0.006] | +0.001 [-0.005, +0.007] | no |  |  |
| cxr_traps/raddino518/Consolidation/X3_crossover.json | raddino518|Consolidation|crossover | -0.002 [-0.030, +0.025] | -0.004 [-0.065, +0.058] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|prevcal|test_rev|-erm | +0.520 [+0.496, +0.544] | +0.512 [+0.466, +0.560] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|prevcal|clean|-erm | -0.031 [-0.064, +0.003] | -0.039 [-0.088, +0.008] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|mask|test_rev|-erm | +0.104 [+0.078, +0.129] | +0.078 [+0.032, +0.125] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|mask|clean|-erm | +0.029 [+0.009, +0.048] | +0.017 [-0.004, +0.038] | **yes** |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|inpaint|test_rev|-erm | +0.164 [+0.144, +0.183] | +0.149 [+0.121, +0.176] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|inpaint|clean|-erm | +0.045 [+0.032, +0.058] | +0.034 [+0.016, +0.052] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|balanced|test_rev|-erm | +0.302 [+0.260, +0.336] | +0.312 [+0.256, +0.365] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|balanced|clean|-erm | +0.077 [+0.049, +0.108] | +0.071 [+0.037, +0.107] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|dfr|test_rev|-erm | +0.244 [+0.208, +0.281] | +0.238 [+0.188, +0.293] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|dfr|clean|-erm | +0.016 [-0.009, +0.042] | -0.019 [-0.059, +0.022] | no | **yes** |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|leace_paired|test_rev|-erm | +0.150 [+0.128, +0.173] | +0.130 [+0.097, +0.165] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|leace_paired|clean|-erm | +0.049 [+0.037, +0.061] | +0.039 [+0.020, +0.059] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|leace_unpaired|test_rev|-erm | +0.409 [+0.380, +0.438] | +0.405 [+0.361, +0.449] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|leace_unpaired|clean|-erm | +0.043 [+0.011, +0.078] | +0.052 [+0.015, +0.091] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|i2e|test_rev|-erm | +0.033 [+0.016, +0.052] | +0.021 [-0.002, +0.045] | **yes** |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|i2e|clean|-erm | +0.006 [-0.009, +0.023] | -0.002 [-0.014, +0.011] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|i2e_balanced|test_rev|-erm | +0.316 [+0.274, +0.358] | +0.327 [+0.273, +0.378] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|i2e_balanced|clean|-erm | +0.077 [+0.050, +0.107] | +0.073 [+0.041, +0.107] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|i2e_rank1|test_rev|-erm | +0.008 [+0.001, +0.017] | +0.008 [-0.001, +0.018] | **yes** |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|i2e_rank1|clean|-erm | +0.001 [-0.005, +0.007] | -0.006 [-0.015, +0.003] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|insert_aug|test_rev|-erm | +0.000 [-0.011, +0.011] | +0.011 [-0.004, +0.024] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapA|insert_aug|clean|-erm | -0.002 [-0.010, +0.006] | -0.002 [-0.014, +0.010] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|prevcal|test_rev|-erm | +0.543 [+0.528, +0.560] | +0.543 [+0.524, +0.564] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|prevcal|clean|-erm | -0.024 [-0.049, -0.000] | -0.024 [-0.055, +0.006] | **yes** |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|mask|test_rev|-erm | +0.100 [+0.092, +0.109] | +0.100 [+0.089, +0.111] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|mask|clean|-erm | +0.030 [+0.021, +0.038] | +0.030 [+0.019, +0.041] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|inpaint|test_rev|-erm | +0.012 [+0.011, +0.012] | +0.012 [+0.011, +0.012] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|inpaint|clean|-erm | +0.003 [+0.002, +0.003] | +0.003 [+0.002, +0.003] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|balanced|test_rev|-erm | +0.315 [+0.305, +0.325] | +0.315 [+0.301, +0.329] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|balanced|clean|-erm | +0.089 [+0.077, +0.099] | +0.089 [+0.073, +0.103] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|dfr|test_rev|-erm | +0.316 [+0.304, +0.327] | +0.316 [+0.300, +0.332] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|dfr|clean|-erm | +0.047 [+0.037, +0.058] | +0.047 [+0.032, +0.063] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|leace_paired|test_rev|-erm | +0.027 [+0.025, +0.029] | +0.027 [+0.025, +0.030] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|leace_paired|clean|-erm | +0.008 [+0.006, +0.009] | +0.008 [+0.006, +0.010] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|leace_unpaired|test_rev|-erm | +0.396 [+0.384, +0.409] | +0.396 [+0.379, +0.413] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|leace_unpaired|clean|-erm | +0.070 [+0.057, +0.082] | +0.070 [+0.051, +0.088] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|i2e|test_rev|-erm | +0.019 [+0.014, +0.023] | +0.019 [+0.013, +0.024] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|i2e|clean|-erm | +0.005 [+0.000, +0.010] | +0.005 [-0.001, +0.011] | **yes** |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|i2e_balanced|test_rev|-erm | +0.315 [+0.306, +0.325] | +0.315 [+0.302, +0.328] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|i2e_balanced|clean|-erm | +0.086 [+0.074, +0.099] | +0.086 [+0.071, +0.101] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|i2e_rank1|test_rev|-erm | +0.001 [+0.000, +0.001] | +0.001 [+0.000, +0.001] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|i2e_rank1|clean|-erm | +0.000 [+0.000, +0.001] | +0.000 [+0.000, +0.001] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|insert_aug|test_rev|-erm | +0.003 [-0.001, +0.007] | +0.003 [-0.001, +0.007] | no |  |  |
| cxr_traps/raddino518/Effusion/bootstrap_vs_erm.csv | trapB|insert_aug|clean|-erm | -0.001 [-0.007, +0.003] | -0.001 [-0.007, +0.003] | no |  |  |
| cxr_traps/raddino518/Effusion/X3_crossover.json | raddino518|Effusion|crossover | -0.004 [-0.027, +0.021] | +0.022 [-0.030, +0.074] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|prevcal|test_rev|-erm | +0.620 [+0.596, +0.643] | +0.632 [+0.597, +0.665] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|prevcal|clean|-erm | -0.020 [-0.043, +0.002] | -0.025 [-0.068, +0.019] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|mask|test_rev|-erm | +0.094 [+0.075, +0.113] | +0.087 [+0.060, +0.116] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|mask|clean|-erm | +0.030 [+0.016, +0.042] | +0.028 [+0.008, +0.047] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|inpaint|test_rev|-erm | +0.109 [+0.093, +0.123] | +0.125 [+0.106, +0.145] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|inpaint|clean|-erm | +0.008 [-0.003, +0.019] | +0.022 [+0.007, +0.039] | **yes** |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|balanced|test_rev|-erm | +0.319 [+0.291, +0.353] | +0.324 [+0.282, +0.371] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|balanced|clean|-erm | +0.066 [+0.043, +0.088] | +0.056 [+0.018, +0.093] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|dfr|test_rev|-erm | +0.314 [+0.284, +0.345] | +0.299 [+0.253, +0.346] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|dfr|clean|-erm | +0.036 [+0.005, +0.065] | +0.002 [-0.034, +0.035] | **yes** | **yes** |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|leace_paired|test_rev|-erm | +0.105 [+0.087, +0.122] | +0.123 [+0.106, +0.141] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|leace_paired|clean|-erm | +0.017 [+0.005, +0.030] | +0.027 [+0.010, +0.043] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|leace_unpaired|test_rev|-erm | +0.369 [+0.333, +0.408] | +0.380 [+0.343, +0.417] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|leace_unpaired|clean|-erm | +0.052 [+0.026, +0.078] | +0.044 [+0.008, +0.079] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|i2e|test_rev|-erm | +0.018 [+0.001, +0.034] | +0.021 [+0.001, +0.040] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|i2e|clean|-erm | +0.008 [-0.003, +0.019] | +0.002 [-0.014, +0.016] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|i2e_balanced|test_rev|-erm | +0.329 [+0.300, +0.363] | +0.316 [+0.279, +0.354] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|i2e_balanced|clean|-erm | +0.066 [+0.047, +0.086] | +0.046 [+0.011, +0.079] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|i2e_rank1|test_rev|-erm | +0.005 [-0.009, +0.016] | +0.007 [-0.001, +0.014] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|i2e_rank1|clean|-erm | +0.005 [-0.003, +0.013] | +0.001 [-0.005, +0.007] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|insert_aug|test_rev|-erm | +0.007 [-0.000, +0.015] | +0.006 [-0.002, +0.016] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapA|insert_aug|clean|-erm | +0.007 [-0.003, +0.020] | +0.001 [-0.005, +0.007] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|prevcal|test_rev|-erm | +0.634 [+0.605, +0.654] | +0.634 [+0.604, +0.659] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|prevcal|clean|-erm | -0.032 [-0.045, -0.019] | -0.032 [-0.055, -0.008] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|mask|test_rev|-erm | +0.063 [+0.058, +0.068] | +0.063 [+0.055, +0.071] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|mask|clean|-erm | +0.010 [+0.001, +0.021] | +0.010 [-0.001, +0.023] | **yes** |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|inpaint|test_rev|-erm | +0.008 [+0.006, +0.010] | +0.008 [+0.005, +0.010] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|inpaint|clean|-erm | -0.002 [-0.010, +0.002] | -0.002 [-0.010, +0.003] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|balanced|test_rev|-erm | +0.234 [+0.221, +0.246] | +0.234 [+0.219, +0.249] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|balanced|clean|-erm | +0.044 [+0.030, +0.058] | +0.044 [+0.028, +0.060] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|dfr|test_rev|-erm | +0.260 [+0.249, +0.271] | +0.260 [+0.244, +0.275] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|dfr|clean|-erm | -0.000 [-0.017, +0.018] | -0.000 [-0.020, +0.022] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|leace_paired|test_rev|-erm | +0.020 [+0.017, +0.022] | +0.020 [+0.016, +0.023] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|leace_paired|clean|-erm | +0.002 [-0.006, +0.006] | +0.002 [-0.007, +0.007] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|leace_unpaired|test_rev|-erm | +0.372 [+0.352, +0.389] | +0.372 [+0.348, +0.394] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|leace_unpaired|clean|-erm | +0.014 [-0.009, +0.033] | +0.014 [-0.013, +0.038] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|i2e|test_rev|-erm | +0.003 [-0.002, +0.007] | +0.003 [-0.003, +0.007] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|i2e|clean|-erm | -0.005 [-0.008, -0.002] | -0.005 [-0.009, -0.001] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|i2e_balanced|test_rev|-erm | +0.221 [+0.206, +0.235] | +0.221 [+0.203, +0.238] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|i2e_balanced|clean|-erm | +0.041 [+0.029, +0.056] | +0.041 [+0.026, +0.058] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|i2e_rank1|test_rev|-erm | +0.002 [+0.002, +0.002] | +0.002 [+0.001, +0.003] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|i2e_rank1|clean|-erm | +0.001 [+0.000, +0.001] | +0.001 [+0.000, +0.001] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|insert_aug|test_rev|-erm | -0.002 [-0.007, +0.002] | -0.002 [-0.007, +0.002] | no |  |  |
| cxr_traps/raddino518/Infiltration/bootstrap_vs_erm.csv | trapB|insert_aug|clean|-erm | -0.004 [-0.015, +0.004] | -0.004 [-0.015, +0.004] | no |  |  |
| cxr_traps/raddino518/Infiltration/X3_crossover.json | raddino518|Infiltration|crossover | -0.032 [-0.051, -0.013] | -0.025 [-0.055, +0.006] | **yes** |  |  |

