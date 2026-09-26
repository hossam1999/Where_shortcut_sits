# Deviations from docs/REPLICATION_SPEC.md (with reasons)

| # | Spec | This repo | Reason / effect |
|---|---|---|---|
| 1 | GPU V100 32 GB | RTX 3090 24 GB | Frozen features; per-seed AUROCs of the rerun match the archive within 0.0005 (E1, E8, E9). |
| 2 | Feature extraction per (seed, env) | per *view* (image × method × overlap), environments assembled by indexing | Mathematically identical (presence is a function of image/label/seed/env only); ~10× fewer backbone passes. |
| 3 | pHash ≤ 8 on the full ISIC 2019 cohort (repo-default analysis) | spec E13 run: pHash ≤ 8 **within each matched trap pool** (as the spec's counts imply); exploratory runs: lesion_id ∪ pHash ≤ 2 on all 25,331 | On all 25,331 images pHash ≤ 8 chains unrelated lesions into one component of 10,974 images (same-lesion precision at distance 8: 2%). |
| 4 | Hair-free A = 0 group (threshold not stated in spec) | ≤ 30 hair/ruler pixels in the native-resolution archive mask | Calibrated on outcome-free counts only: reproduces A0_Y0 709 (spec 708), A0_Y1 157 (157) and all six source proportions to 3 decimals. |
| 5 | Archive mask file names | matched on zero-padded numeric id | Archive names are irregular (`_downsampled_mvig`, `_downsam_inkmark`, typos, truncated ids). 24 truncated ink ids flag their 10 candidate images as ink-uncertain (exploratory runs only). |
| 6 | Images without HAM/BCN lesion_id | source = MSK | Spec lists three sources; ISIC 2019's non-HAM/BCN images are MSK (incl. `_downsampled`). |
| 7 | U-Net (Ronneberger 64–1024, Task 1 only) | same architecture (+BatchNorm), 256 px, 40 epochs, Adam 1e-4, BCE+Dice, 20% held-out pHash groups | Training hyper-parameters are not in the spec. |
| 8 | Derm1M from archive | fresh clone of the official repo | Archived copy lacks `bpe_simple_vocab_16e6.txt.gz` (.gz files were not archived); DermLIP cannot load without it. |
| 9 | §3.1 "leakage inflated the gap by 27%" | not reproduced from the archive | Pre-correction pilot used the identical split; matched-resolution gap ratio 0.96–0.97. A controlled leakage experiment is run instead (`scripts/run_leakage_experiment.py`). |
| 10 | — | added arms: I2E (+balanced, rank-1, insertion augmentation), prevalence calibration (Kina & Petersen 2026) | Proposed method and a 2026 competitor; reported beside, never instead of, the spec arms. |
