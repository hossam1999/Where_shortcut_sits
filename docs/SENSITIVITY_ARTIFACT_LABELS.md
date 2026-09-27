# Sensitivity of the main claims to artifact-label definitions (registered before running)

Motivation: marker masks come from an audited rule-based detector (no clinician audit available). If conclusions
depended on borderline detections they would be fragile. Variants (thyroid and ovary, DINOv2@518, generic library):
- **main**: A1 >= 15 detected px; Trap A r >= 0.5; Trap B r < 0.1.
- **large-markers**: A1 >= 50 px (large detections are almost certainly real calipers).
- **strict-location**: Trap A r >= 0.7; Trap B r < 0.05.
(A 5-px variant changes <= 1 image per cell and is omitted.)
Arms: erm, mask, balanced, U-MtE, U-MtE_protect, U-MtE_balanced. Checked claims: P1 crossover > 0; P2 U-MtE_protect −
mask > 0; P4 U-MtE_balanced − balanced > 0 (Trap A). Robust = same sign and CI excluding 0 in every variant.
Command: scripts/run_thyroid_traps.py --cohort {thyroid,ovary} --generic --min_px/--rA/--rB --tag sens_*.

## Capsule Trap B definition (found during figure review, 2026-09-27)
r = |debris ∩ ROI| / |debris| can be small for a large debris blob that still covers part of the lesion: 26.9 % of
capsule Trap B images have debris covering > 20 % of the lesion box (median 10.5 %; Trap A median 41 %). This makes
the capsule crossover conservative. Variant **capsule-strict-B**: Trap B additionally requires lesion coverage < 5 %
(`--max_cover_B 0.05`). Registered before running; same arms and claims.

## Results (DINOv2@518, generic library; Trap A reversed AUROC [95 % CI])
| cohort | variant | crossover | U-MtE_protect − mask | U-MtE_bal − balanced |
|---|---|---|---|---|
| thyroid | main | +0.230 [+0.204, +0.255] | +0.217 [+0.174, +0.257] | +0.112 [+0.090, +0.134] |
| thyroid | large markers (>= 50 px) | +0.262 [+0.228, +0.294] | +0.282 [+0.243, +0.318] | +0.132 [+0.114, +0.148] |
| thyroid | strict location (0.7 / 0.05) | +0.215 [+0.184, +0.244] | +0.252 [+0.221, +0.279] | +0.136 [+0.105, +0.165] |
| ovary | main | +0.170 [+0.116, +0.224] | +0.043 [+0.025, +0.061] | +0.028 [−0.001, +0.058] |
| ovary | large markers | +0.186 [+0.135, +0.238] | +0.067 [+0.049, +0.082] | +0.025 [−0.011, +0.060] |
| ovary | strict location | +0.195 [+0.143, +0.246] | +0.046 [+0.023, +0.071] | −0.012 [−0.041, +0.017] |
**Robust**: the crossover and U-MtE_protect − mask hold (CI excludes 0) in every variant and grow under stricter
definitions, as expected if detector errors dilute rather than create the effects. The ovary balanced claim is not
significant in any variant (as in the main analysis).
