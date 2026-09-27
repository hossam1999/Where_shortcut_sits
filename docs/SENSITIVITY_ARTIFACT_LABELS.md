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
