# Pre-registration — Self-Localising Artifact Suppression (SLAS) on in-ROI artifact traps

Committed before any SLAS model is evaluated on a trap. Protocol, cohorts, seeds, folds, thresholds, bootstrap:
unchanged from docs/REPLICATION_SPEC.md (E13, ISIC 2019 hair) and docs/PREREGISTRATION_THYROID_TRAPS.md.

## Motivation (evidence available before this registration)
Pixel masking cannot remove an artifact that lies inside the ROI (E13 Trap A: mask − ERM < 0). Pixel artifact
detectors are poor on thin in-ROI artifacts (hair IoU 0.22). Localisation-only test (results/slas/
localisation_isic_hair.json; no trap outcome used): a linear probe on DINOv2@518 patch tokens trained on real
Wegley hair/ruler masks localises hair on held-out images with patch AUROC 0.977 (0.969 inside the lesion);
a probe trained on synthetic generic overlays only reaches 0.64 — so SLAS uses a few real annotated images.

## Method (frozen)
1. Probe: logistic regression (C=1, balanced) on DINOv2 ViT-B/14 @518 patch tokens; patch label = artifact
   coverage >= 0.2 (positive) / 0 (negative). Trained on k = 50 annotated images drawn (seed 20260927) from the
   donor pool (0.1 <= r < 0.5; hair_frac >= 0.02), which is disjoint from both trap pools and shares no leakage
   group with them. The same procedure is used for every artifact type — only the k annotated images change.
2. Representation: L2-normalised mean of patch tokens. Patches with probe probability >= tau = 0.5 are dropped;
   ROI patches = >= 50 % lesion pixels.
3. Arms (heads, thresholds and validation exactly as E13):
   tok_erm (all patches) · tok_mask (ROI patches) · slas (probe-clean patches) · mts (ROI ∩ probe-clean;
   "mask-then-suppress") · mts_oracle (ROI minus real-mask patches; ceiling, not a method) ·
   tok_balanced · mts_balanced.
Secondary: k ∈ {10, 25, 100} (learning curve); thyroid markers (probe trained on 50 donor images with detector
masks); DermLIP / MedSigLIP token versions if the primary is positive.

## Claims (reversed-test AUROC, 95 % hierarchical paired CI, seeds as clusters)
- **S0 (sanity, token space reproduces the thesis)**: tok_mask − tok_erm < 0 in Trap A and > 0 in Trap B.
- **S1 (primary)**: mts − tok_mask > 0 in Trap A.
- **S2**: slas − tok_erm > 0 in Trap A.
- **S3 (no harm out of ROI)**: mts − tok_mask CI lower bound > −0.01 in Trap B.
- **S4 (no gaming)**: clean-AUROC loss of mts vs tok_mask <= 0.02 in both traps.
- **S5**: mts_balanced − tok_balanced > 0 in Trap A.
- **S6**: mts reaches >= 50 % of the oracle gain (mts_oracle − tok_mask) in Trap A.
All outcomes are reported whichever way they go. If S0 fails (token pooling does not reproduce the in-ROI
harm), SLAS results are reported as exploratory only.

## Results — ISIC 2019 hair, DINOv2@518, k = 50 (results/slas/isic_k50/bootstrap.csv)
Reversed-test AUROC deltas [95 % CI]:
- S0 **not supported**: tok_mask − tok_erm = +0.030 [+0.019, +0.040] in Trap A (vs +0.138 [+0.124, +0.151] in
  Trap B). Token-level ROI pooling does not *harm* in Trap A; the location gap (B ≫ A) persists.
  Per the registration, SLAS results on this cohort are therefore **exploratory**.
- S1 supported: mts − tok_mask = +0.022 [+0.012, +0.031] (Trap A).
- S2 not supported: slas − tok_erm = +0.008 [−0.002, +0.017].
- S3 supported: mts − tok_mask = +0.006 [+0.002, +0.011] in Trap B. S4 supported (clean +0.003).
- S5 supported: mts_balanced − tok_balanced = +0.051 [+0.028, +0.072]; mts_balanced has the best reversed
  (0.749) and clean (0.812) AUROC of all token arms.
- S6 supported, but the ceiling is tiny: removing EVERY real-hair patch (oracle) gains only
  +0.015 [+0.002, +0.024] over tok_mask.
**Interpretation.** On real ISIC hair the residual shortcut after masking is mostly *not* carried by the hair
patches themselves (oracle removal ≈ +0.015), consistent with the earlier diagnostic that ~55 % of the hair
shortcut is carried by metadata correlates (site / age / sex). No artifact-removal method can exceed that
ceiling here; group re-weighting (+0.21) is what moves this trap.

## Results — thyroid ultrasound markers, DINOv2@518, k = 50 detector-annotated donors (results/slas/thyroid_k50)
- S0 **not supported**: tok_mask − tok_erm = +0.059 [+0.050, +0.069] in Trap A vs +0.232 [+0.212, +0.252] in
  Trap B (location gap persists; no harm) → exploratory, as registered.
- S1 supported: mts − tok_mask = +0.081 [+0.076, +0.087] (Trap A); larger than the detector-mask "oracle"
  (+0.047 [+0.045, +0.050]) — the probe also removes marker patches the rule-based detector misses.
- S2 not supported in size terms: slas − tok_erm = +0.007 [+0.006, +0.008].
- S3 supported (+0.014 [+0.007, +0.021] in Trap B); S4 supported (clean +0.025).
- S5 supported: mts_balanced − tok_balanced = +0.070 [+0.053, +0.089]; mts_balanced: reversed 0.758, clean 0.796
  (best token arm). For reference the CLS-feature MtE_balanced (real-marker templates) reached 0.818 / 0.812.
