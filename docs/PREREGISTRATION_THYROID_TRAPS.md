# Pre-registration — thyroid ultrasound marker traps (TN3K / TNCD)

Committed before any model is fitted on this cohort. Protocol = the author's E13 protocol
(docs/REPLICATION_SPEC.md) transferred to ultrasound; only the items below differ.

## Cohort (counts are outcome-free; `/root/data/us/tncd/markers_stats.csv`)
- TN3K images with TNCD benign(0)/malignant(1) labels: 3,493 (1,210 malignant). Nodule masks: TN3K manual.
- Marker masks: audited rule-based detector `wtss.data.us_markers.marker_mask` (docs/THYROID_MARKER_AUDIT.md;
  image-level precision/recall ≈ 0.86–0.89 on 20 unseen images).
- A = 0 (marker-free, shared by both traps): no detected marker pixel. A = 1: >= 15 detected marker pixels with
  overlap r = |marker ∩ nodule| / |marker|: **Trap A (in-ROI) r >= 0.5** (1,150 images); **Trap B (out-of-ROI)
  r < 0.1** (326 images). 0.1 <= r < 0.5 images are the donor pool for template insertion (I2E).
- Single source (no site metadata): cells are matched on label only (source matching is vacuous).
- Leakage groups: pHash Hamming <= 8 (largest group 3 images). Patient ids are not published (3,493 images from
  2,421 patients): same-patient different-view leakage cannot be excluded — reported as a limitation.
- Images resized to 518 px (nearest for masks).

## Design (as E13)
5 seeds (42, 123, 456, 789, 2026) × 5-fold group-safe CV; train 90/10, correlated test 90/10, reversed 10/90,
clean 50/50; inner validation 20 % of training groups; fold predictions pooled per seed; hierarchical paired
bootstrap with seeds as clusters; gate: every matched cell >= 25 and >= 40 reversed-test positives.

## Arms
erm, mask (nodule ROI), inpaint (Telea of the detected marker mask; oracle w.r.t. the detector), balanced, dfr,
leace_paired, leace_unpaired, prevcal, i2e / i2e_balanced (real-marker templates from the donor pool),
U-I2E / U-MtE (+ balanced) with the generic artifact library (unchanged from docs/PREREGISTRATION_UNIVERSAL_I2E.md).
Backbones: DINOv2 ViT-B/14 @518 (primary), MedSigLIP-448 (medical foundation model, replication).

## Claims (95 % hierarchical CI)
- **T1** mask − ERM on Trap B (markers outside the nodule) reversed AUROC > 0.
- **T2** mask − ERM on Trap A (markers inside the nodule) reversed AUROC < 0.
- **T3** crossover [mask − ERM]_B − [mask − ERM]_A > 0.
- **T4** balanced and DFR − ERM > 0 in both traps.
- **T5** paired LEACE gain < ½ best label-only gain.
- **T6** U-MtE − mask > 0 in Trap A; U-I2E − ERM > 0 in Trap A; no gaming (clean loss ≤ 0.02, corr ≥ rev − 0.02).
Outcomes reported whichever way they go.

## Amendment 1 (2026-09-27, before any synthetic-thyroid model is fitted) — controlled synthetic calipers
Real-marker results (DINOv2) are known at this point: mask − ERM is positive in both traps, much smaller in Trap A.
To test the location effect under control, the thesis Result-1 design is transferred unchanged:
- Cohort: marker-free images only (1,663 after common support; `wtss.data.thyroid`); split TN3K test -> test,
  trainval -> train/val 80/20 by pHash group.
- Artifact: synthetic sonographer caliper (`wtss.synthetic.draw_caliper`: dotted line + '+' ends), 65x19 box at
  518 px (same geometry as the ISIC ruler); overlap with the nodule r ∈ {0, 0.25, 0.5, 0.75, 1}.
- Seeds 42/123/456; arms erm, mask, inpaint, balanced, dfr, leace + proposed arms (I2E, MtE, U-I2E, U-MtE).
- Claims: **TS1** mask − ERM (reversed AUROC) decreases with overlap (r=1 minus r=0 interaction < 0);
  **TS2** mask − ERM < 0 at r = 1; **TS3** mask − ERM > 0 at r = 0; **TS4** U-MtE − mask > 0 at r = 1.

## Results — DINOv2@518, real markers (results/thyroid/dino518_main, dino518_universal)
Reversed-test AUROC, delta vs ERM [95 % CI]:
- **T1 supported**: mask − ERM, Trap B = +0.341 [+0.320, +0.363].
- **T2 not supported**: mask − ERM, Trap A = +0.111 [+0.097, +0.124] (helps, does not harm). After masking the
  model remains marker-driven in Trap A (test_corr 0.902 vs test_rev 0.400).
- **T3 supported**: crossover [mask − ERM]_B − [mask − ERM]_A = +0.230 [+0.204, +0.255].
- **T4 supported**: balanced +0.387 / +0.269, DFR +0.449 / +0.317 (A / B).
- **T5 supported**: paired LEACE +0.050 (A) vs best label-only +0.449 (DFR; prevcal excluded — it needs A at
  test time and collapses test_corr to 0.258).
- **T6 supported**: MtE − mask (Trap A) = +0.303 [+0.284, ...] with real-marker templates and
  **U-MtE − mask = +0.222 [+0.197, +0.251] with the generic library (no marker example used)**.
  U-MtE_balanced: reversed 0.788, clean 0.807 (template MtE_balanced 0.818 / 0.812); Trap B U-MtE 0.740 vs
  mask 0.747. No gaming: U-MtE corr 0.822 ≥ rev 0.622.
