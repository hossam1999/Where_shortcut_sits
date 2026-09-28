# Pre-registration — round 5: robustness of the ISIC 2020 external result (R12, R13)

Committed on 2026-09-28, after the round-4 results (docs/PREREGISTRATION_ROUND4.md, Stage 7) and before any analysis
below. R10 stays reported exactly as registered; R12 and R13 are sensitivity analyses of it and cannot replace it.
Outputs: `results/round5/`. Changes after this commit are recorded as dated amendments at the end, with the reason.

## Why
R10 found that on ISIC 2020 masking raises overall AUROC but lowers it on shortcut-conflicting pairs (hard-pair
mask − ERM −0.026 [−0.052, −0.000], borderline). Two features of R10 can bias that estimate or cost power:
1. The registered near-duplicate rule (pHash Hamming ≤ 8 to any ISIC 2019 image) removed 7,657 of 32,997 test images,
   6,729 of them at distance 6–8, where the hash also matches merely similar images; melanomas were removed less often
   than benign lesions.
2. The hair labels mix groups: "melanoma without in-lesion hair" includes melanomas with hair beside the lesion (where
   masking is expected to help) and images near the segmenter's threshold. Misclassification dilutes the contrast.

## Common settings (unchanged from R10)
ISIC 2019 training/validation pool, seeds, splits, arms, DINOv2 ViT-B/14 @518, heads, thresholds, hair segmenter
(τ = 200 predicted hair pixels at 518 px), lesion masks, crossed seed × image bootstrap with 10,000 replicates
(operating points 2,000), hard pairs defined by the training association. The only change: every ISIC 2020 image
with a lesion mask (32,997) is scored, so that every duplicate rule below is a subset of one set of predictions.
Check: predictions on the 25,340 R10 test images must equal R10's (max |Δp| ≤ 1e-4 per image and arm); otherwise stop.

## R12 — calibrated near-duplicate removal (primary of this round)
Label-free calibration, computed and committed before any model of this round is fitted:
- pHash distance d19(i) = nearest ISIC 2019 image (as in R10); null d0(i) = nearest ISIC 2020 image of a different
  patient. False-match rate F(t) = share of ISIC 2020 images with d0 ≤ t.
- Embedding similarity s19(i) = largest cosine similarity of the DINOv2 features (original image) to any ISIC 2019
  image; null s0(i) = largest to an ISIC 2020 image of a different patient; G(c) = share with s0 ≥ c.
- Thresholds: t* = the largest t in 0..8 with F(t) ≤ 1 %; c* = the smallest c on the grid 0.900, 0.901, …, 0.999 with
  G(c) ≤ 1 %. Duplicate if d19 ≤ t*, or s19 ≥ c*, or the image name occurs in ISIC 2019.
Tiers reported side by side: exact (d19 = 0 or same name), d19 ≤ 2, **calibrated (primary)**, d19 ≤ 8 (= R10).
- **H12**: on the calibrated test set, hard-pair AUROC mask − ERM < 0.
- Decision: **robust** if the interval lies below 0; **direction-consistent, inconclusive** if the estimate is < 0 and
  the interval includes 0; **not robust** if the estimate is ≥ 0. The paper reports the label whichever it is.
- Descriptive for every tier: all-pair and easy-pair mask − ERM, hard-pair balanced − mask, OP1–OP5 sensitivity among
  melanomas without in-lesion hair, test-set counts by label.

## R13 — cleaner hair groups (secondary; on the calibrated test set)
Groups from the frozen hair masks: **H0** hair-free (hair_px ≤ τ); **Hin** clear in-lesion hair (hair_px ≥ 3τ and
r ≥ 0.5); **Hout** clear hair beside the lesion only (hair_px ≥ 3τ and r < 0.1); all other images ambiguous (kept in
all-pair AUROC, excluded from the group pairs).
- Strict hard pairs: melanomas in H0 vs benign lesions in Hin. Strict easy pairs: melanomas in Hin vs benign in H0.
- **H13**: strict hard-pair AUROC mask − ERM < 0. Reported with and without Holm over {H12, H13}.
- Gate: ≥ 30 melanomas in H0 and ≥ 100 benign lesions in Hin; otherwise descriptive only.
- Descriptive: strict easy pairs; balanced − mask on strict hard pairs; pairs with Hout melanomas or Hout benign
  lesions against H0 (where the location law predicts that masking helps); group counts by label.

## Not changed by this round
No existing result is re-estimated; R10's registered verdicts (H10a supported, borderline; H10b not supported; H10c
supported) stand. The operating-point hypothesis H10b is not re-tested with new thresholds.

## Results (added after the run; the text above is unchanged)
Calibration committed at `a02db23` before any model of this round was fitted; results at `d2b6b24`; write-up in
Stage 7 (Section 5.4). Predictions on the 25,340 R10 test images equal R10's (max |Δp| 2.2e-7 over 1,393,700 rows).

- Calibration: F(8) = 0.41 (41 % of ISIC 2020 images lie within hash distance 8 of another patient's image), so the R10
  rule mostly removed look-alikes. t* = 1, c* = 0.971; the calibrated rule removes 45 images (32,952 kept, 575 melanomas).
- **H12 supported, decision label "robust"**: hard-pair mask − ERM −0.037 [−0.061, −0.012]; the same under exact-copy
  and distance ≤ 2 rules; R10's rule gave −0.026 [−0.052, −0.000]. All pairs +0.016 [−0.000, +0.033] (borderline);
  easy pairs +0.048 [+0.009, +0.090]; balanced − mask on hard pairs +0.072 [+0.048, +0.097]. OP5 sensitivity among
  melanomas without in-lesion hair −0.022 [−0.076, +0.039] (no detectable change).
- **H13 not supported** (gate met: 53 hair-free melanomas, 1,868 benign with clear in-lesion hair): −0.005 [−0.057, +0.048].
  Descriptive: melanomas with hair beside the lesion vs hair-free benign −0.028 [−0.060, +0.004]; hair-free melanomas vs
  benign with hair beside the lesion +0.059 [+0.010, +0.112] — the out-of-ROI half of the law. The hard-pair harm combines
  removal of hair beside melanomas and retention of hair on benign lesions.
