# Pre-registration — capsule endoscopy contamination traps (SEE-AI)

Committed before any classifier is fitted on this cohort. Protocol = E13 (docs/REPLICATION_SPEC.md) as
transferred in docs/PREREGISTRATION_THYROID_TRAPS.md; only the items below differ.

## Cohort (outcome-free; /root/data/capsule/capsule_cohort.csv, scripts/data/prepare_capsule.py)
- SEE-AI (Kaggle capsuleyolo/kyucapsule, CC BY 4.0) single-class frames: erosion (y=1, 3,550) vs polyp-like
  (y=0, 1,931). ROI = union of the expert lesion boxes.
- Artifact = contamination (debris, bubbles, bile). Masks: expert masks (figshare 27645021) where a frame is in the
  expert set (22 frames), else a DINOv2 patch-token probe (wtss.slas) trained on 2,160 expert-masked frames;
  held-out (540 frames) pixel accuracy 0.92, IoU 0.62 (SEE-AI subset 0.61).
- A = 0: contamination < 3 % of the frame (504). A = 1: contamination >= 10 % with r = |contam ∩ ROI| / |contam|:
  **Trap A r >= 0.5** (816), **Trap B r < 0.1** (1,241). 0.1 <= r < 0.5 → donor pool (template I2E / MtE).
- Leakage groups: contiguous blocks of 100 frame numbers ∪ pHash <= 2 (137 groups, largest 173). No video or
  patient ids are published: limitation.
- Cells (A0_Y0, A0_Y1, A1_Y0, A1_Y1): Trap A 166/338/404/412, Trap B 166/338/232/1009; reversed positives 373.

## Arms / backbones
As thyroid: erm, mask, inpaint (Telea of the probe mask; mask-quality-limited), balanced, dfr, leace_paired,
leace_unpaired, prevcal, i2e/i2e_balanced/i2e_rank1/insert_aug, mte/mte_balanced (real-contamination templates),
and the generic library run (U-I2E / U-MtE / mte_aug). DINOv2@518 primary; MedSigLIP-448 replication.

## Claims (reversed-test AUROC, 95 % hierarchical CI)
C1 mask − ERM > 0 in Trap B · C2 mask − ERM < 0 in Trap A · C3 crossover [B] − [A] > 0 ·
C4 balanced, DFR − ERM > 0 · C5 U-MtE − mask > 0 in Trap A · C6 U-MtE − mte_aug > 0 in Trap A ·
no gaming (clean loss <= 0.02; corr >= rev − 0.02). Reported whichever way they go.
Then (user rule): controlled synthetic-artifact test on the same cohort's low-contamination frames.

## Controlled synthetic-debris test (registered with the real-trap design, before any fitting)
- Frames with contamination < 5 % (796 after common support); split 60/20/20 by leakage group (seed 20260927).
- Artifact: `wtss.synthetic.draw_debris` (textured yellow-green blob with bubbles), 48x48 box at 518 px;
  overlap with the lesion box r ∈ {0, 0.25, 0.5, 0.75, 1}; seeds 42/123/456; arms erm, mask, inpaint, balanced,
  dfr, leace + proposed (I2E, MtE, U-I2E, U-MtE).
- Claims: CS1 mask − ERM decreases with overlap (r=1 vs r=0 interaction < 0); CS2 mask − ERM < 0 at r=1;
  CS3 mask − ERM > 0 at r=0; CS4 U-MtE − mask > 0 at r=1. Small test split (137 frames): wide CIs expected.

## Results — DINOv2@518, real contamination, template arms (results/capsule/dino518_main)
Reversed AUROC deltas vs ERM [95 % CI]:
- **C1 supported**: mask − ERM (Trap B) = +0.465 [+0.440, +0.490].
- **C2 not supported**: mask − ERM (Trap A) = +0.097 (0.684 vs 0.587): masking helps, but a large residual
  shortcut remains (test_corr 0.975 vs test_rev 0.684).
- **C3 supported**: crossover = +0.368 [+0.340, +0.397].
- **C4 supported**: balanced +0.19, DFR +0.217 [+0.200, +0.235] (Trap A); DFR is the best arm (0.805).
- **Template MtE fails**: MtE − mask (Trap A) = −0.395 [−0.420, ...]; inpaint also hurts (−0.17 vs ERM).
  Interpretation (post hoc): erosions carry yellow-white fibrin that resembles debris; a subspace estimated from
  real-debris templates / debris-guided inpainting removes disease evidence. Boundary condition for
  erasure-based repair: the artifact subspace must be separable from the disease signal.

## Results — controlled synthetic debris, DINOv2@518 (results/synthetic/capsule/dino518_debris_corr_main)
Reversed AUROC vs ERM [95 % CI]:
- **CS3 supported**: mask − ERM at r=0 = +0.103 [+0.060, +0.149].
- **CS2 supported**: mask − ERM at r=1 = −0.102 [−0.162, −0.040].
- **CS1 supported**: location interaction ([r=0] − [r=1]) = +0.205 [+0.150, +0.259].
- **CS4 supported**: U-MtE − mask at r=1 = +0.081 [+0.050, +0.112]; U-MtE_balanced − mask = +0.193
  [+0.150, +0.235]; U-MtE_balanced − balanced = +0.047 [+0.004, +0.091]. U-MtE_balanced shows no location
  interaction (−0.000 [−0.023, +0.023]): its benefit does not depend on where the artifact sits.

## Replication — MedSigLIP-448 (results/capsule/medsiglip448_main, medsiglip448_universal)
Trap A reversed (clean): ERM 0.593, mask 0.702, template MtE 0.222, U-MtE 0.465, U-MtE_protect 0.667,
U-MtE_protect_balanced 0.871 (0.904), U-MtE_balanced 0.854, balanced 0.810, DFR 0.801, mask+DFR 0.901 (0.908).
Same pattern as DINOv2: masking helps less in Trap A than B; erasure fails on real debris unless protected; with
artifact labels, masking + DFR is the most robust arm.
