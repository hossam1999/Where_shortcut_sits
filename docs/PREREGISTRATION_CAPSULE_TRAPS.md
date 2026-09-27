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
