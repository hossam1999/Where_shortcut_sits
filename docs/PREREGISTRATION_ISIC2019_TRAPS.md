# Pre-registration — ISIC 2019 real-hair traps (re-implementation of thesis Result 3) and Insert-to-Erase

Committed to version control **before any trap AUROC is computed** (check `git log` for this file's
timestamp vs. `results/real/`). The code that produced thesis §9 was not present in the archived
repository (`isic_pcam_code_results/`), so this is a fresh, documented re-implementation. Thesis
numbers are *targets for qualitative agreement*, not for exact reproduction.

## Cohort
- ISIC 2019 training set, 25,331 images; melanoma label = `MEL` column.
- Source from `lesion_id` prefix: HAM, BCN, MSK (incl. `_downsampled`), else `ISIC_legacy`.
- Leakage groups: union-find over `lesion_id` and 64-bit pHash with Hamming ≤ 8 (pilot rule).
- Lesion masks: HAM10000 manual (Tschandl 2020) or ISIC 2018 Task 1 manual where available,
  otherwise U-Net prediction (ResNet34 U-Net, 384 px, trained on those manual masks).
- Artifact masks: Wegley et al. 2026 archive (Scholars' Mine, doi:10.71674/man1-qa33):
  hair+ruler (one channel), ink, vignetting.
- All geometry at 518×518 (nearest-neighbour masks).
- Quality control: exclude images with lesion area fraction < 0.005 or > 0.95 (failed masks) and
  images with any ink mark (ink is a different artifact).

## Artifact groups
- **Hair-free (A=0)**: hair/ruler fraction < 0.001 of the image.
- **Hair-present**: hair/ruler fraction ≥ 0.005.
- Overlap r = |hair ∩ lesion| / |hair|.
- **Trap A (in-ROI)**: hair-present with r ≥ 0.5.  **Trap B (out-of-ROI)**: hair-present with r < 0.1.
- Hair-present images with 0.1 ≤ r < 0.5 are in neither trap. They are the **template-donor pool**
  for Insert-to-Erase, so no donor is ever a train/test image of a trap.

## Splits and environments
- 5-fold StratifiedGroupKFold on (group, y), seed 20260926. Fold k is test; fold (k+1) mod 5 is
  validation; the other three are training. Folds are the clusters of the hierarchical bootstrap.
- Environments per trap, built inside each source (HAM/BCN/MSK/ISIC_legacy) by maximum-size
  subsampling: train 90/10 (P(A=1|Y=1)=0.9, P(A=1|Y=0)=0.1), correlated test 90/10,
  reversed test 10/90. **Source matching:** per source × (A, Y) cell the counts are the minimum
  over the two traps, so both traps have identical cell counts per source; the A=0 images drawn
  are the same for both traps.
- Clean test: every hair-free image of the test fold. Clean validation: every hair-free image of
  the validation fold (used for C, λ and the balanced-accuracy threshold — never artifact data).
- Power: ≥ 40 reversed-test melanomas per trap summed over folds (thesis minimum). Counts are
  reported before any model is fitted.

## Arms (frozen backbone + linear head)
Backbones: DINOv2 ViT-B/14 @518 (full sweep), DermLIP/PanDerm @224 (replication).
| arm | input / supervision |
| --- | --- |
| erm | original image |
| mask | lesion-masked image at train and test (lesion mask) |
| inpaint | Telea inpaint of the true hair mask at train and test (oracle; upper bound) |
| balanced | four (A×Y) groups equally weighted (image-level A at train) |
| dfr | last layer on a group-balanced subset of the validation fold (image-level A on val) |
| leace_paired | LEACE on (original, hair-inpainted) pairs of training hair images |
| leace_unpaired | LEACE on image-level A of the training set (the variant expected to *game* the test) |
| i2e | **proposed**: subspace erasure from (original, hair-inserted) pairs; no artifact label, no mask |
| i2e_balanced | i2e + balanced head (image-level A at train) |
| i2e_rank1 | ablation: LEACE on insertion pairs |
| insert_aug | ablation: label-independent insertion augmentation (WP3 route 2) |

Hair insertion: donor from the donor pool (deterministic hash), random flip/rotation, alpha-composited
with a σ=0.7 px Gaussian-softened donor hair mask. I2E energy threshold 0.90, max rank 64 — fixed now.

## Claims and decision rules (95% hierarchical paired bootstrap CI, 10,000 replicates)
- **C1** mask − ERM on Trap B reversed AUROC > 0 (CI excludes 0).
- **C2** mask − ERM on Trap A reversed AUROC < 0 (CI excludes 0).
- **C3** crossover [mask − ERM]_B − [mask − ERM]_A > 0 (CI excludes 0).
- **C4** balanced − ERM and DFR − ERM > 0 on both traps (CIs exclude 0).
- **C5** leace_paired gain on Trap A < ½ of the balanced gain (linear paired erasure insufficient).
- **P1** i2e − ERM on Trap A > 0 (CI excludes 0) **and** clean AUROC loss ≤ 0.02 vs ERM **and**
  correlated AUROC ≥ reversed AUROC − 0.02 (no inversion, i.e. not gamed).
- **P2** i2e recovers ≥ ½ of the balanced gain on Trap A without any artifact label.
- **P3** i2e_balanced ≥ balanced on Trap A reversed AUROC (point estimate) with clean loss ≤ 0.02.
Outcomes are reported whether positive or negative.
