# Pre-registration — chest radiography real-device traps (multi-disease), NIH ChestX-ray14 × RANZCR-CLiP

Committed before any model is fitted on this cohort (check `git log` against `results/cxr_traps/`).
Protocol = the author's E13 protocol (docs/REPLICATION_SPEC.md) transferred to radiographs; only the items
below differ.

## Cohort
- RANZCR-CLiP training images (30,083; human image-level labels for ETT / NGT / CVC / Swan-Ganz; human
  polylines on 9,095 images) linked to NIH ChestX-ray14 by 64-bit pHash (nearest neighbour ≤ 6 bits, runner-up
  ≥ 4 bits further): 28,786 images, 3,227 patients. Disease labels = NIH findings.
- ROI = both lungs (TorchXRayVision PSPNet, p > 0.5) at 518 px.
- Device masks = CLiP polylines rasterised at 518 px (ETT 12 px, NGT 7 px, CVC / Swan-Ganz 4 px wide).
- Overlap r = |device ∩ lungs| / |device| per device type (observed, outcome-free: median r ETT 0.00,
  NGT 0.00, CVC 0.52).

## Traps (artifact-free comparison group per device, as in E13)
- **Trap A (in-ROI): CVC.** A = 1: CVC present and annotated with r ≥ 0.5. A = 0: no CVC (human label).
- **Trap B (out-of-ROI): ETT.** A = 1: ETT present and annotated with r < 0.1. A = 0: no ETT.
- Source matching: within each label, A = 1 cells are subsampled to the A = 0 **view-position** (AP/PA) mix.
- Groups: NIH patient id; 5-fold group-safe CV per seed; seeds 42, 123, 456, 789, 2026; fold predictions pooled
  per seed (seed = bootstrap cluster). Train 90/10, correlated test 90/10, reversed 10/90, clean 50/50,
  inner validation = 20 % of training patients (clean 50/50 for C and thresholds; all groups for DFR).

## Diseases (co-primary; chosen on counts only)
Every NIH finding with ≥ 40 estimated reversed-test positives in **both** traps: **Infiltration, Effusion,
Atelectasis, Consolidation**. Holm correction over the four diseases for each claim.

## Arms
erm, mask (lung ROI), inpaint (Telea of the trap's device mask; oracle), balanced, dfr, leace_paired
(orig ↔ device-inpainted training A = 1 images), leace_unpaired (A-erasure), prevcal (Kina & Petersen 2026),
i2e / i2e_balanced (device pixels transplanted from donor images of the same device type that are in neither
trap: CVC with 0.1 ≤ r < 0.5; ETT with r ≥ 0.1).
Backbones: RAD-DINO @518 (primary; chest-X-ray foundation model), MedSigLIP-448, DINOv2 ViT-B/14 @518.

## Claims (hierarchical paired bootstrap, 10,000 replicates, 95 % CI; per disease, Holm across diseases)
- **X1** mask − ERM on Trap B (ETT, outside the lungs) reversed AUROC > 0.
- **X2** mask − ERM on Trap A (CVC, inside the lungs) reversed AUROC < 0.
- **X3** crossover [mask − ERM]_B − [mask − ERM]_A > 0.
- **X4** balanced and DFR − ERM > 0 in both traps.
- **X5** paired LEACE gain < ½ best label-only gain.
- **X6** (proposed) i2e − ERM > 0 in Trap A with clean AUROC loss ≤ 0.02 and correlated ≥ reversed − 0.02.
Reported whatever the outcome; the primary backbone is RAD-DINO, the others are replications.

## Amendment 1 (committed before any model fit; stricter)
View matching to the A = 0 view mix is infeasible for Trap B: intubated patients are imaged supine (AP), so
ETT images have almost no PA counterpart (matched cells collapse to 7–35 images). Both traps are therefore
**restricted to AP radiographs**, which removes the view confound instead of re-weighting it. Counts after the
amendment (outcome-free): every matched cell ≥ 25 for all four pre-registered diseases; estimated reversed-test
positives Trap A / Trap B: Infiltration 218 / 3,318, Effusion 125 / 2,316, Atelectasis 140 / 1,918,
Consolidation 66 / 915. Disease selection unchanged.

## Follow-up 1 (committed after the pre-registered Infiltration/RAD-DINO result, before any follow-up fit)
Observed confound (descriptive): in the Trap A pool the CVC-free group carries far more out-of-lung tubes than the
CVC group (ETT 77 % vs 39 %, NGT 83 % vs 42 %), because CLiP's CVC-free images were selected for other tubes. The
A contrast is then partly "fewer tubes outside the lungs", which lung masking removes. The pre-registered results
stand as reported. Follow-up design (stricter, same everything else): cells are matched within label on
**view × presence of the other devices** — Trap A: view × ETT × NGT; Trap B: view × CVC × NGT — so the only device
difference between A = 1 and A = 0 is the trap's own device. Questions: X1–X3 and X6 as above; reported beside,
never instead of, the pre-registered results.

## Results — pre-registered primary (RAD-DINO @518, AP films, 5 seeds × 5 folds)
| Disease | X1 mask−ERM, ETT (out) | X2 mask−ERM, CVC (in) | X3 crossover | balanced A / B | I2E+balanced A / B |
| --- | --- | --- | --- | --- | --- |
| Infiltration | +0.063 [+0.058, +0.068] | +0.094 [+0.075, +0.113] | −0.032 [−0.051, −0.013] | +0.319 / +0.234 | +0.329 / +0.221 |
| Effusion | +0.100 [+0.092, +0.109] | +0.104 [+0.078, +0.129] | −0.004 [−0.027, +0.021] | +0.302 / +0.315 | +0.316 / +0.315 |
| Atelectasis | +0.105 [+0.094, +0.114] | +0.045 [+0.013, +0.077] | +0.060 [+0.028, +0.089] | +0.266 / +0.302 | +0.256 / +0.285 |
| Consolidation | +0.068 [+0.049, +0.086] | +0.070 [+0.044, +0.096] | −0.002 [−0.030, +0.025] | +0.286 / +0.190 | +0.298 / +0.192 |

X1 supported (4/4). **X2 not supported (0/4): lung masking helps even for the in-lung CVC.** X3 supported 1/4
(Atelectasis). X4 supported (4/4). Interpretation (descriptive, see Follow-up 1): the CVC contrast is confounded by
out-of-lung tubes that masking removes; the device-matched follow-up tests this.

## Controlled synthetic tube, RAD-DINO@518, NIH pneumothorax (results/synthetic/nih_ptx/raddino518_tube_corr_main)
Reversed AUROC (ERM / mask / U-MtE): r=0: 0.913 / 0.910 / 0.891; r=0.5: 0.772 / 0.843 / 0.886; r=1: 0.614 / 0.725 / 0.859.
- The thesis sign reversal does **not** replicate here: ERM does not use an out-of-lung tube at all (reversed ≈
  clean at r=0), and lung masking helps more as overlap grows (location interaction [r=0] − [r=1] = −0.113
  [−0.123, −0.103]); masking cannot remove the in-lung tube (r=1: 0.725 vs clean 0.905).
- U-MtE − mask = +0.044 [+0.036, +0.053] (r=0.5), **+0.134 [+0.120, +0.151] (r=1)**; U-MtE − ERM (r=1) = +0.245.
- With artifact labels, balanced alone is better than U-MtE_balanced (−0.031 [−0.038, −0.025]).
Interpretation: with a strong disease signal (clean AUROC 0.91) RAD-DINO ignores extra-pulmonary overlays; the
in-ROI residual shortcut after masking is still present and is what U-MtE removes.
