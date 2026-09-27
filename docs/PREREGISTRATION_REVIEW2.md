# Pre-registration — second external review (confounding, transplant, operating points, fine-tuned hair)

Committed and pushed to GitHub **before any model is fitted for these analyses** (the push time on GitHub is the
external timestamp). Written in response to a second external review of `report/full/full_report.pdf`
(review points and verification: `docs/REVIEW2_RESPONSE.md`). Every analysis below is reported whichever way it goes.

Hardware changed since the original runs: 1× RTX PRO 4000 Blackwell (24 GB), 48 CPU cores, 251 GB RAM; same
software pins (`requirements.txt`, torch 2.11.0+cu128). All data were re-downloaded and every derived object was
rebuilt with `make data data_extra` (chest radiography excluded: not needed for these analyses). The ISIC spec U-Net
is retrained, so non-HAM lesion masks — and hence hair-trap membership — can differ slightly from the original run.

## R0 — reproduction on the rebuilt data (reference for R1, R3)
DINOv2 ViT-B/14 @518, arms erm and mask, the four primary traps with the original protocol (`--tag repro`):
ISIC hair (`run_spec_e13.py`), thyroid, ovary, capsule (`run_thyroid_traps.py`); and the five natural-distribution
tests (`run_natural.py`: thyroid, ISIC BCN/HAM/MSK, capsule). **Criterion:** each crossover has the same sign as the
archived one, its CI excludes 0, and |Δ| ≤ 0.03 from the archived estimate. Unmatched references for R1 are the R0
runs (same machine, same data build), not the archived numbers.

## R1 — covariate balance of the location traps and a covariate-matched crossover
Concern: Trap A (artifact in the ROI) and Trap B (outside) contain *different* artifact-bearing images; in-ROI hair
may go with larger lesions or other body sites, in-ROI calipers with nodule size or measurement practice.

**Covariates (fixed now).**
| cohort | covariates |
|---|---|
| ISIC 2019 hair | lesion area fraction (spec masks), hair amount (native hair fraction), age, sex, anatomical site (6 groups + missing), source (exact), diagnosis within label (benign: NV / BKL / other) |
| Thyroid (TN3K) | nodule area fraction, caliper pixels (log), native width and height (scanner proxy), mean grey level (gain proxy), nodule centroid depth (normalised row) |
| Ovary (MMOTU) | tumour area fraction, caliper pixels (log), native width and height, mean grey level, MMOTU subtype within label |
| Capsule (SEE-AI) | lesion-box area fraction, contamination fraction, mean brightness, frame-number block (1,000 frames; acquisition proxy) |

**R1a (descriptive).** Standardised mean differences (SMD; categorical: per level) between the artifact-bearing
images of Trap A and Trap B, within label and pooled, and between each and the shared artifact-free group.
|SMD| > 0.1 is flagged. Output: `results/review2/covariate_balance.csv`, table in the paper supplement.

**R1b (matched crossover).** Within each label (and exactly within source for ISIC), a logistic propensity model
(standardised covariates, one-hot categories, C = 1) predicts Trap-A membership among the artifact-bearing images of
both traps; greedy 1:1 nearest-neighbour matching on the logit without replacement, calliper 0.2 SD of the logit,
units in stable random order (seed 20260928). The matched images replace the Trap-A and Trap-B artifact-bearing
cells; everything else is unchanged (shared artifact-free group, source matching, 5 seeds × 5 folds, erm and mask,
DINOv2@518; `--tag matched`). Balance after matching is reported (target max |SMD| ≤ 0.1). A cohort with fewer than
30 matched images per label in either trap is reported as infeasible.
- **M1** matched crossover [mask − ERM]_TrapB − [mask − ERM]_TrapA (reversed AUROC) > 0, CI excluding 0, in each
  cohort; Holm over the four cohorts.
- **M2** (descriptive) matched / unmatched (R0) crossover ratio.

## R2 — real-artifact transplant (same images, same artifact, only the location differs)
Real artifact pixels are cut from artifact-bearing images with their masks and pasted inside or outside the ROI of
**the same artifact-free images**. This keeps the realism of the real traps and the causal control of the sweeps.

**Recipients.** The artifact-free group of each cohort (ISIC: ≤ 30 native hair px, QC-passed; thyroid and ovary:
no detected caliper pixel; capsule: contamination < 3 %).
**Donors.** Artifact-bearing images of the same cohort, any location (ISIC: > 30 native hair px, Wegley hair
masks; thyroid/ovary: ≥ 15 detected caliper px; capsule: contamination ≥ 10 %, expert or probe masks). An artifact
*instance* is the original mask pixels inside the largest connected component of the mask after grouping dilation
(calipers 9 px — joins a dotted line and its end crosses; hair and debris 5 px) at 518 px, cropped to its bounding
box. Instances with < 30 px or a bounding-box side > 259 px are discarded.
**Placement.** Positions are restricted to the field of view (median-blurred grey > 15, closed; the instance must
lie ≥ 95 % on it). For recipient *i*, donors are tried in a stable random order (up to 50 donors × 8 dihedral
transforms); the first donor/transform with both an in-ROI position (overlap |instance ∩ ROI| / |instance| ≥ 0.95)
and an out-of-ROI position (overlap = 0) is used, positions drawn at random among the admissible ones (stable seed).
Recipients without a feasible donor are excluded (outcome-free; counts by label are reported). Pixels are
alpha-composited with a Gaussian-blurred mask (σ = 0.7), as the existing `transplant()`.
**Design.** As the controlled sweeps: train 0.9/0.1, correlated test 0.9/0.1, reversed test 0.1/0.9, clean test
without artifact; two locations (in, out). 5-fold grouped cross-validation (StratifiedGroupKFold on label, leakage
groups of each cohort, random_state 20260928; validation = next fold), presence seeds 42, 123, 456, 789, 2026;
fold predictions pooled per seed (seed = bootstrap cluster). DINOv2@518, linear heads, C and threshold on clean
validation. Arms: **erm, mask** (primary); oracle inpainting of the transplanted pixels, balanced, DFR (secondary).
- **T1 (primary)** location interaction [mask − ERM]_out − [mask − ERM]_in (reversed AUROC) > 0, CI excluding 0,
  per cohort (ISIC hair, thyroid, ovary, capsule); Holm over the four.
- **T2** sign and CI of mask − ERM at the in-ROI location (theory: harm iff masking removes disease context).
- **T3** same-head counterfactual reliance |Δp| of the masked vs the ERM model at the in-ROI location.
- **T4** balanced reversed AUROC in vs out (theory: location-invariant; descriptive).
- **Interpretation rule (fixed now):** where T1 holds, the location effect in that cohort cannot be explained by
  population differences between Trap-A and Trap-B images. Where T1 fails but the real-trap crossover holds, the
  paper must say that the real-trap crossover may partly reflect such differences.

## R3 — operating points and clinical metrics on the unaltered test distributions
Predictions from the R0 natural runs (validation predictions saved). Thresholds are fixed on each seed's
**validation** split, never on test: **OP1** maximal balanced accuracy (the existing one), **OP2 / OP3** validation
specificity 0.80 / 0.90, **OP4 / OP5** validation sensitivity 0.80 / 0.90. Metrics on test: sensitivity and
specificity overall; sensitivity among malignant cases and specificity among benign cases in the
shortcut-conflicting subgroups. Uncertainty: hierarchical paired bootstrap (seed clusters, images resampled within
seed, 10,000 replicates) for mask − ERM (and U-MtE_bal − mask, balanced − mask); per-seed values in a table.
- **S1 (thyroid, official split)** mask − ERM sensitivity < 0 at OP1, CI excluding 0 (the reported 0.734 → 0.347).
- **S2** sign and CI of mask − ERM sensitivity at OP2–OP5 (thyroid; all natural cohorts descriptive).
- **S3** at OP2 and OP3, the sensitivity loss is concentrated in shortcut-conflicting malignant nodules (in-ROI
  caliper): mask − ERM sensitivity there < 0.
- **Decision rule (fixed now):** the paper may say "masking lowers sensitivity" only if S1 holds **and** the loss
  holds (CI excluding 0) at ≥ 3 of OP2–OP5. Otherwise it reports the sensitivity change as a consequence of
  transferring a validation threshold (a calibration shift) and bases the clinical claim on AUROC of the
  shortcut-conflicting cases only.

## R4 — a fine-tuned network on the hair traps; absolute performance
- **F1** ResNet-50 fine-tuned end to end on the ISIC 2019 hair traps (spec E13 environments, seed 42, folds 0–4 as
  clusters; 224 px, 8 epochs, same recipe as `PREREGISTRATION_FINETUNE.md`), arms erm, mask, balanced,
  mask_balanced (`run_finetune_spec.py --cohort isic`): crossover > 0. This completes one fine-tuned model per
  cohort (thyroid ViT-S/ResNet-50, ovary and capsule ResNet-50 exist).
- **F2 (descriptive)** clean AUROC of every cohort × model next to published results on the same public data, with
  the caveat that trap subsets and tasks differ from the published ones. No hypothesis.

## Pending registrations run in the same batch
The two amendments committed before this review (`PREREGISTRATION_SCALE.md` ISIC hair ViT-S/L;
`PREREGISTRATION_FT_UMTE.md` Amendment 3, ovary power extension) are run as registered, compute permitting; if not
run, they are listed as not run.

## Not addressable by computation (documented in docs/REVIEW2_RESPONSE.md)
Expert audit of artifact labels with inter-rater agreement (an audit kit is provided), a clinical co-author, and
deposit of the registrations on OSF / Zenodo (a registry table with commit hashes is generated).

## Amendment 1 (2026-09-27, before running) — fine-tuned ovary crossover with more clusters
The archived fine-tuned ResNet-50 ovary crossover is +0.136 [−0.085, +0.329] with five clusters (seed-42 folds), so
F1 ("a fine-tuned model shows the effect in every cohort") cannot yet be claimed for ovary. Added: ResNet-50, arms erm
and mask, both traps, spec split seeds 42, 123 and 456 × folds 0–4 (15 clusters; ids 10·seed + fold for seeds ≠ 42),
same recipe (`run_finetune_spec.py --cohort ovary --tag power --env_seed s --arms erm mask`). **F1b**: ovary
fine-tuned crossover > 0 with 15 clusters. Reported whichever way it goes; the paper states the fine-tuned replication
per cohort as found (significant or not).

## Results (results/review2/*.csv; generated tables paper/tables/review2_*.tex)
**Deviations.** (a) Capsule rebuild: the retrained debris probe (IoU 0.625) moves a handful of frames across the 3 % /
10 % thresholds (A0 164/343 vs 166/338). (b) Hair traps: the retrained spec U-Net (held-out Dice 0.886, HAM agreement
0.895) changes trap membership by about 3 % (Trap A 2,655/942 vs 2,738/989 artifact-bearing benign/melanoma).
(c) Transplant feasibility excluded recipients without a donor instance that fits inside their ROI: hair 3/863,
thyroid 1/1,801, ovary 0/348, capsule 167/507 (37/164 polyp-like, 130/343 erosions — small lesion boxes).
(d) The ISIC queue was split into parallel lanes after it started (no change to any analysis); the FT_UMTE amendment-3
runs were written to `finetune/ovary/resnet50_power_g4` so they cannot collide with amendment 1.

**R0 (criterion met, 4/4).** Crossovers: hair +0.147 [+0.119, +0.177], thyroid +0.239 [+0.212, +0.264], ovary +0.170
[+0.116, +0.223], capsule +0.377 [+0.339, +0.412]; |Δ| to archived 0.003, 0.009, 0.001, 0.008. Natural thyroid:
hard-pair mask − ERM −0.095 [−0.128, −0.061] (archived −0.102); capsule identical to three decimals; BCN −0.025
[−0.036, −0.013] (archived −0.016).

**R1a.** Max |SMD| Trap A vs Trap B (pooled): hair 2.13 (lesion area), thyroid 0.53 (caliper pixels; nodule area
0.40), ovary 0.39, capsule 2.57 (lesion-box area). **R1b / M1 (supported 3/3 feasible; capsule infeasible).** Matched
crossover hair +0.154 [+0.125, +0.184] (ratio 1.05), thyroid +0.238 [+0.207, +0.269] (1.00), ovary +0.163
[+0.087, +0.237] (0.96); max |SMD| after matching 0.11 / 0.04 / 0.08. Capsule: 19 matched pairs among polyp-like
images (< 30) → infeasible; exploratory +0.230 [+0.125, +0.337].

**R2 / T1 (supported 4/4, Holm).** Location interaction hair +0.253 [+0.231, +0.275], thyroid +0.277
[+0.264, +0.290], ovary +0.074 [+0.055, +0.092], capsule +0.497 [+0.455, +0.539]. **T2** mask − ERM in-ROI: hair −0.075
[−0.095, −0.055], thyroid +0.072 [+0.058, +0.086], ovary −0.010 [−0.041, +0.022], capsule −0.175 [−0.210, −0.141].
**T3** |Δp| mask vs ERM in-ROI: hair 0.297 vs 0.340, thyroid 0.271 vs 0.206, ovary 0.116 vs 0.037, capsule 0.551 vs
0.238 (out-of-ROI: masked |Δp| = 0). **T4** balanced interaction −0.052 to −0.011 (not exactly invariant, 5–10× smaller
than masking's). By the interpretation rule, the location effect is not explained by Trap-A/Trap-B population
differences in any cohort.

**R3 (S1 supported; S2 4/4; S3 supported; decision rule met).** Thyroid, mask − ERM sensitivity: OP1 −0.275
[−0.399, −0.182] (0.699 → 0.424), OP2 −0.202 [−0.265, −0.134], OP3 −0.152 [−0.225, −0.071], OP4 −0.373
[−0.431, −0.310], OP5 −0.281 [−0.353, −0.221]; conflicting malignant nodules at OP2 −0.372 [−0.436, −0.291], OP3 −0.236
[−0.345, −0.132]. Capsule: masking raises overall sensitivity but lowers it for conflicting erosions at OP4 (−0.096
[−0.173, −0.029]) and OP5 (−0.104 [−0.178, −0.048]).

**R4 / F1 (supported).** Fine-tuned ResNet-50 on the hair traps: crossover +0.143 [+0.062, +0.219] (5 clusters).
With the archived runs a fine-tuned network now reproduces the crossover in three cohorts: hair (ResNet-50), thyroid
(ViT-S +0.168 [+0.074, +0.265]; ResNet-50 +0.057 [−0.021, +0.130] n.s.) and capsule (ResNet-50 +0.584 [+0.469, +0.691]).
**F1b (Amendment 1, not supported).** Fine-tuned ResNet-50 on ovary with 15 clusters: +0.058 [−0.037, +0.157]; the
network barely learns the task (ERM clean AUROC 0.596 in Trap A, 0.500 in Trap B), so the null is uninformative about
the law. **F2**: thyroid is the only cohort with a published benchmark on the same split (results/review2/literature.csv).
