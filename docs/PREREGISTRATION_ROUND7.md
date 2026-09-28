# Pre-registration — round 7: is the thyroid sensitivity loss a threshold artefact?

Committed on 2026-09-28 before the analysis below was run. Implementation: `scripts/round7/`. Outputs:
`results/round7/`. Only saved predictions are used (no model is trained, no image is read). Changes after this commit
are recorded as dated amendments at the end.

## Why
On the official patient-disjoint thyroid split, masking lowers sensitivity at every validation-fixed operating point
and raises specificity (Stage 5: frozen DINOv2 probe; Stage 8: fine-tuned ConvNeXt-T, FT2–FT3). A reviewer can argue
that this is threshold transport — the masked model's test scores sit differently relative to a threshold learned on
validation images — and would disappear if the threshold were re-tuned for the deployment site. The hard-pair AUROC
(FT1) is threshold-free but does not speak to sensitivity at a clinically fixed specificity.

## Design
The best case for threshold re-tuning is an oracle that sets each model's threshold on the test set itself. If the
subgroup loss survives this, no re-tuning can remove it.
- **Models and predictions** (unchanged, already saved): the fine-tuned model of round 6
  (`results/round6/ft_natural/predictions.csv.gz`; 5 seeds; arms ERM and mask) and the frozen DINOv2 probe of Stage 5
  (`results/natural/thyroid_dino518_repro/predictions.csv.gz`). Test set: the 614 official test images (env `clean`).
- **Matched thresholds.** For each seed and each arm separately, the threshold is chosen on the test images: M-spec80
  (the threshold that maximises sensitivity subject to test specificity ≥ 0.80, `threshold_w` of
  `scripts/analysis/operating_points.py`), M-spec90 (≥ 0.90), M-sens80 (maximises specificity subject to test
  sensitivity ≥ 0.80). Both arms therefore operate at the same test specificity (or sensitivity).
- **Subgroups** (as in Stage 5 and FT3): malignant nodules with an in-ROI caliper (conflicting, the registered FT3
  subgroup, 78 nodules), malignant nodules without one (aligned), all malignant nodules; realised specificity is reported
  to show the matching.
- **Estimator.** Crossed bootstrap as everywhere (seeds resampled, one Poisson weight per test image shared by both arms
  and all seeds), thresholds re-estimated on the weighted test images in every replicate; 10,000 replicates, seed
  20260928; point estimate = mean over seeds of mask − ERM with unit weights.

## Hypotheses
- **TM1 (primary)**: fine-tuned model, M-spec80, sensitivity among the conflicting malignant nodules, mask − ERM < 0.
- **TM2 (secondary)**: the same for the frozen probe of Stage 5.
- Holm over TM1–TM2; one-sided p = share of replicates ≥ 0 (floored at 1/n). Supported if the estimate is < 0, the 95 %
  CI excludes 0 and the Holm-adjusted p < 0.05.
- **Descriptive** (no test): M-spec90 and M-sens80; overall and aligned sensitivity at every matched threshold; the same
  analysis on the other Stage 5 natural test sets (ISIC BCN, HAM and MSK hold-outs; capsule), frozen probe, with each
  test set's own conflicting direction.

## Decision rule for the text
- TM1 supported → the paper may say that the subgroup sensitivity loss is not a threshold artefact: it persists when each
  model's threshold is re-tuned on the test distribution itself.
- TM1 not supported → the paper says the operating-point loss of Stages 5 and 8 is, at least in part, a consequence of
  transporting validation thresholds, and states the threshold-free hard-pair AUROC (FT1) as the evidence of harm.
Both outcomes are reported.

## Not changed by this round
Every earlier registration, estimate and verdict. The exploratory round-6 analyses E1–E2
(`scripts/round6/exploratory.sh`) are run in the same session and remain exploratory.

## Results (added after the run; the text above is unchanged)
Registration at `cbf4229`; results at `68e2432`; write-up in Stage 8 (Section 5.7). Subgroup size 78 in every seed of
both models, as registered.

- **TM1 supported**: fine-tuned model, both arms at test specificity 0.80 (realised 0.806 / 0.804): sensitivity among
  the conflicting malignant nodules 0.544 → 0.390, mask − ERM −0.154 [−0.302, −0.023], Holm p = 0.019. Overall
  sensitivity −0.017 [−0.124, +0.070]; aligned malignant +0.051 [−0.070, +0.150].
- **TM2 supported**: frozen probe 0.505 → 0.331, −0.174 [−0.313, −0.012], Holm p = 0.019; aligned +0.124 [+0.001, +0.231].
- Descriptive: specificity 0.90 (fine-tuned −0.105 [−0.221, +0.019]) and matched sensitivity 0.80 (−0.087
  [−0.172, +0.018]) point the same way without significance. Other natural test sets at specificity 0.80, conflicting
  positives: BCN −0.040 [−0.065, −0.015], HAM +0.067 [+0.032, +0.106], MSK −0.001 [−0.054, +0.053], capsule +0.037
  [−0.013, +0.110].
- Decision rule: the subgroup sensitivity loss is not a threshold artefact; about 30 % of the loss at validation-fixed
  thresholds (FT3, −0.521) survives re-tuning on the test set, the rest is threshold transport.
- Exploratory round-6 E1: thyroid traps without the images both labellers contradict: crossover +0.257 [+0.209, +0.304]
  (Stage 3: +0.239). E2: FT3 in the 36 doubly confirmed nodules −0.394 [−0.570, −0.236] (fine-tuned), −0.189
  [−0.364, −0.011] (frozen).
