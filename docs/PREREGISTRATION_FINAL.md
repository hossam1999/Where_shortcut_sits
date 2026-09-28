# Pre-registration — final runs (corrected bootstrap everywhere, regression-adjusted matching, external validation gate)

Committed and pushed (branch `stages-and-final-runs`) before any of the runs below. Commit times come from this machine
and are not independent proof of order; the GitHub push time of this commit is the external reference.

## A1 — regeneration with the corrected (crossed) bootstrap
- **What.** Per-image predictions of every controlled sweep (dermoscopy ruler: main, proposed, occlusion, dilation,
  stress, DINOv2@224, DermLIP; thyroid, ovary, capsule; chest tube) and every U-MtE / remedy trap run are regenerated
  into `results/rerun_2026-09-28/` (same commands, `WTSS_RESULTS` pointing to the new folder; old results untouched).
  Runs whose per-image predictions already exist on this machine (rebuilt traps, matched traps, natural tests,
  transplant, neutral paste, fine-tuned hair, ovary power runs, hair scale) are copied as predictions and only their
  statistics are recomputed.
- **Estimator.** Every interval is recomputed with `WTSS_BOOTSTRAP=crossed` (`wtss.stats_crossed`: clusters resampled,
  one Poisson weight per test image shared by all clusters; 10,000 replicates unless stated).
- **Rule.** For every interval cited in the paper, supplement or stage reports: old CI, new CI and whether the
  "excludes zero" verdict changed are tabulated (`results/bootstrap_correction/sweeps_umte_old_vs_new.md`). A claim
  whose new CI includes zero is rewritten (no longer stated as supported). An interval that cannot be regenerated is
  removed from every document (the point estimate may be kept and is labelled as not re-estimated).
- **Tolerance.** Regenerated point estimates are compared with the archived ones; differences > 0.03 are listed.

## A2 — regression-adjusted crossover on the matched traps (hair, thyroid, ovary)
- **Outcome.** Per test image of the reversed environment, the placement value of the masked minus the ERM score
  (positives: weighted share of negatives ranked below, ties ½; negatives: share of positives ranked above,
  complemented), so that the mean over images of each class equals the AUROC and the unadjusted between-trap
  difference of means equals the crossover.
- **Model.** Within each seed: weighted least squares of the outcome on an intercept, the Trap-B indicator, the label
  and **all** pre-specified matching covariates of the cohort (standardised; categorical one-hot), fitted on the
  matched sample (calliper 0.2 SD, docs/PREREGISTRATION_REVIEW2.md R1b). The adjusted crossover is the Trap-B
  coefficient averaged over seeds (matching + outcome regression = doubly robust).
- **Uncertainty.** Crossed bootstrap (seeds resampled; one Poisson weight per image shared across seeds and traps;
  placement values and the regression recomputed in every replicate; 2,000 replicates).
- **Claim A2.** Adjusted crossover > 0, CI excluding 0, in each of hair, thyroid, ovary; Holm over the three. Capsule
  stays descriptive (infeasible matching under the pre-registered rule).

## A4 — external validation gate (decided before any external label is read)
Run only if all hold: (1) direct download without registration, application or agreement; (2) licence permits
research use; (3) patient-level identifiers; (4) ROI masks exist or can be produced by our existing dermoscopy U-Net;
(5) cell-count gate: in both traps, at least 50 positive and 50 negative artifact-bearing images and at least 50
positive and 50 negative artifact-free images after source matching. Candidate order: ISIC 2020 (dermoscopy, patient
IDs, CC BY-NC), then thyroid cohorts with patient IDs. For ISIC 2020 the hair masks would come from a hair segmenter
trained on the ISIC 2019 hair masks only (frozen before any ISIC 2020 image is scored); lesion masks from the existing
spec U-Net; groups = patient_id; traps, environments, arms (erm, mask), seeds and folds as the ISIC 2019 spec traps;
**Claim X1**: crossover > 0 (crossed CI). If any condition fails the cohort is not run and is listed for the author.
