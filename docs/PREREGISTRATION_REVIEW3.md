# Pre-registration — third review round (paste-edge control, crossed bootstrap, stricter matching)

Committed and pushed to GitHub before any of the analyses below is computed. Responds to a third set of comments on
the second-review results (`docs/REVIEW3_RESPONSE.md`) and to a statistical issue found while checking them.

## Registration status of the second-review analyses (for the record)
The real-artifact transplant (R2), the covariate-matched traps (R1), the operating points (R3) and the fine-tuned hair
model (R4) were **designed after the earlier results were known** (in response to a review of them) and **registered
before their own results were computed** (`docs/PREREGISTRATION_REVIEW2.md`, pushed in 29cd0da). The subgroup of
malignant nodules with an in-ROI caliper was **defined post hoc** in the natural-distribution analysis
(`PREREGISTRATION_NATURAL.md`, "subset bootstraps, post hoc"); the operating-point test in it (R3/S3) was registered
before the operating points were computed but after the subgroup's AUROC result was known. These analyses form their
own Holm families (T1: four cohorts; M1: four cohorts) and are **not** part of the original 16-test primary family,
which is unchanged. The same holds for everything registered below.

## R5 — paste-edge control for the real-artifact transplant
Concern: the model might respond to the editing seams of a pasted crop rather than to the artifact.
- **Design.** Identical to R2 (same recipients, folds, presence seeds, environments, arms erm and mask, DINOv2@518),
  and for every recipient the **same** donor instance mask, dihedral transform, positions (inside / outside the ROI)
  and alpha compositing (Gaussian-blurred mask, σ = 0.7). Only the pasted pixels differ: instead of the artifact they
  are **neutral tissue** cut from the same donor image with the same (transformed) mask, at a random position in the
  donor's field of view whose window does not touch the donor's artifact mask (dilated by 5 px); if no such window
  exists, from a random artifact-free image of the cohort (stable seeds). The seam geometry is therefore identical to
  the artifact transplant; only the content is neutral.
- **N1** (descriptive) ERM shortcut gap, correlated − reversed AUROC, for neutral pastes at each location.
- **N2** neutral-paste location interaction [mask − ERM]_out − [mask − ERM]_in (reversed AUROC), CI.
- **N3 (primary)** artifact interaction − neutral interaction > 0 (paired: same images and seeds; crossed bootstrap,
  R6), per cohort, Holm over the four cohorts.
- **Decision rule.** In a cohort, the transplant result is attributed to the artifact rather than the seams if N3
  holds **and** the neutral interaction is below one quarter of the artifact interaction (point estimates). Otherwise
  the paper states that pasting seams contribute to the transplant signal in that cohort.

## R6 — crossed (seed × image) bootstrap
Issue found while checking the thyroid subgroup CI: the hierarchical bootstrap (`wtss.stats`, unchanged from the
pilot) resamples images independently within every sampled seed cluster. When the clusters share the same test images
(the official thyroid split, the transplant, and largely the real traps, where every seed's pooled folds cover the
same pool), this divides the test-set sampling variance by the number of clusters and makes CIs too narrow.
- **Estimator.** Per replicate: clusters (seeds, or folds) resampled with replacement; one Poisson(1) weight per
  **image identifier**, shared by every cluster in which that image appears; the statistic (AUROC difference,
  crossover, interaction, sensitivity difference with thresholds re-estimated on weighted validation images) is
  computed per selected cluster with those weights and averaged. 10,000 replicates (2,000 for operating points).
- **Applied to** every headline claim whose predictions are available on this machine: the four real-trap crossovers
  (rebuilt runs), the three feasible matched crossovers, the four transplant interactions and R5 N2/N3, the five
  natural hard-pair mask − ERM contrasts, the thyroid operating points (S1–S3), and the fine-tuned hair crossover.
- **Rule.** The paper reports the crossed CI for these claims (the original CI in the supplement). A claim is stated
  as supported only if its crossed CI excludes 0; claims whose crossed CI includes 0 are downgraded in the text.
  Analyses whose predictions are not on this machine (controlled sweeps, U-MtE arms of the archived runs) keep the
  original CI, and the paper says so.

## R7 — stricter matching (sensitivity)
Within-label balance after R1 matching exceeded |SMD| 0.1 for some covariates (hair 0.16, thyroid 0.15, ovary
0.14). Re-run R1 with a calliper of 0.05 SD of the logit (everything else unchanged; `--tag matched05`). Reported:
images kept, max |SMD| pooled and within label, crossover (crossed CI). Descriptive sensitivity analysis; no new
claim.

## Results (results/review3/*.csv)
- **R5.** Neutral-paste interaction: hair +0.204 [+0.164, +0.244], thyroid +0.145 [+0.130, +0.161], ovary +0.052
  [+0.021, +0.084], capsule +0.459 [+0.383, +0.535] (ratio to artifact 0.81 / 0.52 / 0.71 / 0.92). N3 (artifact −
  neutral, Holm): hair +0.049 [+0.026, +0.072] supported, thyroid +0.132 [+0.116, +0.148] supported, ovary +0.022
  [−0.002, +0.048] not, capsule +0.037 [−0.017, +0.093] not. **Decision rule: not attributable to the artifact alone in
  any cohort** (neutral ratio ≥ 0.25 everywhere); the paper states that pasting seams contribute. N1: neutral pastes are
  learned as shortcuts (ERM correlated − reversed gap 0.09–0.42). Deviation: 213/340 capsule and 56/860 hair neutral
  windows came from artifact-free images (no artifact-free window in the donor).
- **R6.** Crossed CIs 1.1–2.2× wider than per-seed CIs; all headline claims still exclude zero (table S15); downgraded:
  balancing transplant interaction for ovary/capsule, MSK all-pairs mask − ERM. Thyroid subgroup (n = 78): −0.372
  [−0.499, −0.236].
- **R7.** Calliper 0.05: kept 1,777 / 299 / 165 per trap; crossover +0.124 [+0.086, +0.162], +0.218 [+0.165, +0.270],
  +0.152 [+0.041, +0.265]; within-label max |SMD| 0.18 / 0.19 / 0.18 (not improved); capsule still infeasible.
