# Pre-registration — round 4: masking implementations, real-overlap dose-response, external natural test, prospective theory test

Committed on 2026-09-28, before any model is fitted, any feature is extracted or any cell count is computed for these
analyses. Implementation notes: `scripts/round4/README.md`. Outputs: `results/round4/`. Any change after this commit
is recorded as a dated amendment at the end of this file, with its reason, and stated in the results.

## Why these four analyses
Every one answers an objection a reviewer can raise against the location law (masking to the region of interest
removes an out-of-ROI shortcut and keeps, or amplifies, an in-ROI one):

| id | objection | analysis |
|---|---|---|
| R8 | "Mean-colour fill is one implementation; clinical pipelines crop to the lesion or use black backgrounds." | Four other masking implementations on the four real traps. |
| R9 | "Trap A (r ≥ 0.5) and Trap B (r < 0.1) are two hand-picked extremes; the effect may be a threshold artefact." | Dose-response over five bins of the real overlap r. |
| R10 | "Resampled traps are artificial and the data are internal." | Train on ISIC 2019, test on ISIC 2020 (external, patient-level, natural prevalence, no resampling). |
| R11 | "The theory was only checked on results that already existed." | The fixed theory procedure applied to the R8/R9 cells, which do not exist at commit time. |

## Common settings (unchanged from the existing real-trap analyses)
- Encoder DINOv2 ViT-B/14 at 518 px, frozen; logistic heads (`wtss.heads`); regularisation and decision threshold
  chosen on the clean validation environment; seeds 42, 123, 456, 789, 2026; 5-fold group-safe cross-validation.
- Trap environments as in `wtss.data.isic2019_spec.build_spec_envs`: training and correlated test
  P(A=1|Y=1)/P(A=1|Y=0) = 0.9/0.1, reversed test 0.1/0.9, clean 0.5/0.5; artifact-bearing cells matched to the
  artifact-free source mix.
- Cohorts and trap definitions: ISIC 2019 hair (`scripts/run_spec_e13.py`, spec protocol), thyroid calipers,
  capsule debris and ovary calipers (`scripts/run_thyroid_traps.py`, `--cohort thyroid|capsule|ovary`). Same
  artifact-presence rules, artifact-free groups, leakage groups and caches as those scripts.
- Intervals: crossed seed × image bootstrap (`WTSS_BOOTSTRAP=crossed`, `wtss.stats_crossed`), 10,000 replicates;
  one Poisson(1) weight per test image shared by every term in which the image appears.
- Reproducibility gate: every new trap run re-fits ERM and mean-fill masking on identical environments. Their mean
  reversed and clean AUROCs must agree with the stored runs (`results/rerun_2026-09-28/{spec_e13/dino518_spec,
  thyroid/dino518_main, capsule/dino518_main, ovary/dino518_main}/metrics_per_seed.csv`) within 0.03 per trap and
  arm. A larger difference stops the analysis for that cohort until the cause is found and recorded.

## R8 — masking implementations
Views, all computed from the cached 518-px image and its ROI mask:
- `mask` (reference, existing): pixels outside the ROI set to the dataset mean colour.
- `mask_black`: pixels outside the ROI set to black.
- `mask_blur`: pixels outside the ROI replaced by a Gaussian-blurred copy of the image (σ = 16 px at 518 px); the
  ROI stays sharp. Partial removal: coarse context survives, thin structures are attenuated.
- `crop_box`: crop to the ROI bounding box enlarged by 10 % of its width/height on each side (clipped to the image),
  padded to a square with the mean colour, resized to 518 px (bicubic). Out-of-ROI pixels inside the box survive.
- `crop_mask`: `crop_box` applied to the `mask` image: no out-of-ROI pixels, lesion enlarged to fill the frame.
- Images whose ROI is empty keep the full image in the crop views (their number is reported).

Arms: ERM on the original image and a plain ERM head on each view, fitted on identical environments.

Primary endpoint (family of 16 = 4 views × 4 cohorts, Holm): crossover
C_v = [rev(v) − rev(ERM)]_TrapB − [rev(v) − rev(ERM)]_TrapA on reversed-test AUROC.
- **H8a** (full-removal views `mask_black`, `crop_mask`): C_v > 0 in all four cohorts.
- **H8b** (partial-removal views `mask_blur`, `crop_box`): C_v > 0, and C_mask − C_v > 0 (secondary contrast, one
  crossed bootstrap over the same predictions), because part of an out-of-ROI artifact survives.
- Decision: the location law is called **implementation-independent** if all eight H8a tests are positive and
  Holm-significant, **largely implementation-independent** with six or seven, and **implementation-dependent** otherwise.
- Secondary, descriptive with intervals: Trap-A gain rev(v) − rev(ERM); clean-AUROC cost clean(v) − clean(ERM).

## R9 — dose-response over the real overlap r
- Artifact-bearing images (each cohort's own presence rule) are binned by r, the share of artifact pixels inside
  the ROI: b1 [0, 0.1), b2 [0.1, 0.3), b3 [0.3, 0.5), b4 [0.5, 0.75), b5 [0.75, 1]. Each bin is one trap: A = 1 is
  the bin, A = 0 the cohort's shared artifact-free group; matching, seeds, folds and environments as above.
- Count gate, applied to the counts before anything is fitted: every matched A × Y cell ≥ 25 and ≥ 40 reversed-test
  positives summed over the folds of seed 42. A failing bin is merged with its neighbour towards the middle of the
  r range (b1 → b2, b5 → b4, b2 → b3, b4 → b3; b3 with its smaller neighbour) and the gate is re-applied. A cohort
  with fewer than three bins after merging is reported descriptively only. A bin's position x_b is the mean r of its
  artifact-bearing images, fixed from the counts step.
- Arms: ERM, `mask`.
- Primary endpoint (one test per cohort that passes, Holm): least-squares slope β of the mask gain
  g_b = rev(mask) − rev(ERM) on x_b (equal bin weights), written as a fixed linear combination of AUROCs so that the
  crossed bootstrap carries the shared artifact-free images consistently across bins.
- **H9**: β < 0 in every cohort (masking helps less as more of the artifact lies inside the ROI).
- Secondary, descriptive: g_b with intervals per bin; Spearman correlation of x_b and g_b; the r at which g crosses
  zero (linear interpolation between adjacent bins); comparison with the slopes of the controlled overlap sweeps.

## R10 — external natural test (ISIC 2019 → ISIC 2020)
- Training and validation: every quality-controlled image of the ISIC 2019 spec cohort, natural distribution (no
  resampling). Per seed a group-safe 80/20 train/validation split (groups as in `scripts/run_natural.py`).
- Test: every ISIC 2020 image in `results/rerun_2026-09-28/external_isic2020/cohort.csv` with a non-empty lesion
  mask, minus near-duplicates of any ISIC 2019 image (perceptual hash, 8×8, Hamming distance ≤ 8, computed with the
  same function on both 518-px caches). The number removed is reported. Label: melanoma (target = 1) vs benign.
- In-lesion hair A: ISIC 2019 as in `scripts/run_natural.py` (hair_px_native > 30 and r_spec ≥ 0.5); ISIC 2020 as in
  the external analysis (predicted hair pixels > τ of `segmenter.json`, and r ≥ 0.5).
- Arms: as `scripts/run_natural.py` (ERM, mask, balanced, mask_balanced, U-MtE, U-MtE_protect, U-MtE_balanced,
  U-MtE_protect_balanced, DFR, mask+DFR, JTT; generic overlay library).
- Hard pairs (the pairing that conflicts with the training association, in-lesion hair more common in ISIC 2019
  melanomas): melanomas without in-lesion hair vs benign lesions with it; easy pairs the complement.
- Metrics on ISIC 2020: AUROC, cross-group AUROC, AUROC within in-lesion-hair images, hard- and easy-pair AUROC;
  operating points OP1–OP5 of `scripts/analysis/operating_points.py` with thresholds from each seed's ISIC 2019
  validation predictions only; sensitivity among melanomas without in-lesion hair.
- **H10a**: hard-pair AUROC(mask) − AUROC(ERM) < 0. **H10b**: at OP5 (validation sensitivity ≥ 0.90), sensitivity
  among melanomas without in-lesion hair is lower with mask than with ERM. **H10c**: hard-pair AUROC(balanced) −
  AUROC(mask) > 0. Crossed bootstrap; reported whichever way they go.

## R11 — prospective test of the theory
- The fixed procedure of `docs/PREREGISTRATION_THEORY_PREDICTION.md` (per cell: fit S and A to the clean and
  correlated AUROC; predict the reversed AUROC, never used in the fit; no shared or tuned parameter) is applied
  unchanged to every R8 cell (4 cohorts × 2 traps × {ERM, mask and the four views}) and every R9 cell (bins × {ERM,
  mask}). R10 is outside the model (no 0.9/0.1 training set) and is not used.
- **T1'**: reversed AUROC, predicted vs observed, all new cells: MAE, Pearson r, share within ±0.05, against the
  reference predictors (a) rev = clean and (b) rev = 2·clean − corr.
- **T2'**: R8 crossovers C_v, predicted vs observed: sign agreement (16 cells), Pearson r, MAE.
- **T3'**: R9 slopes β and bin gains g_b, predicted vs observed: sign agreement, Pearson r, MAE.
- Decision rule as before: **quantitatively predictive** if every T2' sign agrees, Pearson r ≥ 0.7 for T1' and T2',
  and the T1' MAE is below that of reference (a); otherwise **qualitative account**.

## Not changed by this round
No existing result, table or claim is re-estimated here; the round adds evidence and is reported as a separate
block, including every hypothesis that is not supported.
