# Pre-registration — Round 8: is there a remedy that dominates ROI masking?

Registered 2026-09-28, before any confirmation prediction exists. Design: `docs/ROUND8_DESIGN.md` (Phase 1, approved by the
author on 2026-09-28 with the six changes listed in §0). Code: `scripts/round8/` (`bash scripts/round8/run_all.sh`).
Changes after this commit are made only as dated amendments at the end of this file; the text above them is never edited.

## 0. What was known when this was written, and what changed from the design
- **Everything existing was seen.** Phase 1 recomputed every earlier test result (`results/round8/scoreboard_existing.csv`).
  The candidates below were designed *after* the existing arms' test results were known. The natural test sets
  (thyroid split, capsule, ISIC BCN / HAM / MSK) and ISIC 2019 → 2020 are therefore **semi-confirmatory for every
  candidate**. The **primary confirmation is the new trap resamples** (new environment, fold and validation seeds, §2).
  Their images come from the same pools as before, so images overlap with earlier test sets; the environments, splits
  and models are new.
- **Changes the author required on approval (2026-09-28):**
  1. Selection guard: the objective is worst-group validation AUROC. The guard is validation AUROC on **same-artifact
     pairs** ≥ the masking head's − 0.005 on the same split. The all-pairs guard of the design (§5d) is dropped. A unit
     test shows that on a trap-like validation split (0.9 / 0.1) the new guard admits a shortcut-free head and the old
     one does not (`tests/test_round8.py::test_same_artifact_guard_admits_shortcut_free_head_all_pairs_guard_does_not`).
  2. N3 locrand: instances are pasted only into artifact-free training images, with class-specific probabilities that
     equalise P(A = 1 | Y) after pasting, inside or outside the ROI with probability ½ each. Plus a registered
     pasted-versus-real probe (§6).
  3. N1 on dermoscopy: a descriptive sensitivity run with mean-difference directions estimated only from training
     images that no informative independent source contradicts. It reads only `results/round6/clean_traps` outputs,
     never `results/round6/rating/`, the agreement per-image tables or any audit key.
  4. Primary dominance candidates: **mask_cmc, full_cmc, mask_bal(λ), locrand** (Holm over these four; an
     intersection–union test within each). C2 selector, C3 combinations, C4 AFR / GroupDRO and N2 are descriptive.
  5. Confirmation as stated in the first bullet; **ovary is the held-out cohort**, and the dominance verdict is also
     reported without it.
  6. Novelty is not a condition. Every candidate is judged only by the criterion of §7 and labelled honestly (§3).
- **One correction of a premise, by me.** The approval text assumed that the trap validation split carries the
  0.9 / 0.1 training association. In the existing trap code it does not: heads choose C on `val_clean` (drawn at
  0.5 / 0.5), and `val_groups` is the whole inner validation fold at the pool's own composition. A validation split
  drawn at the training rates would leave 2–20 images in the minority cells (seed 42, fold 0, inner validation):

  | cohort, trap | val_groups: a0y0 / a0y1 / a1y0 / a1y1 | at training rates (0.9 / 0.1) |
  |---|---|---|
  | thyroid A | 165 / 115 / 150 / 45 | 165 / 5 / 18 / 45 |
  | thyroid B | 170 / 115 / 37 / 15 | 169 / 2 / 19 / 14 |
  | capsule A | 29 / 46 / 129 / 62 | 29 / 7 / 3 / 61 |
  | capsule B | 44 / 37 / 61 / 125 | 43 / 14 / 5 / 124 |
  | ovary A | 20 / 32 / 64 / 42 | 20 / 5 / 2 / 41 |
  | ovary B | 22 / 31 / 17 / 17 | 22 / 2 / 2 / 16 |
  | ISIC A | 138 / 44 / 452 / 184 | 138 / 20 / 15 / 184 |
  | ISIC B | 92 / 10 / 496 / 77 | 92 / 9 / 10 / 76 |

  That is too few for a worst-group AUROC. The objective and the guard are therefore both computed on `val_groups`.
  The guard change still matters: the same-artifact statistic does not depend on the validation association at all
  (inside a same-artifact pair a linear score's artifact term cancels), so it is the right guard whatever that
  association is.

## 1. Question and decision rule for the paper
Is there a remedy that is never worse than ROI masking and better where masking fails? Decision (§9): **dominates** /
**wins where masking fails but costs elsewhere** / **no gain**, from the primary family of §7.

## 2. Data, encoders and seeds
- **Traps** (`scripts/round4/common.trap_cohort`, pools unchanged): thyroid, capsule, **ovary (held out)**, ISIC hair.
  Environments from `wtss.data.isic2019_spec.build_spec_envs` with the **confirmation seeds 8101, 8202, 8303, 8404,
  8505**: each seed redraws the grouped 5-fold split, the inner validation groups and every environment subsample.
  Folds are pooled per seed (5 bootstrap clusters). Test environments: reversed (0.1 / 0.9), correlated (0.9 / 0.1),
  clean.
- **Natural test sets** (`scripts/run_natural.py` definitions): thyroid official split, capsule (the group-safe 80 / 20
  split is redrawn with each seed, so capsule test sets are new), ISIC 2019 BCN / HAM / MSK hold-outs, and ISIC 2019 →
  ISIC 2020 (calibrated duplicate tier of round 5). The validation folds are redrawn with the confirmation seeds.
- **Encoders.** Primary: DINOv2-B/14 @ 518 (every cohort). Secondary replication (head-level candidates only, §7 R):
  MedSigLIP (thyroid, capsule, ovary) and ConvNeXt (thyroid, capsule).
- **Development**: the earlier environments (seeds 42, 123, 456, 789, 2026), **validation data only**, on the traps of
  thyroid, capsule and ISIC (DINOv2), and on thyroid and capsule with MedSigLIP and ConvNeXt. Ovary is never run in
  development.
- **Sweeps (D5)**: thyroid caliper, capsule debris, ovary caliper, ISIC 2018 ruler (DINOv2-B/14 @ 518). The NIH chest-tube
  sweep of Stage 2 cannot be rerun: its cohort table (labels) is not on this machine, only its cached features.
  They are cross-fitted: 5 grouped folds, so every cohort image is a test image once (about 5× the earlier test sets).
  Presence seeds are the confirmation seeds; r = 0 and r = 1; the cached renders are reused.
- **Fine-tuned (D6)**: thyroid ConvNeXt-T, recipe of round 6 (`results/round6/ft_recipe.json`, chosen on validation):
  official split with the 5 confirmation seeds; traps with environment seeds 8101 and 8202 × 5 folds (10 clusters).

## 3. Candidates (labels: published / adaptation / new — no adaptation is called new)
Heads are logistic regressions on frozen features; C ∈ {0.01, 0.1, 1, 10} is chosen by AUROC on `val_clean` (natural
data: the validation fold), as for every earlier arm. Every primary candidate's setting list includes the masking head
as fallback (§4).

| candidate | role | what it does | needs | label | closest prior work |
|---|---|---|---|---|---|
| **mask_bal(λ)** | primary | masked features; sample weight (1/n_y)(n_{a,y}/n_y)^−λ, λ ∈ {¼, ½, ¾, 1}; λ = 0 is masking, λ = 1 is mask_balanced | image-level artifact labels (train, validation) | adaptation | group reweighting / balancing: Sagawa et al. 2020 (arXiv 1911.08731), Idrissi et al. 2022 (arXiv 2110.14503) |
| **mask_cmc** | primary | masked features projected onto the complement of the within-class mean differences δ_y = μ(a=1,y) − μ(a=0,y) (setting *pooled*: (δ_0+δ_1)/2; *per_class*: span(δ_0, δ_1)); ordinary head on all images at natural weight, so w·δ_y = 0 | labels at training only | adaptation | first-moment, linear case of the anti-causal conditional-MMD regulariser, Veitch et al. 2021 (arXiv 2106.00545, Eq. 3.2); Makar et al. 2022 (arXiv 2105.06422); ComBat-type harmonisation (doi 10.1093/biostatistics/kxj037) |
| **full_cmc** | primary | the same on unmasked features (keeps the context masking removes) | as mask_cmc | adaptation | as mask_cmc |
| **locrand** | primary | masked head trained after pasting real artifact instances (§3.1) | artifact masks of donor images (existing detectors / published masks), training labels for the pasting rates | adaptation | copy-paste augmentation, Ghiasi et al. 2021 (arXiv 2012.07177); the opposite operation (removal by inpainting), Nauta et al. 2022 (doi 10.3390/diagnostics12010040) |
| locrand_loc | descriptive | as locrand, but in-ROI and out-of-ROI presence each equalised across classes | as locrand | adaptation | as locrand |
| selector_v2 (C2) | descriptive | §4's rule across {mask, U-MtE, protected U-MtE, and the chosen mask_bal, mask_cmc, full_cmc, locrand} | as its members | adaptation | validation-based model selection (Yang et al. 2023, arXiv 2302.12254); this project's U13/U14 |
| mask_dfr, umte_balanced, ens(α) (C3) | descriptive | DFR on masked features; U-MtE + balancing; α·logit(mask) + (1−α)·logit(mask_balanced), α ∈ {¾, ½, ¼} | labels | published / adaptation | DFR, Kirichenko et al. 2023 (arXiv 2204.02937); this project's U-MtE |
| afr (C4) | descriptive | AFR for a linear probe on masked features: ERM head on 80 % of training images, reweighted head (exp(−γ p_true), γ ∈ {½, 1, 2, 4}) on the other 20 % | none in training | published (applied here) | Qiu et al. 2023 (arXiv 2306.11074) |
| groupdro (C4) | descriptive | linear GroupDRO on masked features, groups = artifact × label | labels | published | Sagawa et al. 2020 |
| mask_cfs (N2) | descriptive | head on masked features with penalty wᵀ(I + μΣ̃)w, Σ̃ = normalised covariance of the U-MtE insertion shifts, μ ∈ {1, 10, 100} (a soft U-MtE) | none | adaptation | counterfactual logit pairing, Garg et al. 2019 (arXiv 1809.10610); Ross et al. 2017 (arXiv 1703.03717) |
| mask_cmc_uncontra | descriptive (ISIC traps) | mask_cmc with δ estimated only from training images in `results/round6/clean_traps/isic/predictions.csv.gz` | as mask_cmc | sensitivity run | — |
| erm, mask, balanced, mask_balanced, umte, umte_protect | references | as in rounds 1–7 | — | — | — |

### 3.1 locrand in detail
- **Instances and placements** (`scripts/round8/locrand.py`). Recipients are artifact-free images: thyroid and ovary
  with no detected marker pixel; capsule with contamination < 3 %; ISIC with ≤ 30 native hair pixels. Donors are
  artifact-bearing images whose largest grouped artifact component meets the size limits of `wtss.transplant`
  (Stage 4). Each recipient gets K = 4 versions. Version k uses the k-th distinct donor in a stable random order, with a
  dihedral transform that admits both an in-ROI position (overlap ≥ 0.95) and an out-of-ROI position (overlap 0)
  inside the field of view. Both are rendered and masked; features are cached under
  `$WTSS_CACHE/features/round8/locrand/`.
- **Per training set**: a recipient uses its first version whose donor is a training image of that same training set,
  so no validation or test image contributes pixels. Recipients without one are not pasted. Target
  t = max_y P(A = 1 | y), with A = artifact presence anywhere. Class y receives round(t·n_y − n_{A=1,y}) pastes,
  capped by its number of recipients, chosen at random. Each paste goes inside or outside the ROI with probability ½.
  P(A | Y) before and after, the number of pastes per class and the inside/outside split are written for every training
  set. Validation and test images are never pasted.
- **Prediction from the theory, stated before the run.** Masking deletes the out-of-ROI pastes. Where the real
  artifacts all sit inside the ROI (Trap A), in-ROI presence after pasting and masking stays class-dependent: in Trap A
  training (0.9 / 0.1) the artifact-free class reaches 0.1 + 0.8 × ½ = 0.5 visible in-ROI artifacts against 0.9. So
  locrand is expected to reduce the Trap A shortcut only partly. locrand_loc (descriptive) equalises in-ROI presence
  itself and tests that prediction.
- **Fine-tuned (D6)**: the same plan, with pastes rendered on the fly into the masked training images of the
  ConvNeXt-T recipe (`scripts/round8/ft.py`).

## 4. Validation-only selection rule (replaces §5d of the design)
For every training set and every candidate family, among its settings (the masking head always included):
1. **Objective**: worst-group AUROC on `val_groups` = min(AUROC(Y1A0 vs Y0A1), AUROC(Y1A1 vs Y0A0)).
2. **Guard**: same-artifact AUROC on `val_groups` (the pair-weighted AUROC over Y1A1 vs Y0A1 and Y1A0 vs Y0A0) ≥ the
   masking head's same-artifact AUROC − 0.005. The masking head is always admissible.
3. Choose the admissible setting with the highest objective. Ties within 10⁻⁹ go to the setting closest to masking
   (grid order: smaller λ, fewer directions, smaller γ or μ, larger α).
Implemented in `scripts/round8/common.select_setting`, unit-tested. Natural data use their validation fold as
`val_groups`; the sweeps use the inner validation images with the 0.5 / 0.5 artifact assignment. The fine-tuned
locrand falls back to the fine-tuned masking model by the same rule.

## 5. Development, freeze, confirmation
1. **Development** (`heads_run.py --stage dev`): development seeds, every candidate fitted, the rule applied on
   `val_groups`. Only validation predictions and choices are written; no test environment is predicted. Ovary is refused.
2. **Freeze** (`freeze.py`) writes `results/round8/frozen_choice.json`: candidate list, grids, rule and margin, seeds,
   the checks of §6, the distribution of development choices, and the **held-out settings for ovary**: for mask_bal the
   median λ chosen in development (masking = 0), mapped to the nearest grid value; for every other family the most
   frequent choice (ties to the setting closest to masking). The same settings serve ovary with MedSigLIP.
   `run_all.sh` commits and pushes it; **every confirmation script refuses to run unless that file is committed and
   unchanged**.
3. **Confirmation**: confirmation seeds. Each non-held-out training set applies the rule on its own validation split;
   ovary uses the frozen settings without looking at its validation data. Then `analyse.py`.
- **Smoke runs** (`run_all.sh smoke`) never score an image outside the run's own training and validation data: every
  test environment is replaced by the validation split. Their output is git-ignored and is not a result. The complete
  smoke run (44 steps) passed before this registration was committed; no development or confirmation run had started.

## 6. Registered checks (reported whichever way they go; not part of the dominance test)
- **Pasted-versus-real probe** (per cohort, DINOv2). Positives: recipients with their version-0 instance pasted inside
  the ROI, masked. Negatives: images with a real artifact inside the ROI (present, r ≥ 0.5), masked. The split is
  80 / 20 by image group; the larger class is subsampled in training; logistic regression with C = 1; held-out AUROC
  with a 2,000-replicate image bootstrap. **If the AUROC is > 0.75, every report states that pasted artifacts are
  distinguishable from real ones and that the locrand result may not transfer to real artifacts.** A reference probe
  (recipients' plain masked view versus the same real-artifact images) shows how much of any separation comes from the
  two image populations rather than the instance.
- **Paste balance**: P(A | Y) before and after pasting in every training set; the share of training sets where the
  cap (too few recipients) leaves |P(A | Y=1) − P(A | Y=0)| > 0.02.
- **Fallback rates**: the share of training sets in which each family's rule returned the masking head.
- **N1 dermoscopy sensitivity** (descriptive): mask_cmc_uncontra versus mask_cmc on the ISIC traps.

## 7. Criterion and hypotheses (exact components and margins)
Every contrast is candidate − masking (the masking head of the same training sets), AUROC, crossed seed × image
bootstrap with 10,000 replicates shared by all arms of a source. One-sided p = share of replicates ≤ −margin (NI) or
≤ 0 (SUP), floored at 1/10,000. A component is met if p ≤ 0.025, i.e. its 95% interval's lower bound clears the margin.
min(rev, corr) is the replicate-wise minimum of reversed and correlated AUROC.

**Primary family P — H_c: "candidate c dominates masking", c ∈ {mask_cmc, full_cmc, mask_bal, locrand}** (DINOv2):
| component | cells | test | margin |
|---|---|---|---|
| D1 natural and external all pairs | thyroid split, capsule, BCN, HAM, MSK, ISIC 2019 → 2020 (6) | NI | 0.01 |
| D2 hard pairs | thyroid, BCN, MSK, ISIC 2019 → 2020 (4) | SUP | 0 |
| D2 Trap A min(rev, corr) | thyroid, capsule, ovary, ISIC (4) | SUP | 0 |
| D3 per cell | Trap B reversed, Trap B min(rev, corr), clean (Trap A models), clean (Trap B models) × 4 cohorts (16) | NI | 0.03 |
| D3 pooled | each of the 4 D3 types, mean over the 4 cohorts (4) | NI | 0.01 |
| D4 no shortcut flipping | every trap (8) | point: correlated ≥ reversed − 0.02 | — |

H_c is an intersection–union test: p_c = max over the 34 tested components (D1–D3), and p_c = 1 if any D4 condition
fails. **Holm over the four p_c at one-sided α = 0.025**; c dominates if its Holm-adjusted p ≤ 0.025. The same verdict
**without ovary** (D2 Trap A over 3 cohorts, D3 per cell over 3, pooled over 3) is reported with its own Holm
adjustment; it is secondary.

**Family D5 — controlled sweeps** (Holm over mask_bal, mask_cmc, full_cmc; each an IUT over 12 components): for each
of the 4 sweeps, r = 1 reversed SUP (margin 0); r = 0 reversed NI (0.03); r = 0 clean NI (0.03). locrand is not
defined for synthetic artifacts.

**D6 — fine-tuned thyroid, locrand_ft versus fine-tuned masking** (one IUT, no multiplicity): natural all pairs NI
(0.02); natural hard pairs SUP (hard = malignant with an in-ROI caliper vs benign without, as FT1); Trap A
min(rev, corr) SUP.

**Family R — replication** (Holm within the 15 tests): Trap A min(rev, corr) SUP for mask_bal, mask_cmc and full_cmc ×
{MedSigLIP: thyroid, capsule, ovary; ConvNeXt: thyroid, capsule}.

**Descriptive** (unadjusted 95% intervals): every arm × every cell of every source
(`results/round8/confirm/descriptive_all_cells.csv`), including easy pairs, all losses, the checks of §6 and every
descriptive candidate.

## 8. Theory's predictions, stated before the run (docs/ROUND8_DESIGN.md §4–5)
- mask_cmc / full_cmc: Trap A large gains (as mask_balanced), Trap B ≈ masking, natural all pairs ≈ masking plus
  π_hard·g − π_easy·l (small), hard pairs gain. Capsule at risk (debris resembles fibrin). full_cmc better than
  mask_cmc in dermoscopy (context informative), worse where masking denoises (thyroid, capsule).
- mask_bal(λ): between masking and mask_balanced. The rule should pick small λ on natural data and larger λ on traps.
  A single λ may not satisfy both ISIC 2020 and Trap A.
- locrand: partial Trap A gain (§3.1), little on hair (correlate-carried), none of the reweighting cost. D6 is the only
  path to the fine-tuned model.
- The trap and clean components of D3 may fail through imprecision even for a method as good as masking; the pooled
  0.01 test and the 0.03 per-cell margin are the registered answer to that, not a promise that they will pass.

## 9. Decision rule for the paper
- **Dominates** — at least one primary candidate passes H_c after Holm (all cohorts). The paper recommends it with its
  label (all four are adaptations) and its requirements (image-level artifact labels for three of them; artifact masks
  of donor images for locrand).
- **Wins where masking fails but costs elsewhere** — no candidate dominates, but at least one meets every D2
  component. The paper reports exactly which D1 / D3 / D4 components it fails and by how much, and keeps the decision
  guide.
- **No gain** — otherwise. The paper states that no tested remedy, published or adapted, is better than masking where
  masking fails without costing elsewhere.
The verdict without ovary is reported beside it. D5, D6 and R are separate claims and do not change the decision.

## 10. Compute and order
`bash scripts/round8/run_all.sh`: tests → locrand placements, pasted features and probe (GPU ≈ 15–20 min) →
development (CPU) → freeze, commit, push → confirmation (CPU heads; fine-tuning ≈ 1 GPU hour) → analysis (crossed
bootstraps) → `results/round8/SUMMARY.md`. Predictions and features are never committed; choices, checks, components,
verdicts and summaries are.

## Amendments

### Amendment 1 (2026-09-28, overnight run; before any development or confirmation run)
Made by the author after review of this registration, before any development or confirmation result existed. No
margin, cell, estimand or seed is changed. Implemented in `scripts/round8/{common,analyse,ft}.py` and
`tests/test_round8.py`; the smoke run and the unit tests were re-run before this amendment was committed.
1. **Multiplicity across the four primary candidates**: Holm (§7) is replaced by a **fixed-sequence test** in the order
   of `docs/ROUND8_DESIGN.md` §5c: mask_cmc → full_cmc → mask_bal(λ) → the paste candidate. Each H_c is tested at
   one-sided 0.025 (intersection–union within the candidate, D4 as a condition, unchanged). The sequence stops at the
   first H_c that is not rejected; later candidates are reported as "not tested in the sequence" with their unadjusted
   p_c. The same rule applies to the verdict without ovary. D5 and R keep Holm.
2. **The primary paste candidate is locrand_loc** (in-ROI and out-of-ROI presence each equalised across classes;
   §3, §3.1), because masking deletes out-of-ROI pastes (the prediction of §3.1). locrand becomes descriptive. D6
   (`locrand_ft`) uses the locrand_loc (location-matched) paste plan.
3. **Reporting**: each failed NI or SUP component is labelled **loss** (SUP: estimate < 0; NI: estimate < −margin/2)
   or **inconclusive** (otherwise); a failed D4 condition is a loss. "Wins where masking fails but costs elsewhere"
   (§9) names only the components labelled loss.
Reason: made after review of the registration, before any development or confirmation result existed.

## Results (added after the run; the text above is unchanged)
Amendment 1 at `47ab51c`, frozen development choices at `bdfb4e8`, results at `0d66e42` (`results/round8/SUMMARY.md`);
write-up in Stage 9.
- **No primary candidate dominates masking.** The fixed sequence stopped at mask_cmc (40 of 42 components). Decision:
  *wins where masking fails but costs elsewhere*: mask_cmc (loss: ISIC 2019 → 2020 all pairs −0.021 [−0.027, −0.016];
  thyroid all pairs inconclusive −0.002 [−0.011, +0.007]) and mask_bal (losses: all pairs thyroid −0.008, MSK −0.006,
  ISIC 2020 −0.011).
- mask_cmc − masking where masking fails: Trap A min(rev, corr) thyroid +0.330 [+0.311, +0.351], capsule +0.153
  [+0.131, +0.177], ovary (held out) +0.118 [+0.089, +0.147], ISIC hair +0.193 [+0.166, +0.221]; conflicting pairs
  thyroid +0.084, BCN +0.036, MSK +0.070, ISIC 2020 +0.078 [+0.069, +0.088]. On ISIC 2020 easy pairs −0.116.
- D5 (sweeps): mask_cmc and mask_bal dominate (12 of 12 each). D6 (fine-tuned paste): not supported (1 of 3).
  Replication R: Trap A gain in 5 of 5 cohort-encoder cells for mask_cmc and mask_bal.
- Checks: pasted-versus-real probe AUROC 0.904–0.990 (> 0.75 in every cohort); the paste plans could not balance the
  traps (too few recipients with a training-set donor). The paste results are not evidence about real artifacts.
