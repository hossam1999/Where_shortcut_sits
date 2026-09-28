# Morning summary — unattended run of 2026-09-28

**Push: succeeded.** Everything is on branch `stages-and-final-runs` (not `main`); no force-push, no history rewritten.

## Most important changes first

1. **External validation on new patients worked (A4).** ISIC 2020 met every pre-registered condition. It has 32,997
   images of 2,056 patients, and patient-disjoint folds were used. The smallest cell is 53 hair-free melanomas, against a gate of 50.
   Claim X1 is supported: crossover **+0.250 [+0.174, +0.327]**. Masking harms with hair on the lesion
   (−0.064 [−0.128, −0.002]) and helps with hair beside it (+0.186 [+0.118, +0.249]).
2. **A2 (regression-adjusted matched crossover) is supported in all three cohorts** (Holm p = 0.002 each; table below).
3. **The corrected (crossed) bootstrap changed 179 of 2,466 regenerated intervals.** 147 lost significance and 32 gained it; the median
   interval is 1.56× wider. Every claim that lost significance was rewritten (list below). No interval from the old
   estimator remains in the paper, the supplement or the six stage reports: the number audit checked 914 intervals and
   2,027 numbers and found 0 failures.
4. **Primary family: 11 of 16 supported after Holm (was 12).** P3 in the ovary cohort (erase > augment) lost support.
   All four location-crossover tests (P1) remain supported.
5. **The transplant is not evidence about real artifacts' harm.** A neutral paste of the same shape reproduces 52–92%
   of the transplant interaction. Only the artifact-minus-neutral difference (N3) is attributed to the artifact, and it
   is significant for hair (+0.049 [+0.026, +0.072]) and thyroid (+0.132 [+0.116, +0.148]) only. No harm figure is attributed to the
   real artifact anywhere.
6. **Two new findings from the regenerated remedy analyses:**
   - disease protection *helps* thyroid U-MtE (+0.059 [+0.019, +0.096]) instead of costing a little;
   - balanced U-MtE has a location interaction with MedSigLIP on capsule (−0.165 [−0.300, −0.025]).
   The paper's supplement was rewritten accordingly.
7. **The clinician image audit is prepared but not done** (0 of 240 images rated). Stage 6 has a placeholder that
   `audit/analyse_audit.py` fills.
8. **A disk-full incident at 07:37 stopped the run until you deleted the extracted ISIC 2020 JPEGs.** Thank you. The
   late lane and the ISIC 2020 job were restarted with a disk guard.

## What finished

| Part | Status |
|---|---|
| A1: regenerate all sweeps and U-MtE/remedy runs with the crossed bootstrap | done (`results/rerun_2026-09-28/`; old results untouched) |
| A1: `results/bootstrap_correction/sweeps_umte_old_vs_new.md` | done (2,466 rows) |
| A2: regression-adjusted crossover, matched traps | done, supported 3/3 |
| A3: clinician audit package | prepared (Part B) |
| A4: external validation | ISIC 2020 run, X1 supported; search in `stages/EXTERNAL_VALIDATION_SEARCH.md` |
| A5: paper updates | done (below) |
| Part B: `audit/` (licence check, sampler, blind review sheet, 240 images, analysis script), `figures/examples/` | done |
| Part C: six stage reports, `stages/README.md`, `stages/build_all.sh` | done; all build; number audit passes |
| MORNING_SUMMARY.md, push | done |

**Not finished:**
- The clinician audit itself needs you to recruit reviewers.
- The intervals of runs whose per-image predictions were never saved could not be regenerated and were **withdrawn**. Their point estimates are kept and labelled "not re-estimated". The affected runs:
  - chest-radiograph device traps and chest drains (RANZCR-CLiP is not on this machine and needs competition terms);
  - LaMa inpainting;
  - template-subspace erasure;
  - the archived 5-fold ovary ResNet-50 fine-tuning run.

## Page counts (six stage reports)

| Stage | Pages |
|---|---|
| 1 — problem, pilot, protocol, bootstrap correction | 9 |
| 2 — location law and mechanism | 11 |
| 3 — real artifacts across modalities | 9 |
| 4 — causal checks and the bootstrap correction | 9 |
| 5 — practice and theory | 8 |
| 6 — remedies, audit, external validation, paper | 8 |

The paper is `paper/main.pdf` (22 pages, review format) and `paper/supplement.pdf` (18 pages); the abstract has 249 words.
Rebuild everything with `bash stages/build_all.sh`.

## A2 — regression-adjusted crossover (per cohort)

| Cohort | Unadjusted | Adjusted (doubly robust) | Holm p | Post-matching max \|SMD\| pooled / per label |
|---|---|---|---|---|
| Hair (ISIC 2019) | +0.154 [+0.124, +0.186] | +0.156 [+0.127, +0.187] | 0.002 | 0.108 / 0.164 |
| Thyroid | +0.238 [+0.185, +0.291] | +0.238 [+0.184, +0.291] | 0.002 | 0.045 / 0.150 |
| Ovary | +0.163 [+0.058, +0.268] | +0.165 [+0.056, +0.273] | 0.002 | 0.079 / 0.140 |
| Capsule | descriptive only (matching infeasible: 19 pairs in one label) | | | |

The full balance table is in `results/rerun_2026-09-28/final_a2/balance_after_matching.csv` and in Stage 4.

## Claims changed by the bootstrap correction (all rewritten in the paper and stages)

**Lost significance:**
- **Controlled dermoscopy sweep:** masking at 25% overlap is no longer significantly harmful, so harm starts at 50%. The same holds with DINOv2 at 224 px.
- **Other sweeps:**
  - ovary sweep at 50% overlap: the harm is no longer significant;
  - MedSigLIP thyroid sweep at full overlap: the benefit is no longer significant;
  - capsule U-MtE+balancing − balancing at full overlap: no longer significant.
- **Occlusion control:** masking's small positive effects at 0/50/75/100% are no longer significant. The conclusion (occlusion does not harm) is unchanged.
- **Lesion-size strata:** large lesions at 100% (+0.063 [−0.015, +0.144]) and medium lesions at 50% are no longer significant. What remains is an ordering (harm largest in small lesions), not a monotone dose in every cell.
- **Primary family:** P3 in the ovary cohort (Holm p 0.022 → 0.130). The opposite-direction significance of P2 in capsule disappeared.
- **Natural test sets:**
  - masking's gain on easy thyroid pairs;
  - masking's gain on all MSK pairs;
  - U-MtE's gain over masking on hard thyroid pairs;
  - balancing's loss against masking on MSK.
- **Fine-tuned thyroid comparisons:** several (U-MtE vs masking in Trap B; masked+balanced vs balanced in Trap B).
- **MedSigLIP caliper protection** in the sweeps is no longer significantly negative (−0.041 [−0.095, +0.002]).

**Gained significance:**
- the clean-test cost of masking in the hair Trap A;
- fine-tuned ResNet-50 prediction consistency on thyroid (+0.042 [+0.011, +0.074]);
- two fine-tuned capsule and ViT-S contrasts.

**Point estimates:** 160 regenerated estimates moved by more than 0.03. The largest are in the capsule synthetic sweeps (up to 0.22): the capsule cohort was rebuilt on this machine, and the seeded group shuffle re-dealt the split, so only 32 of the 71 archived test frames recur. The location interaction keeps its sign and significance. P2 thyroid moved from 0.217 to 0.283 because its source run changed from `protect_generic` to `universal`. The full list is in `results/bootstrap_correction/sweeps_umte_old_vs_new.md`.

**Other corrections found during the run:**
- Stage 1's draft cited the wrong commit for the replication spec. It is `2699d51`; hashes are now read from git.
- The theory's held-out error is now 0.052 vs 0.129 for the symmetry heuristic, over 136 cells. It was 0.039 vs 0.108 over 148 cells, a count that included the chest-radiograph cells, which could not be regenerated.
- Zero-shot scores were regenerated (prompt embeddings recomputed). The DermLIP crossover moved from 0.051 to 0.040, and the MedSigLIP capsule crossover is now borderline (+0.045 [+0.001, +0.090]).

## A5 — paper changes

- Every interval was replaced by the regenerated, crossed one. Automated replacements were checked against their source; three coincidental matches were fixed by hand.
- The thyroid subgroup (n = 78, −0.372 [−0.499, −0.236]) is labelled **"a pre-registered test inside a post hoc subgroup (exploratory confirmation)"** and kept out of the 16-test family.
- New "Why this is not obvious" paragraph in the discussion. It covers four points:
  - |Δp| roughly doubles;
  - the operating-point loss on the patient-disjoint thyroid split;
  - the theory's held-out error against the heuristic;
  - the neutral paste, leading to the decision guide.
- Added: the A2 result, the ISIC 2020 external validation (results and abstract), updated CLAIM items 29 and 33, and updated limitations.
- The review-3 supplement table no longer prints the original intervals (width ratio and "verdict changed" instead).

## External validation — decision and reason

- **Run:** ISIC 2020. It had a direct S3 download, a CC BY-NC licence and `patient_id`, and masks could be produced by frozen models. Hair masks come from a segmenter trained only on ISIC 2019 masks and frozen before scoring. The gate was met.
- **Not run; candidates for you:**
  - **ThyUS2Path:** no nodule masks, and the caliper detector fires on its interface. It needs a transferred segmentation and a re-validated detector.
  - **TN-SCUI 2020:** its terms forbid use outside the challenge.
  - **Stanford AIMI:** requires a data-use agreement, which I did not accept.
  - **DDTI:** 134 images with TI-RADS labels; too small.

## Licence decision per dataset (`audit/LICENCES.md`)

| Dataset | Licence | Images committed? |
|---|---|---|
| ISIC 2018/2019 (HAM10000, BCN, MSK), ISIC 2020 | CC BY-NC 4.0 | yes (raw images; ROI outlines) |
| HAM10000 lesion masks | CC BY-NC 4.0 | yes (as outlines) |
| ISIC 2019 hair/ruler masks (Wegley et al.) | no licence stated | **no**: overlays kept in git-ignored `figures_local/`, `audit_local/` |
| TN3K / TNCD (thyroid) | data licence not stated (code MIT only) | **no**: all thyroid images local |
| MMOTU | CC BY 4.0 | yes |
| SEE-AI and expert contamination masks | CC BY 4.0 | yes |
| NIH ChestX-ray14 | not used in figures | — |

**For you to decide:** `report/figures/data_thyroid.png` and `report/figures/data_isic.png` were committed before this check and conflict with the rule.

## Audit results

- **Image audit:** prepared, not performed.
  - 240 images: 4 cohorts × 3 cells × 20, balanced by label, seed 20260928, neutral IDs.
  - The key is kept out of git; its SHA-256 is in `audit/KEY_SHA256.txt`, and `make_audit_sample.py` regenerates it.
  - To run it: `python audit/analyse_audit.py <filled sheet> [second sheet]`, then `bash stages/build_all.sh`.
- **Number audit** (`scripts/verify/audit_numbers_final.py --docs all`): 914 intervals and 2,027 three-decimal numbers checked, 0 failures.
  - I tightened it three times during the run. An interval must now match an explicit (lo, hi) pair of a regenerated result file, together with its estimate.
  - Columns holding archived or pilot values are ignored.
  - Only files actually included in the paper are checked.

## Branch and commits (`stages-and-final-runs`)

| Commit | Content |
|---|---|
| `c45d482` | pre-registration of the final runs (pushed before running) |
| `806c25f` | Part A: regenerated runs, old-vs-new, A2, A4 |
| `06ba438` | Part B: audit package, example images |
| `afb3b2f` | Part A addendum: fine-tuned crossovers |
| `0758aee` | Stage 1 and shared stage infrastructure |
| `a0e45c6` | Stage 2 |
| `074e22e` | Stage 3 |
| `f2cbf6c` | Stage 4 |
| `c72b234` | Stage 5 |
| `55c6417` | A5: paper with corrected intervals; Stage 2 counterfactual correction |
| `e75240f` | Stage 6, `build_all.sh`, refreshed comparison |
| (this file) | MORNING_SUMMARY.md |

Commit times come from this machine and are not independent proof of order. The GitHub push times are the external reference.

## Failed steps and errors (full log: `stages/DECISIONS_LOG.md`)

- **Disk full** (07:37, 317 GB overlay) while writing the ISIC 2020 mask arrays. Every write failed until the extracted JPEGs were deleted. My own attempt to delete them was refused as irreversible and left to you. The late lane crashed at `summary_ledger`, and `umte_ablation` failed on the same error. Both were re-run successfully.
- **5 ISIC copy tasks** (`cp_isic_*`) failed twice with `FileNotFoundError: metrics_per_seed.csv`. They were re-queued with the file copied and succeeded.
- **`t_thy_med_main`, `ft_cap_r50`, `ft_thy_vit`:** CUDA out of memory with three lanes on one GPU. All completed later.
- **`review2_summary`:**
  - It first failed on missing inputs (`PRIMARY_CLAIMS.csv`, `adhoc_bootstraps.json`), which were then generated.
  - Its fine-tuning section then failed with `KeyError: 'finetune|ovary_resnet50|…'`. The archived ovary run has no predictions, so the section now skips it.
- **Zero-shot:** the prompt-embedding files were absent and the SigLIP tokenizer needed `sentencepiece` (installed). Regenerated.
- **Mechanism figure:** the archived combined dermoscopy run was absent; the figure now combines the regenerated main and proposed sweeps.
- **Stage 2 draft error:** it claimed the counterfactual existed for dermoscopy only. That was wrong once the late lane re-ran the analysis; the counterfactual holds in 3 of 5 sweeps. Corrected.

## What needs you

1. Recruit two reviewers for the image audit (about 2.5 h each). The sheet is `audit/review_sheet.csv`.
2. Decide on the two figures that conflict with the licence rule.
3. Confirm the draft submission statements in Stage 6 (ethics waiver, AI-assistance wording).
4. Answer the questions for the supervisor at the end of each stage report.
5. Merge `stages-and-final-runs` into `main` when you are satisfied. I did not touch `main`.
