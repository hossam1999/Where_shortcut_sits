# Round 8 — design: is there a remedy that dominates ROI masking? (Phase 1, no new model, no test scoring)

Status: **Phase 1 design, for the author's approval.** Nothing here is a registered hypothesis yet; the registration
will be `docs/PREREGISTRATION_ROUND8.md` (Phase 2). No model was trained and no test set was scored for this document.
Every number below is generated from a committed file by `scripts/round8/design_tables.py` (tables marked
*generated*) or copied from a paper whose table I opened (literature section); nothing is typed by hand.

Files of this phase (all regenerable with `bash scripts/round8/phase1.sh`):

| file | what |
|---|---|
| `results/round8/scoreboard_existing.csv` | 11,135 rows: every existing arm × setting × cohort × encoder × trap/subset × environment; AUROC, arm − mask and arm − ERM, with crossed seed × image 95% intervals (10,000 replicates) and the source file |
| `results/round8/scoreboard_repro_check.csv` | the recomputed point estimates against the 1,189 stored ones (max \|Δ\| = 0.0005) |
| `results/round8/criterion_existing*.csv` | the proposed criterion (§6) applied to every existing DINOv2 arm |
| `results/round8/precision_by_cell.csv` | interval half-widths per cell (what a margin can demand) |
| `results/round8/pair_decomposition.csv` | all-pairs AUROC split into easy, hard and same-artifact pairs (§4) |
| `scripts/round8/{common,scoreboard,criterion_check,pair_decomposition,design_tables}.py` | code; every script has `--smoke` and is resumable |

## 1. Context (what the earlier rounds established)
- **Location law** (Stages 2–4, 7): masking removes an artifact outside the ROI and concentrates reliance on one inside
  it; the crossover [mask − ERM]_out − [mask − ERM]_in is positive in every modality, for every masking implementation,
  graded in the real overlap, and survives on uncontradicted labels (Stage 8).
- **Theory** (`docs/THEORY.md`): AUROC_rev = Φ((S − Ã)/√(2(S + Ã))). In-ROI masking harms iff it removes disease
  context (S_m < S) while Ã stays; balancing sets Ã = 0 for the artifact *and its correlates*; erasure sets Ã = 0 but
  scales S by (1 − ρ²).
- **Remedies** (Stage 6): U-MtE fixes pixel-carried in-ROI shortcuts (calipers) without annotation, fails for
  pathology-like artifacts without protection and for correlate-carried ones (hair, drains); with artifact labels,
  U-MtE + balancing or mask + DFR is most robust; **no remedy improved on masking uniformly on unaltered data**;
  selectors U13/U14 beat masking in every trap but were never uniformly best.
- **Unaltered data** (Stages 5, 7, 8): masking harms shortcut-conflicting pairs on the thyroid split, BCN, MSK and the
  ISIC 2019 → 2020 external set while leaving all-pairs AUROC nearly unchanged; a fine-tuned thyroid ConvNeXt-T shows
  every registered harm.

## 2. Scoreboard: the numbers to beat
Built from saved per-image predictions only (the stored summary files do not carry arm − mask intervals). One crossed
bootstrap per source run with replicate weights shared across arms, environments and subsets, so every arm − mask
contrast and every min(reversed, correlated) contrast is a function of the same replicates. The recomputed point
estimates reproduce all 1,189 stored ones (bootstrap_vs_erm.csv, natural_boot.json) to within 0.0005.

Coverage: S1 controlled sweeps at r = 0 and r = 1 (14 runs, 5 modalities, 5 encoders); S2 real traps (30 runs:
thyroid, capsule, ovary, ISIC hair; DINOv2 S/B/L, MedSigLIP, ConvNeXt, DermLIP; robustness variants — matched,
reproduction, embedding groups, label sensitivity, ablation — are left out); S3 natural test sets (thyroid split,
BCN, HAM, MSK hold-outs, capsule; all, hard and easy pairs); S4 ISIC 2019 → 2020 under the calibrated duplicate rule
(all/hard/easy) and the ISIC 2020 traps; S5 the fine-tuned thyroid ConvNeXt-T (natural split and traps). Operating
points exist for masking and ERM only (`final_op`, round 7) and are copied as they are. Arm names: `umte*` = universal
mask-then-erase (the stored name `mte` in universal/natural runs), `mte_tpl*` = template MtE (the stored `mte` in
`*_main` runs).

**Criterion cells, DINOv2-B/14 (generated).** Masking's value and the best existing arm per cell:

<!-- INSERT:targets -->
| component | cohort | type | mask AUROC [95% CI] | best existing arm | best − mask [95% CI] |
|---|---|---|---|---|---|
| natural all pairs | thyroid | NI | 0.731 [0.690, 0.770] | umte | +0.004 [-0.011, +0.020] |
| natural all pairs | capsule | NI | 0.967 [0.955, 0.977] | mask_balanced | -0.003 [-0.006, +0.000] |
| natural all pairs | isic_BCN | NI | 0.761 [0.749, 0.772] | erm | +0.025 [+0.016, +0.035] |
| natural all pairs | isic_HAM | NI | 0.782 [0.767, 0.797] | umte_balanced | +0.018 [+0.009, +0.028] |
| natural all pairs | isic_MSK | NI | 0.785 [0.764, 0.806] | umte_protect | -0.003 [-0.012, +0.006] |
| external all pairs | isic2019_to_2020 | NI | 0.807 [0.788, 0.825] | umte_protect | -0.006 [-0.015, +0.003] |
| hard pairs | thyroid | SUP | 0.583 [0.514, 0.651] | balanced | +0.136 [+0.063, +0.207] |
| hard pairs | isic_BCN | SUP | 0.690 [0.672, 0.707] | balanced | +0.043 [+0.026, +0.059] |
| hard pairs | isic_MSK | SUP | 0.631 [0.584, 0.679] | balanced | +0.100 [+0.057, +0.143] |
| hard pairs | isic2019_to_2020 | SUP | 0.735 [0.707, 0.761] | balanced | +0.072 [+0.048, +0.096] |
| Trap A reversed | thyroid | SUP | 0.398 [0.371, 0.425] | mask_dfr | +0.428 [+0.398, +0.457] |
| Trap B reversed | thyroid | NI | 0.752 [0.720, 0.785] | mask_balanced | +0.035 [+0.019, +0.049] |
| Trap A min(rev,corr) | thyroid | NI | 0.398 [0.371, 0.425] | mask_balanced | +0.392 [+0.360, +0.416] |
| Trap B min(rev,corr) | thyroid | NI | 0.752 [0.719, 0.782] | umte_balanced | +0.013 [-0.029, +0.038] |
| clean (Trap A models) | thyroid | NI | 0.689 [0.667, 0.711] | umte_balanced | +0.109 [+0.089, +0.129] |
| clean (Trap B models) | thyroid | NI | 0.767 [0.731, 0.801] | mask_balanced | +0.006 [-0.009, +0.020] |
| Trap A reversed | capsule | SUP | 0.665 [0.629, 0.701] | mask_dfr | +0.228 [+0.198, +0.259] |
| Trap B reversed | capsule | NI | 0.890 [0.865, 0.913] | mask_balanced | +0.030 [+0.022, +0.038] |
| Trap A min(rev,corr) | capsule | NI | 0.665 [0.629, 0.701] | mask_dfr | +0.226 [+0.181, +0.254] |
| Trap B min(rev,corr) | capsule | NI | 0.890 [0.865, 0.913] | mask_balanced | +0.030 [+0.022, +0.038] |
| clean (Trap A models) | capsule | NI | 0.860 [0.837, 0.880] | mask_balanced | +0.053 [+0.042, +0.065] |
| clean (Trap B models) | capsule | NI | 0.932 [0.917, 0.945] | mask_balanced | +0.008 [+0.003, +0.012] |
| Trap A reversed | ovary | SUP | 0.545 [0.494, 0.598] | mask_dfr | +0.206 [+0.163, +0.251] |
| Trap B reversed | ovary | NI | 0.725 [0.667, 0.781] | umte_protect_balanced | +0.051 [+0.018, +0.084] |
| Trap A min(rev,corr) | ovary | NI | 0.545 [0.494, 0.598] | mask_dfr | +0.206 [+0.142, +0.247] |
| Trap B min(rev,corr) | ovary | NI | 0.725 [0.664, 0.770] | umte_balanced | +0.029 [-0.016, +0.067] |
| clean (Trap A models) | ovary | NI | 0.690 [0.656, 0.723] | umte_balanced | +0.084 [+0.047, +0.118] |
| clean (Trap B models) | ovary | NI | 0.733 [0.682, 0.779] | umte | +0.032 [-0.001, +0.068] |
| Trap A reversed | isic_hair | SUP | 0.460 [0.417, 0.505] | balanced | +0.262 [+0.220, +0.305] |
| Trap B reversed | isic_hair | NI | 0.608 [0.567, 0.648] | umte_balanced | +0.142 [+0.112, +0.173] |
| Trap A min(rev,corr) | isic_hair | NI | 0.460 [0.417, 0.505] | balanced | +0.262 [+0.220, +0.305] |
| Trap B min(rev,corr) | isic_hair | NI | 0.608 [0.567, 0.648] | umte_balanced | +0.126 [+0.078, +0.164] |
| clean (Trap A models) | isic_hair | NI | 0.712 [0.682, 0.741] | balanced | +0.073 [+0.042, +0.103] |
| clean (Trap B models) | isic_hair | NI | 0.701 [0.673, 0.730] | balanced | +0.051 [+0.020, +0.082] |
| fine-tuned all pairs | thyroid | NI | 0.745 [0.707, 0.782] | erm | +0.024 [-0.016, +0.063] |
| fine-tuned hard pairs | thyroid | SUP | 0.621 [0.548, 0.692] | erm | +0.102 [+0.026, +0.179] |
<!-- /INSERT:targets -->

**Controlled sweeps, reversed test (generated).** At r = 1 masking fails and many arms repair it; at r = 0 masking is
already right and no arm is reliably better:

<!-- INSERT:sweeps -->
| sweep | encoder | overlap | mask reversed AUROC | best arm | best − mask |
|---|---|---|---|---|---|
| capsule | DINOv2-B/14@518 | r=0 | 0.938 [0.892, 0.975] | umte | +0.013 [-0.002, +0.031] |
| capsule | DINOv2-B/14@518 | r=1 | 0.685 [0.570, 0.791] | umte_balanced | +0.211 [+0.132, +0.296] |
| capsule | MedSigLIP@448 | r=0 | 0.960 [0.929, 0.984] | mte_tpl | -0.003 [-0.030, +0.021] |
| capsule | MedSigLIP@448 | r=1 | 0.518 [0.398, 0.635] | leace | +0.391 [+0.279, +0.504] |
| isic2018 | DINOv2-B/14@224 | r=0 | 0.798 [0.737, 0.855] | leace | +0.032 [-0.023, +0.088] |
| isic2018 | DINOv2-B/14@224 | r=1 | 0.295 [0.224, 0.371] | leace | +0.521 [+0.447, +0.592] |
| isic2018 | DINOv2-B/14@518 | r=0 | 0.849 [0.797, 0.894] | leace | -0.017 [-0.067, +0.033] |
| isic2018 | DINOv2-B/14@518 | r=1 | 0.386 [0.315, 0.461] | leace | +0.423 [+0.353, +0.491] |
| isic2018 | DermLIP@224 | r=0 | 0.858 [0.809, 0.903] | leace | +0.088 [+0.047, +0.133] |
| isic2018 | DermLIP@224 | r=1 | 0.311 [0.242, 0.386] | leace | +0.627 [+0.555, +0.694] |
| nih_ptx | DINOv2-B/14@518 | r=0 | 0.883 [0.864, 0.902] | umte | +0.000 [-0.006, +0.007] |
| nih_ptx | DINOv2-B/14@518 | r=1 | 0.327 [0.298, 0.356] | mte_tpl | +0.553 [+0.528, +0.579] |
| nih_ptx | RAD-DINO@518 | r=0 | 0.910 [0.892, 0.927] | leace | +0.005 [-0.002, +0.013] |
| nih_ptx | RAD-DINO@518 | r=1 | 0.725 [0.691, 0.758] | balanced | +0.199 [+0.174, +0.225] |
| ovary | DINOv2-B/14@518 | r=0 | 0.697 [0.559, 0.825] | umte | +0.060 [+0.011, +0.118] |
| ovary | DINOv2-B/14@518 | r=1 | 0.438 [0.297, 0.589] | balanced | +0.283 [+0.122, +0.437] |
| thyroid | DINOv2-B/14@518 | r=0 | 0.819 [0.759, 0.876] | mte_tpl | +0.006 [-0.016, +0.029] |
| thyroid | DINOv2-B/14@518 | r=1 | 0.294 [0.220, 0.373] | mte_tpl_balanced | +0.501 [+0.424, +0.577] |
| thyroid | MedSigLIP@448 | r=0 | 0.834 [0.776, 0.885] | mte_tpl | -0.001 [-0.023, +0.022] |
| thyroid | MedSigLIP@448 | r=1 | 0.305 [0.224, 0.388] | leace | +0.473 [+0.384, +0.562] |
<!-- /INSERT:sweeps -->

The same table for the other encoders is in `results/round8/design_tables.md` (section `other_encoders`).

## 3. Literature: methods, requirements and published numbers
Numbers are copied from the paper's own table (location given). They come from other datasets and tasks and are
**context, not targets**; the targets are the scoreboard cells above. WGA = worst-group accuracy.

| method | needs | closest benchmark: published number (source) | id |
|---|---|---|---|
| Trap sets, background normalisation, LNTL (Bissoto et al. 2020) | lesion masks; artifact labels (LNTL) | ISIC trap test AUC (%): Inception-v4 unchanged 52.6 ± 1.8, LNTL (ResNet-152) 54.5 ± 3.0; ResNet-18 unchanged 44.7 ± 1.5, normalised 62.4 ± 3.3 (Table 3). Lesion-only inputs lower AUC: ISIC traditional 86.3 ± 1.6 vs skin-only 77.3 ± 1.6, bbox 77.1 ± 1.8 (Table 1) | arXiv 2004.11457 (CVPRW 2020) |
| Artifact-based GroupDRO + NoiseCrop (Bissoto et al. 2022) | image-level artifact labels (7 types); lesion masks; retraining | strong trap test (bias 1) ROC AUC: ERM 0.58, RSC 0.59, GroupDRO 0.68, full pipeline 0.74 (Table 3) | arXiv 2208.09756 |
| CDEP (Rieger et al. 2020) | pixel masks of the patches; retraining with an explanation penalty | ISIC patches, AUC no-patch / all: vanilla 0.87 / 0.93, RRR 0.75 / 0.86, CDEP 0.89 / 0.94 (Table 1) | arXiv 1909.13584 (ICML 2020) |
| Colour-patch removal by inpainting, then retraining (Nauta et al. 2022) | patch masks; inpainting; retraining | full text blocked from this machine (MDPI, PMC, EuropePMC, university repository); **no number copied** — needs a manual check | doi 10.3390/diagnostics12010040 |
| Drain shortcut in chest X-ray (Jiménez-Sánchez et al. 2023) | drain labels (evaluation) | CheXpert→CheXpert, 24k: AUC 81.0 ± 0.6 baseline, 76.7 ± 0.7 without drains, 84.7 ± 0.8 with drains (Table 1) | arXiv 2211.04279 |
| Unlearning heads LNTL / TABE / CLGR (Bevan & Atapour-Abarghouei 2022) | artifact labels; retraining | skewed ISIC, Heid marked AUC: baseline 0.902 ± 0.013, LNTL 0.957 ± 0.023, CLGR 0.949 ± 0.022; Heid ruler: baseline 0.831 ± 0.022, CLGR 0.958 ± 0.018 (Table 1) | arXiv 2109.09818 |
| GroupDRO (Sagawa et al. 2020) | group labels in training and validation | WGA Waterbirds 91.4 (1.1), CelebA 88.9 (2.3) (Table 3, strong ℓ2) | arXiv 1911.08731 |
| JTT (Liu et al. 2021) | none in training; group labels on validation for tuning | WGA Waterbirds 86.7, CelebA 81.1 (ERM 72.6 / 47.2; LfF 78.0 / 77.2) (Table 1) | arXiv 2107.09044 |
| DFR (Kirichenko et al. 2023) | group labels on a held-out set | WGA Waterbirds 92.9 ± 0.2, CelebA 88.3 ± 1.1 (DFR^Val_Tr, Table 2) | arXiv 2204.02937 |
| AFR (Qiu et al. 2023) | no group labels in training; group-labelled validation for γ, λ | WGA Waterbirds 90.4 ± 1.1, CelebA 82.0 ± 0.5, chest X-ray (NIH/CheXpert hospital markings) 56.0 ± 3.4 vs ERM 4.4, JTT 52.3, DFR 59.8 ± 1.8 (Table 1) | arXiv 2306.11074 |
| LfF (Nam et al. 2020) | none (biased-model reweighting) | WGA Waterbirds 78.0, CelebA 77.2 (as reported in JTT Table 1) | arXiv 2007.02561 |
| Group/class balancing SUBG, RWG, RWY (Idrissi et al. 2022) | group labels (SUBG/RWG) or none (RWY) + labelled validation | WGA Waterbirds: SUBG 89.1 ± 1.1, gDRO 87.1 ± 3.4, JTT 85.6 ± 0.2, ERM 85.5 ± 1.0 (Table 2) | arXiv 2110.14503 |
| SELF / class-balanced last-layer retraining (LaBonte et al. 2023) | class labels; few held-out labels | WGA Waterbirds: ES-disagreement SELF 93.0 ± 0.3, CB last layer 92.6 ± 0.8, DFR 92.4 ± 0.9 (Table 1) | arXiv 2309.08534 |
| MaskTune (Asgari et al. 2022) | none (masks salient features, fine-tunes) | Waterbirds WGA: ERM 80.8 ± 1.3, MaskTune 86.4 ± 1.9, GroupDRO 89.3 ± 3.1 (Table 2) | arXiv 2210.00055 |
| EIIL (Creager et al. 2021) | none (infers environments from a reference model) | CMNIST test accuracy: ERM 13.8 ± 0.6, IRM 65.5 ± 2.3, EIIL 68.4 ± 2.7 (Table 2) | arXiv 2010.07249 |
| Subpopulation-shift benchmark (Yang et al. 2023, "Change is Hard") | — | attributes unknown in training *and* validation, WGA: CheXpert ERM 41.7 ± 3.4, GroupDRO 74.7 ± 0.3, JTT 60.4 ± 4.8, DFR 75.8 ± 0.3; Waterbirds ERM 69.1 ± 4.7, JTT 71.2 ± 0.5, DFR 89.0 ± 0.2 (Table 3). Selecting by worst-class validation accuracy loses 1.8 WGA points on average vs the oracle, by overall accuracy 15.0 (Table 5) | arXiv 2302.12254 |
| MEDFAIR (Zong et al. 2023) | sensitive-attribute labels | finding (Sec. 4.3): no method outperforms ERM with statistical significance; model selection by worst-case AUC improves max–min fairness significantly (Sec. 4.2). No table number copied | arXiv 2210.01725 (ICLR 2023) |
| Whac-A-Mole (Li et al. 2023) | shortcut labels (some methods) | finding (Table 5): methods that mitigate the labelled shortcut amplify the unlabelled one on UrbanCars | arXiv 2212.04825 |
| LEACE (Belrose et al. 2023) | concept labels | closed-form linear erasure; no medical benchmark | arXiv 2306.03819 |
| SPLINCE (Holstege et al. 2025) | concept and task labels | text benchmarks; the authors state that SPLINCE "performs relatively worse" on the vision datasets (CelebA, Waterbirds; text next to their Fig. 4 and App. B.5); no number copied | arXiv 2506.10703 |
| Mask of truth (Sourget et al. 2024) | lung / region masks | PadChest effusion models trained without lungs (bbox) still reach AUC 0.85–0.93 (text, Sec. 4) | arXiv 2412.04030 |
| Counterfactual invariance, anti-causal (Veitch et al. 2021) | auxiliary (artifact) labels | conditional MMD penalty MMD(P(f(X)\|Z=0,Y=y), P(f(X)\|Z=1,Y=y)) (Eq. 3.2) | arXiv 2106.00545 |
| Shortcut removal with auxiliary labels (Makar et al. 2022) | auxiliary labels | weighting to the unconfounded distribution plus an MMD penalty (Sec. 3) | arXiv 2105.06422 |
| Counterfactual logit pairing (Garg et al. 2019); Right for the right reasons (Ross et al. 2017) | counterfactual pairs; input masks | penalties on prediction change / input gradients | arXiv 1809.10610; 1703.03717 |
| Copy-paste augmentation (Ghiasi et al. 2021) | instance masks | segmentation, not debiasing | arXiv 2012.07177 |
| Confound regression / ComBat harmonisation (Johnson et al. 2007; Fortin et al. 2018; Snoek et al. 2019) | batch / confound labels | location–scale adjustment preserving covariates of interest | doi 10.1093/biostatistics/kxj037; 10.1016/j.neuroimage.2017.11.024; 10.1016/j.neuroimage.2018.09.074 |

Three lessons for this project. (i) Every group-robust method that works without group labels in training still
tunes on a group-labelled validation set; without one (Yang et al., Table 3) JTT falls to ERM level on Waterbirds and
selection by overall accuracy costs about 15 WGA points (Table 5). Our selection rules must say which labels they use.
(ii) Medical benchmarks report that no fairness method beats ERM significantly (MEDFAIR) — consistent with our Stage 6
finding on unaltered data. (iii) Fixing one shortcut can amplify another (Whac-A-Mole): the criterion must check every
cell, not only the fixed one.

## 4. Feasibility from the theory, checked on the existing predictions
**An exact decomposition.** For any scorer, all-pairs AUROC = Σ_k π_k AUROC_k over the four pair types by artifact
status: *easy* (the pair the training association orders correctly), *hard* (the conflicting pair), and the two
*same-artifact* types (both carry / both lack the artifact). π_k is the share of positive–negative pairs of each type.
So for a remedy R, Δ_all = π_easy Δ_easy + π_hard Δ_hard + π_same Δ_same exactly (verified: max error 6·10⁻¹⁷).
Removing a shortcut gains on hard pairs, loses on easy pairs, and should leave same-artifact pairs alone: inside a
same-artifact pair the artifact term of a linear score cancels. Same-artifact pairs change only when the remedy changes
the *disease* part of the score.

**Theory's prediction for the unavoidable part.** For a linear head with artifact score shift δ and disease margin Δ,
easy pairs have AUROC Φ((Δ + δ)/σ) and hard pairs Φ((Δ − δ)/σ). Removing the shortcut (δ → 0) gains
Φ(Δ/σ) − Φ((Δ − δ)/σ) per hard pair and loses Φ((Δ + δ)/σ) − Φ(Δ/σ) per easy pair. Φ is concave above 0, so the gain
per pair is at least the loss. The unavoidable all-pairs change is therefore about π_hard·g − π_easy·l, and it is
negative only in proportion to π_easy − π_hard (≈ q1 − q0, the test association). On the traps' correlated tests
(0.9 / 0.1) this term is large, and every arm that removes the shortcut loses there (generated):

<!-- INSERT:corr_cost -->
| cohort | cell | arm | arm − mask [95% CI] |
|---|---|---|---|
| capsule | trapA | balanced | -0.056 [-0.085, -0.030] |
| capsule | trapA | mask_balanced | -0.017 [-0.026, -0.009] |
| capsule | trapA | umte_balanced | -0.032 [-0.051, -0.017] |
| capsule | trapB | balanced | -0.083 [-0.111, -0.058] |
| capsule | trapB | mask_balanced | -0.008 [-0.013, -0.004] |
| capsule | trapB | umte_balanced | -0.025 [-0.040, -0.013] |
| isic_hair | trapA | balanced | -0.068 [-0.089, -0.049] |
| isic_hair | trapA | mask_balanced | -0.062 [-0.088, -0.038] |
| isic_hair | trapA | umte_balanced | -0.057 [-0.074, -0.039] |
| isic_hair | trapB | balanced | +0.005 [-0.022, +0.032] |
| isic_hair | trapB | mask_balanced | -0.058 [-0.091, -0.025] |
| isic_hair | trapB | umte_balanced | -0.061 [-0.092, -0.032] |
| ovary | trapA | balanced | -0.103 [-0.155, -0.046] |
| ovary | trapA | mask_balanced | -0.020 [-0.053, +0.021] |
| ovary | trapA | umte_balanced | -0.020 [-0.063, +0.027] |
| ovary | trapB | balanced | -0.042 [-0.118, +0.032] |
| ovary | trapB | mask_balanced | -0.021 [-0.049, +0.009] |
| ovary | trapB | umte_balanced | +0.002 [-0.033, +0.039] |
| thyroid | trapA | balanced | -0.151 [-0.186, -0.118] |
| thyroid | trapA | mask_balanced | -0.111 [-0.132, -0.090] |
| thyroid | trapA | umte_balanced | -0.104 [-0.126, -0.083] |
| thyroid | trapB | balanced | -0.125 [-0.172, -0.077] |
| thyroid | trapB | mask_balanced | -0.024 [-0.043, -0.005] |
| thyroid | trapB | umte_balanced | -0.020 [-0.052, +0.008] |
<!-- /INSERT:corr_cost -->

On the natural sets the association is weak (π_hard = 0.055–0.187, π_easy = 0.101–0.290), so the unavoidable term is
small. **The observed losses of balancing come mostly from the same-artifact pairs**, i.e. from disease signal, not from
removing the shortcut (generated; parts sum exactly to Δ all pairs):

<!-- INSERT:decomposition -->
| cohort | arm | π easy | π hard | easy part | hard part | same-artifact part | Δ all pairs |
|---|---|---|---|---|---|---|---|
| capsule | balanced | 0.208 | 0.087 | -0.019 | -0.001 | -0.066 | -0.086 |
| capsule | mask_balanced | 0.208 | 0.087 | -0.002 | +0.001 | -0.003 | -0.003 |
| capsule | umte | 0.208 | 0.087 | -0.000 | +0.000 | -0.005 | -0.005 |
| capsule | umte_balanced | 0.208 | 0.087 | -0.002 | +0.001 | -0.007 | -0.008 |
| isic_BCN | balanced | 0.271 | 0.154 | +0.001 | +0.007 | +0.014 | +0.022 |
| isic_BCN | mask_balanced | 0.271 | 0.154 | -0.005 | +0.005 | +0.003 | +0.003 |
| isic_BCN | umte | 0.271 | 0.154 | +0.002 | +0.001 | +0.004 | +0.007 |
| isic_BCN | umte_balanced | 0.271 | 0.154 | -0.003 | +0.006 | +0.008 | +0.010 |
| isic_HAM | balanced | 0.224 | 0.134 | -0.006 | -0.002 | -0.023 | -0.032 |
| isic_HAM | mask_balanced | 0.224 | 0.134 | -0.004 | +0.006 | +0.004 | +0.005 |
| isic_HAM | umte | 0.224 | 0.134 | +0.002 | +0.003 | +0.009 | +0.013 |
| isic_HAM | umte_balanced | 0.224 | 0.134 | -0.003 | +0.008 | +0.013 | +0.018 |
| isic_MSK | balanced | 0.101 | 0.066 | -0.013 | +0.006 | -0.012 | -0.018 |
| isic_MSK | mask_balanced | 0.101 | 0.066 | -0.003 | +0.003 | -0.006 | -0.006 |
| isic_MSK | umte | 0.101 | 0.066 | -0.000 | -0.001 | -0.007 | -0.008 |
| isic_MSK | umte_balanced | 0.101 | 0.066 | -0.003 | +0.003 | -0.015 | -0.016 |
| thyroid | balanced | 0.290 | 0.187 | -0.032 | +0.025 | -0.011 | -0.017 |
| thyroid | mask_balanced | 0.290 | 0.187 | -0.014 | +0.010 | -0.005 | -0.010 |
| thyroid | umte | 0.290 | 0.187 | -0.002 | +0.005 | +0.001 | +0.004 |
| thyroid | umte_balanced | 0.290 | 0.187 | -0.013 | +0.013 | -0.005 | -0.005 |
| isic2019_to_2020 | balanced | 0.129 | 0.055 | -0.013 | +0.004 | -0.018 | -0.027 |
| isic2019_to_2020 | mask_balanced | 0.129 | 0.055 | -0.008 | +0.002 | -0.012 | -0.018 |
| isic2019_to_2020 | umte | 0.129 | 0.055 | -0.002 | +0.000 | -0.011 | -0.013 |
| isic2019_to_2020 | umte_balanced | 0.129 | 0.055 | -0.011 | +0.003 | -0.025 | -0.033 |
| thyroid | balanced | 0.290 | 0.187 | -0.012 | +0.017 | -0.003 | +0.002 |
| thyroid | mask_balanced | 0.290 | 0.187 | -0.004 | -0.003 | -0.009 | -0.016 |
<!-- /INSERT:decomposition -->

Reading: on ISIC 2019 → 2020, mask_balanced loses −0.018 on all pairs. The same-artifact pairs contribute −0.012
(disease signal lost by reweighting), the easy pairs −0.008, and the hard pairs only +0.002 (5.5% of pairs). The
reweighting in `mask_balanced` (equal total weight per artifact × label group) lowers the effective sample size and
fits the head to a reweighted population. That costs disease evidence among the 81% of pairs where neither image has
in-lesion hair.

**Verdict by family** (on non-inferiority for natural all pairs together with gains on Trap A and hard pairs):

| family | can it be non-inferior on natural all pairs **and** gain on Trap A / hard pairs? |
|---|---|
| masking alone | defines the reference; fails where masking fails (by construction) |
| reweighting / balancing (balanced, mask_balanced, U-MtE_balanced, DFR) | gains everywhere the shortcut conflicts, but pays a same-artifact (disease) cost that grows with the reweighting strength and with small minority groups. Observed to fail NI in thyroid, MSK and ISIC 2020. **Feasible only in a softened form** (partial reweighting) that trades gain for cost; whether a single strength satisfies every cell is an empirical question |
| pixel/overlay erasure without labels (U-MtE, protected U-MtE) | nearly free on natural sets (disease subspace mostly spared) but **cannot** gain where the shortcut is correlate-carried (hair: hard pairs BCN, MSK, ISIC 2020) or pathology-like (capsule Trap A) — theory §6 and the SLAS oracle ceiling. Fails the hard-pair superiority cells by construction |
| selectors that see only validation data from the training distribution | **cannot** know whether deployment is reversed (trap) or aligned (natural). Validation AUROC on the training distribution always favours keeping the shortcut; worst-group validation AUROC always favours removing it. A selector therefore inherits the candidate it prefers; it can dominate only if one candidate is already non-inferior on aligned data |
| surgical removal of only the artifact's contribution (no reweighting, no loss of disease directions) | the **only family for which theory predicts dominance is possible**: same-artifact pairs unchanged, residual cost: the easy part plus the hard part of mask_balanced in the decomposition table is at most 0.006 in magnitude on every natural and external set |

**Where the target is impossible, plainly:**
1. **Strongly aligned test sets.** When the test association is as strong as the training one (the traps' correlated
   test, 0.9 / 0.1), no remedy that removes the shortcut can be non-inferior to masking on that test. The criterion
   therefore uses min(reversed, correlated), not the correlated test alone.
2. **Where masking already removes everything** (artifact outside the ROI: sweeps at r = 0, Trap B), nothing can be
   *superior*. Only non-inferiority is possible, and only for a method whose predictions stay very close to masking's.
3. **Fine-tuned networks with head-level remedies.** Masking + balancing of the fine-tuned ConvNeXt-T does not gain on
   hard pairs (−0.017 [−0.058, +0.019]); only unmasked models do (ERM +0.102, balanced +0.088), and they lose the
   location benefit. A remedy for S5 must act during fine-tuning.
4. **Statistically, not only substantively.** The proposed margin −0.01 is smaller than the interval half-width of most
   trap and clean cells, even for arms whose predictions track masking closely (generated):

<!-- INSERT:precision -->
| component | min half-width | median half-width | max half-width |
|---|---|---|---|
| Trap A min(rev,corr) | 0.005 | 0.032 | 0.076 |
| Trap A reversed | 0.005 | 0.03 | 0.076 |
| Trap B min(rev,corr) | 0.004 | 0.04 | 0.084 |
| Trap B reversed | 0.004 | 0.034 | 0.084 |
| clean (Trap A models) | 0.002 | 0.022 | 0.044 |
| clean (Trap B models) | 0.002 | 0.028 | 0.068 |
| external all pairs | 0.005 | 0.012 | 0.026 |
| fine-tuned all pairs | 0.016 | 0.039 | 0.042 |
| fine-tuned hard pairs | 0.039 | 0.071 | 0.076 |
| hard pairs | 0.005 | 0.024 | 0.089 |
| natural all pairs | 0.003 | 0.014 | 0.057 |
<!-- /INSERT:precision -->

   With half-width h, a method exactly as good as masking passes "lower bound > −0.01" with probability
   Φ(0.01/(h/1.96) − 1.96): about 10% at h = 0.03, 26% at h = 0.015, 80% at h = 0.007. Sixteen trap and clean cells at
   h ≈ 0.02–0.04 make the conjunction unattainable by design for a merely equal method. The criterion needs the
   margin–precision pairing of §6.

## 5. Candidates
### 5a. Standard candidates (from the literature and from this project's arms), ranked by expected gain per GPU hour

| # | candidate | needs | GPU | CPU (est.) | should win | should lose / risk |
|---|---|---|---|---|---|---|
| C1 | **Partial reweighting on masked features**, `mask_bal(λ)`: sample weights ∝ n_group^−λ, λ ∈ {0, ¼, ½, ¾, 1} (λ = 0 is masking, λ = 1 is mask_balanced); λ and C chosen on validation (§5d). Also the logit ensemble α·mask + (1 − α)·mask_balanced | image-level artifact labels (train, validation) | 0 | ~15 min all cohorts (heads on cached features; ~1 s per 5-C fit on 3,000 images) | Trap A, natural hard pairs (as mask_balanced) with a smaller same-artifact cost | if no single λ meets both ISIC 2020 all pairs and Trap A, it trades one for the other; S5 hard pairs |
| C2 | **Adaptive selector v2** (label-free diagnostic → candidate). Diagnostics fixed in advance: (i) ERM-vs-mask validation disagreement on artifact-bearing images, (ii) U-MtE overlay-subspace score of in-ROI artifacts, (iii) the counterfactual in-ROI sensitivity \|dp\|. Decision guide: artifact outside → masking; inside and pixel-carried → U-MtE; inside and labels available → C1 at the validated λ; correlate signature (U-MtE fails on validation worst group) → DFR | ROI masks; labels for the balanced branches | 0 | ~10 min | inherits the chosen candidate per cohort | cannot beat its best candidate (§4); U13/U14 limits below |
| C3 | **Combinations on one model**: mask + DFR, U-MtE + DFR, U-MtE + balancing, ERM/mask score ensembles | labels (DFR, balancing) | 0 | ~10 min | Trap A (DFR family) | natural all pairs (mask_dfr − mask from −0.017 to −0.057 on the five natural sets; DFR retrains on a small balanced subset) |
| C4a | **AFR on masked features** (Qiu et al. 2023): ERM head, then last-layer reweighting by the ERM's predicted probability of the correct class; γ, λ on validation | none in training; validation group labels for tuning (or worst-class if unavailable) | 0 | ~15 min | label-free Trap A gains | correlate-carried hair on natural sets; AFR's reweighting can cost same-artifact pairs like balancing |
| C4b | **GroupDRO on masked features with inferred groups** (groups from an artifact detector or ERM errors) | detector or none | 0 | ~20 min (linear GroupDRO exists in `heads.fit_groupdro`) | Trap A | as C1 at λ = 1 |
| C5 | **Validation-only model selection by worst-group validation AUROC** for every candidate (a rule, not a method) | validation artifact labels | 0 | — | — | favours shortcut removal everywhere (§4, selectors) — so it is applied with an NI guard on validation all-pairs AUROC (§5d) |

Why **U13/U14 fell short** (checked in `docs/PREREGISTRATION_UNIVERSAL_I2E.md` and `results/adaptive_select_*.csv`):
(1) they picked per (trap, seed, fold) by min(AUROC(Y1A0 vs Y0A1), AUROC(Y1A1 vs Y0A0)) on the trap's validation
split — a score that rewards removing the shortcut and has no term for the aligned-distribution cost, so on natural
data it would favour the balanced arms that fail NI there (§6 table); (2) they were only ever evaluated on traps, so
their natural and external behaviour is unknown; (3) near-ties among 11 candidates made 25 per-fold choices noisy
(thyroid: U-MtE_balanced chosen 14/25); (4) U13 excluded DFR (it trains on validation), so it failed the
correlate-carried regime (hair −0.038 vs DFR, drains); U14 admitted DFR but halved the selection data, which made the
ovary and hair choices worse. Both beat masking in every trap (+0.17 to +0.42) — the shortfall is against the best
candidate, and no candidate dominates.

### 5b. Novel candidates (from this project's own findings)
Each gets (a) the closest prior work and an honest novelty label, (b) the theory's prediction per cell before any
run, (c) cost. They go through the same registration, validation-only development and untouched confirmation data.
A failure is reported as a registered negative result.

**N1 — Conditional-mean-constrained head on masked features (`mask_cmc`), with a context-preserving variant
(`full_cmc`).** *Derived from* the theory (masking harms iff it removes context; balancing removes Ã for the artifact
and its correlates) and the decomposition of §4 (balancing's cost sits in same-artifact pairs). The head is logistic
regression constrained to w·(μ̂_{a=1,y} − μ̂_{a=0,y}) = 0 for y = 0, 1, where μ̂ are group means of the training
features. In words: within each class, the linear score may not differ on average between images with and without the
artifact. Implementation: project the features onto the orthogonal complement of the ≤ 2 within-class mean-difference
directions (shrinkage-estimated; the shrinkage and C are chosen on validation), then fit an ordinary head on all
images with their natural weights. No artifact labels are needed at test time. `full_cmc` does the same on unmasked
features. That is the user's "context-preserving erasure", with a label-based rather than overlay-based direction; the
label-free overlay version of that idea already exists here as `ui2e` in the sweeps.
- (a) *Prior work.* This is the linear, first-moment, hard-constraint special case of the anti-causal
  counterfactual-invariance regulariser of Veitch et al. 2021 (arXiv 2106.00545, Eq. 3.2: conditional MMD between
  P(f(X) | Z, Y)), and it is related to Makar et al. 2022 (arXiv 2105.06422) and to ComBat-style harmonisation with a
  protected covariate (doi 10.1093/biostatistics/kxj037; 10.1016/j.neuroimage.2017.11.024). **Adaptation, not new.**
  The new parts are the combination with masking, the choice of full-image versus masked features by location, and the
  pair-decomposition argument for why it should avoid balancing's cost.
- (b) *Theory.* With artifact mean shift √A·u_a in both classes, the constraint forces w ⟂ u_a, so Ã = 0 as with
  balancing. It removes only ≤ 2 directions (S scaled by 1 − ρ², ρ = overlap of the disease direction with those
  directions; small unless the artifact resembles the disease) and keeps every image at full weight, so same-artifact
  pairs ≈ masking. Because the directions are estimated from labelled groups, they also carry the correlates' mean
  shift (hair ↔ site/age), unlike pixel erasure. Predicted per cell: Trap A reversed ≈ mask_balanced (large gain, all
  four cohorts); Trap B ≈ masking (NI); natural all pairs ≈ masking + (π_hard·g − π_easy·l) (NI in every natural and
  external cell if the same-artifact term is ≈ 0); hard pairs ≈ mask_balanced's gain (thyroid, BCN, MSK, ISIC 2020);
  capsule at risk (debris ~ fibrin, ρ larger); S5 not addressed (frozen heads only). `full_cmc` should beat `mask_cmc`
  where S > S_m (dermoscopy: lesion surroundings informative) and lose where masking denoises (thyroid, capsule).
  *Failure mode:* a shortcut carried by covariance rather than mean shift (a linear head cannot use it either, so
  low risk); tiny minority groups make μ̂ noisy (hence shrinkage).
- (c) *Cost.* 0 GPU; ~10 CPU-minutes for all cohorts and encoders (one projection plus the usual heads).

**N2 — Counterfactual-sensitivity penalty on the masked head (`mask_cfs`).** *Derived from* the mechanism (Stage 2):
after masking, reliance on the in-ROI artifact grows (|dp| about twice ERM's) because competing context is gone. The
head on masked features gets a penalty λ Σ_i (w·(x_i^+ − x_i))², where x_i^+ is the masked image with a generic
overlay inserted inside the ROI (the U-MtE insertion pairs already cached). For a linear head this is a generalised
ridge along the overlay-shift covariance. It interpolates between masking (λ = 0) and U-MtE's hard projection
(λ → ∞), shrinking each direction in proportion to its overlay energy.
- (a) *Prior work.* Counterfactual logit pairing (Garg et al. 2019, arXiv 1809.10610), right-for-the-right-reasons
  input-gradient penalties (Ross et al. 2017, arXiv 1703.03717), CDEP (arXiv 1909.13584); our own U-MtE is its
  λ → ∞ limit. **Adaptation**: new only as the soft, label-free version of U-MtE on masked features.
- (b) *Theory.* Pixel-carried shortcuts: between masking and U-MtE on Trap A (thyroid, ovary gains); pathology-like
  debris: less damage than unprotected U-MtE because low-energy disease directions are barely shrunk (capsule ≈
  masking); correlate-carried hair: **no gain** (hard pairs in BCN, MSK, ISIC 2020 ≈ masking). So it is **predicted
  not to dominate**; registered because it is cheap and tests the mechanism's claim directly.
- (c) *Cost.* 0 GPU (the insertion features exist for U-MtE; re-extraction only for cohorts or encoders without them);
  ~10 CPU-minutes.

**N3 — Training-time location randomisation with real artifact instances (`locrand`), frozen and fine-tuned.**
*Derived from* the transplant (Stage 4: any label-correlated pattern inside the ROI survives masking) and the location
law. During training, real artifact instances (cut with their masks by `wtss.transplant.extract_instance`: calipers,
debris, hair strands) are pasted with probability ½ into images of both classes, and when pasted, inside or outside
the ROI with equal probability. Artifact presence and location then carry no label information; the head (frozen) or
network (fine-tuned) is trained on masked views.
- (a) *Prior work.* Copy-paste augmentation (Ghiasi et al. 2021, arXiv 2012.07177, for segmentation); artifact removal
  by inpainting then retraining (Nauta et al. 2022, doi 10.3390/diagnostics12010040) is the opposite operation;
  ruler synthesis for segmentation (arXiv 2509.12277) is not debiasing; U7 here showed that *generic* overlays as
  augmentation gain ≤ +0.015. I found no published use of *real* instances at randomised locations for shortcut
  debiasing, but the search was limited: **an adaptation of copy-paste augmentation, possibly new in this use**.
- (b) *Theory* (§7 of THEORY.md): augmentation decorrelates exactly the direction of what is pasted. With real
  instances that is u_a itself, so Ã → 0 for the pixel-carried part without removing any direction or reweighting any
  image (S unchanged). Predicted: calipers — large Trap A gain, NI on natural sets and Trap B; debris — the head may
  learn to discount fibrin-like texture (the ρ problem returns as label noise), so capsule is at risk; hair — correlates
  untouched, little gain on hard pairs. **The only candidate that reaches S5** (it works during fine-tuning).
- (c) *Cost.* Frozen: extraction of K = 2 randomised copies of each training image. At the measured ~110–130 images/s
  (round 4 R8 logs) that is ~15 min for all four cohorts with DINOv2, ~45 min for three encoders. Fine-tuned: ~45 s per
  ConvNeXt-T model (round 6: 20 models in 14 min) → ~20 min per arm for the natural split and both traps.

**Rejected idea — per-image location policy** (mask only the images whose artifacts lie outside the ROI). If the view
depends on detected artifact location, the view type itself is a proxy for artifact presence and hence for the label
in the traps: the shortcut returns through the view. Training on both views of every image removes that leak, but then
images with in-ROI artifacts are scored on the unmasked view, which is ERM-like — worse than masking wherever masking
denoises (thyroid Trap A: mask − ERM +0.109). Predicted not to dominate; not registered unless the author wants it.
N1's `full_cmc` versus `mask_cmc` choice is the dataset-level version of the same idea without the leak.

### 5c. Expected ranking (gain per GPU hour)
N1 `mask_cmc` (0 GPU, the only family theory says can dominate) > C1 `mask_bal(λ)` (0 GPU) > C2 selector (0 GPU;
bounded by its candidates) > C4a AFR ≈ C4b GroupDRO (0 GPU) > C3 combinations (0 GPU; mostly known to fail NI) > N2
(0 GPU; predicted not to dominate) > N3 (≈1 GPU h frozen + fine-tuned; the only S5 candidate).

### 5d. Validation-only selection for every candidate
Every hyperparameter (λ, α, shrinkage, C, γ) is chosen per training set on its own validation split (never a test
environment). Rule: maximise worst-group validation AUROC (min over the four artifact × label pair types) **subject to**
validation all-pairs AUROC ≥ the masking head's − 0.005 on the same validation split (an NI guard). If no setting
satisfies the guard, masking is kept (λ = 0 / α = 1). The guard is what U13/U14 lacked. Group labels on validation
are the automatic artifact labels; the label-free candidates (U-MtE, N2, AFR) fall back to worst-class validation
AUROC, as recommended by Yang et al. 2023.

## 6. The criterion "dominates masking" — critique and proposal
**The author's proposal:** NI (lower bound > −0.01) in every natural, external and clean cell; SUP (lower bound > 0)
on Trap A reversed in every cohort and on natural hard pairs in thyroid, BCN, MSK and ISIC 2020; NI on Trap B reversed
(−0.01) and min(rev, corr) ≥ masking in every trap; easy pairs excluded.

**Applied to the existing arms (generated):** no existing arm dominates.

<!-- INSERT:criterion_summary -->
| arm | cells_run | cells_met | cells_total |
|---|---|---|---|
| mask_balanced | 36 | 29 | 36 |
| umte_balanced | 34 | 26 | 36 |
| umte_protect_balanced | 34 | 22 | 36 |
| balanced | 36 | 19 | 36 |
| dfr | 34 | 13 | 36 |
| mask_dfr | 34 | 13 | 36 |
| umte | 34 | 13 | 36 |
| umte_protect | 34 | 13 | 36 |
| umte_aug | 24 | 13 | 36 |
| mask_jtt | 24 | 12 | 36 |
| erm | 36 | 9 | 36 |
| jtt | 34 | 6 | 36 |
| umte_jtt | 24 | 6 | 36 |
<!-- /INSERT:criterion_summary -->

Where the three closest arms fail (generated):

<!-- INSERT:criterion_fails -->
| arm | component | cohort | kind | arm − mask [95% CI] |
|---|---|---|---|---|
| mask_balanced | natural all pairs | thyroid | NI | -0.010 [-0.020, +0.000] |
| umte | natural all pairs | thyroid | NI | +0.004 [-0.011, +0.020] |
| umte_balanced | natural all pairs | thyroid | NI | -0.005 [-0.023, +0.013] |
| umte | natural all pairs | capsule | NI | -0.005 [-0.011, +0.001] |
| umte_balanced | natural all pairs | capsule | NI | -0.008 [-0.013, -0.002] |
| mask_balanced | natural all pairs | isic_MSK | NI | -0.006 [-0.011, -0.002] |
| umte | natural all pairs | isic_MSK | NI | -0.008 [-0.014, -0.002] |
| umte_balanced | natural all pairs | isic_MSK | NI | -0.016 [-0.024, -0.008] |
| mask_balanced | external all pairs | isic2019_to_2020 | NI | -0.018 [-0.022, -0.013] |
| umte | external all pairs | isic2019_to_2020 | NI | -0.013 [-0.018, -0.007] |
| umte_balanced | external all pairs | isic2019_to_2020 | NI | -0.033 [-0.040, -0.026] |
| umte | hard pairs | thyroid | SUP | +0.029 [-0.000, +0.060] |
| umte | hard pairs | isic_BCN | SUP | +0.005 [-0.002, +0.011] |
| umte | hard pairs | isic_MSK | SUP | -0.013 [-0.026, -0.001] |
| umte | hard pairs | isic2019_to_2020 | SUP | +0.006 [-0.002, +0.013] |
| umte | Trap B reversed | thyroid | NI | -0.001 [-0.019, +0.019] |
| umte | Trap B min(rev,corr) | thyroid | NI | -0.001 [-0.019, +0.020] |
| mask_balanced | Trap B min(rev,corr) | thyroid | NI | +0.009 [-0.033, +0.043] |
| umte_balanced | Trap B min(rev,corr) | thyroid | NI | +0.013 [-0.029, +0.038] |
| umte | clean (Trap B models) | thyroid | NI | -0.002 [-0.021, +0.017] |
| umte_balanced | clean (Trap B models) | thyroid | NI | -0.000 [-0.025, +0.022] |
| umte | Trap A reversed | capsule | SUP | -0.126 [-0.145, -0.107] |
| umte | Trap B reversed | capsule | NI | -0.020 [-0.034, -0.007] |
| umte | Trap A min(rev,corr) | capsule | NI | -0.126 [-0.145, -0.107] |
| umte | Trap B min(rev,corr) | capsule | NI | -0.020 [-0.034, -0.007] |
| umte | clean (Trap A models) | capsule | NI | -0.045 [-0.056, -0.034] |
| umte | clean (Trap B models) | capsule | NI | -0.013 [-0.021, -0.005] |
| umte_balanced | clean (Trap B models) | capsule | NI | -0.010 [-0.028, +0.004] |
| umte | Trap A reversed | ovary | SUP | +0.020 [-0.016, +0.053] |
| umte | Trap B reversed | ovary | NI | +0.015 [-0.023, +0.052] |
| umte | Trap A min(rev,corr) | ovary | NI | +0.020 [-0.016, +0.053] |
| umte | Trap B min(rev,corr) | ovary | NI | +0.015 [-0.021, +0.062] |
| mask_balanced | Trap B min(rev,corr) | ovary | NI | +0.006 [-0.038, +0.048] |
| umte_balanced | Trap B min(rev,corr) | ovary | NI | +0.029 [-0.016, +0.067] |
| mask_balanced | fine-tuned all pairs | thyroid | NI | -0.016 [-0.033, +0.000] |
| mask_balanced | fine-tuned hard pairs | thyroid | SUP | -0.017 [-0.058, +0.019] |
<!-- /INSERT:criterion_fails -->

**Critique.**
1. *Margin versus precision* (§4.4). With the −0.01 margin, trap and clean cells cannot be passed by a method that is
   only as good as masking. Two of mask_balanced's seven failures (Trap B min(rev, corr) in thyroid and ovary) are
   intervals of ±0.04 around positive estimates.
2. *min(rev, corr) ≥ masking by point estimate* is too lenient (no interval) and, in the paired form, too noisy (the
   minimum switches between environments across replicates, which widens the interval to ±0.04).
3. *Trap A reversed alone* rewards shortcut flipping (DFR and mask + DFR over-correct on thyroid, Stage 6). Superiority
   should be on min(rev, corr).
4. *Cells without precision* (sweeps: half-widths 0.05–0.15; fine-tuned hard pairs ±0.07) cannot carry NI at 0.01.
5. *Multiplicity.* "Dominates" is an intersection–union test: the conjunction holds only if every component holds, so
   each component at one-sided 0.025 controls the error of the dominance claim without adjustment (Berger 1982, Technometrics, doi 10.2307/1267823). Holm
   is still needed across candidates (several dominance claims) and within each descriptive family.

**Refined criterion (proposal to register).** Components per candidate, DINOv2-B/14 for every cohort; replicated in
MedSigLIP and ConvNeXt where cohorts exist (secondary):

| component | cells | test | margin |
|---|---|---|---|
| D1 natural and external all pairs | thyroid split, capsule, BCN, HAM, MSK, ISIC 2019 → 2020 | NI | 0.01 (half-widths 0.003–0.010 for mask_balanced, which tracks masking closely; wider arms need a real positive effect to pass) |
| D2 superiority where masking fails | natural hard pairs: thyroid, BCN, MSK, ISIC 2020; Trap A **min(rev, corr)**: 4 cohorts | SUP | 0 |
| D3 masking's own territory | Trap B reversed, Trap B min(rev, corr), clean (Trap A and B models): 4 cohorts | NI **pooled** over the four cohorts at margin 0.01 **and** per cell at margin 0.03 (≈ the median half-width) | 0.01 pooled / 0.03 per cell |
| D4 no shortcut flipping | every trap | correlated ≥ reversed − 0.02 (point) — Stage 6 rule | — |
| D5 controlled sweeps (secondary, precision permitting) | r = 1: SUP on reversed in every modality; r = 0: NI at 0.03 on a regenerated sweep with ≥ 4× the test images | SUP / NI | 0 / 0.03 |
| D6 fine-tuned (separate claim) | S5 all pairs NI 0.02, hard pairs SUP, Trap A SUP | as stated | 0.02 / 0 |

Dominance = D1–D4 all met (IUT; one-sided 0.025 per component). D5 and D6 are reported as separate, registered
claims, because no head-level method can reach S5 and the sweeps' precision is limited by their test size. Easy pairs
are excluded as proposed: they are where any shortcut-free method must lose.

## 7. Development / confirmation split
Honest starting point: this Phase 1 has **looked at every existing test set** (the scoreboard), so none of them is
untouched for the *existing* arms. What stays clean is data the search has never scored, and new candidates' predictions
on data whose design decisions do not depend on them.

- **Development (all design choices):** the validation splits of every training set (trap `val_groups` /
  `val_clean`, the natural training splits' validation folds, the ISIC 2019 validation groups), and the existing
  scoreboard for choosing *which candidates* to register (done in this document, before Phase 2). Hyperparameters are
  chosen only by §5d.
- **Frozen before confirmation:** `results/round8/frozen_choice.json` (candidate list, every validation-chosen
  hyperparameter per training set, selector thresholds), committed and pushed before any confirmation prediction exists.
- **Confirmation (never touched by the search):**
  1. **New trap resamples**: every trap re-drawn with a new environment seed (and new fold seed), 5 new seeds. The
     image pools are the same, so images overlap with earlier test sets; the environments, groupings and models are
     new. A limitation to state.
  2. **Held-out cohort: ovary.** No ovary validation data are used in development. Its hyperparameters are the medians
     of the other cohorts' validation choices, so the ovary confirms transfer. Its labels are the least validated
     (Stage 8), which makes it the most informative hold-out and the weakest evidence; the alternative is capsule.
  3. **ISIC 2019 → 2020** with new candidates only (existing arms were scored there, so ISIC 2020 is
     semi-confirmatory; stated as such).
  4. **Fine-tuned thyroid** (N3 only): new fine-tuning seeds on the official split; ERM/mask are refitted with the same
     new seeds as references.
  5. **Optional, truly untouched external:** ThyUS2Path (downloaded to `/root/data/external/thyus2path`) needs nodule
     ROIs and caliper detection first. Proposed as a stretch goal, not a condition.

## 8. Compute plan (this machine: RTX PRO 4000 Blackwell 24 GB, 48 cores, ~134 GB disk free)

| step | GPU | CPU | disk |
|---|---|---|---|
| Phase 2 smoke (all scripts, small subsets) | < 5 min | < 10 min | < 1 GB |
| Development: C1–C5, N1, N2 heads on validation, all cohorts × 3 encoders | 0 | ~1 h (16 workers) | predictions ~1 GB (git-ignored) |
| N3 frozen: randomised copies K = 2, masked view, 4 cohorts × DINOv2 (+ MedSigLIP, ConvNeXt) | ~15 min (~45 min) | ~15 min | ~2 GB features under `$WTSS_CACHE/features/round8/` |
| N3 fine-tuned: natural split (5 seeds) + two traps (10 clusters), 1–2 arms | ~40 min | — | < 1 GB |
| Confirmation: new trap resamples (5 seeds), natural, ISIC 2020, fine-tuned | ~20 min (fine-tuned refits) | ~1.5 h (heads + crossed bootstraps at 10,000) | ~2 GB |
| **Total** | **≈ 1.5–2 GPU h** | **≈ 3–4 CPU h** | **≈ 6 GB** |

## 9. Answer to the Phase 1 question (plain language)
- **Is "never worse, better where masking fails" achievable?** Partly, and the data say where. It is impossible on
  strongly aligned test sets (no method that stops using the shortcut can match masking there), impossible to *show*
  with a 0.01 margin in the trap and sweep cells at their current precision, and out of reach for fine-tuned networks
  with head-level fixes. On the natural and external sets it is possible in principle: the unavoidable cost of removing
  the shortcut is small there, and existing balancing methods lose mainly because reweighting throws away disease
  signal, not because they remove the shortcut.
- **Most likely to achieve it:** N1 `mask_cmc`, a head that uses all training images at full weight but may not score
  images with and without the artifact differently within a class. It removes the artifact's and its correlates' mean
  effect like balancing, without balancing's disease cost. Runner-up: C1 partial reweighting with the validation NI
  guard.
- **Cost:** N1 and C1 are CPU-only (minutes); the whole round, including the fine-tuned candidate N3, is about 2 GPU
  hours and 3–4 CPU hours on this machine.
