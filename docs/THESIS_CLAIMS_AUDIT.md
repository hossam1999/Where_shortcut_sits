# Thesis-proposal claims vs. evidence (audit)

Status legend: **VERIFIED-ARCHIVE** = recomputed exactly from archived predictions/tables;
**VERIFIED-RERUN** = reproduced by rerunning from raw images with this repo;
**NO CODE** = not produced by any code in the archived repository; **NOT SUPPORTED** = archived
evidence contradicts the claim. Filled in as verification proceeds (see `results/verification/`).

| § | Claim in proposal | Status | Evidence / note |
| --- | --- | --- | --- |
| 7 | Masking Δ vs ERM (DINOv2@518): +0.184 / −0.097 / −0.111 / −0.128 at 0/50/75/100% | VERIFIED-ARCHIVE + VERIFIED-RERUN | `scripts/verify/recompute_archived_cis.py`: all 21 bridge CIs recomputed with max abs diff 1e-16 |
| 7 | Interaction 0% vs 100%: +0.312 [+0.272, +0.354] | VERIFIED-ARCHIVE | recomputed +0.312 [+0.272, +0.353] (bootstrap seed only) |
| 7 | Variable-ruler stress +0.147 / −0.155; interaction +0.302 | VERIFIED-ARCHIVE | `results_v3/stress_variability/STRESS_REPORT.md` |
| 8 | Occlusion control +0.035/+0.033/+0.039; cost 0.000–0.004 | pending | v2 `occlusion/` tables |
| 8 | Dilation at 50%: −0.097 → −0.153 as retention 0.50 → 0.95; flat at 100% | VERIFIED-ARCHIVE | `phase1/dilation_mechanism_joined.csv` (margin 25 px: −0.153, retention 0.946) |
| 8 | Same-head \|Δp\| 0.235 (ERM) → 0.490 (mask) at 100% | VERIFIED-ARCHIVE | bridge unified table |
| 6.1 | "Before the [leakage] correction, the measured shortcut gap was inflated by 27%" | **NOT SUPPORTED** | The pre-correction pilot (`pilot_v100`, 224 px) used the *identical* split (confusion matrix diagonal, 2,449/2,449 images; no pHash group spans splits) and identical labels. At matched resolution its ERM shortcut gap is 0.96–0.97× the corrected gap (not 1.27×). The +0.357 vs +0.184 difference at 0% compares 224 px with 518 px. Replaced by a controlled experiment (`scripts/run_leakage_experiment.py`). |
| 6.1 | U-Net held-out Dice 0.876; 0.895 vs manual masks | NO CODE → RE-IMPLEMENTED | ResNet34 U-Net (384 px, 12 epochs): held-out Dice **0.939** (HAM 0.949, ISIC 2018 0.901); vs manual masks on held-out ISIC 2019 images 0.948 (`isic2019/unet/UNET_REPORT.json`, `results/verification/MASK_SOURCE_SENSITIVITY.json`) |
| 9 | ISIC 2019 Traps A/B (25,331 images), C1–C5, DermLIP replication, source stratification, five-seed crossover | NO CODE | re-implemented under pre-registration `docs/PREREGISTRATION_ISIC2019_TRAPS.md` |
| 10 | Hair detector IoU 0.22 | VERIFIED-ARCHIVE | `phase2/real_hair_counterfactual/DETECTOR_REPORT.json` |
| 10 | Training-time repair +0.043; λ=0 control | VERIFIED-ARCHIVE | bridge table / `lambda0_control/` |
| 10 | Hair linearly decodable AUROC 0.943 (DINOv2) | VERIFIED-ARCHIVE | `hair_linear_leace/H1_bissoto_linear_decode.csv` |
| 10 | … 0.883 (DermLIP) | NO CODE | re-implemented on ISIC 2019 |
| 10 | Rank-1 projection 0.812 → 0.814 | VERIFIED-ARCHIVE | `HAIR_LINEAR_REPORT.md` |
| 10 | Unpaired erasure +0.316 (DINOv2), +0.192 (DermLIP) | NO CODE | re-implemented as `leace_unpaired` on ISIC 2019 traps |
| 11 | Synthetic ruler vs real hair at matched area ratio (−0.364 vs −0.059) | NO CODE | — |
| 11 | ≈11% of images change overlap stratum if mask source swapped | NO CODE → RE-IMPLEMENTED | **6.9%** of 824 held-out hair-present images change stratum; **0** Trap A↔B swaps (all changes involve the 0.1 ≤ r < 0.5 donor band) |

## Rerun from raw images (DINOv2@518, all nine pilot arms)
`results/verification/SYNTHETIC_RERUN_VS_ARCHIVE.md`: 252 per-seed AUROCs reproduced within ≤ 0.0005
(fp16 non-determinism) for every sklearn-head arm; the GPU-trained consistency head within ≤ 0.008.
All 21 reversed-test Δ-vs-ERM estimates and CIs agree to 3 decimals; all 21 CI decisions identical.
Location interaction (mask): +0.312 [+0.272, +0.353].

## §9 / E13 under the author's protocol (docs/REPLICATION_SPEC.md), DINOv2 @518, 5 seeds × 5 folds
Outcome-free anchors: hair-free group ≤ 30 native px reproduces A0 cells and all six source proportions;
reversed-test melanomas 172 per trap (spec 172). A1 cells differ by 5–15 % (lesion masks from a re-trained
spec U-Net, held-out Dice 0.890 vs 0.876).

| Claim | Spec | Replication | Verdict (sign, CI status) |
|---|---|---|---|
| C1 mask − ERM, Trap B | +0.124 [+0.105, +0.143] | +0.107 [+0.078, +0.135] | MATCH (point Δ 0.017) |
| C2 mask − ERM, Trap A | −0.064 [−0.079, −0.049] | −0.037 [−0.057, −0.019] | SIGN+CI match (point Δ 0.027) |
| C3 crossover B − A | +0.188 [+0.179, +0.194] | +0.145 [+0.105, +0.181]; seeds 0.087–0.175 | SIGN+CI match |
| C4 balanced / DFR both traps | all CIs > 0 | +0.225 / +0.251 (A), +0.209 / +0.215 (B), all CIs > 0 | MATCH |
| C5 paired LEACE < ½ best label-only | +0.068 vs +0.112 | +0.084 vs +0.117 | MATCH |
| Inpaint (oracle) A / B | +0.101 / +0.108 | +0.103 / +0.096 | MATCH |
| Source-stratified Trap A, BCN | −0.078 [−0.099, −0.058] | −0.051 [−0.075, −0.027] | SIGN+CI match |
| Source-stratified Trap A, HAM | −0.030 [−0.059, −0.003] | −0.009 [−0.036, +0.017] | sign match; CI now includes 0 |
| Unpaired A-erasure Trap A | +0.316, corr < rev | +0.326, corr 0.618 < rev 0.817 | MATCH (gaming reproduced) |

Sensitivity: the author-independent reconstruction (`results/real/isic2019/dino518_main`: hair-free < 0.1 %
of pixels, hair ≥ 0.5 %, artifact-free clean test) gives C1/C3/C4/C5 but a null C2 (+0.012 [−0.042, +0.062]):
the in-ROI harm depends on including lightly-haired images and on the strictly hair-free comparison group.

## Leakage (§3.1) — controlled experiment (`scripts/run_leakage_experiment.py`)
Same 2,437 images, same features and protocol; grouped split vs 5 random image-level splits (93 near-duplicate
pHash ≤ 8 pairs cross train/test). ERM shortcut gap: 0.241 → 0.284 at 0 % (+18 %), 0.422 → 0.465 at 100 % (+10 %);
mask − ERM: +0.184 → +0.223 (0 %), −0.129 → −0.088 (100 %). **Direction of the "27 %" claim supported; supported
magnitude 10–18 %.**

## E12 — overlap-contrast trap
Author protocol, full ISIC 2019 spec cohort: A=1 hair r ≥ 0.5 vs A=0 hair r < 0.1. ERM corr 0.884 / rev 0.479;
mask − ERM −0.083 [−0.097, −0.072]; balanced +0.169, DFR +0.214. **Not null** (spec: null). The spec's E12 predates
E13 and likely used a different cohort; here the contrast is learnable and masking harms, consistent with the
retention mechanism.

## DermLIP availability
`redlessone/DermLIP_PanDerm-base-w-PubMed-256` became gated (`gated="auto"`) on 2026-09-21, after the pilot ran;
access is granted automatically after accepting the terms on the model page.
