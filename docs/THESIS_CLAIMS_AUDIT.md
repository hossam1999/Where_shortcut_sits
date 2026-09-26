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
| 6.1 | U-Net held-out Dice 0.876; 0.895 vs manual masks | NO CODE | re-implemented: `scripts/data/train_lesion_unet.py` |
| 9 | ISIC 2019 Traps A/B (25,331 images), C1–C5, DermLIP replication, source stratification, five-seed crossover | NO CODE | re-implemented under pre-registration `docs/PREREGISTRATION_ISIC2019_TRAPS.md` |
| 10 | Hair detector IoU 0.22 | VERIFIED-ARCHIVE | `phase2/real_hair_counterfactual/DETECTOR_REPORT.json` |
| 10 | Training-time repair +0.043; λ=0 control | VERIFIED-ARCHIVE | bridge table / `lambda0_control/` |
| 10 | Hair linearly decodable AUROC 0.943 (DINOv2) | VERIFIED-ARCHIVE | `hair_linear_leace/H1_bissoto_linear_decode.csv` |
| 10 | … 0.883 (DermLIP) | NO CODE | re-implemented on ISIC 2019 |
| 10 | Rank-1 projection 0.812 → 0.814 | VERIFIED-ARCHIVE | `HAIR_LINEAR_REPORT.md` |
| 10 | Unpaired erasure +0.316 (DINOv2), +0.192 (DermLIP) | NO CODE | re-implemented as `leace_unpaired` on ISIC 2019 traps |
| 11 | Synthetic ruler vs real hair at matched area ratio (−0.364 vs −0.059) | NO CODE | — |
| 11 | ≈11% of images change overlap stratum if mask source swapped | NO CODE | re-implemented (HAM manual vs U-Net) |

## Rerun from raw images (DINOv2@518, all nine pilot arms)
`results/verification/SYNTHETIC_RERUN_VS_ARCHIVE.md`: 252 per-seed AUROCs reproduced within ≤ 0.0005
(fp16 non-determinism) for every sklearn-head arm; the GPU-trained consistency head within ≤ 0.008.
All 21 reversed-test Δ-vs-ERM estimates and CIs agree to 3 decimals; all 21 CI decisions identical.
Location interaction (mask): +0.312 [+0.272, +0.353].
