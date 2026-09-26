# Replication spec (provided by the author, 2026-09-26) — governs all pilot replications

Verbatim copy of the author's README "Replicating the 'Where the Shortcut Sits' pilot". Deviations are
recorded in `CHANGES.md`; match status in `results/SUMMARY.md`.

---

## 0. What the pilot claims

1. **Location decides the method.** Masking an image to the lesion (the region of interest, ROI) removes a shortcut artifact when the artifact lies *outside* the lesion, and makes the model *worse than doing nothing* when it lies *inside*.
2. **Mechanism.** The harm is governed by how much of the artifact survives the mask (retention). It is not caused by occluding lesion tissue, and not by losing surrounding context.
3. **Supervision.** Methods that need only an *image-level* artifact label (balanced training, DFR) fix both regimes with nothing required at test time. Pixel-level artifact localisation fails in practice.
4. **Linear erasure is insufficient** for real artifacts under the sound (paired) protocol.
5. All of the above is shown on a synthetic artifact under full control, and **confirmed on real hair** with pre-registered criteria, on two backbones.

## 1. Environment
- One GPU; original on a V100 32 GB; frozen backbones + linear heads.
- Python 3.10+, PyTorch ≥ 2.1, DINOv2 hub loader, `open_clip_torch` (DermLIP), scikit-learn, opencv, concept-erasure ≥ 0.2.4, imagehash, pandas, numpy.
- Performance pitfall: cache resized images/masks; use AMP.
- Never edit a running script.

## 2. Data sources
| Source | Content | Used for |
|---|---|---|
| ISIC 2018 Task 1 | 2,594 images + manual lesion masks | synthetic experiments; U-Net training |
| ISIC 2016 + 2017 ground truth | melanoma labels for Task 1/2 images | labels (3 conflicting IDs dropped; 2017 takes precedence) |
| ISIC 2019 | 25,331 images, MEL vs rest, sources HAM / BCN / MSK | real-artifact confirmation |
| HAM10000 lesion segmentations (Tschandl) | 10,015 manual masks | lesion masks for the HAM subset of ISIC 2019 |
| ISIC 2019 artifact mask archive (Missouri S&T) | hair+ruler (one channel), ink, vignetting; 255 = artifact | real artifact masks |
| Mendeley hair masks (DOI 10.17632/j5ywpd2p27.2, CC BY 4.0) | 500 manual hair masks (478 join the Task 1 cohort) | real-hair linear analysis |
| Bissoto et al. artifact annotations | image-level presence of 7 artifact types | hair decodability labels |

## 3. Protocol rules
### 3.1 Splits — leakage-safe
Group by lesion_id, merge groups with pHash Hamming ≤ 8; split by group; verify zero cross-split group edges.
Near-duplicate leakage inflated the shortcut gap by 27 % (+0.0896 [+0.014, +0.166]) in an earlier round.
Report threshold counts at 6 / 8 / 10.
### 3.2 Backbones and heads
Frozen DINOv2 ViT-B/14 CLS @518 (primary) and @224; DermLIP/PanDerm (`hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256`,
`encode_image`, native preprocessing @224). L2-normalised features. Head: LogisticRegression(liblinear,
class_weight="balanced", max_iter=3000), C ∈ {0.01, 0.1, 1, 10} on clean validation only.
### 3.3 Environments
train_corr 0.90/0.10; test_corr 0.90/0.10; test_rev 0.10/0.90 (primary); clean — synthetic: no artifact;
**real: artifact uncorrelated with label**. Presence from a stable hash of (image_id, label, seed, env),
independent of the overlap level.
### 3.4 Selection and thresholds
All hyperparameters on clean validation only; thresholds = max balanced accuracy on clean validation, frozen.
### 3.5 Statistics — hierarchical paired bootstrap
Resample seed clusters, then paired test images within cluster; paired AUROC delta within each sampled
cluster; average per replicate; never pool heads. 2.5/97.5 percentiles; seed SD separately. **For k-fold
designs, aggregate fold test predictions per seed before computing AUROC.**
### 3.6 Pre-registration
Decision rules committed to git before each verdict-producing experiment; outcomes reported either way;
amendments only if stricter.

## 4. Known pitfalls
| Pitfall | Symptom | Fix |
|---|---|---|
| Pooled bootstrap across heads | CIs subtly wrong | average seed-specific deltas |
| Artifact placement identical across seeds | seeds less independent | choose uniformly among candidates within tolerance, keyed by (image_id, seed) |
| Placement cache key missing tolerance / n_candidates | stale placements | include them in the cache name |
| Evaluating arms on different image sets | inpaint "+0.53" was really +0.003/−0.008 | inner-join every paired comparison on identical IDs; identity fallback when no mask |
| Near-constant probe for \|Δp\| | \|Δp\| ≈ 1e-4 | report probability SD; flag SD < 0.01 |
| Overlap-contrast trap | null everywhere | comparison group must be artifact-free (E13) |
| Unpaired LEACE | huge reversed gains, clean drops, corr < rev | paired orig↔inpaint erasure |
| Masks for placement ≠ masks for masking | arms see different stimuli | ground-truth lesion mask for every arm |
| Threshold tie-break on a shared cell | decided nothing | tie-break on cells the threshold changes |
| Source imbalance across cells | comparing sites | match cells to a common HAM/BCN/MSK mix |

## 5. Experiments and expected numbers
Tolerance: same versions ±0.01; across versions every sign and CI-excludes-0 verdict matches, points ±0.02.
Seeds: synthetic {42, 123, 456}; real-data traps {42, 123, 456, 789, 2026}.

- **E1** synthetic overlap sweep, DINOv2@518, 2,437 images (154 excluded: median lesion area 0.012 vs 0.152,
  melanoma 4.5 % vs 20.9 %). Mask − ERM: +0.184 [+0.146,+0.224], −0.058 [−0.100,−0.018], −0.097 [−0.138,−0.056],
  −0.111 [−0.153,−0.068], −0.128 [−0.170,−0.086]; interaction +0.312 [+0.272,+0.354]; mask corr=rev=0.849 at 0 %;
  masked corr 0.961 vs ERM 0.937 at 100 %.
- **E2** mask − ERM by backbone (0/25/50/75/100 %): DINOv2@518 +0.184/−0.058/−0.098/−0.111/−0.129;
  DINOv2@224 +0.364/−0.040/−0.076/−0.102/−0.092; DermLIP@224 +0.160/−0.291/−0.300/−0.313/−0.268.
  One of 39 comparisons changed CI status after the bootstrap fix (DINOv2@224, 25 %).
- **E3** appearance randomisation: ERM rev 0.702/0.625/0.576; mask − ERM +0.147/−0.094/−0.155; balanced
  +0.103/+0.171/+0.222; LEACE |corr − rev| 0.002/0.002/0.000; interaction +0.302 [+0.264,+0.340].
- **E4** occlusion (50/50): mask − ERM +0.035 [+0.006,+0.065] (0 %), +0.033 [−0.000,+0.066] (50 %),
  +0.039 [+0.010,+0.067] (100 %); occlusion cost vs clean ≈ 0.000–0.004.
- **E5** dilation: 50 %: margins 0/10/25/50 → retention 0.50/0.73/0.95/1.00, harm −0.097/−0.138/−0.153/−0.142;
  100 %: −0.128/−0.143/−0.144/−0.138; pixel share 0.055 → 0.018.
- **E6** |Δp| at 100 %: ERM 0.235, mask 0.490; at 0 %: mask 0.000, ERM 0.128; ERM 0.195 at 50 %.
- **E7** lesion-size tertiles (large/medium/small): 0 %: +0.249/+0.214/+0.150; 50 %: +0.087/−0.129/−0.325;
  100 %: +0.064/−0.158/−0.364. Artifact-to-lesion ratio ~0.012 (large) vs ~0.106–0.123 (small).
- **E8** inpaint rev 0.806/0.783/0.779/0.759; Δ +0.211/+0.209/+0.244; consistency +0.038/+0.046/+0.031/+0.043,
  |Δp| higher (+0.073/+0.086/+0.085), PARTIAL_SUPPORT; λ=0 +0.030/+0.036/+0.021/+0.040.
- **E9** bridge at 100 %: LEACE +0.294, balanced +0.281, DFR +0.279, inpaint +0.244, GroupDRO +0.197, ERM 0.515,
  mask −0.128.
- **E10** hair decodability (Bissoto) test AUROC 0.943 [0.919,0.964], n=356, base rate 0.567, P(hair|mel)=0.411
  vs 0.608; paired LEACE on 478 Mendeley images (340 train pairs): rank 1/768, A-decoder 0.968→0.500, clean
  0.812→0.814, |Δp| 0.072→0.058; DullRazor detector IoU mean 0.22, median 0.18, P 0.32, R 0.46.
- **E11** ISIC 2019 ∩ HAM10000 atlas (10,015; 5,850 groups; split 6,529/1,310/2,176): hair+ruler n 9,760
  r mean 0.280 median 0.219 share≥0.5 0.199; ink 65, 0.045/0.000/0.031; vignetting 1,134, 0.003/0.000/0.002.
  Vignetting trap (5 seeds): ERM rev 0.305; mask +0.190 [+0.160,+0.219]; balanced +0.347; DFR +0.490;
  LEACE +0.004; inpaint −0.008 [−0.017,+0.001].
- **E12** overlap-contrast trap (A=1 r≥0.5 vs A=0 r<0.1): C1 +0.019 [−0.085,+0.122], C2 +0.047 — null.
- **E13** full ISIC 2019; lesion masks HAM manual, BCN/MSK from a U-Net trained on ISIC 2018 Task 1
  (Ronneberger, widths 64–1024; held-out Dice 0.876, median 0.915; vs HAM manual 0.895 / 0.947); manual↔U-Net
  swap: r MAE 0.042, Pearson 0.929, 10.7 % change stratum. Two traps sharing one hair-free A=0 group; Trap A
  r ≥ 0.5, Trap B r < 0.1 (strict 0.6/0.05 sensitivity; chosen on min A=1 cell 1,131 vs 884). Source matching:
  subsample cells to the A=0 source mix within each label (benign HAM 30.2/BCN 63.3/MSK 6.5 %; melanoma
  25.5/68.2/6.4 %). 5-fold group-safe CV, 90/10 → 10/90; gate ≥ 40 reversed melanomas, every matched cell ≥ 25.
  Matched cells Trap A: A0_Y0 708, A0_Y1 157, A1_Y0 2,903, A1_Y1 1,144 (4,912); Trap B: 708, 157, 4,093, 809
  (5,767). Reversed-test melanomas 172 per trap (~34/fold). Cross-fold groups 0.
  DINOv2@518: ERM clean 0.782 rev 0.516 (Trap B rev 0.510); mask 0.769/0.452, −0.064 [−0.079,−0.049] / B +0.124
  [+0.105,+0.143]; inpaint 0.811/0.616, +0.101 [+0.094,+0.107] / +0.108 [+0.098,+0.117]; balanced 0.812/0.729,
  +0.213 [+0.190,+0.235] / +0.234 [+0.217,+0.252]; DFR 0.724/0.718, +0.202 [+0.176,+0.228] / +0.192
  [+0.168,+0.216]; LEACE paired 0.800/0.574, +0.058 [+0.053,+0.063] / +0.078 [+0.070,+0.085].
  C1–C5 supported; C3 +0.188 [+0.179,+0.194], per seed 0.171–0.195; C5 +0.068 vs +0.112. Source-stratified
  Trap A mask − ERM: HAM −0.030 [−0.059,−0.003], BCN −0.078 [−0.099,−0.058]; Trap B without BCN +0.276;
  HAM paired mask-source check: −0.006 [−0.029,+0.016].
- **E14** area ratio: in-lesion hair median 0.033, ruler 0.030; Trap A tertiles (0.005/0.032/0.116):
  −0.064/−0.057/−0.059; ruler vs hair at matched ratio −0.364 vs −0.059.
- **E15** DermLIP@224 Trap A: ERM 0.845/0.926/0.677; mask 0.800/0.893/0.608, −0.069 [−0.088,−0.051]; balanced
  0.858/0.882/0.803, +0.126 [+0.115,+0.138]; DFR 0.827/0.840/0.792, +0.114 [+0.091,+0.138]; LEACE paired
  0.856/0.924/0.708, +0.031 [+0.028,+0.034]. Hair decodability 0.883 [0.872,0.894]; LEACE rank 1/512 → 0.492.
  LEACE 2×2: DINOv2 paired 0.800/0.920/0.574 (+0.058), A-erasure 0.736/0.624/0.831 (+0.316); DermLIP paired
  0.856/0.924/0.708 (+0.031), A-erasure 0.764/0.643/0.869 (+0.192).

## 6–7. Run order and deliverables
Data/joins/dedup counts → features → E1–E7 → E8–E9 → E10–E11 → U-Net, E12, E13 → E14–E15.
Deliverables: results/SUMMARY.md (expected vs obtained, MATCH/MISMATCH), results/PRECOMMITS/, COMPARABILITY.md,
CHANGES.md, unit tests (placement, bootstrap, LEACE rank, zero cross-split edges, presence independent of overlap).
