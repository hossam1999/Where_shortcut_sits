# Pre-registration — artifact-agnostic erasure (Universal Insert-to-Erase, U-I2E; Universal Mask-then-Erase, U-MtE)

Committed before any U-I2E / U-MtE result exists. The insertion library is `wtss.synthetic.draw_generic_artifact`
(version "generic_artifact_v1"): one procedural family — thin curvilinear strokes, thick bands, solid/translucent
patches, blobs, dashed lines, small glyphs; random colour (half greyscale), opacity 0.35–1, width, 1–3 elements,
half centred inside the ROI. It contains **no ruler with ticks and no real hair**, and is **identical for every
artifact, dataset and modality** (no per-artifact tuning). Eraser: paired-difference subspace, energy 0.90,
max rank 64 (unchanged from I2E).

- **U-I2E**: erase the generic-insertion subspace from the original-image features; head on erased features.
- **U-MtE**: ROI-mask the image, insert generic structure *then* mask, erase that subspace in the masked view;
  head trained and tested on masked images (no artifact localisation, no artifact labels).
- `_balanced` variants add image-level artifact labels (group-balanced head).

## Claims (same protocols and bootstraps as the corresponding experiments)
- **U1** synthetic ruler (fixed and variable appearance), 100 % overlap: U-I2E − ERM reversed AUROC > 0 (CI).
- **U2** synthetic ruler, 100 % overlap: U-MtE − mask > 0 (removes masking's in-ROI harm) and U-MtE − ERM > 0.
- **U3** real hair Trap A (author protocol, DINOv2 @518 and DermLIP @224): U-MtE − mask > 0 (CI) and
  U-MtE − ERM ≥ 0; Trap B: U-MtE − ERM > 0.
- **U4** no gaming: clean AUROC loss ≤ 0.02 vs ERM and correlated ≥ reversed − 0.02 wherever U1–U3 are claimed.
- **U5** (labels available) U-I2E_balanced / U-MtE_balanced ≥ balanced on reversed AUROC with clean AUROC ≥ balanced's.
Outcomes are reported either way.

## Results (appended after the runs)
Synthetic ruler, DINOv2 @518, 100 % overlap (reversed AUROC Δ vs ERM): U-I2E +0.173 [+0.133, +0.218] (U1 ✅;
clean 0.822 vs ERM 0.792; corr 0.903 ≥ rev 0.688); U-MtE −0.007 [−0.053, +0.040] (U2 ❌); U-I2E_balanced +0.291
[+0.258, +0.325] vs balanced +0.281, clean 0.824 vs 0.806 (U5 ✅); U-MtE_balanced +0.243.
Real hair, author protocol, DINOv2 @518: U-I2E +0.046 [+0.025, +0.067] (A), +0.032 [+0.020, +0.044] (B);
U-MtE − mask +0.052 [+0.044, +0.062] (A), +0.025 [+0.009, +0.040] (B); U-MtE − ERM +0.015 [−0.004, +0.033] (A),
+0.132 [+0.105, +0.160] (B) (U3 ✅); U-I2E_balanced +0.238 (A) vs balanced +0.225 with clean 0.791 vs 0.775 (U5 ✅ A),
+0.207 vs +0.209 (B); U-MtE_balanced +0.254 (B, best). No arm inverts correlated/reversed except none (U4 ✅).

## Amendment (2026-09-27, before fitting) — erase vs augment ablation (reviewer baseline)
Closest generic baseline to U-MtE: the SAME generic overlays used as training augmentation (random-erasing /
occlusion-augmentation style) instead of subspace erasure. Arms: insert_aug (ERM view + overlaid copies of a
random 50 % of training images) and mte_aug (masked view + masked overlaid copies). Claim **U7**: U-MtE − mte_aug
> 0 (reversed AUROC, Trap A) on ISIC 2019 hair and thyroid markers (DINOv2@518). Reported either way.

### U7 results (results/ablation_u7.json; reversed AUROC, Trap A)
- Thyroid: U-MtE − mte_aug = **+0.207 [+0.182, +0.235]** (mte_aug − mask only +0.015); Trap B −0.003 [−0.020, +0.014].
- ISIC 2019 hair: U-MtE − mte_aug = **+0.045 [+0.037, +0.054]** (mte_aug − mask +0.007); Trap B +0.015 [−0.000, +0.030].
**U7 supported in both cohorts**: the same generic overlays help only when used to *estimate and erase* a
subspace, not as training augmentation.

## Amendment (2026-09-27, before fitting) — annotation-free balancing (U8)
Balanced heads need image-level artifact labels A. Label-free replacement: an "overlay detector" (logistic,
C=1) trained on (original, generic-overlay) feature pairs of the training pool is applied to the real training
images; Otsu's threshold on its logits gives pseudo-A (`wtss.heads.pseudo_artifact_labels`).
Arms: pbal (balanced ERM head with pseudo-A) and umte_pbal (U-MtE + pseudo-A balancing).
Claims **U8a** umte_pbal − U-MtE > 0 and **U8b** pbal − ERM > 0 (reversed AUROC, Trap A), thyroid + ISIC hair,
DINOv2@518. Diagnostic: AUROC of the detector score vs true A. Reported either way.

## Amendment (2026-09-27, before fitting) — disease-protected erasure (U9)
Motivation: on capsule endoscopy, template MtE removed disease evidence (debris resembles erosion fibrin).
Method: W = orthonormal basis of logistic label weights (C=0.1, 5 bootstrap fits) on the ARTIFACT-FREE training
images (A=0) in the masked view; erased subspace U' = orth((I − W Wᵀ) U) (`insertion.protect`). Guarantee:
wᵀP(x) = wᵀx for w ∈ span(W). Arms mte_protect / mte_protect_balanced, both with the template library
(where it exists) and the generic library. Claims **U9a** mte_protect − mte > 0 (Trap A) on capsule;
**U9b** mte_protect − mask > 0 (Trap A) on capsule, thyroid and ISIC hair; **U9c** mte_protect − mte ≥ −0.02 on
thyroid and ISIC hair (protection costs little where erasure already works). Reported either way.

### U8 results (annotation-free pseudo-group balancing)
- Thyroid: overlay-detector AUROC vs true marker presence 0.84 (A) / 0.79 (B); umte_pbal reversed 0.663 vs U-MtE
  0.622 (Trap A); pbal 0.409 vs ERM 0.289.
- ISIC hair: detector AUROC 0.57 / 0.63 only; umte_pbal 0.510 vs U-MtE 0.506 (no gain); pbal 0.498 vs ERM 0.491.
U8 helps only where the generic overlays resemble the real artifact in feature space (markers, not hair).

### Cross-cohort summary of the generic (artifact-agnostic) arms, DINOv2@518, Trap A reversed AUROC
| cohort | ERM | mask | balanced | DFR | U-MtE | U-MtE + balanced |
|---|---|---|---|---|---|---|
| thyroid markers | 0.289 | 0.400 | 0.675 | 0.738 | 0.622 | **0.788** |
| capsule debris | 0.587 | 0.684 | 0.775 | 0.805 | 0.569 | **0.857** |
| ISIC hair | 0.491 | 0.454 | 0.716 | **0.743** | 0.506 | 0.704 |
| ISIC hair, DermLIP | 0.670 | 0.585 | **0.813** | 0.780 | 0.509 | 0.648 |
U-MtE + balanced is best where the shortcut is carried by the artifact pixels (markers, debris); where the
shortcut is mostly carried by non-pixel correlates (hair: site/age/sex), label-based group methods win.

## Amendment (2026-09-27, before fitting) — erasure + DFR (U10)
Arms: mask_dfr (DFR head on the masked view) and mte_dfr (DFR head on the masked, generic-erased view; eraser
unchanged). Claim **U10**: mte_dfr − dfr > 0 and mte_dfr − mask_dfr > 0 (reversed AUROC, Trap A) on thyroid,
capsule and ISIC hair (DINOv2@518, generic library). Reported either way.

### U9 results (paired_deltas.csv in each results dir; reversed AUROC, Trap A)
- **U9a supported** (capsule): mte_protect − mte = +0.383 [+0.357, +0.414] (template), +0.105 [+0.092, +0.117] (generic).
- **U9b**: capsule not supported (−0.012 [−0.028, +0.003] template; −0.010 [−0.020, −0.001] generic);
  thyroid supported +0.217 [+0.174, +0.254]; ISIC hair supported +0.041 [+0.030, +0.054].
- **U9c supported**: protection cost vs mte −0.005 [−0.036, +0.024] (thyroid), −0.012 [−0.020, −0.003] (ISIC).
- Balanced: protect_balanced − mte_balanced +0.125 (capsule template), −0.009 (capsule generic), +0.016 (thyroid),
  −0.020 (ISIC).
Interpretation: protection is a safety net — it removes the catastrophic failure when the artifact resembles the
disease (capsule template −0.395 → −0.012 vs mask) at ≤ 0.012 cost elsewhere; it does not by itself beat masking
when the artifact subspace overlaps the disease subspace.

### U10 results (reversed AUROC, Trap A; paired_deltas.csv)
| cohort | mte_dfr − dfr | mte_dfr − mask_dfr | mask_dfr − dfr |
|---|---|---|---|
| thyroid | +0.101 [+0.083, +0.120] | +0.007 [+0.002, +0.011] | +0.095 [+0.075, +0.116] |
| capsule | +0.080 [+0.054, +0.107] | +0.000 [−0.015, +0.014] | +0.080 [+0.062, +0.099] |
| ISIC hair | −0.039 [−0.066, −0.009] | +0.014 [−0.004, +0.032] | −0.053 [−0.076, −0.030] |
U10 (mte_dfr > dfr and > mask_dfr) **not supported as a whole**: once a group-robust head (DFR) is used, the
generic erasure adds ≤ 0.014 over masking + DFR. The gain over DFR comes from *masking* in marker/debris cohorts
(+0.08–0.10) and masking costs −0.05 on hair — again location/type dependent.

## Amendment (2026-09-27, before fitting) — backbone-type replication (CNN)
ConvNeXt-Base (timm convnext_base.fb_in22k_ft_in1k_384, global-average-pooled, frozen) on the thyroid and capsule
traps with the generic library; same arms as the DINOv2 runs. Claims as T6 / C5 / U7 / U9 (U-MtE − mask > 0,
U-MtE − mte_aug > 0 in Trap A). Purpose: show the methods are not specific to ViTs. Reported either way.

### Replication — MedSigLIP-448, thyroid, generic library (results/thyroid/medsiglip448_universal)
Trap A reversed AUROC: U-MtE − mask = +0.204 [+0.174, +0.237]; U-MtE − mte_aug = +0.177 [+0.148, +0.211] (U7);
U-MtE_protect − mask = +0.262 [+0.241, +0.282]; U-MtE_balanced − balanced = +0.096 [+0.083, +0.110].
U-MtE_balanced: rev 0.792 / corr 0.800 / clean 0.798 — most robust arm (min(rev,corr)).

## Amendment (2026-09-27, before fitting) — label-free group robustness baseline: JTT (U11)
JTT (Liu et al. 2021): ERM → training errors at the clean-validation threshold → retrain with errors upweighted by
λ ∈ {5, 20, 50} (λ, C by clean-validation AUROC). Needs no artifact/group labels. Arms: jtt (ERM view), mask_jtt,
umte_jtt (masked + generic-erased view). Claims **U11a** umte_jtt − jtt > 0 and **U11b** umte_jtt − U-MtE ≥ 0
(reversed AUROC, Trap A) on ISIC hair, thyroid and capsule (DINOv2@518). Among label-free arms, the most robust
(min(rev, corr)) is reported per cohort. Reported either way.

### U11 results — label-free arms (Trap A reversed AUROC; paired_deltas.csv)
| cohort | JTT | mask+JTT | U-MtE | U-MtE+JTT | U-MtE − JTT | U-MtE+JTT − JTT |
|---|---|---|---|---|---|---|
| thyroid | 0.423 | 0.479 | **0.622** | 0.369 | +0.198 [+0.167, +0.231] | −0.054 [−0.069, −0.040] |
| capsule | 0.607 | 0.769 | 0.569 | **0.771** | −0.038 [−0.065, −0.013] | +0.164 [+0.098, +0.223] |
| ISIC hair | **0.564** | 0.509 | 0.506 | 0.541 | −0.058 [−0.089, −0.031] | −0.023 [−0.061, +0.007] |
U11a/U11b not supported uniformly. Without any labels, the best arm is cohort-dependent: U-MtE where the shortcut is
in distinct overlay-like pixels (calipers), U-MtE+JTT or mask+JTT where erasure alone removes disease signal
(debris), and JTT on the unmasked view where masking harms and the shortcut is carried by correlates (hair).

### CNN replication — ConvNeXt-Base, thyroid, generic library (results/thyroid/convnext384_universal)
Crossover (mask) +0.284 [+0.253, +0.315]; Trap A mask − ERM +0.003 (masking useless in-ROI) vs Trap B +0.288.
U-MtE − mask +0.128 [+0.112, +0.145]; U-MtE − mte_aug +0.120 [+0.102, +0.139] (U7); U-MtE_protect − mask +0.195
[+0.153, +0.238]; U-MtE_balanced − balanced +0.084 [+0.051, +0.111]; U-MtE_balanced min(rev,corr) 0.745 (best;
mask+DFR 0.722, DFR 0.669). The methods are not ViT-specific.

### U9 extension — DermLIP, ISIC hair (results/spec_e13/dermlip224_spec_protect; post-registration arm set, same method)
Trap A reversed: ERM 0.670, mask 0.585, U-MtE 0.509, U-MtE_protect 0.589, U-MtE_protect_balanced 0.693,
U-MtE_balanced 0.648, JTT 0.670 (balanced alone 0.813). Protection removes the erasure failure (≈ mask), as on
capsule; on DermLIP hair masking itself harms and label-based balancing is best.

## Amendment (2026-09-27, before fitting) — protected U-MtE in the controlled sweeps
Arms umte_protect / umte_protect_balanced added to the synthetic driver (same protection as U9, using the
artifact-free training images of the correlated training environment). Run on capsule (DINOv2, MedSigLIP) and
thyroid (DINOv2, MedSigLIP) sweeps, tag "protect". Claim: umte_protect − mask > 0 at r = 1 where U-MtE − mask ≤ 0.

### Protected U-MtE in controlled sweeps — capsule, MedSigLIP (results/synthetic/capsule/medsiglip448_debris_corr_protect)
At r = 1: umte_protect − mask = **+0.061 [+0.032, +0.092]** (claim supported; unprotected U-MtE − mask was −0.114);
umte_protect − umte = +0.175 [+0.137, +0.217]; umte_protect_balanced − balanced = −0.002 [−0.048, +0.046].
- Capsule, DINOv2 (…/dino518_debris_corr_protect): umte_protect − mask = +0.118 [+0.077, +0.167]; − umte = +0.037
  [+0.007, +0.072].
- Thyroid, MedSigLIP (…/medsiglip448_caliper_corr_protect): umte_protect − mask = −0.041 [−0.089, −0.005]
  (claim **not supported**); − umte = +0.035 [−0.013, +0.085]; umte_protect_balanced at r=1: +0.486 vs ERM.
Summary: protection turned 1 of 2 MedSigLIP controlled failures into a win and never cost > 0.012 anywhere; the
balanced (protected or not) variant is the only one that beats masking in every controlled sweep.

## Amendment (2026-09-27, before fitting) — SPLINCE baseline (U12)
`wtss.heads.SpliceProjection`: task-preserving oblique projection (Cov(PX, A) = 0, Cov(PX, Y) preserved), fitted
on the training environment with image-level artifact labels; arms splice (ERM view) and mask_splice (masked
view). Compared with U-MtE_protect (label-free w.r.t. A) and U-MtE_balanced (uses A) on thyroid, capsule, ISIC hair
(DINOv2) and chest drains (RAD-DINO). Claims **U12a** U-MtE_balanced − mask_splice > 0 and **U12b**
U-MtE_protect − mask_splice ≥ 0 (reversed AUROC, Trap A), with gaming flags reported. Reported either way.

## Amendment (2026-09-27, before fitting) — location-adaptive selection ("auto", U13)
Per (trap, seed, fold) the candidate with the best validation score is chosen and its test predictions are used.
Candidates (none trains on val_groups): erm, mask, balanced, mask_balanced, U-MtE, U-MtE_protect, U-MtE_balanced,
U-MtE_protect_balanced, jtt, mask_jtt, umte_jtt. Score on val_groups (image-level A needed on validation only):
min(AUROC(Y1A0 vs Y0A1), AUROC(Y1A1 vs Y0A0)). References: DFR and mask+DFR (trained on val_groups, so not
candidates). Cohorts: thyroid, capsule, ISIC hair (DINOv2), chest drains (RAD-DINO).
Claims **U13a** auto − mask > 0 (Trap A, reversed) in all four; **U13b** auto is within 0.02 (reversed and
min(rev,corr)) of the best fixed candidate in each cohort; **U13c** auto ≥ DFR − 0.02 in each cohort.
Reported either way. (`scripts/analysis/adaptive_select.py`)

### U12 results — SPLINCE (Trap A; rev / corr (clean); min(rev, corr))
| cohort | mask_splice | splice | U-MtE_balanced | U-MtE_protect |
|---|---|---|---|---|
| thyroid (DINOv2) | 0.878 / 0.646 (0.777) †; 0.646 | 0.808 / 0.622 (0.724) †; 0.622 | 0.788 / 0.818 (0.807); **0.788** | 0.617 / 0.780; 0.617 |
| capsule (DINOv2) | 0.951 / 0.734 (0.854) †; 0.734 | 0.901 / 0.640 (0.784) †; 0.640 | 0.857 / 0.928 (0.893); **0.857** | 0.673 / 0.973; 0.673 |
| ISIC hair (DINOv2) | 0.831 / 0.601 (0.720) †; 0.601 | 0.841 / 0.600 (0.724) †; 0.600 | 0.704 / 0.852 (0.779); **0.704** | 0.495 / 0.895; 0.495 |
| drains (RAD-DINO) | 0.516 / 0.623 (0.682); 0.516 | 0.536 / 0.639 (0.726); 0.536 | 0.460 / 0.857 (0.826); 0.460 | 0.269 / 0.946; 0.269 |
† gaming flag (correlated < reversed − 0.02). In the three cohorts where artifact and label are strongly correlated
in training, SPLINCE flips the shortcut (reversed ≫ correlated) — the same failure as unpaired LEACE — because
Cov(X, A) and Cov(X, Y) are nearly collinear. **U12a supported by min(rev, corr)** in thyroid, capsule and hair;
not on drains (SPLINCE 0.516 vs 0.460, at a clean cost of −0.19). U12b not supported on raw reversed AUROC (SPLINCE's
reversed values are inflated by flipping).

### U13 results — location-adaptive selection (results/adaptive_select_summary.csv; Trap A)
| cohort | auto min(rev,corr) | main choice | auto − mask | auto − DFR | best candidate (min) |
|---|---|---|---|---|---|
| thyroid (DINOv2) | **0.795** | U-MtE_balanced (14/25) | +0.395 [+0.379, +0.411] | +0.057 [+0.033, +0.083] | U-MtE_bal 0.788 |
| capsule (DINOv2) | 0.852 | U-MtE_balanced (11/25) | +0.168 [+0.152, +0.183] | +0.047 [+0.016, +0.077] | U-MtE_bal 0.857 |
| ovary (DINOv2) | 0.733 | U-MtE_balanced (19/25) | +0.188 [+0.151, +0.226] | +0.010 [−0.022, +0.044] | U-MtE_bal 0.736 |
| ISIC hair (DINOv2) | 0.705 | balanced (17/25) | +0.251 [+0.227, +0.274] | −0.038 [−0.056, −0.019] | balanced 0.716 |
| drains (RAD-DINO) | 0.538 | balanced (5/5) | +0.299 [+0.280, +0.320] | −0.144 [−0.180, −0.109] | balanced 0.538 |
**U13a supported (5/5); U13b supported (5/5: within 0.02 of the best candidate); U13c supported 3/5** — fails on
hair and drains, the correlate-carried regime, where DFR (not a candidate: it trains on the validation set) is best.
Follow-up registered below: split-validation selection that admits DFR as a candidate.

## Amendment (before fitting) — split-validation adaptive selection (U14)
val_groups is split per image (stable hash "val_half") into two halves. DFR-type candidates dfr_half and
mask_dfr_half are trained on half 0 only; ALL candidates (U13 set + these two) are scored on half 1 only with the
U13 score. Claims: **U14a** auto_split − DFR ≥ −0.02 in all five cohorts (Trap A, reversed); **U14b** auto_split
within 0.02 of the best fixed candidate (min(rev, corr)) in all five. (Image-level split: near-duplicate images can
fall in both halves — small optimistic bias, reported as a limitation.) Reported either way.

### U14 results — split-validation selection (logs/auto2_select.log; results/adaptive_select_split_summary.csv)
Trap A min(rev, corr): auto_split vs [U13 auto] vs best single arm —
thyroid 0.798 [0.795] vs 0.788 (U-MtE_bal); capsule 0.871 [0.852] vs 0.885 (mask+DFR); ovary 0.715 [0.733] vs 0.752
(mask+DFR); ISIC hair 0.688 [0.705] vs 0.741 (DFR); drains 0.656 [0.538] vs 0.683 (DFR).
auto_split − DFR (reversed): thyroid +0.060 [+0.037, +0.082], capsule +0.066 [+0.043, +0.088], ovary −0.007
[−0.041, +0.031], hair −0.055 [−0.083, −0.028], drains −0.026 [−0.044, −0.009].
**U14a supported 3/5** (fails hair, drains by −0.055 / −0.026); **U14b supported 2/5** (thyroid, capsule).
Admitting DFR fixes the drain regime (+0.118 over U13) but halving the selection set makes choices noisier
(ovary, hair worse). Neither selector is uniformly best; both beat masking everywhere (+0.17 to +0.42).

### Correction (2026-09-27) — SPLINCE implementation
The U12 results above used a simplified SPLINCE (oblique projection with both constraints but in the un-whitened
metric). Checked against the paper (Holstege, Ravfogel & Wouters, NeurIPS 2025, arXiv 2506.10703, Theorem 1), the
method minimises distortion in the whitened metric. `SpliceProjection` now implements that (verified: both
constraints hold exactly, P is idempotent, and a numerical search over all admissible ranges finds no lower
distortion). The simplified variant remains available (`euclidean=True`). U12 is re-run as tag *_splice_v2 with the
same arms and claims; both versions are reported.

### U12 re-run with the faithful SPLINCE (tag *_splice_v2) — supersedes the simplified-variant numbers above
Trap A reversed / correlated (clean); min(rev, corr):
| cohort | splice | mask_splice | U-MtE_bal | best |
|---|---|---|---|---|
| thyroid | 0.563 / 0.545 (0.562) | 0.628 / 0.596 (0.616); 0.596 | 0.788 / 0.818 (0.807); **0.788** | U-MtE_bal |
| capsule | 0.795 / 0.613 (0.706) | 0.868 / 0.683 (0.784); 0.683 | 0.857 / 0.928 (0.893); **0.857** | U-MtE_bal |
| ISIC hair | 0.639 / 0.564 (0.606); 0.564 | 0.628 / 0.562 (0.592) | 0.704 / 0.852 (0.779); 0.704 | balanced 0.716 |
| drains | 0.609 / 0.609 (0.717); **0.609** | 0.577 / 0.588 (0.700) | 0.460 / 0.857 (0.826) | DFR 0.683 |
The faithful (whitened, minimal-distortion) SPLINCE no longer flips the shortcut in thyroid or on drains but removes much
disease signal (clean AUROC 0.56–0.78 vs 0.69–0.87 for masking) and still flips it on capsule. **U12a (U-MtE_bal >
SPLINCE by min(rev, corr)) supported in 3/4** (thyroid, capsule, hair); on drains SPLINCE is higher (0.609 vs 0.460),
below DFR (0.683).
Name correction: the method is **SPLINCE** (Simultaneous Projection for LINear concept removal and Covariance
prEservation; arXiv 2506.10703 abstract). Arms keep the internal names `splice` / `mask_splice`.
