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
