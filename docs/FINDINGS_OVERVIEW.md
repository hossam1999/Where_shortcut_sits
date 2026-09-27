# Findings overview (living document; updated 2026-09-27)

Every number below is from a pre-registered analysis (docs/PREREGISTRATION_*.md) with 95 % hierarchical paired
bootstrap CIs; full tables in results/CROSS_COHORT.md and results/SUMMARY.md. "Hard test" = shortcut-reversed
test set (reversed AUROC). Trap A = artifact inside the ROI, Trap B = outside.

## 1. Replication of the original pilot (the PDF)
73 of 88 PDF numbers MATCH (same sign, same CI verdict, |Δ| ≤ 0.02), 12 agree in sign and significance,
3 mismatch (E4 occlusion borderline CI, E12 contrast trap not null, one HAM-only stratum), 0 pending.

## 2. The phenomenon: does ROI masking depend on where the artifact sits?
| cohort | artifact | masking gain, out-of-ROI | masking gain, in-ROI | crossover (B − A) |
|---|---|---|---|---|
| ISIC 2019 dermoscopy (real) | hair | + | −0.037 [−0.057, −0.019] (harm) | +0.145 [+0.105, +0.181] |
| Thyroid US (real, DINOv2) | calipers | +0.341 | +0.111 | +0.230 [+0.204, +0.255] |
| Thyroid US (real, MedSigLIP) | calipers | +0.228 | +0.064 | +0.163 [+0.132, +0.194] |
| Capsule endoscopy (real) | debris | +0.465 | +0.097 | +0.368 [+0.340, +0.397] |
| ISIC 2018 (controlled, DINOv2 / DermLIP) | ruler | + (r=0) | harm (r ≥ 0.25) | sign reversal |
| Thyroid US (real, ConvNeXt CNN) | calipers | +0.288 | +0.003 | +0.284 [+0.253, +0.315] |
| Capsule endoscopy (real, MedSigLIP) | debris | +0.441 | +0.109 | +0.332 [+0.296, +0.365] |
| Capsule endoscopy (real, ConvNeXt CNN) | debris | + | +0.094 | +0.292 [+0.248, +0.337] |
| Capsule (controlled) | debris | +0.103 (r=0) | −0.102 (r=1) | +0.205 [+0.150, +0.259] |
| Thyroid (controlled) | calipers | +0.402 (r=0) | −0.030 (r=1, n.s.) | +0.432 [+0.377, +0.492] |
| Chest X-ray (controlled, RAD-DINO) | tube | −0.002 (not used by ERM) | +0.110 | −0.113 (no reversal) |
Robust claim: masking leaves in-ROI shortcuts largely intact (it removes most of an out-of-ROI shortcut); it is
actively harmful in dermoscopy and in the controlled capsule sweep, not in real thyroid/capsule or in CXR.

## 3. The solution: artifact-agnostic mask-then-erase (U-MtE)
No artifact example, mask or label; procedural overlays → insertion pairs on masked images → erase subspace.
| cohort (Trap A / r=1) | U-MtE − mask | erase − augment (U7) |
|---|---|---|
| Thyroid DINOv2 | +0.222 [+0.197, +0.251] | +0.207 [+0.182, +0.235] |
| Thyroid MedSigLIP | +0.204 [+0.174, +0.237] | +0.177 [+0.148, +0.211] |
| ISIC hair DINOv2 | +0.052 [+0.044, +0.062] | +0.045 [+0.037, +0.054] |
| Thyroid ConvNeXt (CNN) | +0.128 [+0.112, +0.145] | +0.120 [+0.102, +0.139] |
| Thyroid (controlled, r=1) | +0.278 [+0.232, +0.325] | – |
| Capsule (controlled) | +0.081 [+0.050, +0.112] | – |
| Chest X-ray (controlled) | +0.134 [+0.120, +0.151] | – |
| Capsule (real) | −0.115 (fails: debris ≈ fibrin) → protected −0.010 | – |
With image-level artifact labels, U-MtE + group balancing is the most robust model (min of reversed/correlated
AUROC) in thyroid (0.788 DINOv2, 0.792 MedSigLIP); in real capsule masking + DFR is higher (0.885 vs 0.857); on
dermoscopic hair plain DFR is best (0.741) — the hair shortcut is mostly carried by site/age/sex correlates (oracle
removal of every hair patch gains only +0.015).

## 4. Other solution attempts (reported whichever way)
- SLAS (few-shot patch-token artifact suppression): localises hair from 5 annotated images (patch AUROC 0.93),
  but downstream gains are small (+0.02 hair, +0.08 thyroid). Kept as a localisation tool.
- Disease-protected erasure (U9): safety net — removes the capsule failure, costs ≤ 0.012 elsewhere.
- Annotation-free pseudo-group balancing (U8): +0.04 on thyroid, none on hair.
- Erasure + DFR (U10): no gain over masking + DFR.
- JTT label-free baseline (U11): U-MtE beats JTT on thyroid (+0.198); U-MtE+JTT best on capsule (0.771);
  plain JTT best on hair (0.564). The winner follows what carries the shortcut.

## 5. Datasets evaluated and not used (with reasons)
CANDID-PTX (no access), CXR CLiP real devices (masking helps both locations), CXR cardiomegaly (too few labelled),
TCGA/GrandQC pathology (unreliable pen masks), ISIC ink (70 in-lesion images), breast US BUSI/BUS-BRA (marker
labels ≈ 70 % pure), EAD endoscopy (no disease labels), BKAI-IGH NeoPolyp colonoscopy (specular highlights: 0 specular-free images, 41 in-polyp).

## 6. Tools
`python -m wtss.umte` (U-MtE) and `python -m wtss.slas` (SLAS): any backbone (ViT or CNN), two commands.

## 7. Latest (2026-09-27, 03:00–05:30 UTC)
- Real chest drains (NIH pneumothorax): extreme shortcut (ERM reversed 0.19 RAD-DINO, 0.28 DINOv2); lung masking +0.05 / +0.01;
  U-MtE +0.03 over masking with RAD-DINO, none with DINOv2; DFR / mask+DFR best. Drain = treated lung → correlate-carried regime.
- MedSigLIP controlled sweeps (capsule, thyroid): location interaction replicates (+0.19, +0.47); masking does not
  harm at full overlap; unbalanced U-MtE loses to masking (−0.11, −0.08); U-MtE_balanced beats masking (+0.31,
  +0.46) and is location-invariant.
- Disease protection in sweeps: fixes MedSigLIP capsule (+0.061 over masking), helps DINOv2 capsule (+0.118), does
  not fix MedSigLIP calipers (−0.041).
- Colonoscopy (BKAI NeoPolyp, specular): infeasible (no specular-free images).

## 8. Recommendation for practitioners (paper's decision rule)
1. Measure where artifacts sit relative to the ROI. Out-of-ROI → mask.
2. In-ROI, no artifact labels → protected U-MtE (DINOv2/CNN backbones), or JTT when the shortcut is carried by
   correlates (check with the oracle-removal ceiling on a few annotated images, e.g. with SLAS).
3. In-ROI, image-level artifact labels available → U-MtE + balanced (overlay-like artifacts), masking + DFR
   (artifacts resembling the disease), DFR / balancing (correlate-carried shortcuts).
