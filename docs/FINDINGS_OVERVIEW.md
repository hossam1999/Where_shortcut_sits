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

## 9. Session 2 additions (2026-09-27, 06:00–09:30 UTC)
- **4th disease — ovarian tumour ultrasound (MMOTU, calipers)**: real crossover +0.170 [+0.116, +0.224]; controlled
  sweep: masking HARMS at full overlap (−0.197 [−0.292, −0.100]), U-MtE +0.141 over masking. Generic real U-MtE
  +0.019 (n.s.), protected +0.043 [+0.025, +0.061].
- **SPLINCE baseline** (closest published erasure): flips the shortcut in thyroid, capsule and hair (reversed ≫
  correlated); by min(rev, corr) U-MtE_balanced wins in all three; on drains SPLINCE 0.516 vs 0.460 at −0.19 clean.
- **Location-adaptive selection (auto)**: beats masking in 5/5 cohorts (+0.17 to +0.40), within 0.02 of the best
  candidate in 5/5, beats DFR in 3/5 (fails on hair and drains = correlate-carried regime). Split-validation
  variant that admits DFR as a candidate: running (U14).
- **Fine-tuning (ResNet-50, end-to-end)**: thyroid — end-to-end U-MtE +0.143 [+0.099, +0.188] over masking and
  joint-best robust model without labels; capsule — masking HARMS in-ROI (−0.284) with a crossover of +0.584
  [+0.469, +0.691], end-to-end U-MtE fails there (debris ≈ disease).
- Git: 76 commits had silently failed to push (branch name mismatch); fixed and verified.

## 10. Session 3 additions (2026-09-27, 09:30–11:30 UTC) — reviewer-proofing
- **Statistics**: Holm-corrected primary family — 12/16 supported (P1 location crossover 4/4; protected U-MtE >
  mask 3/4; erase > augment 3/4; U-MtE+bal > balanced 2/4). docs/STATISTICAL_PLAN.md, results/PRIMARY_CLAIMS.md.
- **Natural distribution (thyroid, official split, no resampling)**: masking lowers AUROC on shortcut-conflicting
  cases by −0.102 [−0.139, −0.066]; U-MtE recovers +0.051; balancing +0.142. Protected U-MtE did not help (N1/N2 not
  supported).
- **Artifact-label robustness**: crossover and protected-U-MtE gain hold under large-marker and strict-location
  definitions (thyroid, ovary) and a strict capsule Trap B (+0.259). Visual audit: thyroid detections 95 % real.
- **Theory**: AUROC_rev = Φ((S − Ã)/√(2(S + Ã))); simulation agreement 0.009; explains all regimes (docs/THEORY.md).
- **LaMa inpainting** (oracle masks): strong; protected U-MtE matches it on thyroid, below on ovary; U-MtE+bal beats it.
- **Split-validation selector**: fixes drains (0.656), mixed elsewhere; neither selector uniformly best.
- **Paper**: compiles cleanly (14 pages), new figures (dose-response, forest, examples, theory), two review passes.

## Additions after external review (2026-09-27)
A colleague's review raised seven points; what changed:
- **Wording.** U-MtE "needs no artifact example, artifact mask or artifact label" — it does use the ROI mask (now said
  everywhere). Table 1's "I2E" rows renamed "Insert-then-erase (artifact templates)" and explained as U-MtE's
  artifact-specific precursor. GroupDRO is stated to appear only in the controlled dermoscopy sweeps. Chest
  radiography is named as a boundary case. U-MtE is framed as a representation-level intervention for frozen
  foundation-model features; end-to-end results are secondary. The "clean" environment is now described correctly
  (real traps: artifact in half of each class; sweeps: absent).
- **Theory vs real data** (pre-registered, `PREREGISTRATION_THEORY_PREDICTION.md`): fitted per model to clean and
  correlated AUROC only, the closed form matches held-out reversed AUROC (MAE 0.039, r = 0.92, 148 models) and the
  location crossover (r = 0.997, MAE 0.014, 13/14 signs); verdict by the pre-registered rule: qualitative account
  (one near-zero sign miss). Fails for chest drains (correlate-carried shortcut).
- **Leakage** (`PREREGISTRATION_EMBEDDING_GROUPS.md`): stricter DINOv2 near-duplicate groups; 11/12 primary contrasts
  unchanged; ovary erase > augment loses significance.
- **Clinical metrics** (`PREREGISTRATION_NATURAL.md`): AUPRC, Brier, ECE, sensitivity/specificity; masking halves
  thyroid sensitivity at the validation-fixed operating point.
- **Related work**: six verified references added (Pewton 2024; Wang 2024; Germani 2026 MedIA; Lin 2024 MICCAI;
  Zech 2018; Brown 2023).
- **Integrity**: one archived prediction file (thyroid natural) failed a gzip CRC check; regenerated, identical numbers.
- **Model scale** (`PREREGISTRATION_SCALE.md`): DINOv2 ViT-S/B/L — location law 9/9; protected U-MtE > mask on
  thyroid and ovary at every size; larger models exploit in-ROI calipers more (ERM gap thyroid 0.547 → 0.621).
- **U-MtE rank ablation** (`PREREGISTRATION_UMTE_ABLATION.md`): thyroid gain +0.18 to +0.22 for ≥ 16 erased
  directions; protected variant safe on capsule at every rank.
- **Fine-tune, then erase** (`PREREGISTRATION_FT_ERASE.md`): null (+0.004) — U-MtE is for frozen representations.
- **Mechanism figure** (`paper/figures/mechanism.pdf`): masking increases counterfactual reliance on an in-ROI
  artifact in every controlled sweep; U-MtE and balancing remove it.

## Second external review (2026-09-27, evening) — docs/REVIEW2_RESPONSE.md, docs/PREREGISTRATION_REVIEW2.md
- **Verification**: every factual claim of the review matched the result files.
- **Rebuild** on new hardware: all four primary crossovers within 0.01 of the archived values.
- **Confounding** (point 1): Trap A/B populations differ (lesion-area SMD: hair 2.13, capsule 2.57, thyroid 0.40).
  *Real-artifact transplant* (same images, same real artifact, in vs out of ROI): interaction hair +0.253, thyroid
  +0.277, ovary +0.074, capsule +0.497 (4/4 Holm); masking harms with real hair (−0.075) and debris (−0.175) in-ROI.
  *Matched traps*: hair +0.154, thyroid +0.238, ovary +0.163; capsule infeasible (19 pairs).
- **Operating points** (thyroid official, patient-disjoint split): sensitivity lower with masking at 5/5 thresholds;
  conflicting malignant nodules −0.372 at validation specificity 0.80.
- **Paper restructured**: law → mechanism → theory → unaltered data → decision guide; U-MtE to the supplement; theory
  led by held-out MAE (0.039 vs 0.108 heuristic), not r = 0.997; DermLIP explained; CLAIM 2024 checklist; registry.
- **Open**: expert audit (kit ready), clinical co-author, OSF/Zenodo deposit, external cohort with patient IDs
  (ThyUS2Path needs nodule masks and a re-validated caliper detector).
