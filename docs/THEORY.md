# A linear-Gaussian account of location-dependent mitigation

Numerically verified in `scripts/analysis/theory_sim.py` (max |simulation − theory| = 0.009 AUROC over 29 settings;
results/theory/theory_sim.csv; figure paper/figures/theory.pdf).

## Setup
A frozen encoder maps an image to z = (z_s, z_a, ε): a disease block whose class-mean difference has signal-to-noise
ratio **S**, an artifact block with SNR **A** per unit of artifact presence, and independent noise. Training uses
P(A=1 | Y=1) = p1 = 0.9, P(A=1 | Y=0) = p0 = 0.1; the reversed test swaps them. A linear head fitted to the training
distribution (LDA-optimal; logistic regression approximates it) weights each block by its training SNR. The
artifact's *effective* training SNR is

  Ã = (p1 − p0)² A / (1 + p1(1 − p1) A),

and the reversed-environment AUROC is

  **AUROC_rev(S, Ã) = Φ( (S − Ã) / √(2 (S + Ã)) )**,

increasing in S and decreasing in Ã. (Without the artifact, Ã = 0 and AUROC = Φ(√(S/2)).)

## Consequences (each matches an empirical result)
1. **Out-of-ROI artifacts.** Masking removes the artifact (Ã → 0) and changes the disease SNR to S_m:
   AUROC_rev = Φ(√(S_m/2)). It helps whenever S_m > 2·[Φ⁻¹(AUROC_rev^ERM)]², i.e. unless the mask destroys most of the
   disease signal. → masking is the strongest single fix out of the ROI (all cohorts, Trap B).
2. **In-ROI artifacts.** Masking keeps the artifact (Ã unchanged, or larger when the artifact becomes a larger share of
   what remains) and changes S to S_m. Because AUROC_rev is increasing in S, masking **harms iff it removes
   disease-informative context (S_m < S)** and helps only through denoising (S_m > S). → harm in dermoscopy (lesion
   surroundings are informative) and in the controlled sweeps (the synthetic artifact is fully retained), small gains in
   real thyroid/ovary/capsule (masking removes clutter).
3. **The crossover is always positive.** For the same S_m, Φ(√(S_m/2)) − Φ((S_m − Ã)/√(2(S_m + Ã))) > 0 for every Ã > 0:
   masking is always worth more for an out-of-ROI artifact than for an in-ROI one. → P1 supported in all four cohorts.
4. **Erasure.** Projecting out a subspace that contains the artifact direction sets Ã → 0 but scales the disease SNR by
   (1 − ρ²), ρ = overlap between the erased subspace and the disease direction: AUROC_rev = Φ(√(S_m(1 − ρ²)/2)).
   Erasure beats in-ROI masking unless ρ is large. → U-MtE works for calipers (ρ small) and fails for debris that
   resembles erosion fibrin (ρ large); **disease protection** forces ρ → 0 along the protected label directions.
5. **Group balancing** removes the training association between artifact and label (effective p1 = p0 ⇒ Ã = 0) for the
   artifact *and everything correlated with it*: AUROC_rev = Φ(√(S/2)), independent of where the artifact sits.
   → balanced / U-MtE_balanced are location-invariant in every controlled sweep.
6. **Correlate-carried shortcuts.** If part of the shortcut is carried by features that neither masking nor pixel-based
   erasure touch (hair ↔ body site/age/sex; drain ↔ treated lung), a residual Ã_c remains after masking/erasure and only
   label-based reweighting removes it. → DFR/balancing win on hair and drains.
7. **Augmentation vs erasure.** Overlay augmentation only decorrelates the label from the *generated* overlay direction;
   the real artifact direction u_a keeps its association unless u_a lies in that direction. Erasure removes the whole
   subspace spanned by overlay-induced feature changes, which contains u_a whenever overlays and the real artifact move
   the representation along shared "foreign-overlay" directions. → erase ≫ augment (P3).

## Limits of the model
Linear heads on fixed features; Gaussian class-conditional features; the mask is modelled only through (S_m, Ã_m).
It predicts signs and orderings, not the empirical magnitudes of real encoders.
