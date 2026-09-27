# Pre-registration — natural-distribution evaluation (no trap resampling)

Committed before any model is fitted for this analysis.
- Thyroid (TN3K + TNCD): train on the official trainval split with its natural association between in-ROI calipers
  and the label (in-ROI calipers: 37.6 % of benign vs 24.1 % of malignant images); test on the untouched official
  test split (614 images). 5 seeds of a group-safe 80/20 train/validation split = bootstrap clusters.
- A (for group-using arms) = in-ROI caliper (>= 15 px, r >= 0.5).
- Metrics on the natural test set: AUROC; cross-group AUROC (min over the two counter-/pro-shortcut pairings of
  groups; the natural analogue of min(reversed, correlated)); AUROC within the in-ROI-caliper subgroup.
- Arms: erm, mask, balanced, mask_balanced, U-MtE, U-MtE_protect, U-MtE_balanced, U-MtE_protect_balanced, DFR,
  mask+DFR, JTT (DINOv2@518, generic library).
- Claims: **N1** cross-group AUROC(U-MtE_protect) > cross-group AUROC(mask) and **N2** overall AUROC(U-MtE_protect)
  ≥ AUROC(mask) − 0.01 (no cost on the natural distribution). Reported whichever way they go.
