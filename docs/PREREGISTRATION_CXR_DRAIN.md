# Pre-registration — real chest-drain trap (NIH pneumothorax), current arm set

Committed before any model is fitted with this runner (the drain trap was built earlier but never run).
- Cohort (wtss.data.cxr_drain, frozen): 29,687 NIH ChestX-ray14 images; y = pneumothorax; A = chest drain
  (NEATX human labels for PTX+; RAD-DINO drain detector at precision 0.95 / below the median negative score for
  PTX−); view (AP/PA) as the matching source; patient-grouped 5 folds; train 90/10, correlated 90/10,
  reversed 10/90, clean = drain-free. Drains lie inside the lung field: this is an in-ROI trap only (ROI = lungs,
  PSPNet). No pixel drain masks exist, so no inpainting arm.
- Folds act as the bootstrap clusters (5), as in the original run_traps design.
- Backbone: RAD-DINO @518 (chest-X-ray foundation model). Generic overlay library (unchanged).
- Arms: erm, mask, balanced, dfr, U-MtE (mte), mte_balanced, mte_aug, mte_protect(+balanced), mask_dfr, jtt, mask_jtt.
- Claims (reversed AUROC, 95 % CI): **D1** mask − ERM ≤ small (in-ROI; reported with CI, no direction claimed);
  **D2** U-MtE − mask > 0; **D3** U-MtE − mte_aug > 0; **D4** U-MtE_protect − mask > 0;
  **D5** U-MtE − JTT > 0 (label-free comparison). Reported whichever way they go.

## Results — RAD-DINO@518 (results/cxr_drain/raddino518_universal)
Reversed (clean) AUROC: ERM 0.191 (0.886) — an extreme shortcut; mask 0.239; U-MtE 0.268; U-MtE_protect 0.269
(0.867); JTT 0.175; balanced 0.538; DFR 0.683 (corr 0.738); mask+DFR 0.646.
- D1: mask − ERM = +0.049 [+0.035, +0.061] (lung masking barely helps; drains lie in the lungs).
- **D2 supported (small)**: U-MtE − mask = +0.029 [+0.021, +0.036]. **D3 supported**: U-MtE − mte_aug = +0.031
  [+0.022, +0.042]. **D4 supported**: U-MtE_protect − mask = +0.030 [+0.022, +0.037] at no clean cost
  (−0.003 [−0.009, +0.002]). **D5 supported**: U-MtE − JTT = +0.094 [+0.077, +0.109].
- With artifact labels, DFR is best (0.683); U-MtE_balanced is worse than balanced (−0.079).
Interpretation: a real drain co-occurs with a treated (re-expanded) pneumothorax, so the shortcut is not only the
tube's pixels; pixel/insertion-based repair recovers little (+0.03), label-based group methods much more.

## Amendment (before fitting): DINOv2@518 replication of the drain trap — same arms and claims D1–D5.

## Results — DINOv2@518 replication (results/cxr_drain/dino518_universal)
Reversed: ERM 0.280, mask 0.292, U-MtE 0.285, U-MtE_protect 0.281, JTT 0.294, DFR 0.549, mask+DFR 0.566 (best),
U-MtE_balanced 0.443 (vs balanced 0.399: +0.044 [+0.025, +0.064]).
D1 mask − ERM +0.012 [+0.002, +0.023]; **D2 not supported** (U-MtE − mask −0.008 [−0.015, +0.000]); **D3 not
supported** (−0.010); **D4 not supported** (−0.011); **D5 not supported** (−0.010). With RAD-DINO D2–D5 held but
with small effects (+0.03): on real drains, insertion-based erasure does not transfer across backbones; group
methods with drain labels are required.
