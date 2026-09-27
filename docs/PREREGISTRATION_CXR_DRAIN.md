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
