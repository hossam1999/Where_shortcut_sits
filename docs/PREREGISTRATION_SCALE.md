# Pre-registration — backbone scale (DINOv2 ViT-S/14 and ViT-L/14)

Committed before any feature is extracted. Question: do the location law and U-MtE depend on the size of the frozen
foundation model? (DINOv3 would be the newest family but its weights are gated and access was refused, 403.)

## Design (frozen)
- Backbones: DINOv2 ViT-S/14 (21 M parameters, 384-d) and ViT-L/14 (304 M, 1024-d) at 518 px, alongside the main
  ViT-B/14 (86 M). Same preprocessing, CLS token, linear heads, seeds, folds and traps.
- Cohorts: thyroid, ovary, capsule (the three cohorts run by `run_thyroid_traps.py`; ISIC is too large for the time
  budget). Arms: erm, mask, balanced, U-MtE (mte), U-MtE_balanced, U-MtE_protect, overlay augmentation (mte_aug);
  generic overlays. `run_thyroid_traps.py --backbone {dinos518,dinol518} --cohort ... --generic --tag scale`.

## Claims (reversed AUROC, 95 % CI)
- **S1** location crossover > 0 for both backbones in all three cohorts (6 tests).
- **S2** U-MtE_protect − mask > 0 (Trap A) in thyroid and ovary (distinct overlays), both backbones.
- **S3** descriptive: does the ERM shortcut gap (correlated − reversed, Trap A) or the crossover grow or shrink with
  scale (S → B → L)? No directional prediction.
Reported whichever way they go.
