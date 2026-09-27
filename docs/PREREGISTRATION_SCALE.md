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

## Results (results/scale/scale_compare.csv; ViT-B = main runs)
| cohort | size | crossover [95 % CI] | U-MtE_protect − mask (Trap A) | U-MtE − mask | ERM gap corr − rev (Trap A) |
|---|---|---|---|---|---|
| thyroid | S | +0.244 [+0.209, +0.275] | +0.181 [+0.146, +0.218] | +0.182 | 0.547 |
| thyroid | B | +0.230 [+0.204, +0.255] | +0.217 [+0.174, +0.257] | +0.222 | 0.590 |
| thyroid | L | +0.271 [+0.234, +0.308] | +0.222 [+0.195, +0.256] | +0.213 | 0.621 |
| ovary | S | +0.298 [+0.229, +0.365] | +0.103 [+0.078, +0.128] | +0.120 | 0.319 |
| ovary | B | +0.170 [+0.116, +0.224] | +0.043 [+0.026, +0.061] | +0.019 | 0.409 |
| ovary | L | +0.202 [+0.153, +0.250] | +0.060 [+0.043, +0.077] | +0.094 | 0.570 |
| capsule | S | +0.370 [+0.341, +0.401] | −0.008 [−0.014, −0.003] | −0.153 | 0.353 |
| capsule | B | +0.368 [+0.340, +0.397] | −0.010 [−0.021, −0.001] | −0.115 | 0.354 |
| capsule | L | +0.408 [+0.378, +0.438] | −0.040 [−0.046, −0.035] | −0.093 | 0.325 |

- **S1 supported 6/6** (and 9/9 with ViT-B): the location law does not depend on model size.
- **S2 supported 4/4**: protected U-MtE beats masking on thyroid and ovary with ViT-S and ViT-L.
- **S3**: larger models exploit the in-ROI caliper shortcut *more* (ERM gap thyroid 0.547 → 0.590 → 0.621; ovary
  0.319 → 0.409 → 0.570); capsule is flat. Scale does not remove the shortcut. On capsule, disease protection is less
  complete with ViT-L (−0.040 vs masking).
