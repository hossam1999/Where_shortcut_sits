# P4 (Saab, ISIC, A13 item 8a): arm a = mask_cmc on their Segmentation-ERM encoder (machine 2, A27)

Lesion-masked images through our reproduced Seg-ERM's SE-ResNeXt50 encoder (best validation-Dice epoch), pooled 2,048-d features, mask_cmc + class-balanced logistic regression (C grid {0.01, 0.1, 1, 10}, mode and C by validation AUROC). Robust AUROC on the 772-image trap test, scored once per split; image-only paired bootstrap vs Seg-ERM of the same split (same test images). Descriptive until the P4 gate (A13). Validation (259 images) lies inside training (Bissoto's trap sets, A13), so validation-based choices say little.

| split | best epoch | mode | C (edge) | robust AUROC | Seg-ERM | Δ Seg-ERM [95% CI] | Δ Binary-ERM [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 125 | pooled | 10 (highest) | 0.528 | 0.609 | -0.081 [-0.140, -0.022] | +0.174 [+0.108, +0.240] |
