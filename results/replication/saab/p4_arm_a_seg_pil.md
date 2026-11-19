# P4 (Saab, ISIC, A13 item 8a): arm a = mask_cmc on their Segmentation-ERM encoder (machine 2, A27)

Lesion-masked images through our reproduced Seg-ERM's SE-ResNeXt50 encoder (best validation-Dice epoch), pooled 2,048-d features, mask_cmc + class-balanced logistic regression (C grid {0.01, 0.1, 1, 10}, mode and C by validation AUROC). Robust AUROC on the 772-image trap test, scored once per split; image-only paired bootstrap vs Seg-ERM of the same split (same test images). Descriptive until the P4 gate (A13). Validation (259 images) lies inside training (Bissoto's trap sets, A13), so validation-based choices say little.

| split | best epoch | mode | C (edge) | robust AUROC | Seg-ERM | Δ Seg-ERM [95% CI] | Δ Binary-ERM [95% CI] |
|---|---|---|---|---|---|---|---|
| 1 | 146 | pooled | 10 (highest) | 0.530 | 0.626 | -0.096 [-0.159, -0.031] | +0.175 [+0.105, +0.247] |
| 2 | 135 | pooled | 10 (highest) | 0.570 | 0.673 | -0.103 [-0.162, -0.044] | +0.209 [+0.144, +0.273] |
| 3 | 131 | pooled | 10 (highest) | 0.554 | 0.641 | -0.087 [-0.153, -0.021] | +0.181 [+0.109, +0.251] |

Mean over 3 splits: arm a 0.552 ± 0.020, Seg-ERM 0.647 ± 0.024
