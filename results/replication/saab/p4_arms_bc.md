# P4 (Saab, ISIC, A13): our arms b and c on Bissoto trap sets 1-3

Robust AUROC on the 772-image trap test per split; image-only paired bootstrap per split (one network per split). Descriptive until the P4 gate (A13). Validation (259 images) lies inside training (Bissoto's trap sets, A13), so validation-based choices say little.

| split | arm | mode | C | robust AUROC | Δ Binary-ERM [95% CI] |
|---|---|---|---|---|---|
| 1 | b_dino518_mask_cmc | pooled | 1 | 0.659 | +0.305 [+0.246, +0.365] |
| 1 | c_umte | k = 64 | 1 | 0.611 | +0.256 [+0.195, +0.319] |
| 2 | b_dino518_mask_cmc | pooled | 10 | 0.660 | +0.298 [+0.237, +0.358] |
| 2 | c_umte | k = 64 | 10 | 0.606 | +0.245 [+0.183, +0.307] |
| 3 | b_dino518_mask_cmc | pooled | 1 | 0.654 | +0.281 [+0.222, +0.340] |
| 3 | c_umte | k = 64 | 1 | 0.599 | +0.225 [+0.167, +0.286] |

Mean over splits: b_dino518_mask_cmc 0.658 ± 0.003, c_umte 0.605 ± 0.006
