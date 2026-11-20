# P4 second-tier comparison (A40; DESCRIPTIVE) — our arms vs our A31 Seg-ERM run of Saab's official code — machine 2

P4 is not reproduced (Seg-ERM 0.647 vs 0.727; A38). Robust AUROC on the 772-image trap tests, trap sets 1–3; image-only bootstrap with one Poisson weight per image shared across sets and methods, sets fixed; Holm over 4. No claim against 0.727.

| arm | set 1 | set 2 | set 3 | 3-set mean | Seg-ERM mean | Δ [95 % CI] | p Holm |
|---|---|---|---|---|---|---|---|
| a_segerm_mask_cmc | 0.530 | 0.570 | 0.554 | 0.552 | 0.647 | -0.095 [-0.141, -0.049] | 0.0004 |
| b_dino518_mask_cmc | 0.674 | 0.679 | 0.657 | 0.670 | 0.647 | +0.023 [-0.029, +0.076] | 0.8802 |
| c_umte | 0.625 | 0.633 | 0.613 | 0.623 | 0.647 | -0.023 [-0.075, +0.029] | 0.8802 |
| d_effb0_mask_cmc | 0.630 | 0.639 | 0.594 | 0.621 | 0.647 | -0.026 [-0.073, +0.022] | 0.8802 |
