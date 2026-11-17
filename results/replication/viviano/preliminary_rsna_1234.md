# P1 RSNA pneumonia (view shortcut), seed 1234: preliminary head-to-head

**Preliminary, 1 seed, no verdict** (A14: P1's head-to-heads are secondary — global shortcut — and stay at one seed). Paired image-only bootstrap (10,000 replicates); training-seed variance is not included.

Target: **ActDiff** (A11: their RSNA Masked test input is label-dependent; shown, flagged, not a target)

Rows: their methods; `a_mask_cmc_their_net` = **registered arm a** (A11/A12: mask_cmc on the ActDiff network's features of PSPNet-lung-masked images, every split); the 'diagnostic' row is the ActDiff network's own logits on the same lung-masked images (the feature pass that feeds arm a), not a method; b = DINOv2 + mask_cmc; c = U-MtE.

| method | AUROC (their definition) | AUROC | clean: within-site AUROC | paper | Δ vs ActDiff [95% CI] | Δ vs Standard [95% CI] |
|---|---|---|---|---|---|---|
| Standard | 0.186 | 0.186 | 0.654 | 0.20 ± 0.03 | -0.523 [-0.566, -0.479] | — |
| Masked ⚠ label-dependent test input | 0.759 | 0.840 | 0.800 | 0.56 ± 0.16 | +0.132 [+0.092, +0.172] | +0.655 [+0.618, +0.692] |
| ActDiff | 0.708 | 0.708 | 0.471 | 0.68 ± 0.05 | — | +0.523 [+0.479, +0.566] |
| ActDiff net on lung-masked images (diagnostic, not a method) | 0.690 | 0.689 | 0.571 |  | -0.019 [-0.041, +0.003] | +0.504 [+0.464, +0.543] |
| a_mask_cmc_their_net | — | 0.610 | 0.623 |  | -0.099 [-0.132, -0.065] | +0.424 [+0.389, +0.459] |
| b_dino518_mask_cmc | — | 0.767 | 0.763 |  | +0.059 [+0.021, +0.098] | +0.582 [+0.551, +0.613] |
| c_umte | — | 0.378 | 0.723 |  | -0.330 [-0.375, -0.285] | +0.193 [+0.166, +0.220] |
