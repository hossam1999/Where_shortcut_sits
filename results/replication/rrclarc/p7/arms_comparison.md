# P7 (RR-ClArC, A28): our arms vs our reproduced RR-ClArC — machine 2

Biased test = LSB trigger in every image of every class (Table 1 'biased'); accuracy; image-only paired bootstrap (10,000), Holm over the 9 comparisons; 'beaten' = higher accuracy, 95 % CI above 0, Holm p < 0.05 (section 2); VGG-16 descriptive (RR-ClArC not reproduced). Clean drop vs RR-ClArC > 0.01 flagged.

| arch | arm | C (edge) | biased acc | RR-ClArC biased | Δ [95 % CI] | p Holm | clean acc (Δ vs RR-ClArC) | verdict |
|---|---|---|---|---|---|---|---|---|
| resnet18 | a_vanilla_mask_cmc | 0.003  | 0.709 | 0.713 | -0.004 [-0.019, +0.012] | 1.0000 | 0.812 (+0.032) | not beaten |
| resnet18 | a2_rrclarc_mask_cmc | 0.003  | 0.709 | 0.713 | -0.004 [-0.019, +0.012] | 1.0000 | 0.812 (+0.032) | not beaten |
| resnet18 | b_dino224_mask_cmc | 10 (highest) | 0.614 | 0.713 | -0.099 [-0.122, -0.078] | 0.0009 | 0.672 (-0.108) ⚠ | not beaten |
| efficientnet_b0 | a_vanilla_mask_cmc | 0.001 (lowest) | 0.712 | 0.728 | -0.017 [-0.028, -0.005] | 0.0128 | 0.833 (+0.005) | not beaten |
| efficientnet_b0 | a2_rrclarc_mask_cmc | 0.001 (lowest) | 0.712 | 0.728 | -0.017 [-0.028, -0.005] | 0.0128 | 0.833 (+0.005) | not beaten |
| efficientnet_b0 | b_dino224_mask_cmc | 10 (highest) | 0.668 | 0.728 | -0.060 [-0.082, -0.038] | 0.0009 | 0.673 (-0.156) ⚠ | not beaten |
| vgg16 | a_vanilla_mask_cmc | 1  | 0.571 | 0.531 | +0.039 [+0.022, +0.057] | 0.0009 | 0.734 (-0.015) ⚠ | descriptive (RR-ClArC not reproduced) |
| vgg16 | a2_rrclarc_mask_cmc | 1  | 0.571 | 0.531 | +0.039 [+0.022, +0.057] | 0.0009 | 0.734 (-0.015) ⚠ | descriptive (RR-ClArC not reproduced) |
| vgg16 | b_dino224_mask_cmc | 10 (highest) | 0.616 | 0.531 | +0.085 [+0.062, +0.108] | 0.0009 | 0.671 (-0.078) ⚠ | descriptive (RR-ClArC not reproduced) |
