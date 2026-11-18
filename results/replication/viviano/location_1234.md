# P1 location test (A15/A16): marker inside vs outside the lungs + heart ROI, P1's pipeline, seed 1234

Artifact: a synthetic bar-shaped radio-opaque marker (43 x 6 px at 224 px; straight, bright, with a central stripe), not a realistic tube. Outside placements lie near the image edge: below the lungs + heart near the bottom edge (47.5% of test images), above the lungs near the top edge (38.2%) or beside the ROI (14.3%). Realism is covered elsewhere in the thesis (real device traps on NIH / RANZCR; transplant of real artifacts).

One seed: paired image-only bootstrap (10,000 replicates); training-seed variance is not included (A3). Crossover = gain over Standard outside − gain inside (reversed test). 'Kept by circle' = mean share of marker pixels inside P1's circle mask (what Masked and arm a keep); 'kept by lungs + heart' = the same for the PSPNet lungs ∪ heart mask (what arms b and c keep).

| position | method | kept by circle | kept by lungs + heart | correlated | reversed | clean | reversed gain vs Standard [95% CI] | Δ vs Masked (reversed) [95% CI] | crossover [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| inside | Standard (theirs) | 0.986 | 1.000 | 0.943 | 0.514 | 0.866 | +0.000  | +0.045 [+0.035, +0.054] | — |
| inside | Masked (theirs, A5) | 0.986 | 1.000 | 0.935 | 0.469 | 0.837 | -0.045 [-0.054, -0.035] | — | +0.032 [+0.023, +0.041] |
| inside | a: mask_cmc on their net | 0.986 | 1.000 | 0.893 | 0.856 | 0.884 | +0.342 [+0.324, +0.360] | +0.387 [+0.369, +0.405] | -0.328 [-0.347, -0.310] |
| inside | b: DINOv2 + mask_cmc | 0.986 | 1.000 | 0.904 | 0.877 | 0.895 | +0.364 [+0.343, +0.384] | +0.408 [+0.387, +0.430] | -0.359 [-0.379, -0.338] |
| inside | c: U-MtE | 0.986 | 1.000 | 0.960 | 0.598 | 0.881 | +0.084 [+0.063, +0.105] | +0.129 [+0.107, +0.150] | -0.070 [-0.092, -0.049] |
| inside | d: EfficientNet-B0 trained on ROI-masked images + mask_cmc (A20) | 0.986 | 1.000 | 0.961 | 0.735 | 0.906 | +0.222 [+0.202, +0.241] | +0.266 [+0.245, +0.287] | -0.195 [-0.216, -0.174] |
| outside | Standard (theirs) | 0.000 | 0.000 | 0.901 | 0.894 | 0.898 | +0.000  | +0.013 [+0.006, +0.019] | — |
| outside | Masked (theirs, A5) | 0.000 | 0.000 | 0.881 | 0.881 | 0.881 | -0.013 [-0.019, -0.006] | — | +0.032 [+0.023, +0.041] |
| outside | a: mask_cmc on their net | 0.000 | 0.000 | 0.908 | 0.908 | 0.908 | +0.014 [+0.007, +0.021] | +0.027 [+0.021, +0.033] | -0.328 [-0.347, -0.310] |
| outside | b: DINOv2 + mask_cmc | 0.000 | 0.000 | 0.899 | 0.899 | 0.899 | +0.005 [-0.004, +0.015] | +0.018 [+0.007, +0.029] | -0.359 [-0.379, -0.338] |
| outside | c: U-MtE | 0.000 | 0.000 | 0.908 | 0.908 | 0.908 | +0.014 [+0.004, +0.023] | +0.026 [+0.015, +0.037] | -0.070 [-0.092, -0.049] |
| outside | d: EfficientNet-B0 trained on ROI-masked images + mask_cmc (A20) | 0.000 | 0.000 | 0.921 | 0.921 | 0.921 | +0.027 [+0.017, +0.036] | +0.039 [+0.029, +0.050] | -0.195 [-0.216, -0.174] |
