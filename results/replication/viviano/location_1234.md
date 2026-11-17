# P1 location test (A15/A16): tube inside vs outside the lungs + heart ROI, P1's pipeline, seed 1234

One seed: paired image-only bootstrap (10,000 replicates); training-seed variance is not included (A3). Crossover = gain over Standard outside − gain inside (reversed test). 'Kept by circle' = mean share of tube pixels inside P1's circle mask (what Masked and arm a keep); 'kept by lungs + heart' = the same for the PSPNet lungs ∪ heart mask (what arms b and c keep).

| position | method | kept by circle | kept by lungs + heart | correlated | reversed | clean | reversed gain vs Standard | Δ vs Masked (reversed) [95% CI] | crossover [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| inside | Standard (theirs) | 0.986 | 1.000 | 0.943 | 0.514 | 0.866 | +0.000 | +0.045 [+0.035, +0.054] | — |
| inside | Masked (theirs, A5) | 0.986 | 1.000 | 0.935 | 0.469 | 0.837 | -0.045 | — | — |
| inside | a: mask_cmc on their net | 0.986 | 1.000 | 0.893 | 0.856 | 0.884 | +0.342 | +0.387 [+0.369, +0.405] | — |
| inside | b: DINOv2 + mask_cmc | 0.986 | 1.000 | 0.904 | 0.877 | 0.895 | +0.364 | +0.408 [+0.387, +0.430] | — |
| inside | c: U-MtE | 0.986 | 1.000 | 0.960 | 0.598 | 0.881 | +0.084 | +0.129 [+0.107, +0.150] | — |
