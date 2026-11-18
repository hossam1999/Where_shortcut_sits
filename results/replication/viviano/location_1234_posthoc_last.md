# POST HOC sensitivity analysis (A17) — P1 location test on their epoch-100 models (`last_model_1234.pth.tar`)

**Every number in this table is POST HOC** (decided after the registered table was seen). The registered table (`location_1234.md`, their selection rule: epoch 11 inside, epoch 8 outside) is the primary result; this one is descriptive only, with no verdict. Standard, Masked and arm a use their epoch-100 network at both positions; arms b and c do not use their network and are the registered predictions, reused unchanged.

Artifact: a synthetic bar-shaped radio-opaque marker (43 x 6 px at 224 px; straight, bright, with a central stripe), not a realistic tube. Outside placements lie near the image edge: below the lungs + heart near the bottom edge (47.5% of test images), above the lungs near the top edge (38.2%) or beside the ROI (14.3%). Realism is covered elsewhere in the thesis (real device traps on NIH / RANZCR; transplant of real artifacts).

One seed: paired image-only bootstrap (10,000 replicates); training-seed variance is not included (A3). Crossover = gain over Standard outside − gain inside (reversed test). 'Kept by circle' = mean share of marker pixels inside P1's circle mask (what Masked and arm a keep); 'kept by lungs + heart' = the same for the PSPNet lungs ∪ heart mask (what arms b and c keep).

| position | method | kept by circle | kept by lungs + heart | correlated | reversed | clean | reversed gain vs Standard [95% CI] | Δ vs Masked (reversed) [95% CI] | crossover [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| POST HOC inside | Standard (theirs) | 0.986 | 1.000 | 0.966 | 0.468 | 0.897 | +0.000  | +0.042 [+0.025, +0.058] | — |
| POST HOC inside | Masked (theirs, A5) | 0.986 | 1.000 | 0.929 | 0.426 | 0.799 | -0.042 [-0.058, -0.025] | — | +0.211 [+0.193, +0.229] |
| POST HOC inside | a: mask_cmc on their net | 0.986 | 1.000 | 0.921 | 0.633 | 0.844 | +0.166 [+0.150, +0.182] | +0.208 [+0.198, +0.218] | +0.006 [-0.013, +0.025] |
| POST HOC inside | b: DINOv2 + mask_cmc | 0.986 | 1.000 | 0.904 | 0.877 | 0.895 | +0.410 [+0.390, +0.429] | +0.452 [+0.430, +0.473] | -0.225 [-0.242, -0.207] |
| POST HOC inside | c: U-MtE | 0.986 | 1.000 | 0.960 | 0.598 | 0.881 | +0.130 [+0.112, +0.149] | +0.172 [+0.149, +0.194] | +0.063 [+0.042, +0.085] |
| POST HOC outside | Standard (theirs) | 0.000 | 0.000 | 0.962 | 0.714 | 0.910 | +0.000  | -0.169 [-0.183, -0.156] | — |
| POST HOC outside | Masked (theirs, A5) | 0.000 | 0.000 | 0.884 | 0.884 | 0.884 | +0.169 [+0.156, +0.183] | — | +0.211 [+0.193, +0.229] |
| POST HOC outside | a: mask_cmc on their net | 0.000 | 0.000 | 0.886 | 0.886 | 0.886 | +0.172 [+0.157, +0.187] | +0.003 [-0.006, +0.011] | +0.006 [-0.013, +0.025] |
| POST HOC outside | b: DINOv2 + mask_cmc | 0.000 | 0.000 | 0.899 | 0.899 | 0.899 | +0.185 [+0.169, +0.201] | +0.016 [+0.004, +0.027] | -0.225 [-0.242, -0.207] |
| POST HOC outside | c: U-MtE | 0.000 | 0.000 | 0.908 | 0.908 | 0.908 | +0.193 [+0.177, +0.210] | +0.024 [+0.013, +0.036] | +0.063 [+0.042, +0.085] |
