# P4 (Saab et al., MLHC 2022, ISIC) — reproduction gate (A13 + addendum); Seg-ERM on machine 2 (A27)

| method | split | robust AUROC | note |
|---|---|---|---|
| Binary-ERM | 1 | 0.3548 | machine 1 (git) |
| Seg-ERM | 1 | 0.6090 | machine 2, best val Dice 0.841 at epoch 125.0 |
| Binary-ERM | 2 | 0.3614 | machine 1 (git) |
| Binary-ERM | 3 | 0.3735 | machine 1 (git) |

- Binary-ERM: mean 0.363 ± 0.009 (SD over splits) vs paper 0.359 (|Δ| 0.004) / notebook 0.31 (|Δ| 0.053) → **pass**
- Seg-ERM: 1 of 3 splits scored — pending

## Arm d (EfficientNet-B0 + mask_cmc, A20/A21; machine 1's scores) vs Seg-ERM split 1 (paired, same 772 images)
arm d 0.666 vs Seg-ERM 0.609: Δ +0.057 [+0.003, +0.111], p = 0.0392. One network each, image-only bootstrap (10,000). Descriptive until the P4 gate.
