# P4 (Saab et al., MLHC 2022, ISIC) — reproduction gate (A13 + addendum); Seg-ERM on machine 2 (A27)

| method | split | robust AUROC | note |
|---|---|---|---|
| Binary-ERM | 1 | 0.3548 | machine 1 (git) |
| Seg-ERM | 1 | 0.6090 | machine 2, best val Dice 0.841 at epoch 125.0 |
| Binary-ERM | 2 | 0.3614 | machine 1 (git) |
| Seg-ERM | 2 | 0.6105 | machine 2, best val Dice 0.867 at epoch 138.0 |
| Binary-ERM | 3 | 0.3735 | machine 1 (git) |
| Seg-ERM | 3 | 0.6758 | machine 2, best val Dice 0.843 at epoch 148.0 |

- Binary-ERM: mean 0.363 ± 0.009 (SD over splits) vs paper 0.359 (|Δ| 0.004) / notebook 0.31 (|Δ| 0.053) → **pass**
- Seg-ERM: mean 0.632 ± 0.038 (SD over splits) vs paper 0.727 (|Δ| 0.095) / notebook 0.772 (|Δ| 0.140) → **FAIL**

**P4 gate: FAILED — P4 stays descriptive** (A13 item 4 + addendum).

## Arm d (EfficientNet-B0 + mask_cmc, A20/A21; machine 1's scores) vs Seg-ERM split 1 (paired, same 772 images)
arm d 0.666 vs Seg-ERM 0.609: Δ +0.057 [+0.003, +0.111], p = 0.0392. One network each, image-only bootstrap (10,000). Descriptive until the P4 gate.
