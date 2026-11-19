# P4 (Saab et al., MLHC 2022, ISIC) — reproduction gate (A13 + addendum); Seg-ERM on machine 2 (A27) — A31 reruns (PIL-resized cache)

| method | split | robust AUROC | note |
|---|---|---|---|
| Binary-ERM | 1 | 0.3548 | machine 1 (git) |
| Seg-ERM | 1 | 0.6258 | machine 2, best val Dice 0.843 at epoch 146.0 |
| Binary-ERM | 2 | 0.3614 | machine 1 (git) |
| Seg-ERM | 2 | 0.6734 | machine 2, best val Dice 0.869 at epoch 135.0 |
| Binary-ERM | 3 | 0.3735 | machine 1 (git) |
| Seg-ERM | 3 | 0.6411 | machine 2, best val Dice 0.843 at epoch 131.0 |

- Binary-ERM: mean 0.363 ± 0.009 (SD over splits) vs paper 0.359 (|Δ| 0.004) / notebook 0.31 (|Δ| 0.053) → **pass**
- Seg-ERM: mean 0.647 ± 0.024 (SD over splits) vs paper 0.727 (|Δ| 0.080) / notebook 0.772 (|Δ| 0.125) → **FAIL**

**P4 gate: FAILED — P4 stays descriptive** (A13 item 4 + addendum).

## Arm d (EfficientNet-B0 + mask_cmc, A20/A21; machine 1's scores) vs Seg-ERM split 1 (paired, same 772 images)
arm d 0.666 vs Seg-ERM 0.626: Δ +0.040 [-0.015, +0.096], p = 0.1564. One network each, image-only bootstrap (10,000). Descriptive until the P4 gate.
