# Round 6 Part B — fine-tuned thyroid

Recipe: `convnext_tiny.fb_in22k_ft_in1k` lr=0.0001 epochs=8 validation AUROC 0.8653628857018687.

| hypothesis | estimate | 95% CI | p | Holm p | verdict |
|---|---|---|---|---|---|
| FT0 | 0.7685005829073625 |  | None |  | below benchmark |
| FT1 | -0.1022645578720345 | [-0.178, -0.026] | 0.004 | 0.004 | SUPPORTED |
| FT2 | 4 |  | 0.0005 | 0.0015 | SUPPORTED |
| FT3 | -0.5205128205128206 | [-0.642, -0.382] | 0.0005 | 0.0015 | SUPPORTED |
| FT4 | 0.18738743540065037 | [+0.111, +0.268] | 0.0001 | 0.0004 | SUPPORTED |

FT0 is descriptive (benchmark-level if the mean ERM test AUROC is at least 0.773) and is not in the Holm family.
FT1–FT4 use the crossed bootstrap. Holm is applied to those four one-sided p-values.
Hard pairs are malignant nodules with an in-ROI caliper versus benign nodules without (`hard_pos_a=1`).
