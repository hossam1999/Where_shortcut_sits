# P1 cardiomegaly (site shortcut), seed 1234: preliminary head-to-head

**Preliminary, 1 of 3 seeds, no verdict.** Paired image-only bootstrap (10,000 replicates); training-seed variance is not included. The verdict is the registered 3-seed rule.

Target: **Masked** (A5: Standard model scored with test masking)

| method | AUROC (their definition) | AUROC | clean: within-site AUROC | paper | Δ vs Masked [95% CI] | Δ vs Standard [95% CI] |
|---|---|---|---|---|---|---|
| Standard | 0.627 | 0.627 | 0.887 | 0.70 ± 0.05 | -0.202 [-0.217, -0.188] | — |
| Masked | 0.829 | 0.829 | 0.871 | 0.77 ± 0.09 | — | +0.202 [+0.188, +0.217] |
| a_mask_cmc_their_net | — | 0.866 | 0.890 |  | +0.037 [+0.031, +0.043] | +0.239 [+0.224, +0.253] |
| b_dino518_mask_cmc | — | 0.869 | 0.880 |  | +0.040 [+0.026, +0.054] | +0.242 [+0.224, +0.260] |
| c_umte | — | 0.593 | 0.868 |  | -0.236 [-0.256, -0.216] | -0.034 [-0.053, -0.014] |
