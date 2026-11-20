# P8: our arms vs our reproduced DISC (A32, A35, A37) — machine 2

**Gate FAILED (ERM 0.507 vs 0.364, DISC 0.489 vs 0.551; A37 reading): P8 is DESCRIPTIVE.** The comparison is against our run of DISC's official code under identical conditions (reporting tier 2), not against the published 0.551; the last column says whether the registered rule's conditions hold, not a verdict.

5-set mean test ROC-AUC (618 test images per trap set, DISC's split); paired image-only bootstrap (10,000 replicates, one Poisson weight per image shared across trap sets and methods, trap sets fixed). Primary rule = DISC's protocol (selection on 154 test-distribution validation images, as DISC and ERM); secondary = fit on 1,563 / select on Bissoto's 259 held-out validation images (no test-distribution data). Holm family = the 4 primary comparisons; arms b and d primary, a and c secondary.

| rule | arm | arm mean | DISC mean | Δ [95 % CI] | p | p Holm | rule conditions met |
|---|---|---|---|---|---|---|---|
| primary | a_disc_mask_cmc | 0.621 | 0.489 | +0.132 [+0.097, +0.165] | 0.0001 | 0.0004 | yes |
| primary | b_dino518_mask_cmc | 0.736 | 0.489 | +0.247 [+0.204, +0.290] | 0.0001 | 0.0004 | yes |
| primary | c_umte | 0.630 | 0.489 | +0.141 [+0.094, +0.189] | 0.0001 | 0.0004 | yes |
| primary | d_effb0_mask_cmc | 0.691 | 0.489 | +0.201 [+0.154, +0.248] | 0.0001 | 0.0004 | yes |
| secondary | a_disc_mask_cmc | 0.608 | 0.489 | +0.118 [+0.084, +0.152] | 0.0001 | — | — |
| secondary | b_dino518_mask_cmc | 0.757 | 0.489 | +0.268 [+0.226, +0.310] | 0.0001 | — | — |
| secondary | c_umte | 0.661 | 0.489 | +0.171 [+0.129, +0.214] | 0.0001 | — | — |
| secondary | d_effb0_mask_cmc | 0.648 | 0.489 | +0.159 [+0.118, +0.201] | 0.0001 | — | — |

Per trap set: `arms_per_set.csv`; per-image test scores (arms, DISC, ERM): `scores_test.csv.gz`; selection grids: `selection.csv`. ERM 5-set mean (same images): 0.507.
