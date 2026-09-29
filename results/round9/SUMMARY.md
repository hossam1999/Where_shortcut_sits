# Round 9 — do the search's candidates dominate ROI masking?

Registration: `docs/PREREGISTRATION_ROUND9.md` (the last remedy round). Seeds 9101–9505. Every contrast is arm − masking with a crossed seed × image 95% interval; one-sided p; dominance = intersection–union of D1–D4; fixed sequence mask_condadv → mask_irm → mask_vrex (one-sided 0.025 each). mask_cmc and mask_bal (round 8) are descriptive references on the same data.

**Decision (registered rule, all cohorts): wins where masking fails but costs elsewhere: mask_irm, mask_vrex**

- **mask_condadv**: 27/42 met. Losses: D1 all pairs isic_MSK -0.010; D1 all pairs isic2020 -0.029; D3 Trap B reversed ovary -0.016; D3 Trap B min(rev,corr) ovary -0.016; D3 clean (Trap B models) ovary -0.021; D3 clean (Trap B models) (pooled) thyroid+capsule+ovary+isic -0.011. Inconclusive: D1 all pairs isic_HAM -0.004; D2 hard pairs thyroid +0.001; D2 hard pairs isic_BCN +0.003; D2 hard pairs isic_MSK +0.010; D2 hard pairs isic2020 +0.005; D2 Trap A min(rev,corr) ovary +0.039; D3 Trap B reversed (pooled) thyroid+capsule+ovary+isic -0.000; D3 Trap B min(rev,corr) (pooled) thyroid+capsule+ovary+isic -0.000; D3 clean (Trap B models) capsule -0.012.
- **mask_irm**: 41/42 met. Losses: D1 all pairs isic2020 -0.009. Inconclusive: none.
- **mask_vrex**: 40/42 met. Losses: D1 all pairs isic2020 -0.012. Inconclusive: D1 all pairs thyroid -0.001.
- **mask_cmc**: 41/42 met. Losses: D1 all pairs isic2020 -0.015. Inconclusive: none.
- **mask_bal**: 38/42 met. Losses: D1 all pairs thyroid -0.010; D1 all pairs isic_MSK -0.006; D1 all pairs isic2020 -0.015. Inconclusive: D3 Trap B min(rev,corr) ovary +0.024.

## Verdicts

| scope | candidate | role | p_iut | no_flipping | components | components_met | losses | all_D2_met | sequence | dominates |
|---|---|---|---|---|---|---|---|---|---|---|
| all cohorts | mask_condadv | candidate | 1.000 | True | 42 | 27 | 6 | False | tested | False |
| all cohorts | mask_irm | candidate | 0.343 | True | 42 | 41 | 1 | True | not tested in the sequence | False |
| all cohorts | mask_vrex | candidate | 0.800 | True | 42 | 40 | 1 | True | not tested in the sequence | False |
| all cohorts | mask_cmc | reference | 0.849 | True | 42 | 41 | 1 | True | descriptive reference |  |
| all cohorts | mask_bal | reference | 0.980 | True | 42 | 38 | 3 | True | descriptive reference |  |
| without ovary | mask_condadv | candidate | 1.000 | True | 35 | 26 | 3 | False | tested | False |
| without ovary | mask_irm | candidate | 0.343 | True | 35 | 34 | 1 | True | not tested in the sequence | False |
| without ovary | mask_vrex | candidate | 0.800 | True | 35 | 33 | 1 | True | not tested in the sequence | False |
| without ovary | mask_cmc | reference | 0.849 | True | 35 | 34 | 1 | True | descriptive reference |  |
| without ovary | mask_bal | reference | 0.980 | True | 35 | 32 | 3 | True | descriptive reference |  |

## ISIC 2019 → 2020: all-pairs contrast with masking, decomposed (π × Δ per pair type)

| arm | pi_easy | pi_hard | pi_same | easy_part | hard_part | same_artifact_part | all_pairs_delta |
|---|---|---|---|---|---|---|---|
| erm | 0.129 | 0.055 | 0.817 | -0.005 | 0.002 | -0.010 | -0.014 |
| mask_bal | 0.129 | 0.055 | 0.817 | -0.007 | 0.002 | -0.010 | -0.015 |
| mask_cmc | 0.129 | 0.055 | 0.817 | -0.011 | 0.003 | -0.007 | -0.015 |
| mask_condadv | 0.129 | 0.055 | 0.817 | -0.006 | 0.000 | -0.024 | -0.029 |
| mask_irm | 0.129 | 0.055 | 0.817 | -0.005 | 0.002 | -0.005 | -0.009 |
| mask_vrex | 0.129 | 0.055 | 0.817 | -0.009 | 0.002 | -0.006 | -0.012 |

## Seed stability (Trap A reversed, arm − masking)

| cohort | arm | trapA_rev_delta_mean | trapA_rev_delta_seed_sd |
|---|---|---|---|
| thyroid | erm | -0.107 | 0.014 |
| thyroid | mask_bal | 0.393 | 0.028 |
| thyroid | mask_cmc | 0.333 | 0.014 |
| thyroid | mask_condadv | 0.122 | 0.020 |
| thyroid | mask_irm | 0.229 | 0.024 |
| thyroid | mask_vrex | 0.322 | 0.015 |
| capsule | erm | -0.097 | 0.015 |
| capsule | mask_bal | 0.164 | 0.024 |
| capsule | mask_cmc | 0.143 | 0.030 |
| capsule | mask_condadv | 0.161 | 0.041 |
| capsule | mask_irm | 0.030 | 0.027 |
| capsule | mask_vrex | 0.156 | 0.022 |
| ovary | erm | -0.123 | 0.034 |
| ovary | mask_bal | 0.134 | 0.030 |
| ovary | mask_cmc | 0.094 | 0.023 |
| ovary | mask_condadv | 0.039 | 0.028 |
| ovary | mask_irm | 0.057 | 0.019 |
| ovary | mask_vrex | 0.081 | 0.021 |
| isic | erm | 0.047 | 0.009 |
| isic | mask_bal | 0.218 | 0.011 |
| isic | mask_cmc | 0.205 | 0.013 |
| isic | mask_condadv | 0.096 | 0.010 |
| isic | mask_irm | 0.120 | 0.018 |
| isic | mask_vrex | 0.193 | 0.008 |

## Replication family R (Trap A min(rev, corr), Holm within)

| encoder | cohort | candidate | estimate | ci95_lo | ci95_hi | margin | test | p_one_sided | met | label | p_holm | met_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| medsiglip448 | thyroid | mask_condadv | 0.167 | 0.137 | 0.198 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| medsiglip448 | thyroid | mask_irm | 0.265 | 0.216 | 0.305 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| medsiglip448 | thyroid | mask_vrex | 0.366 | 0.339 | 0.391 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| medsiglip448 | capsule | mask_condadv | 0.146 | 0.113 | 0.177 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| medsiglip448 | capsule | mask_irm | 0.019 | 0.006 | 0.032 | 0.000 | SUP | 0.003 | True | met | 0.005 | True |
| medsiglip448 | capsule | mask_vrex | 0.164 | 0.144 | 0.185 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| medsiglip448 | ovary | mask_condadv | 0.161 | 0.126 | 0.195 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| medsiglip448 | ovary | mask_irm | 0.166 | 0.135 | 0.198 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| medsiglip448 | ovary | mask_vrex | 0.234 | 0.200 | 0.270 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| convnext384 | thyroid | mask_condadv | 0.150 | 0.128 | 0.176 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| convnext384 | thyroid | mask_irm | 0.245 | 0.222 | 0.269 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| convnext384 | thyroid | mask_vrex | 0.299 | 0.280 | 0.317 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| convnext384 | capsule | mask_condadv | 0.159 | 0.119 | 0.197 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| convnext384 | capsule | mask_irm | 0.037 | 0.008 | 0.068 | 0.000 | SUP | 0.009 | True | met | 0.009 | True |
| convnext384 | capsule | mask_vrex | 0.144 | 0.123 | 0.169 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |

## mask_condadv — all cohorts

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | 0.001 | -0.002 | 0.007 | 0.000 | met |
| D1 all pairs | capsule | NI | 0.010 | 0.000 | -0.002 | 0.002 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | -0.000 | -0.005 | 0.003 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | -0.004 | -0.011 | 0.004 | 0.042 | inconclusive |
| D1 all pairs | isic_MSK | NI | 0.010 | -0.010 | -0.019 | -0.000 | 0.476 | loss |
| D1 all pairs | isic2020 | NI | 0.010 | -0.029 | -0.039 | -0.020 | 1.000 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.001 | -0.007 | 0.009 | 0.602 | inconclusive |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.003 | -0.013 | 0.015 | 0.339 | inconclusive |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.010 | -0.008 | 0.029 | 0.138 | inconclusive |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.005 | -0.006 | 0.017 | 0.203 | inconclusive |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.122 | 0.102 | 0.143 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.161 | 0.125 | 0.199 | 0.000 | met |
| D2 Trap A min(rev,corr) | ovary | SUP | 0.000 | 0.039 | -0.005 | 0.081 | 0.041 | inconclusive |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.096 | 0.077 | 0.116 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.004 | -0.007 | 0.014 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.005 | -0.020 | 0.027 | 0.005 | met |
| D3 Trap B reversed | ovary | NI | 0.030 | -0.016 | -0.063 | 0.029 | 0.264 | loss |
| D3 Trap B reversed | isic | NI | 0.030 | 0.006 | -0.011 | 0.022 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | -0.000 | -0.014 | 0.013 | 0.084 | inconclusive |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.004 | -0.008 | 0.014 | 0.000 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.005 | -0.020 | 0.027 | 0.005 | met |
| D3 Trap B min(rev,corr) | ovary | NI | 0.030 | -0.016 | -0.063 | 0.026 | 0.286 | loss |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.006 | -0.011 | 0.022 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | -0.000 | -0.014 | 0.012 | 0.092 | inconclusive |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.015 | 0.002 | 0.029 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | 0.040 | 0.022 | 0.060 | 0.000 | met |
| D3 clean (Trap A models) | ovary | NI | 0.030 | 0.022 | -0.007 | 0.050 | 0.001 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.012 | -0.000 | 0.024 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.022 | 0.013 | 0.032 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.000 | -0.012 | 0.011 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | -0.012 | -0.032 | 0.006 | 0.037 | inconclusive |
| D3 clean (Trap B models) | ovary | NI | 0.030 | -0.021 | -0.056 | 0.014 | 0.308 | loss |
| D3 clean (Trap B models) | isic | NI | 0.030 | -0.011 | -0.025 | 0.003 | 0.005 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | -0.011 | -0.022 | 0.000 | 0.560 | loss |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.331 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.027 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.112 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.042 |  |  |  | met |
| D4 no flipping | ovary trapA | point | 0.020 | 0.273 |  |  |  | met |
| D4 no flipping | ovary trapB | point | 0.020 | 0.035 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.333 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.151 |  |  |  | met |

## mask_irm — all cohorts

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | 0.002 | -0.007 | 0.010 | 0.006 | met |
| D1 all pairs | capsule | NI | 0.010 | -0.000 | -0.001 | 0.000 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | 0.002 | -0.001 | 0.005 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | 0.005 | 0.001 | 0.009 | 0.000 | met |
| D1 all pairs | isic_MSK | NI | 0.010 | -0.002 | -0.007 | 0.002 | 0.004 | met |
| D1 all pairs | isic2020 | NI | 0.010 | -0.009 | -0.013 | -0.006 | 0.343 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.049 | 0.030 | 0.073 | 0.000 | met |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.017 | 0.012 | 0.022 | 0.000 | met |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.036 | 0.024 | 0.048 | 0.000 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.028 | 0.022 | 0.034 | 0.000 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.229 | 0.205 | 0.255 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.030 | 0.008 | 0.053 | 0.004 | met |
| D2 Trap A min(rev,corr) | ovary | SUP | 0.000 | 0.057 | 0.037 | 0.080 | 0.000 | met |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.120 | 0.096 | 0.143 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | -0.000 | -0.010 | 0.010 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.000 | -0.004 | 0.004 | 0.000 | met |
| D3 Trap B reversed | ovary | NI | 0.030 | 0.022 | -0.005 | 0.048 | 0.000 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.028 | 0.018 | 0.039 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.012 | 0.005 | 0.020 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | -0.000 | -0.017 | 0.010 | 0.001 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.000 | -0.004 | 0.004 | 0.000 | met |
| D3 Trap B min(rev,corr) | ovary | NI | 0.030 | 0.022 | -0.012 | 0.046 | 0.001 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.028 | 0.018 | 0.039 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.012 | 0.002 | 0.019 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.059 | 0.044 | 0.074 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | 0.005 | -0.003 | 0.014 | 0.000 | met |
| D3 clean (Trap A models) | ovary | NI | 0.030 | 0.031 | 0.018 | 0.044 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.022 | 0.008 | 0.037 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.029 | 0.023 | 0.036 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | -0.006 | -0.018 | 0.005 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | -0.001 | -0.004 | 0.000 | 0.000 | met |
| D3 clean (Trap B models) | ovary | NI | 0.030 | 0.011 | -0.011 | 0.035 | 0.000 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | 0.008 | -0.001 | 0.017 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.003 | -0.004 | 0.010 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.216 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.020 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.256 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.072 |  |  |  | met |
| D4 no flipping | ovary trapA | point | 0.020 | 0.250 |  |  |  | met |
| D4 no flipping | ovary trapB | point | 0.020 | 0.027 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.306 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.140 |  |  |  | met |

## mask_vrex — all cohorts

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | -0.001 | -0.011 | 0.009 | 0.039 | inconclusive |
| D1 all pairs | capsule | NI | 0.010 | 0.000 | -0.000 | 0.001 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | -0.005 | -0.006 | -0.003 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | -0.001 | -0.005 | 0.003 | 0.001 | met |
| D1 all pairs | isic_MSK | NI | 0.010 | 0.001 | -0.002 | 0.004 | 0.000 | met |
| D1 all pairs | isic2020 | NI | 0.010 | -0.012 | -0.017 | -0.008 | 0.800 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.088 | 0.061 | 0.114 | 0.000 | met |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.022 | 0.020 | 0.025 | 0.000 | met |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.059 | 0.050 | 0.067 | 0.000 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.043 | 0.037 | 0.050 | 0.000 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.322 | 0.303 | 0.343 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.156 | 0.135 | 0.181 | 0.000 | met |
| D2 Trap A min(rev,corr) | ovary | SUP | 0.000 | 0.081 | 0.055 | 0.108 | 0.000 | met |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.193 | 0.170 | 0.216 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.014 | 0.007 | 0.022 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.015 | 0.007 | 0.026 | 0.000 | met |
| D3 Trap B reversed | ovary | NI | 0.030 | 0.004 | -0.020 | 0.024 | 0.006 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.053 | 0.038 | 0.070 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.021 | 0.014 | 0.029 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.014 | -0.001 | 0.022 | 0.000 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.015 | 0.007 | 0.026 | 0.000 | met |
| D3 Trap B min(rev,corr) | ovary | NI | 0.030 | 0.004 | -0.019 | 0.024 | 0.005 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.053 | 0.038 | 0.070 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.021 | 0.013 | 0.029 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.104 | 0.087 | 0.122 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | 0.044 | 0.033 | 0.055 | 0.000 | met |
| D3 clean (Trap A models) | ovary | NI | 0.030 | 0.041 | 0.021 | 0.060 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.046 | 0.032 | 0.061 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.059 | 0.051 | 0.067 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.007 | 0.001 | 0.014 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | 0.004 | -0.001 | 0.009 | 0.000 | met |
| D3 clean (Trap B models) | ovary | NI | 0.030 | 0.002 | -0.020 | 0.020 | 0.005 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | 0.018 | 0.007 | 0.029 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.008 | 0.001 | 0.013 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.127 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.017 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.121 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.054 |  |  |  | met |
| D4 no flipping | ovary trapA | point | 0.020 | 0.222 |  |  |  | met |
| D4 no flipping | ovary trapB | point | 0.020 | 0.049 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.218 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.115 |  |  |  | met |

## mask_cmc — all cohorts

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | -0.002 | -0.010 | 0.005 | 0.022 | met |
| D1 all pairs | capsule | NI | 0.010 | 0.001 | -0.001 | 0.002 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | -0.005 | -0.006 | -0.003 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | 0.000 | -0.003 | 0.004 | 0.000 | met |
| D1 all pairs | isic_MSK | NI | 0.010 | -0.001 | -0.005 | 0.002 | 0.000 | met |
| D1 all pairs | isic2020 | NI | 0.010 | -0.015 | -0.024 | -0.006 | 0.849 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.062 | 0.030 | 0.088 | 0.000 | met |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.037 | 0.034 | 0.039 | 0.000 | met |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.055 | 0.026 | 0.078 | 0.000 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.056 | 0.026 | 0.079 | 0.001 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.333 | 0.313 | 0.353 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.143 | 0.118 | 0.172 | 0.000 | met |
| D2 Trap A min(rev,corr) | ovary | SUP | 0.000 | 0.094 | 0.066 | 0.122 | 0.000 | met |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.205 | 0.181 | 0.229 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.017 | 0.010 | 0.025 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.023 | 0.014 | 0.032 | 0.000 | met |
| D3 Trap B reversed | ovary | NI | 0.030 | 0.004 | -0.019 | 0.024 | 0.004 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.053 | 0.037 | 0.070 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.024 | 0.017 | 0.031 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.017 | -0.003 | 0.024 | 0.000 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.023 | 0.014 | 0.032 | 0.000 | met |
| D3 Trap B min(rev,corr) | ovary | NI | 0.030 | 0.004 | -0.018 | 0.024 | 0.004 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.053 | 0.037 | 0.070 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.024 | 0.015 | 0.031 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.106 | 0.089 | 0.124 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | 0.040 | 0.029 | 0.052 | 0.000 | met |
| D3 clean (Trap A models) | ovary | NI | 0.030 | 0.046 | 0.025 | 0.066 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.047 | 0.032 | 0.062 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.060 | 0.051 | 0.068 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.008 | 0.002 | 0.015 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | 0.007 | 0.001 | 0.013 | 0.000 | met |
| D3 clean (Trap B models) | ovary | NI | 0.030 | 0.004 | -0.017 | 0.022 | 0.002 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | 0.014 | -0.000 | 0.026 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.008 | 0.002 | 0.014 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.109 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.013 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.136 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.047 |  |  |  | met |
| D4 no flipping | ovary trapA | point | 0.020 | 0.206 |  |  |  | met |
| D4 no flipping | ovary trapB | point | 0.020 | 0.048 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.201 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.107 |  |  |  | met |

## mask_bal — all cohorts

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | -0.010 | -0.019 | -0.001 | 0.470 | loss |
| D1 all pairs | capsule | NI | 0.010 | -0.000 | -0.003 | 0.002 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | 0.003 | -0.000 | 0.006 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | 0.001 | -0.003 | 0.004 | 0.000 | met |
| D1 all pairs | isic_MSK | NI | 0.010 | -0.006 | -0.010 | -0.001 | 0.036 | loss |
| D1 all pairs | isic2020 | NI | 0.010 | -0.015 | -0.019 | -0.010 | 0.980 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.046 | 0.030 | 0.063 | 0.000 | met |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.028 | 0.021 | 0.035 | 0.000 | met |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.049 | 0.038 | 0.060 | 0.000 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.039 | 0.030 | 0.048 | 0.000 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.393 | 0.363 | 0.414 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.164 | 0.140 | 0.189 | 0.000 | met |
| D2 Trap A min(rev,corr) | ovary | SUP | 0.000 | 0.134 | 0.101 | 0.170 | 0.000 | met |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.218 | 0.194 | 0.245 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.026 | 0.011 | 0.043 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.027 | 0.019 | 0.035 | 0.000 | met |
| D3 Trap B reversed | ovary | NI | 0.030 | 0.036 | 0.002 | 0.070 | 0.000 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.093 | 0.067 | 0.119 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.045 | 0.034 | 0.057 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.020 | -0.016 | 0.039 | 0.001 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.027 | 0.019 | 0.035 | 0.000 | met |
| D3 Trap B min(rev,corr) | ovary | NI | 0.030 | 0.024 | -0.038 | 0.064 | 0.047 | inconclusive |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.093 | 0.067 | 0.119 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.041 | 0.021 | 0.053 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.114 | 0.095 | 0.134 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | 0.048 | 0.036 | 0.061 | 0.000 | met |
| D3 clean (Trap A models) | ovary | NI | 0.030 | 0.051 | 0.031 | 0.072 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.050 | 0.034 | 0.067 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.066 | 0.057 | 0.075 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.009 | -0.002 | 0.020 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | 0.007 | 0.002 | 0.012 | 0.000 | met |
| D3 clean (Trap B models) | ovary | NI | 0.030 | 0.007 | -0.022 | 0.035 | 0.009 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | 0.016 | -0.002 | 0.035 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.010 | 0.001 | 0.019 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.013 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | -0.006 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.116 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.039 |  |  |  | met |
| D4 no flipping | ovary trapA | point | 0.020 | 0.148 |  |  |  | met |
| D4 no flipping | ovary trapB | point | 0.020 | -0.012 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.179 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.040 |  |  |  | met |

## mask_condadv — without ovary

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | 0.001 | -0.002 | 0.007 | 0.000 | met |
| D1 all pairs | capsule | NI | 0.010 | 0.000 | -0.002 | 0.002 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | -0.000 | -0.005 | 0.003 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | -0.004 | -0.011 | 0.004 | 0.042 | inconclusive |
| D1 all pairs | isic_MSK | NI | 0.010 | -0.010 | -0.019 | -0.000 | 0.476 | loss |
| D1 all pairs | isic2020 | NI | 0.010 | -0.029 | -0.039 | -0.020 | 1.000 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.001 | -0.007 | 0.009 | 0.602 | inconclusive |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.003 | -0.013 | 0.015 | 0.339 | inconclusive |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.010 | -0.008 | 0.029 | 0.138 | inconclusive |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.005 | -0.006 | 0.017 | 0.203 | inconclusive |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.122 | 0.102 | 0.143 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.161 | 0.125 | 0.199 | 0.000 | met |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.096 | 0.077 | 0.116 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.004 | -0.007 | 0.014 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.005 | -0.020 | 0.027 | 0.005 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.006 | -0.011 | 0.022 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.005 | -0.006 | 0.015 | 0.003 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.004 | -0.008 | 0.014 | 0.000 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.005 | -0.020 | 0.027 | 0.005 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.006 | -0.011 | 0.022 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.005 | -0.006 | 0.014 | 0.004 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.015 | 0.002 | 0.029 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | 0.040 | 0.022 | 0.060 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.012 | -0.000 | 0.024 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.022 | 0.014 | 0.031 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.000 | -0.012 | 0.011 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | -0.012 | -0.032 | 0.006 | 0.037 | inconclusive |
| D3 clean (Trap B models) | isic | NI | 0.030 | -0.011 | -0.025 | 0.003 | 0.005 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+isic | NI | 0.010 | -0.007 | -0.016 | 0.001 | 0.270 | loss |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.331 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.027 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.112 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.042 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.333 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.151 |  |  |  | met |

## mask_irm — without ovary

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | 0.002 | -0.007 | 0.010 | 0.006 | met |
| D1 all pairs | capsule | NI | 0.010 | -0.000 | -0.001 | 0.000 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | 0.002 | -0.001 | 0.005 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | 0.005 | 0.001 | 0.009 | 0.000 | met |
| D1 all pairs | isic_MSK | NI | 0.010 | -0.002 | -0.007 | 0.002 | 0.004 | met |
| D1 all pairs | isic2020 | NI | 0.010 | -0.009 | -0.013 | -0.006 | 0.343 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.049 | 0.030 | 0.073 | 0.000 | met |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.017 | 0.012 | 0.022 | 0.000 | met |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.036 | 0.024 | 0.048 | 0.000 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.028 | 0.022 | 0.034 | 0.000 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.229 | 0.205 | 0.255 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.030 | 0.008 | 0.053 | 0.004 | met |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.120 | 0.096 | 0.143 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | -0.000 | -0.010 | 0.010 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.000 | -0.004 | 0.004 | 0.000 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.028 | 0.018 | 0.039 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.009 | 0.004 | 0.014 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | -0.000 | -0.017 | 0.010 | 0.001 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.000 | -0.004 | 0.004 | 0.000 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.028 | 0.018 | 0.039 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.009 | 0.003 | 0.014 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.059 | 0.044 | 0.074 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | 0.005 | -0.003 | 0.014 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.022 | 0.008 | 0.037 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.029 | 0.021 | 0.037 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | -0.006 | -0.018 | 0.005 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | -0.001 | -0.004 | 0.000 | 0.000 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | 0.008 | -0.001 | 0.017 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.000 | -0.005 | 0.005 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.216 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.020 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.256 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.072 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.306 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.140 |  |  |  | met |

## mask_vrex — without ovary

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | -0.001 | -0.011 | 0.009 | 0.039 | inconclusive |
| D1 all pairs | capsule | NI | 0.010 | 0.000 | -0.000 | 0.001 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | -0.005 | -0.006 | -0.003 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | -0.001 | -0.005 | 0.003 | 0.001 | met |
| D1 all pairs | isic_MSK | NI | 0.010 | 0.001 | -0.002 | 0.004 | 0.000 | met |
| D1 all pairs | isic2020 | NI | 0.010 | -0.012 | -0.017 | -0.008 | 0.800 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.088 | 0.061 | 0.114 | 0.000 | met |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.022 | 0.020 | 0.025 | 0.000 | met |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.059 | 0.050 | 0.067 | 0.000 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.043 | 0.037 | 0.050 | 0.000 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.322 | 0.303 | 0.343 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.156 | 0.135 | 0.181 | 0.000 | met |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.193 | 0.170 | 0.216 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.014 | 0.007 | 0.022 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.015 | 0.007 | 0.026 | 0.000 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.053 | 0.038 | 0.070 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.027 | 0.021 | 0.034 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.014 | -0.001 | 0.022 | 0.000 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.015 | 0.007 | 0.026 | 0.000 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.053 | 0.038 | 0.070 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.027 | 0.020 | 0.034 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.104 | 0.087 | 0.122 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | 0.044 | 0.033 | 0.055 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.046 | 0.032 | 0.061 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.065 | 0.056 | 0.073 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.007 | 0.001 | 0.014 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | 0.004 | -0.001 | 0.009 | 0.000 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | 0.018 | 0.007 | 0.029 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.009 | 0.005 | 0.014 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.127 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.017 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.121 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.054 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.218 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.115 |  |  |  | met |

## mask_cmc — without ovary

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | -0.002 | -0.010 | 0.005 | 0.022 | met |
| D1 all pairs | capsule | NI | 0.010 | 0.001 | -0.001 | 0.002 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | -0.005 | -0.006 | -0.003 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | 0.000 | -0.003 | 0.004 | 0.000 | met |
| D1 all pairs | isic_MSK | NI | 0.010 | -0.001 | -0.005 | 0.002 | 0.000 | met |
| D1 all pairs | isic2020 | NI | 0.010 | -0.015 | -0.024 | -0.006 | 0.849 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.062 | 0.030 | 0.088 | 0.000 | met |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.037 | 0.034 | 0.039 | 0.000 | met |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.055 | 0.026 | 0.078 | 0.000 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.056 | 0.026 | 0.079 | 0.001 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.333 | 0.313 | 0.353 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.143 | 0.118 | 0.172 | 0.000 | met |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.205 | 0.181 | 0.229 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.017 | 0.010 | 0.025 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.023 | 0.014 | 0.032 | 0.000 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.053 | 0.037 | 0.070 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.031 | 0.024 | 0.038 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.017 | -0.003 | 0.024 | 0.000 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.023 | 0.014 | 0.032 | 0.000 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.053 | 0.037 | 0.070 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.031 | 0.022 | 0.037 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.106 | 0.089 | 0.124 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | 0.040 | 0.029 | 0.052 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.047 | 0.032 | 0.062 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.064 | 0.056 | 0.073 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.008 | 0.002 | 0.015 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | 0.007 | 0.001 | 0.013 | 0.000 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | 0.014 | -0.000 | 0.026 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.010 | 0.004 | 0.015 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.109 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.013 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.136 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.047 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.201 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.107 |  |  |  | met |

## mask_bal — without ovary

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | -0.010 | -0.019 | -0.001 | 0.470 | loss |
| D1 all pairs | capsule | NI | 0.010 | -0.000 | -0.003 | 0.002 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | 0.003 | -0.000 | 0.006 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | 0.001 | -0.003 | 0.004 | 0.000 | met |
| D1 all pairs | isic_MSK | NI | 0.010 | -0.006 | -0.010 | -0.001 | 0.036 | loss |
| D1 all pairs | isic2020 | NI | 0.010 | -0.015 | -0.019 | -0.010 | 0.980 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.046 | 0.030 | 0.063 | 0.000 | met |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.028 | 0.021 | 0.035 | 0.000 | met |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.049 | 0.038 | 0.060 | 0.000 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.039 | 0.030 | 0.048 | 0.000 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.393 | 0.363 | 0.414 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.164 | 0.140 | 0.189 | 0.000 | met |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.218 | 0.194 | 0.245 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.026 | 0.011 | 0.043 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.027 | 0.019 | 0.035 | 0.000 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.093 | 0.067 | 0.119 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.048 | 0.038 | 0.059 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.020 | -0.016 | 0.039 | 0.001 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.027 | 0.019 | 0.035 | 0.000 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.093 | 0.067 | 0.119 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.046 | 0.031 | 0.057 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.114 | 0.095 | 0.134 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | 0.048 | 0.036 | 0.061 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.050 | 0.034 | 0.067 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.071 | 0.062 | 0.080 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.009 | -0.002 | 0.020 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | 0.007 | 0.002 | 0.012 | 0.000 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | 0.016 | -0.002 | 0.035 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.011 | 0.004 | 0.018 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.013 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | -0.006 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.116 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.039 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.179 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.040 |  |  |  | met |
