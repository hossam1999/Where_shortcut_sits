# Round 8 — does any remedy dominate ROI masking?

Registration: `docs/PREREGISTRATION_ROUND8.md`. Frozen development choices: `results/round8/frozen_choice.json`. Every contrast is candidate − masking with a crossed seed × image 95% interval; one-sided p; dominance = intersection–union of D1–D4; fixed-sequence test over the four primary candidates in the order mask_cmc, full_cmc, mask_bal, locrand_loc (one-sided 0.025 each; Amendment 1).

**Decision (registered rule, all cohorts): wins where masking fails but costs elsewhere: mask_cmc, mask_bal**

- mask_cmc costs (components labelled loss): D1 all pairs isic2020 -0.021
- mask_bal costs (components labelled loss): D1 all pairs thyroid -0.008; D1 all pairs isic_MSK -0.006; D1 all pairs isic2020 -0.011

## Verdicts

| scope | candidate | p_iut | no_flipping | components | components_met | all_D2_met | sequence | dominates | p_holm |
|---|---|---|---|---|---|---|---|---|---|
| all cohorts | mask_cmc | 1.000 | True | 42 | 40 | True | tested | False |  |
| all cohorts | full_cmc | 1.000 | True | 42 | 36 | False | not tested in the sequence | False |  |
| all cohorts | mask_bal | 0.645 | True | 42 | 39 | True | not tested in the sequence | False |  |
| all cohorts | locrand_loc | 1.000 | True | 42 | 39 | False | not tested in the sequence | False |  |
| without ovary | mask_cmc | 1.000 | True | 35 | 33 | True | tested | False |  |
| without ovary | full_cmc | 1.000 | True | 35 | 30 | False | not tested in the sequence | False |  |
| without ovary | mask_bal | 0.645 | True | 35 | 32 | True | not tested in the sequence | False |  |
| without ovary | locrand_loc | 0.669 | True | 35 | 33 | False | not tested in the sequence | False |  |
| D5 | mask_bal | 0.000 |  | 12 | 12 |  |  | True | 0.000 |
| D5 | mask_cmc | 0.000 |  | 12 | 12 |  |  | True | 0.000 |
| D5 | full_cmc | 0.431 |  | 12 | 7 |  |  | False | 0.431 |
| D6 | locrand_ft | 0.478 |  | 3 | 1 |  |  | False | 0.478 |

## mask_cmc — all cohorts

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | -0.002 | -0.011 | 0.007 | 0.045 | inconclusive |
| D1 all pairs | capsule | NI | 0.010 | -0.000 | -0.001 | 0.001 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | -0.005 | -0.006 | -0.003 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | -0.003 | -0.008 | 0.002 | 0.008 | met |
| D1 all pairs | isic_MSK | NI | 0.010 | -0.002 | -0.006 | 0.002 | 0.000 | met |
| D1 all pairs | isic2020 | NI | 0.010 | -0.021 | -0.027 | -0.016 | 1.000 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.084 | 0.069 | 0.101 | 0.000 | met |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.036 | 0.033 | 0.040 | 0.000 | met |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.070 | 0.059 | 0.082 | 0.000 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.078 | 0.069 | 0.088 | 0.000 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.330 | 0.311 | 0.351 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.153 | 0.130 | 0.177 | 0.000 | met |
| D2 Trap A min(rev,corr) | ovary | SUP | 0.000 | 0.118 | 0.089 | 0.147 | 0.000 | met |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.193 | 0.166 | 0.221 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.015 | 0.006 | 0.027 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.022 | 0.014 | 0.030 | 0.000 | met |
| D3 Trap B reversed | ovary | NI | 0.030 | -0.002 | -0.017 | 0.011 | 0.000 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.045 | 0.032 | 0.058 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.020 | 0.014 | 0.026 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.015 | -0.000 | 0.026 | 0.000 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.022 | 0.014 | 0.030 | 0.000 | met |
| D3 Trap B min(rev,corr) | ovary | NI | 0.030 | -0.002 | -0.017 | 0.012 | 0.000 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.045 | 0.032 | 0.058 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.020 | 0.014 | 0.026 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.095 | 0.079 | 0.111 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | 0.048 | 0.037 | 0.061 | 0.000 | met |
| D3 clean (Trap A models) | ovary | NI | 0.030 | 0.043 | 0.023 | 0.062 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.039 | 0.023 | 0.056 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.056 | 0.048 | 0.065 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.009 | -0.003 | 0.023 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | 0.007 | 0.002 | 0.012 | 0.000 | met |
| D3 clean (Trap B models) | ovary | NI | 0.030 | -0.002 | -0.015 | 0.011 | 0.000 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | 0.013 | 0.003 | 0.022 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.007 | 0.002 | 0.012 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.103 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.029 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.141 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.056 |  |  |  | met |
| D4 no flipping | ovary trapA | point | 0.020 | 0.207 |  |  |  | met |
| D4 no flipping | ovary trapB | point | 0.020 | 0.051 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.216 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.113 |  |  |  | met |

Losses: D1 all pairs isic2020 -0.021
Inconclusive: D1 all pairs thyroid -0.002

## full_cmc — all cohorts

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D1 all pairs | capsule | NI | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | 0.004 | 0.000 | 0.012 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | -0.038 | -0.053 | -0.023 | 1.000 | loss |
| D1 all pairs | isic_MSK | NI | 0.010 | -0.020 | -0.044 | 0.003 | 0.803 | loss |
| D1 all pairs | isic2020 | NI | 0.010 | -0.032 | -0.053 | -0.012 | 0.985 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | inconclusive |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.014 | 0.000 | 0.045 | 0.335 | inconclusive |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.126 | 0.085 | 0.168 | 0.000 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.091 | 0.068 | 0.114 | 0.000 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.041 | 0.026 | 0.058 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.020 | 0.006 | 0.034 | 0.001 | met |
| D2 Trap A min(rev,corr) | ovary | SUP | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | inconclusive |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.240 | 0.191 | 0.288 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B reversed | ovary | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.029 | -0.003 | 0.065 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.007 | -0.001 | 0.016 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B min(rev,corr) | ovary | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.029 | -0.003 | 0.065 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.007 | -0.001 | 0.016 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | -0.002 | -0.011 | 0.006 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | -0.003 | -0.011 | 0.005 | 0.000 | met |
| D3 clean (Trap A models) | ovary | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.072 | 0.044 | 0.101 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.017 | 0.009 | 0.025 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap B models) | ovary | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | 0.021 | -0.003 | 0.046 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.005 | -0.001 | 0.011 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.444 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.045 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.270 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.081 |  |  |  | met |
| D4 no flipping | ovary trapA | point | 0.020 | 0.343 |  |  |  | met |
| D4 no flipping | ovary trapB | point | 0.020 | 0.050 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.184 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.165 |  |  |  | met |

Losses: D1 all pairs isic_HAM -0.038; D1 all pairs isic_MSK -0.020; D1 all pairs isic2020 -0.032
Inconclusive: D2 hard pairs thyroid +0.000; D2 hard pairs isic_BCN +0.014; D2 Trap A min(rev,corr) ovary +0.000

## mask_bal — all cohorts

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | -0.008 | -0.017 | 0.001 | 0.287 | loss |
| D1 all pairs | capsule | NI | 0.010 | -0.001 | -0.004 | 0.000 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | 0.003 | 0.000 | 0.007 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | 0.003 | -0.001 | 0.008 | 0.000 | met |
| D1 all pairs | isic_MSK | NI | 0.010 | -0.006 | -0.011 | -0.000 | 0.056 | loss |
| D1 all pairs | isic2020 | NI | 0.010 | -0.011 | -0.016 | -0.005 | 0.645 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.048 | 0.032 | 0.066 | 0.000 | met |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.033 | 0.026 | 0.040 | 0.000 | met |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.045 | 0.030 | 0.058 | 0.000 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.035 | 0.024 | 0.046 | 0.000 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.380 | 0.355 | 0.404 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.166 | 0.148 | 0.185 | 0.000 | met |
| D2 Trap A min(rev,corr) | ovary | SUP | 0.000 | 0.161 | 0.135 | 0.187 | 0.000 | met |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.213 | 0.187 | 0.239 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.031 | 0.020 | 0.042 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.022 | 0.013 | 0.032 | 0.000 | met |
| D3 Trap B reversed | ovary | NI | 0.030 | 0.043 | 0.018 | 0.069 | 0.000 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.085 | 0.063 | 0.106 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.045 | 0.036 | 0.054 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.031 | -0.012 | 0.041 | 0.000 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.022 | 0.013 | 0.032 | 0.000 | met |
| D3 Trap B min(rev,corr) | ovary | NI | 0.030 | 0.036 | -0.023 | 0.064 | 0.011 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.085 | 0.063 | 0.105 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.043 | 0.024 | 0.052 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.107 | 0.089 | 0.125 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | 0.051 | 0.040 | 0.063 | 0.000 | met |
| D3 clean (Trap A models) | ovary | NI | 0.030 | 0.051 | 0.030 | 0.071 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.044 | 0.026 | 0.062 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.063 | 0.055 | 0.072 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.011 | -0.002 | 0.025 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | 0.003 | -0.003 | 0.010 | 0.000 | met |
| D3 clean (Trap B models) | ovary | NI | 0.030 | 0.012 | -0.009 | 0.033 | 0.000 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | 0.012 | -0.006 | 0.029 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.010 | 0.002 | 0.017 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.027 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.004 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.122 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.047 |  |  |  | met |
| D4 no flipping | ovary trapA | point | 0.020 | 0.146 |  |  |  | met |
| D4 no flipping | ovary trapB | point | 0.020 | -0.008 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.187 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.040 |  |  |  | met |

Losses: D1 all pairs thyroid -0.008; D1 all pairs isic_MSK -0.006; D1 all pairs isic2020 -0.011
Inconclusive: none

## locrand_loc — all cohorts

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | -0.001 | -0.007 | 0.005 | 0.003 | met |
| D1 all pairs | capsule | NI | 0.010 | -0.002 | -0.003 | -0.000 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | -0.002 | -0.003 | -0.001 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | 0.000 | -0.001 | 0.002 | 0.000 | met |
| D1 all pairs | isic_MSK | NI | 0.010 | 0.001 | -0.001 | 0.003 | 0.000 | met |
| D1 all pairs | isic2020 | NI | 0.010 | -0.001 | -0.002 | 0.000 | 0.000 | met |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.024 | 0.010 | 0.040 | 0.001 | met |
| D2 hard pairs | isic_BCN | SUP | 0.000 | -0.000 | -0.001 | 0.001 | 0.669 | loss |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.005 | 0.002 | 0.009 | 0.002 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.003 | 0.000 | 0.005 | 0.007 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.164 | 0.137 | 0.193 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.001 | -0.001 | 0.005 | 0.242 | inconclusive |
| D2 Trap A min(rev,corr) | ovary | SUP | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | inconclusive |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.029 | 0.021 | 0.037 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B reversed | ovary | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.000 | -0.000 | 0.000 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.000 | -0.000 | 0.000 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B min(rev,corr) | ovary | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.000 | -0.000 | 0.000 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.000 | -0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.030 | 0.013 | 0.046 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | -0.001 | -0.004 | 0.001 | 0.000 | met |
| D3 clean (Trap A models) | ovary | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.007 | 0.002 | 0.012 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.009 | 0.005 | 0.013 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap B models) | ovary | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | -0.000 | -0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+ovary+isic | NI | 0.010 | 0.000 | -0.000 | 0.000 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.284 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.045 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.296 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.081 |  |  |  | met |
| D4 no flipping | ovary trapA | point | 0.020 | 0.343 |  |  |  | met |
| D4 no flipping | ovary trapB | point | 0.020 | 0.050 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.439 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.177 |  |  |  | met |

Losses: D2 hard pairs isic_BCN -0.000
Inconclusive: D2 Trap A min(rev,corr) capsule +0.001; D2 Trap A min(rev,corr) ovary +0.000

## mask_cmc — without ovary

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | -0.002 | -0.011 | 0.007 | 0.045 | inconclusive |
| D1 all pairs | capsule | NI | 0.010 | -0.000 | -0.001 | 0.001 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | -0.005 | -0.006 | -0.003 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | -0.003 | -0.008 | 0.002 | 0.008 | met |
| D1 all pairs | isic_MSK | NI | 0.010 | -0.002 | -0.006 | 0.002 | 0.000 | met |
| D1 all pairs | isic2020 | NI | 0.010 | -0.021 | -0.027 | -0.016 | 1.000 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.084 | 0.069 | 0.101 | 0.000 | met |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.036 | 0.033 | 0.040 | 0.000 | met |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.070 | 0.059 | 0.082 | 0.000 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.078 | 0.069 | 0.088 | 0.000 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.330 | 0.311 | 0.351 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.153 | 0.130 | 0.177 | 0.000 | met |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.193 | 0.166 | 0.221 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.015 | 0.006 | 0.027 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.022 | 0.014 | 0.030 | 0.000 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.045 | 0.032 | 0.058 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.027 | 0.021 | 0.033 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.015 | -0.000 | 0.026 | 0.000 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.022 | 0.014 | 0.030 | 0.000 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.045 | 0.032 | 0.058 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.027 | 0.020 | 0.033 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.095 | 0.079 | 0.111 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | 0.048 | 0.037 | 0.061 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.039 | 0.023 | 0.056 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.061 | 0.052 | 0.070 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.009 | -0.003 | 0.023 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | 0.007 | 0.002 | 0.012 | 0.000 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | 0.013 | 0.003 | 0.022 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.010 | 0.004 | 0.015 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.103 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.029 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.141 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.056 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.216 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.113 |  |  |  | met |

Losses: D1 all pairs isic2020 -0.021
Inconclusive: D1 all pairs thyroid -0.002

## full_cmc — without ovary

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D1 all pairs | capsule | NI | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | 0.004 | 0.000 | 0.012 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | -0.038 | -0.053 | -0.023 | 1.000 | loss |
| D1 all pairs | isic_MSK | NI | 0.010 | -0.020 | -0.044 | 0.003 | 0.803 | loss |
| D1 all pairs | isic2020 | NI | 0.010 | -0.032 | -0.053 | -0.012 | 0.985 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | inconclusive |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.014 | 0.000 | 0.045 | 0.335 | inconclusive |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.126 | 0.085 | 0.168 | 0.000 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.091 | 0.068 | 0.114 | 0.000 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.041 | 0.026 | 0.058 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.020 | 0.006 | 0.034 | 0.001 | met |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.240 | 0.191 | 0.288 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.029 | -0.003 | 0.065 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.010 | -0.001 | 0.022 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.029 | -0.003 | 0.065 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.010 | -0.001 | 0.022 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | -0.002 | -0.011 | 0.006 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | -0.003 | -0.011 | 0.005 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.072 | 0.044 | 0.101 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.022 | 0.012 | 0.033 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | 0.021 | -0.003 | 0.046 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.007 | -0.001 | 0.015 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.444 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.045 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.270 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.081 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.184 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.165 |  |  |  | met |

Losses: D1 all pairs isic_HAM -0.038; D1 all pairs isic_MSK -0.020; D1 all pairs isic2020 -0.032
Inconclusive: D2 hard pairs thyroid +0.000; D2 hard pairs isic_BCN +0.014

## mask_bal — without ovary

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | -0.008 | -0.017 | 0.001 | 0.287 | loss |
| D1 all pairs | capsule | NI | 0.010 | -0.001 | -0.004 | 0.000 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | 0.003 | 0.000 | 0.007 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | 0.003 | -0.001 | 0.008 | 0.000 | met |
| D1 all pairs | isic_MSK | NI | 0.010 | -0.006 | -0.011 | -0.000 | 0.056 | loss |
| D1 all pairs | isic2020 | NI | 0.010 | -0.011 | -0.016 | -0.005 | 0.645 | loss |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.048 | 0.032 | 0.066 | 0.000 | met |
| D2 hard pairs | isic_BCN | SUP | 0.000 | 0.033 | 0.026 | 0.040 | 0.000 | met |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.045 | 0.030 | 0.058 | 0.000 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.035 | 0.024 | 0.046 | 0.000 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.380 | 0.355 | 0.404 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.166 | 0.148 | 0.185 | 0.000 | met |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.213 | 0.187 | 0.239 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.031 | 0.020 | 0.042 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.022 | 0.013 | 0.032 | 0.000 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.085 | 0.063 | 0.106 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.046 | 0.037 | 0.054 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.031 | -0.012 | 0.041 | 0.000 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.022 | 0.013 | 0.032 | 0.000 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.085 | 0.063 | 0.105 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.046 | 0.031 | 0.053 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.107 | 0.089 | 0.125 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | 0.051 | 0.040 | 0.063 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.044 | 0.026 | 0.062 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.067 | 0.058 | 0.077 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.011 | -0.002 | 0.025 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | 0.003 | -0.003 | 0.010 | 0.000 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | 0.012 | -0.006 | 0.029 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.009 | 0.001 | 0.016 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.027 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.004 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.122 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.047 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.187 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.040 |  |  |  | met |

Losses: D1 all pairs thyroid -0.008; D1 all pairs isic_MSK -0.006; D1 all pairs isic2020 -0.011
Inconclusive: none

## locrand_loc — without ovary

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D1 all pairs | thyroid | NI | 0.010 | -0.001 | -0.007 | 0.005 | 0.003 | met |
| D1 all pairs | capsule | NI | 0.010 | -0.002 | -0.003 | -0.000 | 0.000 | met |
| D1 all pairs | isic_BCN | NI | 0.010 | -0.002 | -0.003 | -0.001 | 0.000 | met |
| D1 all pairs | isic_HAM | NI | 0.010 | 0.000 | -0.001 | 0.002 | 0.000 | met |
| D1 all pairs | isic_MSK | NI | 0.010 | 0.001 | -0.001 | 0.003 | 0.000 | met |
| D1 all pairs | isic2020 | NI | 0.010 | -0.001 | -0.002 | 0.000 | 0.000 | met |
| D2 hard pairs | thyroid | SUP | 0.000 | 0.024 | 0.010 | 0.040 | 0.001 | met |
| D2 hard pairs | isic_BCN | SUP | 0.000 | -0.000 | -0.001 | 0.001 | 0.669 | loss |
| D2 hard pairs | isic_MSK | SUP | 0.000 | 0.005 | 0.002 | 0.009 | 0.002 | met |
| D2 hard pairs | isic2020 | SUP | 0.000 | 0.003 | 0.000 | 0.005 | 0.007 | met |
| D2 Trap A min(rev,corr) | thyroid | SUP | 0.000 | 0.164 | 0.137 | 0.193 | 0.000 | met |
| D2 Trap A min(rev,corr) | capsule | SUP | 0.000 | 0.001 | -0.001 | 0.005 | 0.242 | inconclusive |
| D2 Trap A min(rev,corr) | isic | SUP | 0.000 | 0.029 | 0.021 | 0.037 | 0.000 | met |
| D3 Trap B reversed | thyroid | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B reversed | capsule | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B reversed | isic | NI | 0.030 | 0.000 | -0.000 | 0.000 | 0.000 | met |
| D3 Trap B reversed (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.000 | -0.000 | 0.000 | 0.000 | met |
| D3 Trap B min(rev,corr) | thyroid | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B min(rev,corr) | capsule | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 Trap B min(rev,corr) | isic | NI | 0.030 | 0.000 | -0.000 | 0.000 | 0.000 | met |
| D3 Trap B min(rev,corr) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.000 | -0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap A models) | thyroid | NI | 0.030 | 0.030 | 0.013 | 0.046 | 0.000 | met |
| D3 clean (Trap A models) | capsule | NI | 0.030 | -0.001 | -0.004 | 0.001 | 0.000 | met |
| D3 clean (Trap A models) | isic | NI | 0.030 | 0.007 | 0.002 | 0.012 | 0.000 | met |
| D3 clean (Trap A models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.012 | 0.006 | 0.018 | 0.000 | met |
| D3 clean (Trap B models) | thyroid | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap B models) | capsule | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap B models) | isic | NI | 0.030 | -0.000 | -0.000 | 0.000 | 0.000 | met |
| D3 clean (Trap B models) (pooled) | thyroid+capsule+isic | NI | 0.010 | 0.000 | -0.000 | 0.000 | 0.000 | met |
| D4 no flipping | thyroid trapA | point | 0.020 | 0.284 |  |  |  | met |
| D4 no flipping | thyroid trapB | point | 0.020 | 0.045 |  |  |  | met |
| D4 no flipping | capsule trapA | point | 0.020 | 0.296 |  |  |  | met |
| D4 no flipping | capsule trapB | point | 0.020 | 0.081 |  |  |  | met |
| D4 no flipping | isic trapA | point | 0.020 | 0.439 |  |  |  | met |
| D4 no flipping | isic trapB | point | 0.020 | 0.177 |  |  |  | met |

Losses: D2 hard pairs isic_BCN -0.000
Inconclusive: D2 Trap A min(rev,corr) capsule +0.001

## mask_bal — D5

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D5 r=1 test_rev | thyroid | SUP | 0.000 | 0.467 | 0.431 | 0.503 | 0.000 | met |
| D5 r=0 test_rev | thyroid | NI | 0.030 | -0.002 | -0.006 | 0.000 | 0.000 | met |
| D5 r=0 clean | thyroid | NI | 0.030 | -0.002 | -0.006 | 0.000 | 0.000 | met |
| D5 r=1 test_rev | capsule | SUP | 0.000 | 0.156 | 0.136 | 0.177 | 0.000 | met |
| D5 r=0 test_rev | capsule | NI | 0.030 | -0.001 | -0.003 | 0.001 | 0.000 | met |
| D5 r=0 clean | capsule | NI | 0.030 | -0.001 | -0.003 | 0.001 | 0.000 | met |
| D5 r=1 test_rev | ovary | SUP | 0.000 | 0.114 | 0.076 | 0.152 | 0.000 | met |
| D5 r=0 test_rev | ovary | NI | 0.030 | -0.003 | -0.012 | 0.006 | 0.000 | met |
| D5 r=0 clean | ovary | NI | 0.030 | -0.003 | -0.012 | 0.006 | 0.000 | met |
| D5 r=1 test_rev | isic2018 | SUP | 0.000 | 0.369 | 0.343 | 0.393 | 0.000 | met |
| D5 r=0 test_rev | isic2018 | NI | 0.030 | -0.001 | -0.003 | 0.001 | 0.000 | met |
| D5 r=0 clean | isic2018 | NI | 0.030 | -0.001 | -0.003 | 0.001 | 0.000 | met |

Losses: none
Inconclusive: none

## mask_cmc — D5

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D5 r=1 test_rev | thyroid | SUP | 0.000 | 0.433 | 0.391 | 0.474 | 0.000 | met |
| D5 r=0 test_rev | thyroid | NI | 0.030 | -0.000 | -0.002 | 0.002 | 0.000 | met |
| D5 r=0 clean | thyroid | NI | 0.030 | -0.000 | -0.002 | 0.002 | 0.000 | met |
| D5 r=1 test_rev | capsule | SUP | 0.000 | 0.152 | 0.127 | 0.181 | 0.000 | met |
| D5 r=0 test_rev | capsule | NI | 0.030 | -0.000 | -0.001 | 0.001 | 0.000 | met |
| D5 r=0 clean | capsule | NI | 0.030 | -0.000 | -0.001 | 0.001 | 0.000 | met |
| D5 r=1 test_rev | ovary | SUP | 0.000 | 0.090 | 0.051 | 0.125 | 0.000 | met |
| D5 r=0 test_rev | ovary | NI | 0.030 | -0.003 | -0.010 | 0.001 | 0.000 | met |
| D5 r=0 clean | ovary | NI | 0.030 | -0.003 | -0.010 | 0.001 | 0.000 | met |
| D5 r=1 test_rev | isic2018 | SUP | 0.000 | 0.391 | 0.369 | 0.413 | 0.000 | met |
| D5 r=0 test_rev | isic2018 | NI | 0.030 | -0.002 | -0.005 | 0.000 | 0.000 | met |
| D5 r=0 clean | isic2018 | NI | 0.030 | -0.002 | -0.005 | 0.000 | 0.000 | met |

Losses: none
Inconclusive: none

## full_cmc — D5

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D5 r=1 test_rev | thyroid | SUP | 0.000 | 0.007 | 0.000 | 0.023 | 0.335 | inconclusive |
| D5 r=0 test_rev | thyroid | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D5 r=0 clean | thyroid | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D5 r=1 test_rev | capsule | SUP | 0.000 | 0.003 | -0.014 | 0.021 | 0.431 | inconclusive |
| D5 r=0 test_rev | capsule | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D5 r=0 clean | capsule | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D5 r=1 test_rev | ovary | SUP | 0.000 | 0.027 | -0.005 | 0.061 | 0.050 | inconclusive |
| D5 r=0 test_rev | ovary | NI | 0.030 | -0.017 | -0.035 | -0.003 | 0.068 | loss |
| D5 r=0 clean | ovary | NI | 0.030 | -0.016 | -0.034 | -0.002 | 0.055 | loss |
| D5 r=1 test_rev | isic2018 | SUP | 0.000 | 0.311 | 0.226 | 0.393 | 0.000 | met |
| D5 r=0 test_rev | isic2018 | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |
| D5 r=0 clean | isic2018 | NI | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | met |

Losses: D5 r=0 test_rev ovary -0.017; D5 r=0 clean ovary -0.016
Inconclusive: D5 r=1 test_rev thyroid +0.007; D5 r=1 test_rev capsule +0.003; D5 r=1 test_rev ovary +0.027

## locrand_ft — D6

| component | cohort | test | margin | estimate | ci95_lo | ci95_hi | p_one_sided | label |
|---|---|---|---|---|---|---|---|---|
| D6 all pairs | thyroid (fine-tuned) | NI | 0.020 | -0.006 | -0.023 | 0.010 | 0.053 | inconclusive |
| D6 hard pairs | thyroid (fine-tuned) | SUP | 0.000 | 0.001 | -0.026 | 0.028 | 0.478 | inconclusive |
| D6 Trap A min(rev,corr) | thyroid (fine-tuned) | SUP | 0.000 | 0.119 | 0.088 | 0.150 | 0.000 | met |

Losses: none
Inconclusive: D6 all pairs thyroid (fine-tuned) -0.006; D6 hard pairs thyroid (fine-tuned) +0.001

## Replication (MedSigLIP, ConvNeXt; Trap A min(rev,corr), Holm within)

| encoder | cohort | candidate | estimate | ci95_lo | ci95_hi | margin | test | p_one_sided | met | label | p_holm | met_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| medsiglip448 | thyroid | mask_bal | 0.428 | 0.400 | 0.455 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| medsiglip448 | thyroid | mask_cmc | 0.397 | 0.373 | 0.424 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| medsiglip448 | thyroid | full_cmc | 0.119 | 0.056 | 0.170 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| medsiglip448 | capsule | mask_bal | 0.162 | 0.134 | 0.187 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| medsiglip448 | capsule | mask_cmc | 0.159 | 0.128 | 0.189 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| medsiglip448 | capsule | full_cmc | 0.023 | 0.006 | 0.042 | 0.000 | SUP | 0.004 | True | met | 0.008 | True |
| medsiglip448 | ovary | mask_bal | 0.286 | 0.250 | 0.323 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| medsiglip448 | ovary | mask_cmc | 0.248 | 0.212 | 0.285 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| medsiglip448 | ovary | full_cmc | 0.000 | 0.000 | 0.000 | 0.000 | SUP | 1.000 | False | inconclusive | 1.000 | False |
| convnext384 | thyroid | mask_bal | 0.383 | 0.351 | 0.413 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| convnext384 | thyroid | mask_cmc | 0.318 | 0.296 | 0.341 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| convnext384 | thyroid | full_cmc | 0.125 | 0.090 | 0.161 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| convnext384 | capsule | mask_bal | 0.149 | 0.116 | 0.179 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| convnext384 | capsule | mask_cmc | 0.132 | 0.091 | 0.168 | 0.000 | SUP | 0.000 | True | met | 0.002 | True |
| convnext384 | capsule | full_cmc | 0.035 | 0.013 | 0.059 | 0.000 | SUP | 0.001 | True | met | 0.002 | True |

## Registered checks (section 6)

### Pasted-versus-real probe (DINOv2)

| cohort | pasted_vs_real AUROC | reference plain_vs_real AUROC | distinguishable (> 0.75) |
|---|---|---|---|
| capsule | 0.990 [0.979, 0.998] | 0.955 | True |
| isic2019 | 0.975 [0.963, 0.985] | 0.939 | True |
| ovary | 0.904 [0.855, 0.947] | 0.937 | True |
| thyroid | 0.906 [0.881, 0.930] | 0.952 | True |

**Pasted artifacts are distinguishable from real ones (capsule, isic2019, ovary, thyroid): the locrand / locrand_loc results may not transfer to real artifacts.**

### Placements

| cohort | recipients_placed | versions_mean | donors_used |
|---|---|---|---|
| capsule | 369 | 3.734 | 259 |
| isic2019 | 860 | 3.994 | 3070 |
| ovary | 348 | 4.000 | 555 |
| thyroid | 1800 | 4.000 | 1329 |

### Paste balance (confirmation training sets)

| run | family | training sets | mean |P(A|Y=1)-P(A|Y=0)| after | share > 0.02 | mean pastes |
|---|---|---|---|---|---|
| natural/capsule_dino518 | locrand | 5 | 0.000 | 0.000 | 52.600 |
| natural/capsule_dino518 | locrand_loc | 5 | 0.031 | 0.600 | 235.400 |
| natural/isic2020_dino518 | locrand | 5 | 0.000 | 0.000 | 19.600 |
| natural/isic2020_dino518 | locrand_loc | 5 | 0.000 | 0.000 | 703.400 |
| natural/isic_BCN_dino518 | locrand | 5 | 0.000 | 0.000 | 10.400 |
| natural/isic_BCN_dino518 | locrand_loc | 5 | 0.002 | 0.000 | 217.000 |
| natural/isic_HAM_dino518 | locrand | 5 | 0.000 | 0.000 | 75.800 |
| natural/isic_HAM_dino518 | locrand_loc | 5 | 0.001 | 0.000 | 458.000 |
| natural/isic_MSK_dino518 | locrand | 5 | 0.000 | 0.000 | 22.000 |
| natural/isic_MSK_dino518 | locrand_loc | 5 | 0.000 | 0.000 | 655.400 |
| natural/thyroid_dino518 | locrand | 5 | 0.000 | 0.000 | 135.400 |
| natural/thyroid_dino518 | locrand_loc | 5 | 0.000 | 0.000 | 135.400 |
| trap/capsule_dino518 | locrand | 50 | 0.463 | 1.000 | 39.220 |
| trap/capsule_dino518 | locrand_loc | 50 | 0.463 | 1.000 | 39.220 |
| trap/isic_dino518 | locrand | 50 | 0.419 | 1.000 | 197.560 |
| trap/isic_dino518 | locrand_loc | 50 | 0.419 | 1.000 | 197.560 |
| trap/ovary_dino518 | locrand | 50 | 0.144 | 0.600 | 79.100 |
| trap/ovary_dino518 | locrand_loc | 50 | 0.144 | 0.600 | 79.100 |
| trap/thyroid_dino518 | locrand | 50 | 0.195 | 0.580 | 451.620 |
| trap/thyroid_dino518 | locrand_loc | 50 | 0.195 | 0.580 | 451.620 |

### Fallback rates of the selection rule (confirmation)

| run | family | training sets | fallback to masking |
|---|---|---|---|
| natural/capsule_dino518 | full_cmc | 5 | 1.000 |
| natural/capsule_dino518 | locrand | 5 | 0.600 |
| natural/capsule_dino518 | locrand_loc | 5 | 0.200 |
| natural/capsule_dino518 | mask_bal | 5 | 0.200 |
| natural/capsule_dino518 | mask_cmc | 5 | 0.200 |
| natural/isic2020_dino518 | full_cmc | 5 | 0.000 |
| natural/isic2020_dino518 | locrand | 5 | 0.600 |
| natural/isic2020_dino518 | locrand_loc | 5 | 0.200 |
| natural/isic2020_dino518 | mask_bal | 5 | 0.000 |
| natural/isic2020_dino518 | mask_cmc | 5 | 0.000 |
| natural/isic_BCN_dino518 | full_cmc | 5 | 0.800 |
| natural/isic_BCN_dino518 | locrand | 5 | 0.600 |
| natural/isic_BCN_dino518 | locrand_loc | 5 | 0.200 |
| natural/isic_BCN_dino518 | mask_bal | 5 | 0.000 |
| natural/isic_BCN_dino518 | mask_cmc | 5 | 0.000 |
| natural/isic_HAM_dino518 | full_cmc | 5 | 0.000 |
| natural/isic_HAM_dino518 | locrand | 5 | 0.000 |
| natural/isic_HAM_dino518 | locrand_loc | 5 | 0.400 |
| natural/isic_HAM_dino518 | mask_bal | 5 | 0.000 |
| natural/isic_HAM_dino518 | mask_cmc | 5 | 0.200 |
| natural/isic_MSK_dino518 | full_cmc | 5 | 0.000 |
| natural/isic_MSK_dino518 | locrand | 5 | 0.400 |
| natural/isic_MSK_dino518 | locrand_loc | 5 | 0.000 |
| natural/isic_MSK_dino518 | mask_bal | 5 | 0.000 |
| natural/isic_MSK_dino518 | mask_cmc | 5 | 0.000 |
| natural/thyroid_dino518 | full_cmc | 5 | 1.000 |
| natural/thyroid_dino518 | locrand | 5 | 0.000 |
| natural/thyroid_dino518 | locrand_loc | 5 | 0.200 |
| natural/thyroid_dino518 | mask_bal | 5 | 0.000 |
| natural/thyroid_dino518 | mask_cmc | 5 | 0.000 |
| sweeps/capsule | full_cmc | 50 | 0.920 |
| sweeps/capsule | mask_bal | 50 | 0.240 |
| sweeps/capsule | mask_cmc | 50 | 0.240 |
| sweeps/isic2018 | full_cmc | 50 | 0.580 |
| sweeps/isic2018 | mask_bal | 50 | 0.300 |
| sweeps/isic2018 | mask_cmc | 50 | 0.280 |
| sweeps/ovary | full_cmc | 50 | 0.720 |
| sweeps/ovary | mask_bal | 50 | 0.380 |
| sweeps/ovary | mask_cmc | 50 | 0.340 |
| sweeps/thyroid | full_cmc | 50 | 0.980 |
| sweeps/thyroid | mask_bal | 50 | 0.200 |
| sweeps/thyroid | mask_cmc | 50 | 0.140 |
| trap/capsule_convnext384 | full_cmc | 50 | 0.860 |
| trap/capsule_convnext384 | mask_bal | 50 | 0.040 |
| trap/capsule_convnext384 | mask_cmc | 50 | 0.080 |
| trap/capsule_dino518 | full_cmc | 50 | 0.920 |
| trap/capsule_dino518 | locrand | 50 | 0.460 |
| trap/capsule_dino518 | locrand_loc | 50 | 0.920 |
| trap/capsule_dino518 | mask_bal | 50 | 0.000 |
| trap/capsule_dino518 | mask_cmc | 50 | 0.100 |
| trap/capsule_medsiglip448 | full_cmc | 50 | 0.880 |
| trap/capsule_medsiglip448 | mask_bal | 50 | 0.040 |
| trap/capsule_medsiglip448 | mask_cmc | 50 | 0.060 |
| trap/isic_dino518 | full_cmc | 50 | 0.220 |
| trap/isic_dino518 | locrand | 50 | 0.060 |
| trap/isic_dino518 | locrand_loc | 50 | 0.400 |
| trap/isic_dino518 | mask_bal | 50 | 0.000 |
| trap/isic_dino518 | mask_cmc | 50 | 0.060 |
| trap/ovary_dino518 | full_cmc | 50 | 1.000 |
| trap/ovary_dino518 | locrand | 50 | 0.000 |
| trap/ovary_dino518 | locrand_loc | 50 | 1.000 |
| trap/ovary_dino518 | mask_bal | 50 | 0.000 |
| trap/ovary_dino518 | mask_cmc | 50 | 0.000 |
| trap/ovary_medsiglip448 | full_cmc | 50 | 1.000 |
| trap/ovary_medsiglip448 | mask_bal | 50 | 0.000 |
| trap/ovary_medsiglip448 | mask_cmc | 50 | 0.000 |
| trap/thyroid_convnext384 | full_cmc | 50 | 0.680 |
| trap/thyroid_convnext384 | mask_bal | 50 | 0.140 |
| trap/thyroid_convnext384 | mask_cmc | 50 | 0.120 |
| trap/thyroid_dino518 | full_cmc | 50 | 0.880 |
| trap/thyroid_dino518 | locrand | 50 | 0.220 |
| trap/thyroid_dino518 | locrand_loc | 50 | 0.540 |
| trap/thyroid_dino518 | mask_bal | 50 | 0.080 |
| trap/thyroid_dino518 | mask_cmc | 50 | 0.080 |
| trap/thyroid_medsiglip448 | full_cmc | 50 | 0.780 |
| trap/thyroid_medsiglip448 | mask_bal | 50 | 0.060 |
| trap/thyroid_medsiglip448 | mask_cmc | 50 | 0.040 |
