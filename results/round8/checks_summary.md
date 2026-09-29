
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
