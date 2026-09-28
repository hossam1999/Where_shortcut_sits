<!-- T:targets -->
| component | cohort | type | mask AUROC [95% CI] | best existing arm | best − mask [95% CI] |
|---|---|---|---|---|---|
| natural all pairs | thyroid | NI | 0.731 [0.690, 0.770] | umte | +0.004 [-0.011, +0.020] |
| natural all pairs | capsule | NI | 0.967 [0.955, 0.977] | mask_balanced | -0.003 [-0.006, +0.000] |
| natural all pairs | isic_BCN | NI | 0.761 [0.749, 0.772] | erm | +0.025 [+0.016, +0.035] |
| natural all pairs | isic_HAM | NI | 0.782 [0.767, 0.797] | umte_balanced | +0.018 [+0.009, +0.028] |
| natural all pairs | isic_MSK | NI | 0.785 [0.764, 0.806] | umte_protect | -0.003 [-0.012, +0.006] |
| external all pairs | isic2019_to_2020 | NI | 0.807 [0.788, 0.825] | umte_protect | -0.006 [-0.015, +0.003] |
| hard pairs | thyroid | SUP | 0.583 [0.514, 0.651] | balanced | +0.136 [+0.063, +0.207] |
| hard pairs | isic_BCN | SUP | 0.690 [0.672, 0.707] | balanced | +0.043 [+0.026, +0.059] |
| hard pairs | isic_MSK | SUP | 0.631 [0.584, 0.679] | balanced | +0.100 [+0.057, +0.143] |
| hard pairs | isic2019_to_2020 | SUP | 0.735 [0.707, 0.761] | balanced | +0.072 [+0.048, +0.096] |
| Trap A reversed | thyroid | SUP | 0.398 [0.371, 0.425] | mask_dfr | +0.428 [+0.398, +0.457] |
| Trap B reversed | thyroid | NI | 0.752 [0.720, 0.785] | mask_balanced | +0.035 [+0.019, +0.049] |
| Trap A min(rev,corr) | thyroid | NI | 0.398 [0.371, 0.425] | mask_balanced | +0.392 [+0.360, +0.416] |
| Trap B min(rev,corr) | thyroid | NI | 0.752 [0.719, 0.782] | umte_balanced | +0.013 [-0.029, +0.038] |
| clean (Trap A models) | thyroid | NI | 0.689 [0.667, 0.711] | umte_balanced | +0.109 [+0.089, +0.129] |
| clean (Trap B models) | thyroid | NI | 0.767 [0.731, 0.801] | mask_balanced | +0.006 [-0.009, +0.020] |
| Trap A reversed | capsule | SUP | 0.665 [0.629, 0.701] | mask_dfr | +0.228 [+0.198, +0.259] |
| Trap B reversed | capsule | NI | 0.890 [0.865, 0.913] | mask_balanced | +0.030 [+0.022, +0.038] |
| Trap A min(rev,corr) | capsule | NI | 0.665 [0.629, 0.701] | mask_dfr | +0.226 [+0.181, +0.254] |
| Trap B min(rev,corr) | capsule | NI | 0.890 [0.865, 0.913] | mask_balanced | +0.030 [+0.022, +0.038] |
| clean (Trap A models) | capsule | NI | 0.860 [0.837, 0.880] | mask_balanced | +0.053 [+0.042, +0.065] |
| clean (Trap B models) | capsule | NI | 0.932 [0.917, 0.945] | mask_balanced | +0.008 [+0.003, +0.012] |
| Trap A reversed | ovary | SUP | 0.545 [0.494, 0.598] | mask_dfr | +0.206 [+0.163, +0.251] |
| Trap B reversed | ovary | NI | 0.725 [0.667, 0.781] | umte_protect_balanced | +0.051 [+0.018, +0.084] |
| Trap A min(rev,corr) | ovary | NI | 0.545 [0.494, 0.598] | mask_dfr | +0.206 [+0.142, +0.247] |
| Trap B min(rev,corr) | ovary | NI | 0.725 [0.664, 0.770] | umte_balanced | +0.029 [-0.016, +0.067] |
| clean (Trap A models) | ovary | NI | 0.690 [0.656, 0.723] | umte_balanced | +0.084 [+0.047, +0.118] |
| clean (Trap B models) | ovary | NI | 0.733 [0.682, 0.779] | umte | +0.032 [-0.001, +0.068] |
| Trap A reversed | isic_hair | SUP | 0.460 [0.417, 0.505] | balanced | +0.262 [+0.220, +0.305] |
| Trap B reversed | isic_hair | NI | 0.608 [0.567, 0.648] | umte_balanced | +0.142 [+0.112, +0.173] |
| Trap A min(rev,corr) | isic_hair | NI | 0.460 [0.417, 0.505] | balanced | +0.262 [+0.220, +0.305] |
| Trap B min(rev,corr) | isic_hair | NI | 0.608 [0.567, 0.648] | umte_balanced | +0.126 [+0.078, +0.164] |
| clean (Trap A models) | isic_hair | NI | 0.712 [0.682, 0.741] | balanced | +0.073 [+0.042, +0.103] |
| clean (Trap B models) | isic_hair | NI | 0.701 [0.673, 0.730] | balanced | +0.051 [+0.020, +0.082] |
| fine-tuned all pairs | thyroid | NI | 0.745 [0.707, 0.782] | erm | +0.024 [-0.016, +0.063] |
| fine-tuned hard pairs | thyroid | SUP | 0.621 [0.548, 0.692] | erm | +0.102 [+0.026, +0.179] |

<!-- T:criterion_summary -->
| arm | cells_run | cells_met | cells_total |
|---|---|---|---|
| mask_balanced | 36 | 29 | 36 |
| umte_balanced | 34 | 26 | 36 |
| umte_protect_balanced | 34 | 22 | 36 |
| balanced | 36 | 19 | 36 |
| dfr | 34 | 13 | 36 |
| mask_dfr | 34 | 13 | 36 |
| umte | 34 | 13 | 36 |
| umte_protect | 34 | 13 | 36 |
| umte_aug | 24 | 13 | 36 |
| mask_jtt | 24 | 12 | 36 |
| erm | 36 | 9 | 36 |
| jtt | 34 | 6 | 36 |
| umte_jtt | 24 | 6 | 36 |

<!-- T:criterion_fails -->
| arm | component | cohort | kind | arm − mask [95% CI] |
|---|---|---|---|---|
| mask_balanced | natural all pairs | thyroid | NI | -0.010 [-0.020, +0.000] |
| umte | natural all pairs | thyroid | NI | +0.004 [-0.011, +0.020] |
| umte_balanced | natural all pairs | thyroid | NI | -0.005 [-0.023, +0.013] |
| umte | natural all pairs | capsule | NI | -0.005 [-0.011, +0.001] |
| umte_balanced | natural all pairs | capsule | NI | -0.008 [-0.013, -0.002] |
| mask_balanced | natural all pairs | isic_MSK | NI | -0.006 [-0.011, -0.002] |
| umte | natural all pairs | isic_MSK | NI | -0.008 [-0.014, -0.002] |
| umte_balanced | natural all pairs | isic_MSK | NI | -0.016 [-0.024, -0.008] |
| mask_balanced | external all pairs | isic2019_to_2020 | NI | -0.018 [-0.022, -0.013] |
| umte | external all pairs | isic2019_to_2020 | NI | -0.013 [-0.018, -0.007] |
| umte_balanced | external all pairs | isic2019_to_2020 | NI | -0.033 [-0.040, -0.026] |
| umte | hard pairs | thyroid | SUP | +0.029 [-0.000, +0.060] |
| umte | hard pairs | isic_BCN | SUP | +0.005 [-0.002, +0.011] |
| umte | hard pairs | isic_MSK | SUP | -0.013 [-0.026, -0.001] |
| umte | hard pairs | isic2019_to_2020 | SUP | +0.006 [-0.002, +0.013] |
| umte | Trap B reversed | thyroid | NI | -0.001 [-0.019, +0.019] |
| umte | Trap B min(rev,corr) | thyroid | NI | -0.001 [-0.019, +0.020] |
| mask_balanced | Trap B min(rev,corr) | thyroid | NI | +0.009 [-0.033, +0.043] |
| umte_balanced | Trap B min(rev,corr) | thyroid | NI | +0.013 [-0.029, +0.038] |
| umte | clean (Trap B models) | thyroid | NI | -0.002 [-0.021, +0.017] |
| umte_balanced | clean (Trap B models) | thyroid | NI | -0.000 [-0.025, +0.022] |
| umte | Trap A reversed | capsule | SUP | -0.126 [-0.145, -0.107] |
| umte | Trap B reversed | capsule | NI | -0.020 [-0.034, -0.007] |
| umte | Trap A min(rev,corr) | capsule | NI | -0.126 [-0.145, -0.107] |
| umte | Trap B min(rev,corr) | capsule | NI | -0.020 [-0.034, -0.007] |
| umte | clean (Trap A models) | capsule | NI | -0.045 [-0.056, -0.034] |
| umte | clean (Trap B models) | capsule | NI | -0.013 [-0.021, -0.005] |
| umte_balanced | clean (Trap B models) | capsule | NI | -0.010 [-0.028, +0.004] |
| umte | Trap A reversed | ovary | SUP | +0.020 [-0.016, +0.053] |
| umte | Trap B reversed | ovary | NI | +0.015 [-0.023, +0.052] |
| umte | Trap A min(rev,corr) | ovary | NI | +0.020 [-0.016, +0.053] |
| umte | Trap B min(rev,corr) | ovary | NI | +0.015 [-0.021, +0.062] |
| mask_balanced | Trap B min(rev,corr) | ovary | NI | +0.006 [-0.038, +0.048] |
| umte_balanced | Trap B min(rev,corr) | ovary | NI | +0.029 [-0.016, +0.067] |
| mask_balanced | fine-tuned all pairs | thyroid | NI | -0.016 [-0.033, +0.000] |
| mask_balanced | fine-tuned hard pairs | thyroid | SUP | -0.017 [-0.058, +0.019] |

<!-- T:precision -->
| component | min half-width | median half-width | max half-width |
|---|---|---|---|
| Trap A min(rev,corr) | 0.005 | 0.032 | 0.076 |
| Trap A reversed | 0.005 | 0.03 | 0.076 |
| Trap B min(rev,corr) | 0.004 | 0.04 | 0.084 |
| Trap B reversed | 0.004 | 0.034 | 0.084 |
| clean (Trap A models) | 0.002 | 0.022 | 0.044 |
| clean (Trap B models) | 0.002 | 0.028 | 0.068 |
| external all pairs | 0.005 | 0.012 | 0.026 |
| fine-tuned all pairs | 0.016 | 0.039 | 0.042 |
| fine-tuned hard pairs | 0.039 | 0.071 | 0.076 |
| hard pairs | 0.005 | 0.024 | 0.089 |
| natural all pairs | 0.003 | 0.014 | 0.057 |

<!-- T:decomposition -->
| cohort | arm | π easy | π hard | easy part | hard part | same-artifact part | Δ all pairs |
|---|---|---|---|---|---|---|---|
| capsule | balanced | 0.208 | 0.087 | -0.019 | -0.001 | -0.066 | -0.086 |
| capsule | mask_balanced | 0.208 | 0.087 | -0.002 | +0.001 | -0.003 | -0.003 |
| capsule | umte | 0.208 | 0.087 | -0.000 | +0.000 | -0.005 | -0.005 |
| capsule | umte_balanced | 0.208 | 0.087 | -0.002 | +0.001 | -0.007 | -0.008 |
| isic_BCN | balanced | 0.271 | 0.154 | +0.001 | +0.007 | +0.014 | +0.022 |
| isic_BCN | mask_balanced | 0.271 | 0.154 | -0.005 | +0.005 | +0.003 | +0.003 |
| isic_BCN | umte | 0.271 | 0.154 | +0.002 | +0.001 | +0.004 | +0.007 |
| isic_BCN | umte_balanced | 0.271 | 0.154 | -0.003 | +0.006 | +0.008 | +0.010 |
| isic_HAM | balanced | 0.224 | 0.134 | -0.006 | -0.002 | -0.023 | -0.032 |
| isic_HAM | mask_balanced | 0.224 | 0.134 | -0.004 | +0.006 | +0.004 | +0.005 |
| isic_HAM | umte | 0.224 | 0.134 | +0.002 | +0.003 | +0.009 | +0.013 |
| isic_HAM | umte_balanced | 0.224 | 0.134 | -0.003 | +0.008 | +0.013 | +0.018 |
| isic_MSK | balanced | 0.101 | 0.066 | -0.013 | +0.006 | -0.012 | -0.018 |
| isic_MSK | mask_balanced | 0.101 | 0.066 | -0.003 | +0.003 | -0.006 | -0.006 |
| isic_MSK | umte | 0.101 | 0.066 | -0.000 | -0.001 | -0.007 | -0.008 |
| isic_MSK | umte_balanced | 0.101 | 0.066 | -0.003 | +0.003 | -0.015 | -0.016 |
| thyroid | balanced | 0.290 | 0.187 | -0.032 | +0.025 | -0.011 | -0.017 |
| thyroid | mask_balanced | 0.290 | 0.187 | -0.014 | +0.010 | -0.005 | -0.010 |
| thyroid | umte | 0.290 | 0.187 | -0.002 | +0.005 | +0.001 | +0.004 |
| thyroid | umte_balanced | 0.290 | 0.187 | -0.013 | +0.013 | -0.005 | -0.005 |
| isic2019_to_2020 | balanced | 0.129 | 0.055 | -0.013 | +0.004 | -0.018 | -0.027 |
| isic2019_to_2020 | mask_balanced | 0.129 | 0.055 | -0.008 | +0.002 | -0.012 | -0.018 |
| isic2019_to_2020 | umte | 0.129 | 0.055 | -0.002 | +0.000 | -0.011 | -0.013 |
| isic2019_to_2020 | umte_balanced | 0.129 | 0.055 | -0.011 | +0.003 | -0.025 | -0.033 |
| thyroid | balanced | 0.290 | 0.187 | -0.012 | +0.017 | -0.003 | +0.002 |
| thyroid | mask_balanced | 0.290 | 0.187 | -0.004 | -0.003 | -0.009 | -0.016 |

<!-- T:sweeps -->
| sweep | encoder | overlap | mask reversed AUROC | best arm | best − mask |
|---|---|---|---|---|---|
| capsule | DINOv2-B/14@518 | r=0 | 0.938 [0.892, 0.975] | umte | +0.013 [-0.002, +0.031] |
| capsule | DINOv2-B/14@518 | r=1 | 0.685 [0.570, 0.791] | umte_balanced | +0.211 [+0.132, +0.296] |
| capsule | MedSigLIP@448 | r=0 | 0.960 [0.929, 0.984] | mte_tpl | -0.003 [-0.030, +0.021] |
| capsule | MedSigLIP@448 | r=1 | 0.518 [0.398, 0.635] | leace | +0.391 [+0.279, +0.504] |
| isic2018 | DINOv2-B/14@224 | r=0 | 0.798 [0.737, 0.855] | leace | +0.032 [-0.023, +0.088] |
| isic2018 | DINOv2-B/14@224 | r=1 | 0.295 [0.224, 0.371] | leace | +0.521 [+0.447, +0.592] |
| isic2018 | DINOv2-B/14@518 | r=0 | 0.849 [0.797, 0.894] | leace | -0.017 [-0.067, +0.033] |
| isic2018 | DINOv2-B/14@518 | r=1 | 0.386 [0.315, 0.461] | leace | +0.423 [+0.353, +0.491] |
| isic2018 | DermLIP@224 | r=0 | 0.858 [0.809, 0.903] | leace | +0.088 [+0.047, +0.133] |
| isic2018 | DermLIP@224 | r=1 | 0.311 [0.242, 0.386] | leace | +0.627 [+0.555, +0.694] |
| nih_ptx | DINOv2-B/14@518 | r=0 | 0.883 [0.864, 0.902] | umte | +0.000 [-0.006, +0.007] |
| nih_ptx | DINOv2-B/14@518 | r=1 | 0.327 [0.298, 0.356] | mte_tpl | +0.553 [+0.528, +0.579] |
| nih_ptx | RAD-DINO@518 | r=0 | 0.910 [0.892, 0.927] | leace | +0.005 [-0.002, +0.013] |
| nih_ptx | RAD-DINO@518 | r=1 | 0.725 [0.691, 0.758] | balanced | +0.199 [+0.174, +0.225] |
| ovary | DINOv2-B/14@518 | r=0 | 0.697 [0.559, 0.825] | umte | +0.060 [+0.011, +0.118] |
| ovary | DINOv2-B/14@518 | r=1 | 0.438 [0.297, 0.589] | balanced | +0.283 [+0.122, +0.437] |
| thyroid | DINOv2-B/14@518 | r=0 | 0.819 [0.759, 0.876] | mte_tpl | +0.006 [-0.016, +0.029] |
| thyroid | DINOv2-B/14@518 | r=1 | 0.294 [0.220, 0.373] | mte_tpl_balanced | +0.501 [+0.424, +0.577] |
| thyroid | MedSigLIP@448 | r=0 | 0.834 [0.776, 0.885] | mte_tpl | -0.001 [-0.023, +0.022] |
| thyroid | MedSigLIP@448 | r=1 | 0.305 [0.224, 0.388] | leace | +0.473 [+0.384, +0.562] |

<!-- T:other_encoders -->
| cohort | encoder | run | trap | mask reversed AUROC | best arm | best − mask |
|---|---|---|---|---|---|---|
| capsule | ConvNeXt@384 | convnext384_universal | trapA | 0.612 [0.572, 0.651] | umte_dfr | +0.252 [+0.220, +0.283] |
| capsule | ConvNeXt@384 | convnext384_universal | trapB | 0.835 [0.803, 0.865] | mask_dfr | +0.049 [+0.028, +0.072] |
| capsule | DINOv2-L/14@518 | dinol518_scale | trapA | 0.616 [0.577, 0.655] | umte_balanced | +0.222 [+0.200, +0.245] |
| capsule | DINOv2-L/14@518 | dinol518_scale | trapB | 0.865 [0.837, 0.891] | umte_balanced | +0.025 [+0.008, +0.042] |
| capsule | DINOv2-S/14@518 | dinos518_scale | trapA | 0.626 [0.590, 0.663] | umte_balanced | +0.192 [+0.165, +0.219] |
| capsule | DINOv2-S/14@518 | dinos518_scale | trapB | 0.875 [0.848, 0.901] | umte_balanced | +0.026 [+0.007, +0.046] |
| capsule | MedSigLIP@448 | medsiglip448_main | trapA | 0.687 [0.651, 0.723] | prevcal | +0.245 [+0.212, +0.278] |
| capsule | MedSigLIP@448 | medsiglip448_main | trapB | 0.916 [0.893, 0.937] | prevcal | -0.011 [-0.036, +0.016] |
| capsule | MedSigLIP@448 | medsiglip448_universal | trapA | 0.687 [0.651, 0.723] | mask_dfr | +0.228 [+0.201, +0.257] |
| capsule | MedSigLIP@448 | medsiglip448_universal | trapB | 0.916 [0.893, 0.936] | umte_protect_balanced | +0.015 [+0.003, +0.026] |
| isic_hair | DINOv2-L/14@518 | dinol518_spec_scale | trapA | 0.459 [0.416, 0.502] | balanced | +0.272 [+0.237, +0.307] |
| isic_hair | DINOv2-L/14@518 | dinol518_spec_scale | trapB | 0.608 [0.567, 0.647] | umte_balanced | +0.138 [+0.110, +0.167] |
| isic_hair | DINOv2-S/14@518 | dinos518_spec_scale | trapA | 0.466 [0.422, 0.510] | balanced | +0.257 [+0.217, +0.298] |
| isic_hair | DINOv2-S/14@518 | dinos518_spec_scale | trapB | 0.622 [0.581, 0.662] | umte_balanced | +0.126 [+0.092, +0.159] |
| isic_hair | DermLIP@224 | dermlip224_spec | trapA | 0.561 [0.513, 0.609] | prevcal | +0.386 [+0.341, +0.432] |
| isic_hair | DermLIP@224 | dermlip224_spec | trapB | 0.644 [0.604, 0.685] | prevcal | +0.310 [+0.269, +0.351] |
| isic_hair | DermLIP@224 | dermlip224_spec_universal | trapA | 0.561 [0.512, 0.610] | balanced | +0.250 [+0.208, +0.293] |
| isic_hair | DermLIP@224 | dermlip224_spec_universal | trapB | 0.644 [0.603, 0.684] | balanced | +0.200 [+0.159, +0.241] |
| ovary | DINOv2-L/14@518 | dinol518_scale | trapA | 0.471 [0.426, 0.514] | umte_balanced | +0.243 [+0.205, +0.282] |
| ovary | DINOv2-L/14@518 | dinol518_scale | trapB | 0.709 [0.649, 0.767] | umte_balanced | +0.032 [+0.002, +0.061] |
| ovary | DINOv2-S/14@518 | dinos518_scale | trapA | 0.528 [0.480, 0.578] | umte_balanced | +0.257 [+0.214, +0.299] |
| ovary | DINOv2-S/14@518 | dinos518_scale | trapB | 0.743 [0.685, 0.800] | umte_protect | +0.020 [+0.002, +0.040] |
| ovary | MedSigLIP@448 | medsiglip448_universal | trapA | 0.428 [0.384, 0.473] | umte_dfr | +0.382 [+0.338, +0.427] |
| ovary | MedSigLIP@448 | medsiglip448_universal | trapB | 0.732 [0.674, 0.789] | umte_dfr | +0.075 [+0.032, +0.119] |
| thyroid | ConvNeXt@384 | convnext384_universal | trapA | 0.353 [0.328, 0.379] | umte_dfr | +0.429 [+0.402, +0.457] |
| thyroid | ConvNeXt@384 | convnext384_universal | trapB | 0.700 [0.665, 0.735] | umte_dfr | +0.044 [+0.010, +0.077] |
| thyroid | DINOv2-L/14@518 | dinol518_scale | trapA | 0.381 [0.355, 0.409] | umte_balanced | +0.419 [+0.390, +0.451] |
| thyroid | DINOv2-L/14@518 | dinol518_scale | trapB | 0.745 [0.713, 0.776] | umte_balanced | +0.027 [+0.007, +0.048] |
| thyroid | DINOv2-S/14@518 | dinos518_scale | trapA | 0.420 [0.393, 0.446] | umte_balanced | +0.375 [+0.353, +0.397] |
| thyroid | DINOv2-S/14@518 | dinos518_scale | trapB | 0.744 [0.712, 0.776] | umte_balanced | +0.019 [-0.010, +0.046] |
| thyroid | MedSigLIP@448 | medsiglip448_main | trapA | 0.341 [0.318, 0.366] | prevcal | +0.549 [+0.520, +0.578] |
| thyroid | MedSigLIP@448 | medsiglip448_main | trapB | 0.729 [0.696, 0.761] | prevcal | +0.191 [+0.156, +0.225] |
| thyroid | MedSigLIP@448 | medsiglip448_universal | trapA | 0.341 [0.317, 0.366] | umte_dfr | +0.468 [+0.442, +0.492] |
| thyroid | MedSigLIP@448 | medsiglip448_universal | trapB | 0.729 [0.696, 0.761] | umte_balanced | +0.065 [+0.044, +0.087] |

<!-- T:corr_cost -->
| cohort | cell | arm | arm − mask [95% CI] |
|---|---|---|---|
| capsule | trapA | balanced | -0.056 [-0.085, -0.030] |
| capsule | trapA | mask_balanced | -0.017 [-0.026, -0.009] |
| capsule | trapA | umte_balanced | -0.032 [-0.051, -0.017] |
| capsule | trapB | balanced | -0.083 [-0.111, -0.058] |
| capsule | trapB | mask_balanced | -0.008 [-0.013, -0.004] |
| capsule | trapB | umte_balanced | -0.025 [-0.040, -0.013] |
| isic_hair | trapA | balanced | -0.068 [-0.089, -0.049] |
| isic_hair | trapA | mask_balanced | -0.062 [-0.088, -0.038] |
| isic_hair | trapA | umte_balanced | -0.057 [-0.074, -0.039] |
| isic_hair | trapB | balanced | +0.005 [-0.022, +0.032] |
| isic_hair | trapB | mask_balanced | -0.058 [-0.091, -0.025] |
| isic_hair | trapB | umte_balanced | -0.061 [-0.092, -0.032] |
| ovary | trapA | balanced | -0.103 [-0.155, -0.046] |
| ovary | trapA | mask_balanced | -0.020 [-0.053, +0.021] |
| ovary | trapA | umte_balanced | -0.020 [-0.063, +0.027] |
| ovary | trapB | balanced | -0.042 [-0.118, +0.032] |
| ovary | trapB | mask_balanced | -0.021 [-0.049, +0.009] |
| ovary | trapB | umte_balanced | +0.002 [-0.033, +0.039] |
| thyroid | trapA | balanced | -0.151 [-0.186, -0.118] |
| thyroid | trapA | mask_balanced | -0.111 [-0.132, -0.090] |
| thyroid | trapA | umte_balanced | -0.104 [-0.126, -0.083] |
| thyroid | trapB | balanced | -0.125 [-0.172, -0.077] |
| thyroid | trapB | mask_balanced | -0.024 [-0.043, -0.005] |
| thyroid | trapB | umte_balanced | -0.020 [-0.052, +0.008] |
