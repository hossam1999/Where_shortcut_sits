# Round 6

Commit `2caabe604d6e150e6ab3d372c6e9422c0ed3dacb`. GPU: NVIDIA RTX PRO 4000 Blackwell.

## Sources and coverage
| source | licence | matched | note |
|---|---|---|---|
| BUSClean | MIT | 4695 |  |
| MedGemma 1.5 4B-it | model terms of use | 0 |  |
| Kabir hair masks | CC BY 4.0 | 485 | Already on the machine (Amendment 1). Licence read from the Mendeley dataset page. |
| Multicentre clear/contaminated capsule masks | CC BY 4.0 | 29 |  |
| DermArtifactDB | CC BY 4.0 | 20519 | Licence read from README.md inside the Zenodo archive before the labels were used. Heatmaps were not extracted. |
| IMA++ | cc-by-nc-nd-4.0 | 241 | CC BY-NC-ND: measurement only. Masks and pixel-level derivatives stay under the data root and are never committed. |

| cohort | source | n | gate |
|---|---|---|---|
| isic | DermArtifactDB | 20519 | analysed |
| isic | IMA++ | 241 | analysed |
| isic | Kabir | 485 | analysed |
| capsule | Mendeley_vmxhn95j8z | 29 | not_feasible |
| thyroid | BUSClean | 3493 | analysed |
| ovary | BUSClean | 1202 | analysed |
| thyroid | MedGemma | 3492 | analysed |
| ovary | MedGemma | 1198 | analysed |

Gate (Amendment 1): ≥200 analysed as registered, 50–199 descriptive only, <50 not feasible.

## Agreement
### isic
```json
{
  "cohort": "isic",
  "derm_kappa": {
    "all": {
      "estimate": 0.07579205864816269,
      "ci95_lo": 0.06972656240985095,
      "ci95_hi": 0.08157959610878171,
      "n": 20519,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.08299687634103907,
      "ci95_lo": 0.07581870770960963,
      "ci95_hi": 0.0900728292297856,
      "n": 16788,
      "n_boot": 2000
    },
    "y1": {
      "estimate": 0.05051466120575197,
      "ci95_lo": 0.04134650119075958,
      "ci95_hi": 0.059831444771276124,
      "n": 3731,
      "n_boot": 2000
    }
  },
  "ima_dice": {
    "all": {
      "estimate": 0.8513349618654565,
      "ci95_lo": 0.829700080124656,
      "ci95_hi": 0.8704650701823344,
      "n": 241,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.8510280006086165,
      "ci95_lo": 0.8251794946394916,
      "ci95_hi": 0.87238819939721,
      "n": 208,
      "n_boot": 2000
    },
    "y1": {
      "estimate": 0.8532697479691757,
      "ci95_lo": 0.8185347692675867,
      "ci95_hi": 0.8881424971354862,
      "n": 33,
      "n_boot": 2000
    }
  },
  "ima_iou": {
    "all": {
      "estimate": 0.767307980700534,
      "ci95_lo": 0.742050514663776,
      "ci95_hi": 0.7903301171174686,
      "n": 241,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.7688741772625008,
      "ci95_lo": 0.7393532005261985,
      "ci95_hi": 0.7951394888339034,
      "n": 208,
      "n_boot": 2000
    },
    "y1": {
      "estimate": 0.7574361963099538,
      "ci95_lo": 0.7054243598037195,
      "ci95_hi": 0.8093624525908913,
      "n": 33,
      "n_boot": 2000
    }
  },
  "ima_trap_class": {
    "all": {
      "estimate": 0.8410041841004184,
      "ci95_lo": 0.7907949790794979,
      "ci95_hi": 0.8870292887029289,
      "n": 239,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.855072463768116,
      "ci95_lo": 0.8067632850241546,
      "ci95_hi": 0.8985507246376812,
      "n": 207,
      "n_boot": 2000
    },
    "y1": {
      "estimate": 0.75,
      "ci95_lo": 0.59375,
      "ci95_hi": 0.875,
      "n": 32,
      "n_boot": 2000
    }
  },
  "ima_dice_by_roi_source": {
    "MSK": {
      "estimate": 0.8513349618654565,
      "ci95_lo": 0.829700080124656,
      "ci95_hi": 0.8704650701823344,
      "n": 241,
      "n_boot": 2000
    }
  },
  "kabir_dice": {
    "all": {
      "estimate": 0.6183865431578199,
      "ci95_lo": 0.5995366604792202,
      "ci95_hi": 0.6367709182017555,
      "n": 485,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.6165181435457368,
      "ci95_lo": 0.5950586038309382,
      "ci95_hi": 0.6377167600252539,
      "n": 412,
      "n_boot": 2000
    },
    "y1": {
      "estimate": 0.6289314834342336,
      "ci95_lo": 0.5916853652020572,
      "ci95_hi": 0.6651738632073788,
      "n": 73,
      "n_boot": 2000
    }
  },
  "kabir_kappa": {
    "all": {
      "estimate": 0.0,
      "ci95_lo": 0.0,
      "ci95_hi": 0.0,
      "n": 485,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.0,
      "ci95_lo": 0.0,
      "ci95_hi": 0.0,
      "n": 412,
      "n_boot": 2000
    },
    "y1": {
      "estimate": NaN,
      "ci95_lo": NaN,
      "ci95_hi": NaN,
      "n": 73,
      "n_boot": 0
    }
  },
  "informative_sources": [
    "cell_derm",
    "cell_ima",
    "cell_kabir"
  ],
  "gates": {
    "DermArtifactDB": "analysed",
    "IMA++": "analysed",
    "Kabir": "analysed"
  }
}
```

### thyroid
```json
{
  "cohort": "thyroid",
  "busclean_kappa": {
    "all": {
      "estimate": 0.482615222829655,
      "ci95_lo": 0.45561174377709246,
      "ci95_hi": 0.5089952959689544,
      "n": 3492,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.4406354353083524,
      "ci95_lo": 0.4078639649999519,
      "ci95_hi": 0.4724775887675859,
      "n": 2282,
      "n_boot": 2000
    },
    "y1": {
      "estimate": 0.5623632045451019,
      "ci95_lo": 0.5154524346459913,
      "ci95_hi": 0.6101710016897784,
      "n": 1210,
      "n_boot": 2000
    }
  },
  "busclean_location": {
    "all": {
      "estimate": 0.8181818181818182,
      "ci95_lo": 0.7896103896103897,
      "ci95_hi": 0.8454545454545455,
      "n": 770,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.800369685767098,
      "ci95_lo": 0.7652495378927912,
      "ci95_hi": 0.833641404805915,
      "n": 541,
      "n_boot": 2000
    },
    "y1": {
      "estimate": 0.8602620087336245,
      "ci95_lo": 0.8122270742358079,
      "ci95_hi": 0.9039301310043668,
      "n": 229,
      "n_boot": 2000
    }
  },
  "busclean_positive_rate_on_caliper_free": 0.05219322598556358,
  "medgemma_kappa": {
    "all": {
      "estimate": 0.6628845376171416,
      "ci95_lo": 0.638455059717024,
      "ci95_hi": 0.6858348918718679,
      "n": 3492,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.6318225388560057,
      "ci95_lo": 0.5990704088002639,
      "ci95_hi": 0.6623709115435747,
      "n": 2282,
      "n_boot": 2000
    },
    "y1": {
      "estimate": 0.6933704457600603,
      "ci95_lo": 0.6551925789655394,
      "ci95_hi": 0.7330412132505622,
      "n": 1210,
      "n_boot": 2000
    }
  },
  "medgemma_location": {
    "all": {
      "estimate": 0.6525722339675828,
      "ci95_lo": 0.6272022551092319,
      "ci95_hi": 0.6765503875968991,
      "n": 1419,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.6434699714013346,
      "ci95_lo": 0.6139180171591992,
      "ci95_hi": 0.6720924690181124,
      "n": 1049,
      "n_boot": 2000
    },
    "y1": {
      "estimate": 0.6783783783783783,
      "ci95_lo": 0.6297297297297297,
      "ci95_hi": 0.7243243243243244,
      "n": 370,
      "n_boot": 2000
    }
  },
  "informative": {
    "BUSClean": {
      "status": "informative",
      "kappa": 0.5045849877103421,
      "n": 2899
    },
    "MedGemma": {
      "status": "informative",
      "kappa": 0.6536918491094588,
      "n": 2601
    }
  },
  "informative_sources": [
    "cell_bus",
    "cell_mg"
  ],
  "gates": {
    "BUSClean": "analysed",
    "MedGemma": "analysed"
  }
}
```

### capsule
```json
{
  "cohort": "capsule",
  "capsule_excluded_expert_labelled": 0,
  "capsule_iou": {
    "all": {
      "estimate": 0.0,
      "ci95_lo": 0.0,
      "ci95_hi": 0.0,
      "n": 29,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.0,
      "ci95_lo": 0.0,
      "ci95_hi": 0.0,
      "n": 8,
      "n_boot": 2000
    },
    "y1": {
      "estimate": 0.0,
      "ci95_lo": 0.0,
      "ci95_hi": 0.0,
      "n": 21,
      "n_boot": 2000
    }
  },
  "informative_sources": [],
  "gates": {
    "Mendeley_vmxhn95j8z": "not_feasible"
  }
}
```

### ovary
```json
{
  "cohort": "ovary",
  "busclean_kappa": {
    "all": {
      "estimate": 0.09846646147170873,
      "ci95_lo": 0.056716939874824596,
      "ci95_hi": 0.1394728003685184,
      "n": 1198,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.14142205756625845,
      "ci95_lo": 0.07872700857685923,
      "ci95_hi": 0.20830313155002195,
      "n": 618,
      "n_boot": 2000
    },
    "y1": {
      "estimate": 0.056780595369349714,
      "ci95_lo": 0.0039374199279251015,
      "ci95_hi": 0.11424086382779217,
      "n": 580,
      "n_boot": 2000
    }
  },
  "busclean_location": {
    "all": {
      "estimate": 0.8121059268600253,
      "ci95_lo": 0.7843631778058008,
      "ci95_hi": 0.8398802017654475,
      "n": 793,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.823943661971831,
      "ci95_lo": 0.7887323943661971,
      "ci95_hi": 0.8591549295774648,
      "n": 426,
      "n_boot": 2000
    },
    "y1": {
      "estimate": 0.7983651226158038,
      "ci95_lo": 0.7547683923705722,
      "ci95_hi": 0.8392370572207084,
      "n": 367,
      "n_boot": 2000
    }
  },
  "busclean_positive_rate_on_caliper_free": 0.9051724137931034,
  "medgemma_kappa": {
    "all": {
      "estimate": 0.19427568636533707,
      "ci95_lo": 0.1475854980345875,
      "ci95_hi": 0.24660393048846677,
      "n": 1198,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.29396719298843904,
      "ci95_lo": 0.21896801545909386,
      "ci95_hi": 0.37460308090782274,
      "n": 618,
      "n_boot": 2000
    },
    "y1": {
      "estimate": 0.0954326030449308,
      "ci95_lo": 0.042312049968070455,
      "ci95_hi": 0.15270339417634884,
      "n": 580,
      "n_boot": 2000
    }
  },
  "medgemma_location": {
    "all": {
      "estimate": 0.442643391521197,
      "ci95_lo": 0.4077306733167082,
      "ci95_hi": 0.4763092269326683,
      "n": 802,
      "n_boot": 2000
    },
    "y0": {
      "estimate": 0.4369158878504673,
      "ci95_lo": 0.3901869158878505,
      "ci95_hi": 0.48130841121495327,
      "n": 428,
      "n_boot": 2000
    },
    "y1": {
      "estimate": 0.44919786096256686,
      "ci95_lo": 0.3983957219251337,
      "ci95_hi": 0.5,
      "n": 374,
      "n_boot": 2000
    }
  },
  "informative": {
    "BUSClean": {
      "status": "invalid",
      "reason": "positive on more than half of the caliper-free images: it detects on-screen annotation, not calipers (validity guard, Amendment 2)"
    },
    "MedGemma": {
      "status": "pending_human",
      "reason": "only one model labeller is available; judged against the blinded rating (A4/A4b) with the same threshold (Amendment 2)"
    }
  },
  "informative_sources": [],
  "gates": {
    "BUSClean": "analysed",
    "MedGemma": "analysed"
  }
}
```

## Cleaned traps (HA2)
| cohort | ran | estimate | ci95_lo | ci95_hi | p | p_holm | verdict |
|---|---|---|---|---|---|---|---|
| isic | True | 0.199 | 0.164 | 0.233 | 0.000 | 0.000 | SUPPORTED |
| thyroid | False | nan | nan | nan | 1.000 | nan | gate not met |
| capsule | True | 0.377 | 0.327 | 0.425 | 0.000 | 0.000 | SUPPORTED |
| ovary | True | 0.170 | 0.089 | 0.255 | 0.000 | 0.000 | SUPPORTED |

## Differential error and attenuation (A3)
| cohort | source | cell | y | n | error_rate | ci95_lo | ci95_hi | flagged |
|---|---|---|---|---|---|---|---|---|
| isic | DermArtifactDB | diff_y1_minus_y0_trapA | diff | 4255 | 0.133 | 0.100 | 0.168 | True |
| isic | DermArtifactDB | diff_y1_minus_y0_trapB | diff | 6995 | 0.166 | 0.134 | 0.198 | True |
| isic | DermArtifactDB | e_A | all | 4255 | 0.000 | nan | nan | nan |
| isic | DermArtifactDB | e_B | all | 6995 | 0.000 | nan | nan | nan |
| isic | IMA++ | diff_y1_minus_y0_trapA | diff | 15 | 0.000 | 0.000 | 0.000 | False |
| isic | IMA++ | diff_y1_minus_y0_trapB | diff | 158 | 0.046 | -0.041 | 0.208 | False |
| isic | IMA++ | e_A | all | 15 | 0.000 | nan | nan | nan |
| isic | IMA++ | e_B | all | 158 | 0.025 | nan | nan | nan |
| isic | Kabir | diff_y1_minus_y0_trapA | diff | 31 | 0.000 | 0.000 | 0.000 | False |
| isic | Kabir | diff_y1_minus_y0_trapB | diff | 233 | -0.061 | -0.096 | -0.032 | False |
| isic | Kabir | e_A | all | 31 | 0.000 | nan | nan | nan |
| isic | Kabir | e_B | all | 233 | 0.004 | nan | nan | nan |
| thyroid | BUSClean | diff_y1_minus_y0_trapA | diff | 1150 | -0.163 | -0.229 | -0.099 | True |
| thyroid | BUSClean | diff_y1_minus_y0_trapB | diff | 326 | 0.052 | -0.063 | 0.160 | False |
| thyroid | BUSClean | e_A | all | 1150 | 0.106 | nan | nan | nan |
| thyroid | BUSClean | e_B | all | 326 | 0.055 | nan | nan | nan |
| thyroid | MedGemma | diff_y1_minus_y0_trapA | diff | 1150 | -0.042 | -0.108 | 0.018 | False |
| thyroid | MedGemma | diff_y1_minus_y0_trapB | diff | 326 | 0.042 | -0.074 | 0.156 | False |
| thyroid | MedGemma | e_A | all | 1150 | 0.352 | nan | nan | nan |
| thyroid | MedGemma | e_B | all | 326 | 0.270 | nan | nan | nan |

- isic: corrected crossover by source {
  "by_source": {
    "DermArtifactDB": {
      "note": "Differential error flagged; the cohort conclusion rests on A2. Correction not reported.",
      "e_A": 0.0,
      "e_B": 0.0,
      "flagged": true
    },
    "IMA++": {
      "c_obs": 0.1474504004676915,
      "c_from": "original (Stage 3)",
      "e_A": 0.0,
      "e_B": 0.02531645569620253,
      "n_A": 15,
      "n_B": 158,
      "corrected": 0.15128028099931987,
      "ci95_lo": 0.11412840110195926,
      "ci95_hi": 0.1901586601279685,
      "note": "First-order attenuation (approximation); reported because differential error was not flagged."
    },
    "Kabir": {
      "c_obs": 0.1474504004676915,
      "c_from": "original (Stage 3)",
      "e_A": 0.0,
      "e_B": 0.004291845493562232,
      "n_A": 31,
      "n_B": 233,
      "corrected": 0.14808596253867293,
      "ci95_lo": 0.11171855326114422,
      "ci95_hi": 0.18614341561304468,
      "note": "First-order attenuation (approximation); reported because differential error was not flagged."
    }
  }
}
- thyroid: model labellers {"BUSClean": {"status": "informative", "kappa": 0.5045849877103421, "n": 2899}, "MedGemma": {"status": "informative", "kappa": 0.6536918491094588, "n": 2601}}
- thyroid: corrected crossover by source {
  "by_source": {
    "BUSClean": {
      "note": "Differential error flagged; the cohort conclusion rests on A2. Correction not reported.",
      "e_A": 0.10608695652173913,
      "e_B": 0.05521472392638037,
      "flagged": true
    },
    "MedGemma": {
      "c_obs": 0.2385752065741104,
      "c_from": "original (Stage 3)",
      "e_A": 0.3521739130434783,
      "e_B": 0.26993865030674846,
      "n_A": 1150,
      "n_B": 326,
      "corrected": 0.6313393445657796,
      "ci95_lo": 0.5196358175062586,
      "ci95_hi": 0.7418725843123622,
      "note": "First-order attenuation (approximation); reported because differential error was not flagged."
    }
  }
}
- capsule: corrected crossover by source {
  "by_source": {}
}
- ovary: model labellers {"BUSClean": {"status": "invalid", "reason": "positive on more than half of the caliper-free images: it detects on-screen annotation, not calipers (validity guard, Amendment 2)"}, "MedGemma": {"status": "pending_human", "reason": "only one model labeller is available; judged against the blinded rating (A4/A4b) with the same threshold (Amendment 2)"}}
- ovary: corrected crossover by source {
  "by_source": {}
}

## Part B
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


## Rating
```json
{
  "main": {
    "status": "both passes are not saved yet; nothing scored"
  },
  "a4b": {
    "status": "both passes are not saved yet; nothing scored"
  }
}
```
{
  "ran": true,
  "n": 240,
  "model": "google/medgemma-1.5-4b-it"
}

The 240-image rating tool is `scripts/round6/rating_app.py` on 127.0.0.1:8765. Intra-rater analysis runs only after both passes exist. A4b is optional; if `a4b_KEY_SHA256.txt` is present and the author has not rated it, that is reported here.
A4b key SHA-256: `1f6ae8beb59e2c6ff7fc403db69992292e23f6bf318f1825492a776e085be3f0`.
The author has not rated the A4b caliper sample.
