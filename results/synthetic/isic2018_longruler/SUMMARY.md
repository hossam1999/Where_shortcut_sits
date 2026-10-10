# Long-ruler sweep: results (registered in docs/PREREGISTRATION_LONG_RULER.md)

Run on 2026-10-10 from commit bc36df2 (branch `claude/festive-bohr-95evo5`). The registered design was not changed, and
each command was run once.

## Setup and checks
- 1x RTX 3090 (24 GB), Python 3.12.14, torch 2.11.0+cu128 (from `requirements.txt`). DINOv2 ViT-B/14 loaded through torch.hub.
- Data: ISIC 2018 Task 1 training input and ground truth from the public S3 bucket. The input zip was finished with
  parallel ranged `curl` requests after `wget` slowed down. It is 11,165,358,566 bytes, the Content-Length the server reports,
  and it unzipped without errors (2,596 entries in each folder).
- `python -m pytest -q tests`: **68 passed**.
- Placement check: `find_best_placements_for_mask(load_mask(mask, 518) > 0, id, OVERLAPS, 200, 10, 2500)` was
  recomputed for the first 50 ids of `frozen/longruler_common_ids.csv`. **0 mismatches** with `frozen/placements_518_200x10.json`.
- Cohort as registered: test set of 187 images with 48 melanomas, seeds 42/123/456, 4 arms, 5 overlaps.
  `WTSS_BOOTSTRAP=crossed`, 10,000 replicates (all valid), 95% percentile intervals.

Commands (as registered):
```
PYTHONPATH=src WTSS_BOOTSTRAP=crossed python scripts/run_synthetic.py --cohort isic2018_longruler --backbone dino518 \
    --artifact ruler_long --tag longruler --arms erm mask inpaint balanced
PYTHONPATH=src WTSS_BOOTSTRAP=crossed python scripts/run_synthetic.py --cohort isic2018_longruler --backbone dino518 \
    --artifact ruler_fixed --tag longruler_subset --arms erm mask inpaint balanced
PYTHONPATH=src WTSS_BOOTSTRAP=crossed python scripts/analysis/longruler_report.py   # tables below, L3, absolute AUROCs
```
`scripts/analysis/longruler_report.py` only reads the two runs' outputs. It writes `longruler_tables.csv` and
`longruler_report.json`. Its estimators come from the repository: arm − ERM and the interaction from
`bootstrap_vs_erm.csv` / `location_interaction.csv`, and absolute AUROC from `hierarchical_auc_bootstrap` (crossed).
L3 uses the crossed `replicates` helper: the long and standard interactions are computed on the same images and seeds,
with one Poisson weight per image shared across all four terms and seeds as clusters (bootstrap seed 20260918).
Point estimates are the mean of the per-seed estimates.

## Claims
| Claim | Estimate [95% CI] | Rule | Verdict |
|---|---|---|---|
| **L1** (primary): long-ruler interaction of masking, [mask − ERM]₀% − [mask − ERM]₁₀₀%, reversed test | **+0.224 [+0.161, +0.285]** | > 0 with CI above 0 | **Supported** |
| **L2**: long-ruler mask − ERM at 100% overlap | **−0.054 [−0.123, +0.015]** | < 0 with CI below 0 | **Not supported**: the estimate is negative but the CI includes 0 |
| **L3** (descriptive): long − standard interaction, same 1,314 images | **−0.038 [−0.071, −0.005]** | no direction claimed | The long ruler's interaction is slightly smaller (by about 15%) |

Reading: the location result holds for a long, thin ruler with real proportions. Masking helps a lot when the ruler is
outside the lesion (+0.170) and does not help when the ruler is inside it. With the long ruler fully inside the lesion,
masking is not clearly worse than ERM (−0.054, CI includes 0). On the same images the standard ruler gives −0.102
[−0.175, −0.032]. The long ruler covers more of the lesion, so masking keeps more of it, and the interaction is
about 0.04 AUROC smaller. Its sign and size are otherwise similar.

### Arm − ERM, reversed-test AUROC (mean of per-seed deltas, crossed 95% CI)

| Ruler | Arm | 0% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|
| Long 200×10 | mask − ERM | +0.170 [+0.094, +0.249] | -0.042 [-0.122, +0.035] | -0.070 [-0.149, +0.005] | -0.064 [-0.143, +0.011] | -0.054 [-0.123, +0.015] |
| Long 200×10 | inpaint − ERM | +0.103 [+0.053, +0.157] | +0.128 [+0.074, +0.185] | +0.155 [+0.099, +0.214] | +0.214 [+0.168, +0.263] | +0.237 [+0.190, +0.288] |
| Long 200×10 | balanced − ERM | +0.068 [+0.014, +0.121] | +0.106 [+0.052, +0.162] | +0.159 [+0.105, +0.212] | +0.191 [+0.132, +0.248] | +0.238 [+0.176, +0.300] |
| Standard 65×19 | mask − ERM | +0.160 [+0.084, +0.239] | +0.010 [-0.068, +0.087] | -0.055 [-0.131, +0.019] | -0.068 [-0.147, +0.007] | -0.102 [-0.175, -0.032] |
| Standard 65×19 | inpaint − ERM | +0.106 [+0.059, +0.159] | +0.167 [+0.113, +0.226] | +0.159 [+0.103, +0.217] | +0.153 [+0.099, +0.209] | +0.237 [+0.186, +0.292] |
| Standard 65×19 | balanced − ERM | +0.076 [+0.024, +0.129] | +0.128 [+0.072, +0.187] | +0.127 [+0.073, +0.183] | +0.142 [+0.087, +0.199] | +0.190 [+0.134, +0.249] |

### Location interaction [arm − ERM]₀% − [arm − ERM]₁₀₀% (location_interaction.csv)

| Arm | Long ruler | Standard ruler (same images) | Long − standard (L3) |
|---|---|---|---|
| mask | +0.224 [+0.161, +0.285] | +0.262 [+0.200, +0.325] | -0.038 [-0.071, -0.005] |
| inpaint | -0.134 [-0.190, -0.081] | -0.131 [-0.184, -0.083] | -0.002 [-0.030, +0.028] |
| balanced | -0.171 [-0.234, -0.125] | -0.114 [-0.147, -0.084] | -0.057 [-0.103, -0.023] |

### Absolute AUROC (mean over seeds, crossed 95% CI)

| Ruler | Arm | Reversed 0% | Reversed 100% | Correlated 0% | Correlated 100% |
|---|---|---|---|---|---|
| Long 200×10 | erm | 0.690 [0.602, 0.771] | 0.499 [0.410, 0.588] | 0.884 [0.830, 0.931] | 0.928 [0.891, 0.960] |
| Long 200×10 | mask | 0.860 [0.799, 0.913] | 0.445 [0.360, 0.529] | 0.860 [0.799, 0.913] | 0.959 [0.933, 0.979] |
| Long 200×10 | inpaint | 0.793 [0.720, 0.859] | 0.736 [0.654, 0.811] | 0.825 [0.758, 0.886] | 0.856 [0.793, 0.911] |
| Long 200×10 | balanced | 0.757 [0.679, 0.830] | 0.737 [0.660, 0.811] | 0.860 [0.800, 0.913] | 0.863 [0.795, 0.921] |
| Standard 65×19 | erm | 0.699 [0.613, 0.779] | 0.534 [0.444, 0.624] | 0.886 [0.831, 0.933] | 0.931 [0.892, 0.963] |
| Standard 65×19 | mask | 0.859 [0.799, 0.913] | 0.433 [0.348, 0.519] | 0.859 [0.799, 0.913] | 0.966 [0.937, 0.985] |
| Standard 65×19 | inpaint | 0.805 [0.734, 0.869] | 0.772 [0.693, 0.843] | 0.818 [0.749, 0.881] | 0.834 [0.767, 0.893] |
| Standard 65×19 | balanced | 0.776 [0.696, 0.846] | 0.725 [0.642, 0.800] | 0.854 [0.794, 0.908] | 0.876 [0.821, 0.925] |

Notes on the absolute AUROCs:
- Masking at 0% gives the same AUROC on the reversed and correlated tests (0.860). The ruler is outside the lesion and
  gets masked away, so the test environment does not matter.
- At 100% both rulers sit inside the lesion and stay visible after masking. Mask then reaches 0.959/0.966 on the
  correlated test and drops to 0.445/0.433 on the reversed test, which is below ERM.
- Inpainting and group balancing beat ERM at every overlap for both rulers. Their gains grow as the ruler moves inside
  the lesion, so their interactions are negative.

## Files
Committed, per run directory (`dino518_ruler_long_corr_longruler/`, `dino518_ruler_fixed_corr_longruler_subset/`):
`bootstrap_vs_erm.csv`, `location_interaction.csv`, `summary_auc.csv`, `metrics.csv`, `heads.csv`,
`counterfactual_summary.csv`, `counterfactual_per_image.csv.gz`. Also committed at the top level: `longruler_tables.csv` and
`longruler_report.json`.

**Not committed (git-ignored by `results/**/predictions*.csv.gz`). Download these before the machine is released:**
- `results/synthetic/isic2018_longruler/dino518_ruler_long_corr_longruler/predictions.csv.gz` (394 KB)
- `results/synthetic/isic2018_longruler/dino518_ruler_fixed_corr_longruler_subset/predictions.csv.gz` (395 KB)

Feature/image caches are outside the repository (`$WTSS_DATA/cache/`) and were not committed.
