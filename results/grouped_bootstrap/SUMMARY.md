# Grouped-bootstrap sensitivity analysis

Same predictions and replicate seed; only the resampling unit changes (image vs fold group).

| Result | Estimate | Image-level 95% CI | Group-level 95% CI | Width ratio | p (group) | Images / groups |
|---|---|---|---|---|---|---|
| Skin hair/ruler, DINOv2 | +0.147 | [+0.110, +0.185] | [+0.110, +0.197] | 1.16 | 0.0002 | 8084 / 4034 |
| Thyroid calipers, DINOv2 | +0.239 | [+0.195, +0.280] | [+0.196, +0.279] | 0.97 | 0.0002 | 2645 / 2622 |
| Ovary calipers, DINOv2 | +0.170 | [+0.088, +0.252] | [+0.086, +0.254] | 1.02 | 0.0002 | 892 / 804 |
| Capsule debris, DINOv2 | +0.377 | [+0.327, +0.425] | [+0.304, +0.451] | 1.50 | 0.0002 | 1476 / 129 |
| Thyroid calipers, MedSigLIP | +0.195 | [+0.153, +0.236] | [+0.153, +0.237] | 1.01 | 0.0002 | 2645 / 2622 |
| Ovary calipers, MedSigLIP | +0.178 | [+0.109, +0.248] | [+0.107, +0.252] | 1.04 | 0.0002 | 892 / 804 |
| Capsule debris, MedSigLIP | +0.335 | [+0.286, +0.383] | [+0.251, +0.408] | 1.63 | 0.0002 | 1476 / 129 |
| Thyroid calipers, ConvNeXt | +0.275 | [+0.230, +0.320] | [+0.230, +0.322] | 1.02 | 0.0002 | 2645 / 2622 |
| Capsule debris, ConvNeXt | +0.320 | [+0.269, +0.372] | [+0.248, +0.391] | 1.40 | 0.0002 | 1476 / 129 |
| Skin hair/ruler, DermLIP | +0.014 | [-0.037, +0.063] | [-0.035, +0.095] | 1.29 | 0.5639 | 8084 / 4034 |
| Pasted ruler sweep, DINOv2 518 (location effect) | +0.313 | [+0.258, +0.370] | [+0.252, +0.373] | 1.09 | 0.0002 | 356 / 310 |
| Pasted ruler sweep, DINOv2 224 (location effect) | +0.456 | [+0.390, +0.525] | [+0.388, +0.524] | 1.00 | 0.0002 | 356 / 310 |

Four registered crossovers, group-level, Holm over four: all significant (adjusted p: 0.0008, 0.0008, 0.0008, 0.0008).

## How this run was made (2026-10-31, rebuilt machine)

- **Predictions.** The 12 `predictions.csv.gz` files above were taken from the release archive (`wtss_outputs.tar`)
  and placed at `results/rerun_2026-09-28/<run>/`; each file's size and SHA256 match `results/ARCHIVE_MANIFEST.csv`.
  No classifier was retrained. Prediction files are not committed.
- **Reproduction check (image level vs the reported numbers).** Estimates identical; interval ends within 0.001:
  skin DINOv2 +0.147 [+0.110, +0.185] (reported [+0.111, +0.185]); thyroid +0.239 [+0.195, +0.280] ([+0.196, +0.280]);
  ovary +0.170 [+0.088, +0.252] ([+0.087, +0.252]); capsule +0.377 [+0.327, +0.425] ([+0.327, +0.426]);
  DermLIP +0.014 [-0.037, +0.063] (identical); pasted-ruler sweep 518 px +0.313 [+0.258, +0.370] (identical).
- **Groups.** Every test image found its group in the rebuilt cohorts, so the fallback (groups over all ISIC 2019
  images) was **not** needed. Groups are exactly those of the script: ISIC 2019 lesion_id ∪ pHash ≤ 8 within each
  trap's matched pool, joined across both traps; thyroid `group_ph8`; ovary and capsule `group`; ISIC 2018 sweep:
  the frozen pilot manifest's pHash group.
- **Cohorts rebuilt from re-downloaded data**, each download checked against `docs/DATA_MANIFEST.md` (file count, size,
  listing SHA256; all match after the original "delete archives after extraction" step; NIH differs only in Hugging
  Face cache metadata files that store the download time). Thyroid 3,493 images, ovary 1,202, capsule 5,481 (as
  preregistered). ISIC 2019: `prepare_isic2019.py` stages unzip, groups, cache, artifact_stats; `native_hair_stats.py`;
  spec U-Net masks from the **archived** weights (`weights/isic2019_lesion_unet_spec_256.pt`, renamed to
  `unet_spec_256.pt`; recomputed agreement with the HAM masks, Dice 0.89484 mean / 0.94700 median on 10,015 images,
  is identical to the archived report); `prepare_spec_cohort.py`. The older lesion U-Net (`train_lesion_unet.py`) was
  not retrained: `prepare_spec_cohort.py` replaces its masks with the spec U-Net's (or HAM manual masks), and the
  cohort table, `r_spec` and `qc_ok` do not depend on it.
- **Upstream change found:** the ISIC challenge files moved on 2026-10-28 to
  `https://isic-archive.s3.amazonaws.com/challenges/<year>/...`; the old URLs in `scripts/data/download_data.sh` now
  return a small HTML "moved" page. The files at the new address match the manifest.
- Width ratios of the four registered DINOv2 crossovers: 0.97-1.50 (largest for capsule: 1,476 test frames in only
  129 groups). In each, none of the 10,000 group-level replicates reached zero, so p = 2/10,001 ≈ 0.0002 (the smallest
  value this formula can give) and 0.0008 after Holm over four.
