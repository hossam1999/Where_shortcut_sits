# Registration: does the location result hold for a long, thin, realistic ruler?

Committed and pushed before any model is fitted on the long-ruler images (2026-10-10).

## Why
The pasted ruler of the controlled dermoscopy sweep is 65 × 19 px at 518 px: 13% of the image width. Real dermoscopy
scale bars are longer and thinner, often a third to half of the image width. A reviewer can ask whether the result
depends on the ruler's shape. This experiment repeats the sweep with a ruler of real proportions.

## Design (fixed before fitting)
- **Ruler**: `ruler_long`, 200 × 10 px at 518 px (39% of the width, 0.75% of the image area), light strip with a dark
  outline and millimetre ticks (`wtss.synthetic.draw_ruler_long`).
- **Images**: the 2,437-image ISIC 2018 sweep cohort, restricted to images where the long ruler can be placed at
  0, 25, 50, 75 and 100% overlap with the lesion within 10 percentage points (same placement search,
  `find_best_placements_for_mask`, 2,500 candidates, same seed rule). This leaves **1,314 images** (train 941 / val
  186 / test 187; 48 test melanomas), with the frozen pHash-grouped splits unchanged. The subset has larger lesions
  (median 30% of the image versus 15%), because a long ruler fits fully only inside a large lesion. Placements:
  `frozen/placements_518_200x10.json`; image list: `frozen/longruler_common_ids.csv` (computed from the public
  ISIC 2018 Task 1 masks, outcome-free: no label or model output was used except the split's melanoma count).
- **Same-subset control**: the standard 65 × 19 ruler is run on the **same 1,314 images**, so that ruler shape is
  separated from lesion size.
- **Model**: DINOv2 ViT-B/14 @518, logistic heads, C and threshold chosen on clean validation data, seeds 42/123/456,
  overlaps 0/0.25/0.5/0.75/1, trap 0.9/0.1, as in the main sweep.
- **Arms**: erm, mask, inpaint, balanced.
- **Statistics**: crossed seed × image bootstrap (`WTSS_BOOTSTRAP=crossed`), 10,000 replicates, 95% percentile
  intervals.

## Claims and what counts as support
- **L1 (primary)**: location interaction of masking for the long ruler,
  [mask − ERM]₀% − [mask − ERM]₁₀₀% of reversed-test AUROC, is > 0 with a 95% interval above 0. Supported if so.
- **L2**: masking harms with the long ruler fully inside: mask − ERM at 100% < 0 with the interval below 0.
- **L3 (descriptive)**: long-ruler interaction minus standard-ruler interaction on the same images, with a
  crossed interval. No direction is claimed.
- Both rulers' results are reported whichever way they go.

## Commands
```
PYTHONPATH=src WTSS_BOOTSTRAP=crossed python scripts/run_synthetic.py --cohort isic2018_longruler --backbone dino518 \
    --artifact ruler_long --tag longruler --arms erm mask inpaint balanced
PYTHONPATH=src WTSS_BOOTSTRAP=crossed python scripts/run_synthetic.py --cohort isic2018_longruler --backbone dino518 \
    --artifact ruler_fixed --tag longruler_subset --arms erm mask inpaint balanced
```
Outputs: `results/synthetic/isic2018_longruler/dino518_ruler_long_corr_longruler/` and
`.../dino518_ruler_fixed_corr_longruler_subset/` (`location_interaction.csv`, bootstrap tables, predictions).
