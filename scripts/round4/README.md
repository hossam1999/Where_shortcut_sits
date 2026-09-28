# Round 4 — how to run

Plan, hypotheses and decision rules: `docs/PREREGISTRATION_ROUND4.md` (frozen; changes only as dated amendments at
its end).

| script | analysis | output |
|---|---|---|
| `masking_variants.py` | R8 masking implementations (black, blurred background, box crop, box crop of the masked image) | `results/round4/masking_variants/` |
| `dose_response.py` | R9 mask gain over five bins of the real overlap r (gate + merge rule, crossed-bootstrap slope) | `results/round4/dose_response/` |
| `natural_isic2020.py` | R10 train on ISIC 2019, test on ISIC 2020 (natural prevalence, near-duplicates removed, operating points) | `results/round4/natural_isic2020/` |
| `theory_round4.py` | R11 the fixed theory procedure on the new R8/R9 cells | `results/round4/theory/` |
| `run_all.sh` | everything in order, logs in `results/round4/logs/`; `smoke` = small subsets | |
| `common.py` | cohort loaders (same definitions as the trap scripts), feature assembly, Holm | |

```bash
export WTSS_DATA=/path/to/data          # contains isic2019/, us/tncd/, capsule/, ovary/, external/isic2020_work/
bash scripts/round4/run_all.sh smoke    # every script on small subsets -> results/round4/_smoke/ (git-ignored)
bash scripts/round4/run_all.sh          # the full analyses (resumable: re-running skips finished steps)
```

Design notes
- Environments, seeds, folds, heads and thresholds are those of `scripts/run_spec_e13.py` and
  `scripts/run_thyroid_traps.py`; `run_spec` gained one backward-compatible hook (`extra_views`, `extra_renderers`
  in `extra_ctx`) so plain ERM heads can be fitted on extra rendered views.
- Feature caches are written only under `$WTSS_CACHE/features/round4/`. Rows of views that already exist
  (original image, mean-fill mask, generic-overlay view) are copied from the existing caches of the same view and
  image; only missing images go through the encoder. No existing cache file is modified.
- R8 stops a cohort if re-fitted ERM or mean-fill masking differs from the stored run by more than 0.03 AUROC
  (`repro_check.csv`). Example renderings are written only for capsule and ovary images, to
  `results/round4/_local_examples/` (git-ignored; licences).
- R9: x_b is the mean r of the artifact-bearing images of the bin's matched pool (the images that enter the trap).
  `run_all.sh` commits the bin counts and the gate outcome before any dose-response model is fitted.
- R10: ISIC 2020 IDs carry the prefix `i20_` inside the pipeline; hard pairs are fixed by the training association
  (melanomas without in-lesion hair vs benign lesions with it). Operating-point thresholds come only from the ISIC 2019
  validation predictions of each seed.
- `predictions*.csv.gz` stay git-ignored; every metrics, counts and analysis CSV/JSON/MD is committed.
