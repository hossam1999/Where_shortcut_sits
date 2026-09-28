# Round 4 — implementation runbook (for the GPU machine)

Scientific plan, hypotheses and decision rules: `docs/PREREGISTRATION_ROUND4.md`. That file is frozen: do not edit
its existing text. Anything that has to change is added as a dated "Amendment" section at its end, with the reason.

## 0. Ground rules
- Work on the branch `round4-results`, created from `origin/claude/festive-bohr-95evo5`. Never push to `main`.
- New code goes in `scripts/round4/`. The only permitted change to existing code is the backward-compatible hook in
  section 1; every existing script must behave exactly as before.
- New results go only to `results/round4/<analysis>/...`. New feature caches go only to
  `$WTSS_CACHE/features/round4/<analysis>/<cohort>/dinov2_b14_518/`. Never write into an existing feature directory:
  `extract_view` overwrites a cache file whose image IDs do not cover the requested pool. Copying an existing
  `erm.npz` / `mask.npz` into the new directory first is allowed (it is reused only if it covers the pool).
- Commit only code, CSV, JSON and Markdown. Never commit images, image-derived arrays, feature files or figures that
  show thyroid (TN3K/TNCD) or ISIC images (dataset licences). `predictions*.csv.gz` stay git-ignored (size); commit
  every `metrics_per_seed.csv`, `counts.csv` and analysis CSV/JSON so the numbers can be traced without them.
- Environment for every command:
  `export PYTHONPATH=$PWD/src WTSS_BOOTSTRAP=crossed WTSS_DATA=<data root>` (the directory that contains
  `isic2019/prepared/cohort_spec.csv`, `us/tncd/thyroid_cohort.csv`, `capsule/`, `ovary/`, `external/isic2020_work/`).
- Every script: a `--smoke` flag (1 seed, 1 fold, a few hundred images, output under `results/round4/_smoke/`), a
  `--counts_only` flag where counts exist, and resumability (skip a step whose output file exists).
- Seeds for bootstraps: a fixed integer per test written into the output CSV.

## 1. Library hook (the only edit to existing code)
In `src/wtss/experiments/spec_traps.py`, `run_spec`, directly after `rend = make_renderers(...)` and before the
`need` set is used, add

```python
    xc = extra_ctx or {}
    need = need | set(xc.get("extra_views", ()))
    rend.update(xc.get("extra_renderers", {}))
```

(move the `need` / `cached` computation below this if needed so the extra views are included in the cache check).
`extra_ctx` is already copied into `_CTX`, so `view_arms` entries `(method_name, view_name, False)` fit one plain ERM
head per extra view. With `extra_ctx=None` nothing changes. Add a unit test in `tests/` that checks run_spec's
default `need` set is unchanged.

## 2. R8 — masking implementations (`scripts/round4/masking_variants.py`)
- `--cohort isic|thyroid|capsule|ovary`. Cohort table, `envs`, cache and donors exactly as in
  `scripts/run_spec_e13.py` (isic: `load_spec_cohort()`, `build_spec_envs(c)`, cache
  `isic2019/prepared/cache_518` with `roi_spec.npy`) and `scripts/run_thyroid_traps.py` (import its `cohort`,
  `capsule_cohort`, `ovary_cohort`; `build_spec_envs(c, group_col="group")`; caches and art files as there).
- Renderers (from `cache.get(i)` → `rgb, roi, art`; output PIL image at 518 px):
  - `mask_black`: `apply_roi_mask(img, roi, fill=(0, 0, 0))`.
  - `mask_blur`: `np.where(roi[..., None] > 0, rgb, cv2.GaussianBlur(rgb, (0, 0), 16 * rgb.shape[0] / 518))`.
  - `crop_box`: bounding box of `roi > 0`, enlarged by 10 % of its height/width on each side, clipped; crop; pad to
    a square with `MEAN_RGB` (centred); resize to 518 bicubic. Empty ROI → full image (count them).
  - `crop_mask`: the same crop applied to `apply_roi_mask(img, roi)`.
  Save one PNG per view of 3 capsule or ovary images to `results/round4/_local_examples/` for a visual check
  (git-ignored; never commit it).
- `run_spec(envs, cache, "dino518", out, fdir, donors, arms=("erm", "mask"), extra_ctx={"extra_views": V,
  "extra_renderers": R, "view_arms": [(v, v, False) for v in V]})` with `out = results/round4/masking_variants/<cohort>`.
- Analysis written to the same directory:
  - `repro_check.csv`: mean clean and reversed AUROC of `erm` and `mask` per trap vs the stored run named in the
    pre-registration, and the absolute difference; stop with a clear message if any difference exceeds 0.03.
  - `crossovers.csv`: per view v in (mask, mask_black, mask_blur, crop_box, crop_mask):
    `difference_of_deltas(P[trapB], P[trapA], v, "erm", "test_rev", 10000, seed)`, plus Trap-A and Trap-B gains
    `hierarchical_paired_bootstrap(P[trap], v, "erm", "test_rev")` and clean cost (env "clean").
  - `contrast_vs_mask.csv`: `difference_of_deltas(P[trapB], P[trapA], "mask", v, "test_rev", ...)` = C_mask − C_v.
- `scripts/round4/masking_summary.py`: collects the four cohorts, applies Holm over the 16 primary tests, writes
  `results/round4/masking_variants/SUMMARY.csv` and `SUMMARY.md` (the H8a/H8b verdicts and the decision label).

## 3. R9 — overlap dose-response (`scripts/round4/dose_response.py`)
- Same cohort loaders. Presence rule per cohort (isic: `~A0`; thyroid/ovary: `marker_px >= 15`; capsule:
  `contam_frac >= 0.10`) and overlap column (isic: `r_spec`; others: `r`).
- Bins b1..b5 as in the pre-registration; column `b<k>_A1 = present & (r in bin)`; the shared `A0` column is the
  artifact-free group. `--counts_only` writes `counts.csv` (per bin: matched A × Y cells via `matched_pool`, reversed
  positives of seed 42 summed over folds, x_b = mean r of the bin's A = 1 images) and applies the gate and the merge
  rule, writing `bins_final.json`. Merged bins get new names (e.g. `b4b5`) and new `_A1` columns.
- Full run: `build_spec_envs(c, traps=final_bins, group_col=...)` (isic: no group_col, as in run_spec_e13) and
  `run_spec(..., arms=("erm", "mask"), traps=final_bins)` to `results/round4/dose_response/<cohort>`.
- Slope: weights w_b = (x_b − mean x)/Σ(x_j − mean x)^2; terms =
  Σ_b `stats_crossed._pair_terms(P[b], "mask", "erm", "test_rev", "seed", coef=w_b)`;
  `stats_crossed._summary(*stats_crossed.replicates(terms, 10000, seed), {...})`. Identical image IDs across bins
  share their Poisson weight (intended: the artifact-free images appear in every bin).
- Also per bin: g_b with interval (`hierarchical_paired_bootstrap`), Spearman(x_b, g_b), zero crossing.
- `scripts/round4/dose_summary.py`: Holm over cohorts, `results/round4/dose_response/SUMMARY.{csv,md}`, and a figure
  `results/round4/dose_response/dose_response.pdf` (g_b vs x_b per cohort with intervals; plot only, no images).

## 4. R10 — ISIC 2019 → ISIC 2020 natural test (`scripts/round4/natural_isic2020.py`)
- Train/val pool: the `isic_*` branch of `scripts/run_natural.py::load` without the source split (all ISIC 2019
  images). Per seed in `run_natural.SEEDS`: group-safe 80/20 train/val split exactly as in `run_natural.main`.
- Test pool: `results/rerun_2026-09-28/external_isic2020/cohort.csv` → `image_id = "i20_" + image_name`,
  `y = target`, `lesion_frac > 0`, `a = (hair_px > tau) & (r >= 0.5)` with tau from `segmenter.json`,
  `source = "ISIC2020"`. Drop near-duplicates of ISIC 2019: pHash (`imagehash.phash`, hash_size 8) of the 518-px
  cached RGB of both sets, Hamming ≤ 8 → drop the ISIC 2020 image; also drop identical image names. Write
  `dedup.csv` (dropped IDs and their nearest ISIC 2019 match) and the counts.
- One cache object for both sets: a small class with `get(i)` that routes `i20_*` IDs (prefix stripped) to
  `RealCache($WTSS_DATA/external/isic2020_work/cache_518, roi_file="roi.npy", art_file="hair.npy")` and every other
  ID to the ISIC 2019 cache used by run_natural. If the ISIC 2020 cache is missing, re-run `scripts/final/external_isic2020.py`, which is resumable
  and reuses the frozen hair segmenter (never delete `hair_unet.pt` or `segmenter.json`, never re-train).
- `envs[("natural", s, 0, k)]` as in run_natural with `test_corr = test_rev = clean =` the ISIC 2020 test pool;
  `run_spec(..., arms=run_natural.ARMS, traps=("natural",), insert_fn=<generic overlay as run_natural>,
  insert_tag="_generic", folds=[0], save_val=True)` to `results/round4/natural_isic2020/`.
- Analysis: `natural_metrics.csv` (per arm: AUROC, cross-group AUROC, AUROC among in-lesion-hair images, hard- and
  easy-pair AUROC; hard = melanomas without in-lesion hair vs benign with it); `natural_boot.csv` (crossed bootstrap
  on hard / easy / all for mask − ERM, balanced − mask, mte_balanced − mask, mte − mask);
  `operating_points.csv` using `scripts/analysis/operating_points.py`'s `threshold` and `rates` with thresholds
  from each seed's `val_groups` predictions, OP1–OP5, sensitivity overall and among melanomas without in-lesion hair,
  with crossed intervals for mask − ERM and balanced − mask. `SUMMARY.md` with the H10a–c verdicts.

## 5. R11 — prospective theory test (`scripts/round4/theory_round4.py`)
- Load `scripts/analysis/theory_predict.py` with importlib; call its `cells("trap", runs)` with
  `runs = {(cohort, "DINOv2"): "round4/masking_variants/<cohort>"}` and `{(cohort, "DINOv2 bins"):
  "round4/dose_response/<cohort>"}`. Keep its EXCLUDE list and fitting code unchanged.
- T1' (`summarise`, plus the two reference predictors), T2' (per view: predicted vs observed C_v), T3' (predicted
  vs observed slope and g_b). Write `results/round4/theory/{cells.csv, summary.json, SUMMARY.md}` with the decision.

## 6. Final
- `results/round4/SUMMARY.md`: one section per analysis — what was run, the counts, every pre-registered hypothesis
  with its estimate, 95 % interval, Holm-adjusted p where applicable and SUPPORTED / NOT SUPPORTED, deviations,
  runtimes, GPU, commit hash.
- `python -m pytest -q tests` must pass. Commit and push `round4-results`.
