# Round 6 — implementation runbook (for the GPU machine)

Plan, hypotheses and decision rules: `docs/PREREGISTRATION_ROUND6.md` (frozen; changes only as dated amendments at
its end, reported to the author). Ground rules are those of round 4 (`scripts/round4/README.md`): new code in
`scripts/round6/`, results in `results/round6/`, feature caches only under `$WTSS_CACHE/features/round6/` (copy rows
from existing caches with `scripts/round4/common.assemble_views`), never commit images, masks of external datasets,
features or predictions; commit code, CSV, JSON and MD. Every script gets a `--smoke` mode (output under
`results/round6/_smoke/`, git-ignored) and is resumable.

```bash
export PYTHONPATH=$PWD/src WTSS_BOOTSTRAP=crossed WTSS_DATA=<data root>
```

## Part A — labels without a clinician

### A0 `scripts/round6/sources.py`
- Download into `$WTSS_DATA/external/round6/`: DermArtifactDB (Zenodo 18324962), IMA++ (Zenodo 14201693), the capsule
  clear/contaminated masks (Mendeley vmxhn95j8z, version 3), BUSClean (`git clone https://github.com/hawaii-ai/bus-cleaning`).
- Read each licence from the source itself (record page, `LICENSE` file). Write `results/round6/sources.json`: name,
  URL, version or commit, download date, licence text or identifier, number of our cohort images matched. BUSClean
  without a licence that permits research use → skip it and record why. MedGemma (`google/medgemma-1.5-4b-it`) is gated:
  if the Hugging Face token or accepted terms are missing, stop that part and tell the author (they must run
  `huggingface-cli login` after accepting the terms on the model page).
- Matching: ISIC by ID (strip `_downsampled`); capsule by pHash of the original SEE-AI image resized to 256 (Hamming ≤ 2,
  exactly as `scripts/data/prepare_capsule.py`). Check mask polarity on three images before use (the capsule dataset
  marks contaminated pixels black in its binary masks). IMA++: majority mask over its annotators, resized to 518 px
  with nearest-neighbour interpolation.

### A1 `scripts/round6/label_agreement.py --cohort isic|thyroid|capsule|ovary`
Cohort tables and caches from `scripts/round4/common.trap_cohort` (unchanged definitions). Outputs
`results/round6/agreement/<cohort>/{per_image.csv, agreement.json}` with κ / Dice / IoU and 95 % bootstrap CIs
(2,000 image-level replicates), overall and by Y.
- ISIC: hair-free vs hair against DermArtifactDB; our ROI vs IMA++ (Dice); r recomputed at 518 px with the cached hair
  mask and **both** lesion masks (ours and IMA++), so the comparison is like for like; trap-class agreement.
- Capsule: IoU of `contam.npy` with the expert mask; contamination share, r and cell recomputed with the expert mask.
- Thyroid, ovary: our detector (`marker_px`, `r`) against BUSClean (presence; positions if returned) and MedGemma.
  MedGemma, greedy decoding, `max_new_tokens=3`, answer parsed as yes/no (anything else = missing), prompts fixed:
  - presence (plain image): "This is an ultrasound image. Are there caliper or measurement marks overlaid on it
    (small plus signs, small crosses, or dotted measurement lines)? Answer only yes or no."
  - location (nodule or tumour outlined with a 2-px green contour): "The green contour outlines the nodule. Is any
    caliper or measurement mark (a small plus sign or cross) inside the green contour? Answer only yes or no."
    (ovary: "tumour" instead of "nodule").
  - κ ≥ 0.40 against the images where the other two caliper labels agree → informative (A1 rule), else uninformative.

### A2 `scripts/round6/clean_traps.py --cohort ...`
Build the cleaned cohort (drop contradicted images per the pre-registration; keep unverified ones), then exactly the
Stage 3 trap protocol with arms `erm`, `mask` (`build_spec_envs`, `run_spec`), count gate, and
`stats_crossed.difference_of_deltas(P[trapB], P[trapA], "mask", "erm", "test_rev", 10000, seed)`. Features: copy rows
from the existing caches of the same view (`scripts/round4/common.trap_cohort(...)["old"]` plus the round-4 dirs).
Output `results/round6/clean_traps/<cohort>/` and a `SUMMARY.csv` with Holm over the cohorts, the verified /
contradicted / unverified shares per cell, and the original Stage 3 crossover next to the cleaned one.

### A3 `scripts/round6/bias_analysis.py`
Error rates of our cells against each informative source (and against the human rating once it exists) by trap and
Y, bootstrap CIs, the differential-error flag, e_A + e_B and the corrected crossover (pre-registration A3).

### A4 `scripts/round6/rating_app.py` (human rating; built now, used later by the author)
- A small local web page (Python standard library or Flask) bound to `127.0.0.1:8765` only. The author opens it with
  `ssh -p <port> root@<host> -L 8765:localhost:8765` and then `http://localhost:8765` in a browser.
- Reads `audit/review_sheet.csv` (images from `audit/` and `audit_local/`; regenerate `audit_local/` with
  `python audit/make_audit_sample.py` if missing). Shows one image at a time in a random order fixed per pass
  (`--pass 1|2`, different seeds); step 1 the raw image (artifact_present, artifact_location), step 2 the overlay
  (mask_correct, roi_mask_correct), notes; never shows the cell. Saves after every image to
  `results/round6/rating/pass<k>.csv` (answers only, no images), resumable.
- `scripts/round6/medgemma_audit.py`: MedGemma answers presence and location questions for the 240 images (cohort-
  specific wording: hair on the lesion; caliper marks in the nodule/tumour; bubbles/debris/turbid fluid in the lesion
  box) → `results/round6/rating/medgemma.csv`.
- `scripts/round6/analyse_rating.py`: intra-rater κ (pass 1 vs pass 2), rater vs MedGemma, rater vs our labels (after
  unblinding with `python audit/make_audit_sample.py --unblind` and checking the key against `audit/KEY_SHA256.txt`),
  error rates by cell and Y. Runs only when both passes exist.

## Part B — benchmark-level fine-tuned thyroid model

### B1 `scripts/round6/ft_thyroid.py --select`
- Natural thyroid split as `scripts/run_natural.py` (`load("thyroid")`; seed 42 train/validation split by
  `stable_int("natural_val", s, group) % 5 == 0`). Reuse `wtss.experiments.finetune.train_eval` unchanged except for an
  optional, backward-compatible model-kwargs argument so that `vit_base_patch14_dinov2.lvd142m` can be created with
  `img_size=224`.
- Pass `E = {"train_corr": train, "clean_val": val, "clean_test": val, "test_corr": val, "test_rev": val}` during
  selection, so that **no test image is scored**. Grid and selection rule exactly as registered (validation AUROC of
  ERM). Write `results/round6/ft_recipe.json` (all 12 validation AUROCs and the choice) and COMMIT it before B2.

### B2 `scripts/round6/ft_thyroid.py --natural` and `--traps`
- Natural: 5 seeds × arms {erm, mask, balanced, mask_balanced} with the chosen recipe; `E = {"train_corr": train,
  "clean_val": val, "clean_test": test, "test_corr": val, "test_rev": test}` so the saved predictions contain the
  validation images (env `test_corr` → rename to `val_groups`) for operating-point thresholds and the official test
  images (env `clean_test` → rename to `clean`). Predictions to `results/round6/ft_natural/predictions.csv.gz`.
- Analysis as Stage 5 / R10: hard pairs (malignant with in-ROI caliper vs benign without), crossed bootstraps
  (mask − erm, balanced − mask, mask_balanced − mask on all/hard/easy), OP1–OP5 with thresholds from validation only
  and the crossed operating-point bootstrap (reuse `scripts/round4/natural_isic2020.op_crossed` with `hard_pos_a=1`),
  sensitivity among malignant nodules with an in-ROI caliper; FT0 test AUROC vs 0.773.
- Traps: `scripts/run_finetune_spec.py --cohort thyroid --arch <chosen> --lr <chosen> --epochs <chosen> --arms erm mask
  --env_seed 42` and again with `--env_seed 123`, `--tag round6`; crossover with the crossed bootstrap over the 10
  clusters.
- `results/round6/ft_SUMMARY.md`: recipe, FT0–FT4 with estimates, 95 % CIs, Holm-adjusted p and verdicts.

## Final
`results/round6/SUMMARY.md` (Part A: sources and coverage, agreement tables, cleaned-trap crossovers with Holm,
differential-error flags, e_A + e_B; Part B: ft_SUMMARY), `python -m pytest -q tests`, commit, push.

## Additions from the pre-run review (Amendment 2)
- `scripts/round6/medgemma_cohort.py` runs MedGemma on every non-gap thyroid and ovary image (presence on the plain
  image, location on a contour-only image) → `results/round6/matches/{thyroid,ovary}_medgemma.csv`; `run_all.sh` runs
  it right after A0. `label_agreement.py` applies the informativeness rule, the BUSClean validity guard and the
  like-for-like 518-px rule; model labellers enter A2 only when informative.
- `run_all.sh` never starts the rating server (`rating_app.py --check` only verifies the image files).
- Rating, later: `python scripts/round6/rating_app.py --pass 1` (and `--sheet a4b` for the caliper sample); pass 2
  at least 7 days later. Then `bash scripts/round6/after_rating.sh` scores the rating (keys checked against their
  committed SHA-256) and re-runs A1–A3 for thyroid and ovary.
- `run_finetune_spec.py --img_size 224` is passed automatically when the DINOv2 ViT is the chosen recipe.
