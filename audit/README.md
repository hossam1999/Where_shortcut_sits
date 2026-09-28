# Image audit for manual checking

**What.** 240 images — 4 cohorts (ISIC 2019 hair, thyroid calipers, ovarian calipers, capsule debris) × 3 cells
(artifact inside the ROI, outside the ROI, artifact-free, *by our automatic labels*) × 20 images (10 positive, 10
negative diagnosis), drawn at random with a fixed seed by `make_audit_sample.py`.

**Blind review.** Images are named by neutral IDs (`A0000.png` …). The key that maps each ID to its cell is not in the
repository: it is written to the git-ignored `audit_local/KEY_open_after_review.csv` by `make_audit_sample.py` (fixed
seed, so it can always be regenerated), and its SHA-256 is committed in `KEY_SHA256.txt` so that the key used after the
review can be shown to be the one fixed before it. The `in_roi/`, `out_roi/`, `artifact_free/` folders stay empty until
`python audit/make_audit_sample.py --unblind` is run after reviewing.

**How to review (about 30–40 s per image, ~2.5 hours in total; one cohort ≈ 40 min).**
For each row of `review_sheet.csv`:
1. Open `image_file` (the raw 518-px image the models saw). Fill `artifact_present` (yes/no) for the cohort's artifact
   (hair; sonographer caliper or measurement mark; capsule debris/bubbles) and `artifact_location` relative to the
   lesion/nodule/tumour (inside / outside / both / none).
2. Open `overlay_file` (green = ROI outline, red = our automatic artifact mask). Fill `mask_correct` (yes / partly / no:
   does red cover the artifact and little else?) and `roi_mask_correct` (yes / partly / no: does green follow the
   lesion?). `notes` is free text.
Save the filled sheet (e.g. `audit/review_sheet_filled.csv`) and run
`python audit/analyse_audit.py audit/review_sheet_filled.csv [second_reviewer_sheet.csv]`; the results fill the
placeholder of Stage 6.

**Licences.** Thyroid images and every ISIC overlay (which draws hair masks without a stated licence) are not in the
repository; `make_audit_sample.py` regenerates them into `audit_local/` from the original data (see `LICENCES.md`).
The sheet's paths point there for those rows.
