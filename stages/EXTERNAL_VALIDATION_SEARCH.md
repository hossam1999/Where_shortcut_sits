# External validation — search record and decision (A4)

Rule (docs/PREREGISTRATION_FINAL.md, A4, committed and pushed before any external label was read): run only if all of
(1) direct download without registration, application or agreement; (2) licence permits research use; (3) patient-level
identifiers; (4) ROI masks exist or can be produced by our existing dermoscopy U-Net; (5) cell-count gate: in both
traps at least 50 positive and 50 negative artifact-bearing images and at least 50 positive and 50 negative
artifact-free images. No licence, agreement or registration form was accepted and no credentials were entered.

## Candidates searched

| Dataset | Modality | (1) direct download | (2) licence | (3) patient IDs | (4) ROI masks | Decision |
|---|---|---|---|---|---|---|
| **ISIC 2020 Challenge training set** (Rotemberg et al., Sci Data 2021) | dermoscopy, 33,126 images, 2,056 patients, 584 melanomas | yes (public S3 bucket, no login) | CC BY-NC 4.0 | yes (`patient_id`) | lesion masks from the existing spec U-Net (trained on ISIC 2018 Task 1 only); hair masks from a segmenter trained on ISIC 2019 hair masks only | **run** (gate met, below) |
| ThyUS2Path (Hou et al. 2024) | thyroid ultrasound, 842 patients, pathology | yes | CC BY 4.0 | yes | **no nodule masks**; the frozen caliper detector responds to the scanner interface | not run: fails (4); candidate for the author (needs a transferred nodule segmentation and a re-validated detector) |
| TN-SCUI 2020 | thyroid ultrasound, one image per patient, masks | registration | terms forbid use outside the challenge | yes | yes | not run: fails (1) and (2) |
| DDTI | thyroid ultrasound, 134 images | yes | research | partly | yes | not run: TI-RADS labels, no pathology; too small for the gate |
| Stanford AIMI datasets | several | data-use agreement | agreement | yes | varies | not run: fails (1); an agreement cannot be accepted on the author's behalf |

## ISIC 2020 — what was done

- **Frozen before any ISIC 2020 image was scored.** Hair segmenter: U-Net (ImageNet ResNet-34 encoder), 3,000 steps on
  90% of the ISIC 2019 518-px images and their published hair masks; the image-level hair-free threshold (predicted
  hair pixels ≤ 200 at 518 px) chosen on the other 10% by balanced accuracy against the paper's hair-free definition
  (0.93; pixel IoU 0.60) — `results/rerun_2026-09-28/external_isic2020/segmenter.json`. Lesion masks: the existing
  spec U-Net (ISIC 2018 Task 1 only), 256 px, upsampled.
- **Gate** (`gate.json`): hair-free 53 melanomas / 1,954 other; Trap A (hair on the lesion, r ≥ 0.5) 79 / 2,057;
  Trap B (hair outside, r < 0.1) 231 / 18,685 → **met** (smallest cell 53 ≥ 50). ISIC 2020 has a single source, so
  source matching does not subsample.
- **Traps**: the ISIC 2019 spec protocol unchanged (DINOv2 ViT-B/14 @518, logistic heads, arms ERM and ROI masking,
  5 seeds × 5 folds), groups = `patient_id` (patient-disjoint folds).
- **Result (claim X1, crossed bootstrap)**: see `results/rerun_2026-09-28/external_isic2020/traps_dino518/SUMMARY.json`
  and `bootstrap_vs_erm.csv`; reported in Stage 6 and MORNING_SUMMARY.md.

## Limitations of this external test
- The artifact (hair) masks are automatic; the segmenter was validated on ISIC 2019, not on ISIC 2020.
- ISIC 2020 has few melanomas among hair-free images (53), so the artifact-free cell is small.
- Same modality and artifact as the main dermoscopy analysis: this is external validation of the dermoscopy result on
  new patients and a new acquisition campaign, not of the other modalities.
