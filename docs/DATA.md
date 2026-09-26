# Data sources, licences, and derived objects

| Dataset | Use | Source | Licence / terms |
| --- | --- | --- | --- |
| ISIC 2018 Task 1/2 images + Task 1 masks | synthetic-ruler pilot cohort; U-Net training | ISIC challenge S3 | CC-BY-NC |
| ISIC 2016/2017 Part 3 labels | pilot melanoma labels (union; 3 conflicts dropped) | ISIC | CC-BY-NC |
| ISIC 2019 training (25,331) | real-hair traps | ISIC challenge S3 | CC-BY-NC |
| HAM10000 lesion segmentations | manual lesion masks (10,015) | Tschandl et al., Harvard Dataverse doi:10.7910/DVN/DBW86T | CC-BY-NC |
| ISIC 2019 artifact masks (hair+ruler, ink, vignetting) | artifact geometry, oracle inpainting, I2E donors | Wegley et al. 2026, Scholars' Mine doi:10.71674/man1-qa33 | see source |
| NIH ChestX-ray14 (112,120) | CXR cohorts | NIH CC (HF mirror `alkzar90/NIH-Chest-X-ray-dataset`) | NIH terms (unrestricted use, cite) |
| NEATX | chest-drain labels for 3,543 NIH pneumothorax images | Jiménez-Sánchez et al., zenodo 14944064 | CC BY-NC-SA 2.0 |
| torchxrayvision PSPNet | lung ROI masks | Cohen et al. | Apache-2.0 |

## Derived objects (all rebuilt by `make data`)
- `isic2019/prepared/cohort_groups.csv` — pHash (64-bit) and leakage groups (lesion_id ∪ pHash ≤ 2).
- `isic2019/unet/UNET_REPORT.json` — lesion U-Net held-out Dice.
- `isic2019/prepared/cohort_518.csv` — per image: source, lesion-mask source, hair fraction, hair∩lesion overlap r, ink, vignetting.
- `cxr/prepared/DRAIN_DETECTOR.json` — drain detector CV AUROC and thresholds; `nih_drain_scores.csv`.
- `cache/features/**` — per-view frozen features (npz); `cache/images/**` — memmapped resized images.
