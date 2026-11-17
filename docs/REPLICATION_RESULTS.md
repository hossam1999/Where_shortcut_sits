# Replication results: Viviano 2021 (P1), Pahde 2025 (P2), Bassi/ISNet 2024 (P3)

Status: **in progress**, filled in as runs finish. Every number traces to a file under `results/replication/`. Rules:
`docs/PREREGISTRATION_REPLICATION.md` (with amendments A1–A11). Phase-0 checks, deviations and engineering notes:
`docs/REPLICATION_STATUS.md`.

## How to read this report

| comparison | role | why |
|---|---|---|
| P2 head-to-head on its own biased test | **primary** | localized artifact (inserted "microscope" vignette) |
| P3 head-to-head on its own o.o.d. test | **primary** | localized artifact (corner marks) |
| P1 location test (tube inside vs outside the lungs + heart ROI, phase 4) | **primary for P1** | it addresses the thesis question (where the shortcut sits) |
| P1 head-to-head on its original tests (site / AP-PA view) | **secondary** | P1's shortcuts are global (hospital source, AP/PA view): they change the whole image, the ROI included, rather than sitting at a location |

"Preliminary" tables (one seed of P1) carry no verdict. With one training seed (P2, P3, phase 4) the bootstrap resamples test
images only, so training-seed variance is not included and "beaten" means "beaten for this training run" (A3).

## P1 — Viviano et al., ICLR 2021 (secondary head-to-head; primary location test)

### Finding: P1's RSNA "Masked" test input depends on the label

In P1's RSNA pneumonia task, "Masked" multiplies each image by its pneumonia bounding boxes. Negatives have no box.

- **Code** (P1 at `32f7b7d`, TorchXRayVision 0.0.18 as pinned by P1):
  - `torchxrayvision/datasets.py`, `RSNA_Pneumonia_Dataset.get_mask_dict` (lines 492–515): every mask starts as
    `np.zeros([size, size])`, and only the boxes are set to 1.
  - `activmask/datasets/xray.py`, `JointXRayRSNADataset.__getitem__`: `seg = sample["pathology_masks"][0]` (line 650),
    then `img *= seg` when `mask_all` is set (lines 669–671).
  - The Masked config sets `mask_all` on the test split. Because of a YAML duplicate key it is set on the test split *only*;
    see `REPLICATION_STATUS.md`.
- **Statistic** (seed 1234 test split): 200 of the 676 test negatives were inspected, and **every one becomes the same
  constant image (pixel SD 0, value 0.0)**. Positives keep only their box contents (3,396 of 50,176 pixels on average).
  By the code above this holds for every negative.
- **Figure:** `figures/replication/p1_rsna_masked_test_masks.png` shows the masks applied (4 negatives: all zeros;
  4 positives: box masks). No X-ray pixels are committed; the masked X-ray examples stay under `$WTSS_DATA`.
- **Consequence** (A11): their RSNA Masked score (paper 0.56 ± 0.16) is reported as **label-dependent at test time** and is
  not a head-to-head target. The RSNA target is ActDiff (0.68 ± 0.05). Our arm a on RSNA uses label-free PSPNet lung ROIs.
  Cardiomegaly (circle mask, label-free) is unaffected.

### Reproduction (their code; 3 seeds: 1234, 3232, 3221)
*pending (gate after seed 1234; A8)*

### Head-to-head on P1's original tests (secondary)
*preliminary seed-1234 tables: `results/replication/viviano/preliminary_<task>_1234.md`; 3-seed verdicts pending*

### Location test (primary for P1; phase 4)
*pending*

## P2 — Pahde et al., Machine Learning 2025 (primary)
*pending (Vanilla training, seed 0)*

## P3 — Bassi et al., Nature Communications 2024 (primary)
*pending (data ready; ISNet starts after the P1 gate; BIMCV test images fetching)*
