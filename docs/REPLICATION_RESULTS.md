# Replication results: Viviano 2021 (P1), Pahde 2025 (P2), Bassi/ISNet 2024 (P3)

Status: **in progress**, filled in as runs finish. Every number traces to a file under `results/replication/`. Rules:
`docs/PREREGISTRATION_REPLICATION.md` (with amendments A1–A11). Phase-0 checks, deviations and engineering notes:
`docs/REPLICATION_STATUS.md`.

## How to read this report

| comparison | role | why |
|---|---|---|
| P2 head-to-head on its own biased test | **primary** | localized artifact (inserted "microscope" vignette) |
| P3 head-to-head on its own o.o.d. test | **primary** | localized artifact (corner marks) |
| P3 location test (marks moved inside the lungs, phase 4) | **primary; the strongest location result** | ISNet's own design assumes the mark lies outside the lungs |
| P1 location test (tube inside vs outside the lungs + heart ROI, phase 4) | **primary for P1** | it addresses the thesis question (where the shortcut sits) |
| P1 head-to-head on its original tests | **secondary**: *global shortcut (hospital / AP-PA view), not a localized artifact* | the signal is spread over the whole image, inside and outside the mask at once, so "inside vs outside the ROI" does not apply |

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

### Head-to-head on P1's original tests (secondary: global shortcut (hospital / AP-PA view), not a localized artifact)
**Preliminary, 1 seed (1234), no verdict (A14).** Paired image-only bootstrap; training-seed variance not included. Full tables:
`results/replication/viviano/preliminary_xray_1234.md`, `preliminary_rsna_1234.md`; per-image test scores in
`results/replication/viviano/scores/`.

| task (target) | target AUROC | Standard | arm a | arm b | arm c (U-MtE) |
|---|---|---|---|---|---|
| cardiomegaly (Masked, A5) | 0.829 (paper 0.77 ± 0.09) | 0.627 (paper 0.70) | 0.866, Δ +0.037 [+0.031, +0.043] | 0.869, Δ +0.040 [+0.026, +0.054] | 0.593, Δ −0.236 [−0.256, −0.216] |
| RSNA pneumonia (ActDiff, A11) | 0.708 (paper 0.68 ± 0.05) | 0.186 (paper 0.20) | 0.610, Δ −0.099 [−0.132, −0.065] | 0.767, Δ +0.059 [+0.021, +0.098] | 0.378, Δ −0.330 [−0.375, −0.285] |

Notes:
- **Arm a on RSNA** (`a_mask_cmc_their_net`) is the registered arm a: mask_cmc on the ActDiff network's features of
  PSPNet-lung-masked images (A11 ROI, A12 network). Verified: the ActDiff network's final layer reproduces the saved logits of
  the feature pass to 5e-7, the Masked network's does not (1.7). The table's *diagnostic* row (`arma_lungs`) is the ActDiff
  network's own logits on those lung-masked images (0.689): lung masking alone barely changes ActDiff; it is not a method.
- **RSNA Masked** (0.759 in their definition, 0.840 as logit-difference AUROC) is label-dependent at test time (A11). The two
  definitions disagree because every negative gets the identical score, and ties are handled differently.
- **Clean (within-site / within-view) AUROC:** RSNA ActDiff 0.471 (below chance): its reversed-test AUROC comes mostly from the
  view, not the disease. Arm b keeps 0.763.
- U-MtE is designed for localized overlays inside the ROI and does not remove a global shortcut; it fails on both P1 tasks.

### Location test (primary for P1; phase 4)
*pending*

## P2 — Pahde et al., Machine Learning 2025 (primary)
*pending (Vanilla training, seed 0)*

## P3 — Bassi et al., Nature Communications 2024 (primary)
*pending (data ready; ISNet starts after the P1 gate; BIMCV test images fetching)*
