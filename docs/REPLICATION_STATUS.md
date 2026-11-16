# Replication status (three papers: Viviano 2021, Pahde 2025, Bassi/ISNet 2024)

Live log for the replication described in `docs/REPLICATION_PLAN.md`. Updated 2026-11-16.

## Machine

| item | value |
|---|---|
| GPU | 1x NVIDIA GeForce RTX 5060 Ti, 16 GB (16311 MiB), compute capability 12.0 (Blackwell) |
| driver / max CUDA | 580.159.03 / CUDA 13.0 |
| CPU | AMD EPYC 7K62, 96 logical cores visible |
| RAM | 251 GB |
| disk | `/workspace`: 1.1 TB host volume (persists through recycle/destroy); `/`: 107 GB container disk (not persistent) |
| OS | unprivileged Docker container (Vast.ai base image), kernel 5.15.0-119 |
| layout | repo at `/root/Where_shortcut_sits`; `WTSS_DATA=/workspace/data`; third-party clones at `/workspace/third_party/<paper>` (linked as `third_party/<paper>/code`) |

**GPU constraint for third-party code.** Blackwell (cc 12.0) needs CUDA ≥ 12.8 builds (torch ≥ 2.7). The pinned environments
of the papers cannot run on this GPU: Viviano pins CUDA 10.1 / Python 3.7 (`environment.yml`); ISNet pins torch 1.11+cu113,
pytorch-lightning 1.6.3, Python 3.8; Pahde pins nothing (unversioned `requirements.txt`). Each paper therefore gets its own
env with the **paper's Python-level code unchanged** and only torch/lightning/captum raised to cu128 builds; every code change
needed for that is kept as a `.patch` in `third_party/<paper>/`. This is a recorded deviation, not a choice.

The GPU has 16 GB, less than the 24 GB of the machine that produced the thesis results (`docs/ENVIRONMENT.md`). The DINOv2
ViT-B/14 @518 arm b) may need smaller batches; this does not change results for frozen features.

## Phase 0: verification against the papers and the official code

Sources were opened directly: arXiv 1910.00199 (Viviano), arXiv 2501.13818 (Pahde), the Nature Communications PDF and
Supplementary Information (Bassi), and the code linked in each paper's code-availability section.

### P1 Viviano et al., ICLR 2021

- Code: https://github.com/josephdviviano/saliency-red-herring (linked on p. 2, footnote 1), commit `339860257f1ae4404a7f17171c21e4fcbf1b3add`.
- **Mask: a central circle (ellipse), not a lung mask** for the cardiomegaly (site) task. Paper p. 6 (§4.2): "we hypothesized
  we could improve overfitting performance by masking out the edges of the image using a circular mask". Code
  `activmask/datasets/xray.py` (`JointDataset`): `skimage.draw.ellipse(c, c, 224/2.5, 224/2.5)`, a circle of radius 89.6 px at 224 px.
  For the pneumonia task the mask is the **RSNA pneumonia bounding boxes** (`JointXRayRSNADataset`, `pathology_masks`).
- **The pneumonia task is on RSNA Pneumonia (Kaggle), not PadChest/NIH** (§4.3, p. 7): "we made use of the RSNA Pneumonia
  challenge (Shih et al., 2019) dataset … AP and PA views". Config `rsna-seeds/base_rsna.yml` reads `stage_2_train_images_jpg`.
  This dataset was not in our data list (see Decisions needed).
- **Limitation quote** (§5, pp. 8–9): "In particular, the methods we explored are not guaranteed to negate any confounding
  variables that exist within the mask, which could explain these results, and might additionally give rise to underspecified
  trained models, as evidenced by the models with high test set variability across seeds (D'Amour et al., 2020). This is
  particularly true for the Masked models, for which the representations constructed from inside of the masks were not
  regularized in any way."
- **Split files: not in the repo.** The splits are generated in code with `np.random.seed(1234)` from the NIH and PadChest CSVs
  (50/25/25, classes and sites balanced, 90/10 site–label link in train, reversed in valid and test). Reproducible only if
  our CSVs have the same rows in the same order: the config uses NIH `Data_Entry_2017.csv` (the original, not `_v2020`) and
  PadChest `PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`. PadChest keeps PA views only; labels "cardiomegaly" vs "normal";
  NIH "Cardiomegaly" vs "No Finding". Paper sizes: N_train 5542, N_valid 2770, N_test 2770; RSNA 2696/1348/1348.
- **Resolution: the 224-px TorchXRayVision copies are what the paper used.** Configs `xray-seeds/base_xray.yml` read
  `NIH/images-224` and `PC/images-224`; the paper says "All images were 224 × 224 pixels" (p. 6). The Academic Torrents
  `images-224.tar` (PadChest, ~12.8 GB) is sufficient; **no full-resolution download is needed.**
- **Validation is the reversed distribution** (p. 6: "produce validation and test set where the reverse relationship is true").
  Model selection (best valid epoch, Bayesian hyperparameter search) is done on this reversed validation set. We keep their protocol
  for their methods; for our arms the registered rule must say which validation we select on (see Phase 1).
- **Numbers (paper wins).** Table 1 (p. 7, cardiomegaly, with SPC, AUROC mean ± SD over 10 seeds): Baseline 0.70 ± 0.05,
  **Masked 0.77 ± 0.09**, ActDiff 0.73 ± 0.03, Adversarial 0.68 ± 0.07, GradMask 0.74 ± 0.03, RRR 0.57 ± 0.09. No SPC:
  0.93 / 0.90 / 0.63 / 0.92 / 0.94 / 0.93. **Pneumonia numbers are in Table 2 (p. 8), not Fig. 7.** With VPC: Baseline
  0.20 ± 0.03, Masked 0.56 ± 0.16, **ActDiff 0.68 ± 0.05**, Adversarial 0.62 ± 0.10, GradMask 0.49 ± 0.06, RRR 0.21 ± 0.04.
  The list in the plan matches except for the table reference, and the masked pneumonia SD (± 0.16) was missing.
- Setup: ResNet-18 (torchvision), Adam, 100 epochs, best valid epoch, batch 16, 10 seeds. Tuned per-method settings are in
  `activmask/config/xray-seeds/*.yml` (e.g. baseline lr 0.009992; masked `mask_all: True`, lr 0.01; ActDiff λ 10, lr 0.000583;
  GradMask λ 0.385, lr 0.009238; RRR λ 7.74, lr 0.001355).

### P2 Pahde et al., Machine Learning 114:206 (2025)

- Code: https://github.com/frederikpahde/medical-ai-safety (paper p. 2), commit `4990898b0f0b88ba42458ebb9caa89d1f99c393f`.
- **Artifact placement.** The controlled ISIC 2019 artifact is a "microscope-like black circle" (p. 10). Code
  `utils/artificial_artifact.py::insert_microscope`: everything outside a **centred** circle of random radius
  [W/2 − 3, W/2 + 15) is set to black (a vignette). Position fixed (image centre), radius random. In dermoscopy the lesion is
  usually central, so **the artifact lies outside the lesion by construction**: it only reaches a lesion that touches the image edge.
  (Appendix A.5 says "brightness artifact inserted into 20% of melanoma"; Table 2 and the code say microscope. We follow the
  code and Table 2.)
- **Splits: not released as files, but deterministic.** `datasets/base_dataset.py::do_train_val_test_split` uses
  `np.random.default_rng(0)`: 10% validation, then 10% test, over the 25,331 ISIC 2019 images in CSV order. Artifact presence
  is drawn with `np.random.seed(0)` (`datasets/isic/isic_attacked.py`), p_artifact = 0.2 for class MEL. We can rebuild these exactly.
- **Numbers (paper wins).** Table 1 (p. 13, ResNet-50, ISIC 2019 column, accuracy): Vanilla clean 0.87 / biased 0.28; RRR
  0.84 / 0.51; **RR-ClArC 0.86 / 0.76**; P-ClArC 0.82 / 0.59; rP-ClArC 0.87 / 0.60. **No spread or seed count is reported** (one run).
  The plan's "clean 0.86" is RR-ClArC's clean accuracy; Vanilla clean is 0.87.
- Configuration: ResNet-50 is **timm `resnet50d.a3_in1k`** (A.6), 224 px, 300 epochs, lr /10 at epochs 150 and 250; chosen
  ISIC-attacked setting Adam, lr 0.001 (`config_files/chosen_model_params.json`). Mitigation (RRR, RR-ClArC) fine-tunes the
  Vanilla model for 10 epochs with λ ∈ {10, 50}; RR-ClArC: `direction_mode=signal`, `cav_mode=cavs_max`,
  `criterion=all_logits_random`, at one of the last 3 ResNet layers; RRR (`RRR_ExpMax`) uses the ground-truth artifact masks.
  **Which λ and layer produced Table 1 is not stated in the paper or the code** (open item; we will choose on validation and register the rule).

### P3 Bassi et al., ISNet, Nature Communications 15:291 (2024)

- Code: https://github.com/PedroRASB/ISNet (code-availability section), commit `7ad0a9ee15ab57fa055c9974f7c336aee26b34af`.
- **Does the code run?** The repo contains the ISNet library (`code/`, `backup/`), the benchmark models
  (`backup/AlternativeModels`: U-Net, multi-task U-Net, RRR, GAIN, AG-Sononet, ViT) and an MNIST demo. It pins torch 1.11+cu113,
  which does not run on this GPU (see above); not yet tested after porting. **It contains no COVID-19 data pipeline, no split
  lists and no code for the geometric marks**: the benchmark must be rebuilt from the Supplementary text.
- **Datasets.** Training/validation COVID-19 from **BrixIA** (4,644 frontal X-rays with Brixia score > 0, Supp. Note 5.1, p. 27);
  normal and pneumonia from CheXpert (Supp. Note 5.1). **The Table 1 test set is the o.o.d. set** (Supp. Note 10, p. 48: n = 3,126:
  1,515 COVID-19 from **BIMCV COVID-19+** iterations 1+2, 1,295 pneumonia from ChestX-ray14, 406 normal from Montgomery (80) +
  Shenzhen (326)). So **P3's own test cannot be built without BIMCV COVID-19+**.
- **Marks** (Supp. Note 5.1, p. 29): white geometric shapes in the **upper-left corner**, height and width 10% of the image;
  triangle = COVID-19, square = normal, circle = pneumonia; on all training and validation images. Deceiving test: circle on
  normal, triangle on pneumonia, square on COVID-19. Images 224 × 224; augmentation (rotation, translation, flip) can remove marks.
- **U-Net lungs-only pipeline** (Supp. Note 2.1, Algorithm 1): U-Net (original architecture) trained beforehand on the same
  classification dataset, with targets made by a lung U-Net from a previous study (threshold 0.4); U-Net output thresholded to a
  binary mask, image multiplied by it, then DenseNet-121 classifies; U-Net frozen; no pre-training (Supp. Note 6.4).
- **Numbers (paper wins).** Table 1 (COVID-19 rows, macro-F1 over biased / standard / deceiving tests): ISNet 0.775 ± 0.008 on
  all three; standard DenseNet-121 0.775 ± 0.008 / 0.434 ± 0.01 / 0.195 ± 0.004; segmentation–classification pipeline
  0.618 ± 0.009 / **0.619** ± 0.009 / 0.618 ± 0.009 (plan said 0.618 on all three). TB o.o.d. AUROC: ISNet 0.809 ± 0.031.
  **The ± is a Bayesian posterior SD over test images (Supp. Note 10), not a spread over training seeds**; the paper reports one
  trained model per method.

## Data status

| dataset | needed for | status |
|---|---|---|
| ISIC 2019 (+ HAM10000 masks) | P2, P2 location | downloading (our pipeline) |
| NIH ChestX-ray14 | P1 cardiomegaly; P3 o.o.d. pneumonia | downloading (our pipeline); P1 needs the original `Data_Entry_2017.csv` |
| PadChest 224 px (Academic Torrents) | P1 cardiomegaly | to download |
| RSNA Pneumonia (Kaggle) | P1 pneumonia | **not in the agreed list: needs your OK** |
| CheXpert small | P3 normal/pneumonia training | to download (Stanford registration) |
| BrixIA | P3 COVID-19 training | **waiting for your link** |
| BIMCV COVID-19+ | P3 test (1,515 COVID-19) | **skipped per instruction: P3's own test cannot be built** |
| Montgomery + Shenzhen | P3 test normals | small, open (NLM) |

## Blocked / decisions needed

1. **P3 test set needs BIMCV COVID-19+.** Without it we cannot score on P3's test, so there is no valid head-to-head number for P3.
   Options: (a) you approve the BIMCV download (iterations 1+2, registration required, large), (b) we run P3's design on an
   i.i.d. held-out split of BrixIA + CheXpert and label it as *not* P3's test (no "reproduced"/"beaten" claim).
2. **P3 BrixIA link** (from you).
3. **P1 pneumonia needs RSNA Pneumonia** (Kaggle competition data, ~3.7 GB; your Kaggle account must accept the rules).
4. **Third-party envs on Blackwell** use cu128 builds instead of the pinned versions (above). Accept as a recorded deviation?
