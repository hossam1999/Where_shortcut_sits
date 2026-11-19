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
- **P1 split check (RSNA pneumonia, seed 1234, their code, TorchXRayVision 0.0.18).** The balanced pool matches the
  paper exactly: n_per_category = 1,348, so 4 × 1,348 = 5,392 = 2,696 + 1,348 + 1,348. **But the code does not build the
  splits the paper describes:**
  - *Sizes:* the code gives train 1,346, valid 672, test 3,374 (the test is all images left over). The paper's
    2,696 / 1,348 / 1,348 are the 50/25/25 fractions of the pool, not the sets the code trains and tests on.
  - *Correlation:* **validation has the same link as training**, not the reversed one: 303 of 336 validation positives are
    AP, as in training (606 of 673). The **test is reversed only at ~74/26** (439 of 1,687 positives are AP), not 10/90,
    because it is the remainder. The paper text (p. 6, p. 7) says validation and test are reversed.
  - Their model selection (best validation epoch) is therefore done on a validation set that carries the training shortcut.
  - We **run their code as is** (it produced their numbers) and keep the paper's numbers as the targets. Our arms use the same
    validation split, so the registered rule "select on the paper's validation split" applies unchanged, although it is
    the correlated split and not a reversed one.
  - The cardiomegaly (site) split is checked the same way once PadChest is on disk.
  - Their "Masked" input for pneumonia multiplies the image by the pneumonia boxes. Negatives have no box, so a masked
    training negative is an all-zero image (seg sum 0). This is their method as coded; we keep it.
- **P1 split check (cardiomegaly, seed 1234).** n_per_category = 2,772 (paper's sizes imply 2,771: 5,542 = 2 × 2,771); the
  one-image difference per cell probably comes from PadChest-224 having 160,845 images against 160,868 in the full release.
  The pattern is the same as RSNA: train 2,770 (1,247 of 1,385 positives from PadChest); valid 1,384 with the same link
  (623 of 692); test 6,934, the remainder, reversed only ~74/26 (902 of 3,467 positives from PadChest).
  **Input size: `JointDataset` defaults to `new_size=128` and no xray config overrides it, so cardiomegaly images enter
  the network at 128 × 128, not 224 × 224 as the paper states (p. 6).** The circle mask is drawn at 128 px too. Run as coded.
- **P1 "Masked" masks only the test images (as coded).** Both `xray-seeds_resnet_maskall.yml` and
  `rsna-seeds_resnet_maskall.yml` contain three top-level `dataset:` keys (train, valid, test). YAML keeps only the last, so
  the loaded config has `mask_all` = None / None / True for train / valid / test (checked with their `load_config`). The
  "Masked" model is therefore **trained and selected on unmasked images and scored on masked images**. This contradicts the
  paper's description of masking as a training intervention. The files date from 2020-09-24 (`eb307f5`, "cleaned up configs
  and added rrr baseline"), before the ICLR camera-ready. PyYAML of that time also kept the last duplicate key, so this is
  very likely how the published Masked numbers were produced. For pneumonia, a test negative (no box) becomes an all-zero image.
  - We reproduce **their code as is**.
  - As an exploratory check (labelled so), we also run "Masked" with the mask applied to train, valid and test, as the paper describes.
  - **Open for arm a (P1).** The registered text says arm a uses "the Masked ResNet-18 with its own input pipeline". As coded,
    that model was trained on unmasked images. Extracting its training features on unmasked images and its test features on
    masked images is what "its own pipeline" literally means. This must be settled (amendment) before arm a runs; see
    "Blocked / waiting".
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

- **P2 data check (their code, our ISIC 2019 download).** ISIC 2019: 25,331 images and 25,331 ground-truth rows (official
  count). Their split: train 20,265, validation 2,533, test 2,533. Microscope artifact on 916 of 4,522 melanomas (20.3%) and on
  no other class (train 745, validation 81, test 90).
  Their training sets no random seed (the paper reports one run), so a `seed` key was added (`third_party/pahde/cu128_port.patch`);
  with no seed the code behaves as before. The batch size comes from `local_config.yaml`; the released value is 12, the paper
  does not state it, and we use 12.

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

## BIMCV COVID-19+ for P3's test (2026-11-16)

**Source.** The owners' free mirror (linked from github.com/BIMCV-CSUSP/BIMCV-COVID-19, `BIMCV-COVID19+/Readme.md`):
B2DROP public share `BIMCV-COVID19-cIter_1_2_3`, via WebDAV. It is reachable. It has 571 subject archives
(`covid19_posi_subjects_partaa…partvy.tar.gz`), each with a `-tvf.txt` listing, plus metadata, sessions, derivatives
and SHA1 sums. Licence: BIMCV-COVID19 Research Use Agreement (academic use, cite, no redistribution). No image is
ever committed.

**Small files downloaded:** all 571 listings, `covid19_posi_metadata.tar.gz` (participants.tsv), `covid19_posi_sessions_tsv.tar.gz`,
`sha1sum.txt`, `sha1sum_b.txt`, `LICENSE.md`, `00_extract_data.sh` (66 MB). Also `covid19_posi_derivative.tar.gz` (74 MB,
SHA1 OK), from which only `derivatives/labels/Labels_covid_123.tsv` (the NLP report labels) and `partitions.tsv` were kept.

**P3's rule applied** (`scripts/replication/bimcv_select.py`; Supp. Note 5.1). Iteration 1+2 = 5-digit subject ids
(S03044–S12845; iteration 3 has 6-digit ids and was added in v1.0.2, 2022-11-20; all 10 figshare test images have 5-digit ids).

| step | images |
|---|---|
| PNGs in all listings | 57,183 |
| iterations 1+2 | 14,308 |
| frontal (vp-ap / vp-pa) | 12,914 |
| report NLP label "COVID 19" | 2,508 |
| DICOM sidecar (.json) present | 2,508 |
| adult (youngest recorded age ≥ 18, participants.tsv) | **2,501** (1,631 patients) |

**The count does not match P3: 2,501 images and 1,631 patients against P3's 1,515 and 1,145.** Further filters from P3's
text do not close the gap:
- study date within P3's stated period (16 Feb – 8 Jun 2020): 2,423 images, 1,591 patients;
- also excluding reports labelled "COVID 19 uncertain": 2,374 images, 1,560 patients;
- also one image per session: 2,183 images, 1,560 patients.

**All 10 figshare test images are in every variant.** P3 most likely used an earlier state of the release (labels and
metadata as of 2020–21) that the mirror no longer serves. The test is therefore labelled **"rebuilt by P3's rule, not the
identical image list; 2,501 vs 1,515 images"**. The primary rebuilt test is the 2,501-image rule above, fixed now before
any scoring. The date-window variant is reported as a sensitivity check.

**Download cost.** The 2,501 images (36.9 GB of PNG) are spread over **531 of the 571 archives (~2.8 TB)**. A `.tgz` cannot
be read by byte range, and SHA1 verification needs the whole archive. `scripts/replication/bimcv_fetch.sh` therefore
processes one archive at a time: download (≥ 18 MB/s measured), SHA1 check against the owners' `sha1sum.txt`, extract only
the selected PNG + JSON, delete the archive. Disk use stays ≤ ~6 GB at a time; the expected time is about 1–1.5 days. Every
archive's bytes, SHA1 result and files kept are logged in `$WTSS_DATA/bimcv/fetch_log.tsv`. Started 2026-11-16.

**NIH pneumonia part of P3's test.** All 10 figshare NIH images carry a Pneumonia label, some with other findings, so the rule is
"any Pneumonia label, age ≥ 18". With the original `Data_Entry_2017.csv` it gives 1,247 images and 890 patients. With NIH's
2020 re-issue `Data_Entry_2017_v2020.csv` (corrected ages) it gives **1,319 images and 938 patients, mean age 47.2 ± 15.4,
58.8% male**, against P3's 1,295 images, 941 patients, 48 ± 15.5 and 58.7% (Supp. Note 5.1). We use the v2020 rule; the
24-image gap is unexplained and reported.

**P3 pre-trained weights.** From the Google Drive folder in the P3 README: `COVIDIsNet.pt` (28,506,899 bytes, SHA256
`760e9b62b79a7ed4d7a07961193aa103cc6706b758ff7705e6bfa087c95f87b9`). The folder has **no synthetic-bias COVID model**.
`COVIDIsNet.pt` is the COVID ISNet without synthetic bias, scored in **Table 2 on the same o.o.d. test** (macro-F1
0.773 ± 0.009; per-class F1 (Table 2 column order): normal 0.555, pneumonia 0.858, COVID-19 0.907). Scoring it on the rebuilt test checks how close
our test is to theirs.

## Data status

| dataset | needed for | status |
|---|---|---|
| ISIC 2019 (+ HAM10000 masks) | P2, P2 location | downloading from the ISIC S3 bucket (our pipeline's source) |
| NIH ChestX-ray14, 224 px + original `Data_Entry_2017.csv` | P1 cardiomegaly | **done** (Academic Torrents `e615d3ae…`, "NIH Chest X-ray Dataset (Resized to 224x224)", the `NIH/images-224` folder P1's config reads) |
| PadChest 224 px "Fixed cropping" + `PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv` | P1 cardiomegaly | downloading (Academic Torrents `96ebb4f9…`, 12.9 GB) |
| RSNA Pneumonia JPG (`kaggle-pneumonia-jpg/stage_2_train_images_jpg`) | P1 pneumonia | downloading (Academic Torrents `95588a73…`, 3.9 GB); approved |
| NIH ChestX-ray14 pneumonia (1,295 images) | P3 test | from the NIH copy above (selection rule: Pneumonia label, age ≥ 18) |
| Montgomery + Shenzhen | P3 test normals (406) | downloading (Academic Torrents `ac786f74…`, `462728e8…`) |
| CheXpert small | P3 training normal/pneumonia | waiting (registration) |
| BrixIA | P3 training COVID-19 | waiting for the author's link |
| BIMCV COVID-19+ iter. 1+2 | P3 test COVID-19 (rebuilt: 2,501) | fetching from the owners' B2DROP mirror, one archive at a time (see BIMCV section) |

## Decisions (author, 2026-11-16)

1. **P3 test: get BIMCV COVID-19+**, under these conditions. The author registers on IEEE DataPort/BIMCV and provides
   the link or credentials; we never register for them. We stream the archive and keep only the frontal COVID-19 images
   P3 used. We never store the full archive, and we log the bytes downloaded and the files kept.
   - If P3's own list of the 1,515 images exists, we match it 1:1 and record missing images.
   - Otherwise we rebuild the selection by P3's stated rule and label the test "rebuilt by P3's rule, not the identical
     image list". "Reproduced"/"beaten" verdicts are still allowed, but only if our reproduced ISNet lands within the
     pre-registered tolerance of P3's Table 1.
   - The external test is completed with the 1,295 NIH pneumonia images and the 406 Montgomery/Shenzhen normals.
   - Until BIMCV arrives, the proxy (held-out BrixIA + CheXpert split, "not P3's test", no verdicts) and the P3 location
     extension run in parallel. The head-to-head is rerun on the real test once BIMCV is available.
   - On the original P3 benchmark (artifact = label) only U-MtE runs; mask_cmc runs on the 90/10 variant and in the location extension.
2. **Third-party code on Blackwell: approved.** Their code runs unchanged except torch/lightning/captum at cu128 builds;
   every change is kept as a `.patch` and recorded as a deviation.
3. **RSNA Pneumonia: approved** (Academic Torrents JPG copy, the folder P1's config reads).

### Search for P3's list of the 1,515 BIMCV images
- Paper, Supplementary Information (MOESM1), peer-review file (MOESM2), reporting summary (MOESM3, MOESM5), Source Data
  (MOESM4: P-hyperparameter sweep table only): **no image list**.
- ISNet repository (commit above): no data files or lists.
- figshare 10.6084/m9.figshare.24243895 (Supplementary Data 1, 385 MB): ISNet heatmaps for 30 example test images only.
  These are 10 BIMCV COVID-19 images (e.g. `sub-S03258_ses-E06425_run-1_bp-chest_vp-pa_dx.png`), 10 ChestX-ray14
  pneumonia images and 10 Montgomery/Shenzhen normals.
- **Verdict: no list of the 1,515 BIMCV images is published.** The P3 test will be **rebuilt by P3's rule, not the
  identical image list**. The rule (Supp. Note 5.1, p. 28): BIMCV COVID-19+ iterations 1+2; AP or PA views (no lateral);
  NLP report label "COVID-19" (not "uncertain"/"negative"); DICOM metadata present; patients ≥ 18 years. Partial check:
  the rebuilt set must contain the 10 BIMCV test images named in the figshare deposit; any that are missing will be
  reported. The same check applies to the 10 NIH and 10 Montgomery/Shenzhen names.

## Blocked / waiting

| item | blocks | waiting for |
|---|---|---|
| BrixIA download link | all P3 work (COVID-19 training class) | author |
| CheXpert (small) | P3 normal and pneumonia training classes | Stanford AIMI registration (author's account; a third-party mirror is not used) |

| P1 arm a input: "their Masked model" is trained unmasked (config bug) | P1 arm a only | author: (i) literal pipeline (train features unmasked, test masked), (ii) all features masked, or (iii) use the exploratory masked-everywhere model |

P1 and P2 are not blocked: their data is downloading, and their code is being ported to cu128.

## Measured timings (2026-11-16; one 16 GB RTX 5060 Ti)

| method (paper's own code and settings) | measured | per run | GPU memory |
|---|---|---|---|
| P1 Standard, RSNA, 100 epochs | full run, sharing the GPU with one job | 1.9 h | 5.6 GB |
| P1 ActDiff, RSNA, 100 epochs | full run, sharing | 2.5 h | 9.4 GB |
| P1 ActDiff, cardiomegaly (128 px) | 7-epoch test, sharing: 80 s/epoch + test scoring | ~2.3 h | ~4 GB |
| P2 Vanilla ResNet-50d, 300 epochs | 2-epoch test, sharing: ~3.0 min/epoch + 1.5 min eval every 5 epochs | ~16.5 h | 2.8 GB |
| P3 ISNet DenseNet-121, 48 epochs, batch 16, mixed precision (P3: batch 10–16, AMP) | 60 iterations alone, random inputs of the real shape | 11.3 h (14.1 min/epoch) | 10.4 GiB |
| P3 ISNet, batch 10 | same | 16.6 h | 6.7 GiB |

ISNet fits on 16 GB and is under the author's 12 h limit at batch 16 when run alone. It is paired only with the small P2
job (2.8 GB), never with P1 ActDiff.

## P1 diagnostics (2026-11-16): why the first runs did not match, and the port fix

The first P1 runs (RSNA, seed 1234) gave Standard 0.355 (paper 0.20 ± 0.03) and ActDiff 0.308 (paper 0.68 ± 0.05). Four checks
were run without re-tuning anything.

**(a) Split proportions (their code at repository HEAD; identical for seeds 1234, 3232, 3221).** RSNA share of AP:
- train: positives 0.90, negatives 0.10;
- valid: **0.90 / 0.10, the same as training**;
- test: **0.26 / 0.74**.

The paper describes training 90/10 with validation and test reversed.

**(b) Paper text vs code, and which code produced the published numbers.** In the repository history, commit `81ecbd3`
(2021-02-03, "syncing xray") changed the split code in `activmask/datasets/xray.py`. **Before it, the training cells were drawn
at twice the size and the validation cells were reversed** (`ratio` and `1 − ratio` swapped). This gives exactly the paper's
design and sizes. The published numbers predate the change: arXiv v2 (2020-10-06) already contains Table 1 (Masked
0.77 ± 0.09), Table 2 (ActDiff 0.68 ± 0.05) and N_train = 5,542. The repository has no GitHub issues.

The commit current at arXiv v2 is `32f7b7d` (2020-10-01). Its only behavioural difference from HEAD is this split code:
- `loss.py` at HEAD only adds an unused function;
- `training.py` differs only in formatting; batch 16 in both (hard-coded at `32f7b7d`, from the configs at HEAD);
- the batch size was 128 only between 2020-10-14 and 2021-02-03, after arXiv v2;
- the Standard / Masked / ActDiff configs are unchanged.

**Our port therefore deviated from the code behind the published numbers. Fix (bug fix, not tuning):**
`activmask/datasets/xray.py` is restored from `32f7b7d` (in `third_party/viviano/cu128_port.patch`). With it, seed 1234 gives:

| task | train | valid | test | correlation |
|---|---|---|---|---|
| RSNA (paper 2,696 / 1,348 / 1,348) | 2,694 | 1,346 | 1,352 | train 0.90 / 0.10; valid and test reversed 0.10 / 0.90 |
| cardiomegaly (paper 5,542 / 2,770 / 2,770) | 5,542 | 2,770 | 2,776 | train 0.90 / 0.10; valid and test reversed 0.10 / 0.90 |

**All P1 runs made with the HEAD split code are superseded** and kept, labelled, in
`$WTSS_DATA/replication/viviano/runs_superseded_head_splits/`. They are not used in any result. They were 8 runs, ~16 GPU-h.
The P1 queue restarted from scratch at 19:16 on the fixed code. The earlier "P1 split check" notes above describe the HEAD
code and are kept as history.

**(c) ActDiff loss is active.** The saved stats of the (superseded) ActDiff run show λ = 1.60256, lr 4.1e-5 and activation
layer [5], as configured. The training ActDiff term is non-zero and falls over training (1.56 at epoch 0, 0.33 at epoch 14,
0.15 at epoch 99); it is 0 at validation and test, as coded. The term is computed between each image's activations and those
of the same image with the pixels outside its mask shuffled (`shuffle_outside_mask`).

**(d) Image size and masks.** Both versions of the code construct cardiomegaly images at **128 × 128** (`JointDataset`
default `new_size=128`, not overridden by any config) and RSNA images at 224 × 224. The paper says 224 for both. We run as
coded. RSNA masks are the challenge's pneumonia boxes, as the paper describes; negatives have no box, so their mask is empty.
The Masked configs mask only the test split (YAML duplicate keys, see above); this is the same at `32f7b7d`.

## P3 data (2026-11-16)

**CheXpert-v1.0-small (verified).** Source: Kaggle mirror `ashery/chexpert` (Kaggle API, author's account; Stanford CheXpert
Research Use Agreement accepted by the author).
- 223,414 train + 234 valid JPGs; `train.csv` (223,414 rows) and `valid.csv` (234 rows); paths prefixed `CheXpert-v1.0-small/`.
- Sample image sizes ~390 × 320 (200 sampled; 390 × 320 most common), so this is the small version.
- One macOS metadata file was skipped when unpacking.
- P3 does not state the resolution beyond "jpg"; every image is resized to 224, so the small version suffices.

**BrixIA (authors' private OSF link; the link is kept outside the repository).**
- The folder zip OSF builds on the fly stopped at 17.0 GB: two of the 15 DICOM archive parts were missing or cut.
- Instead, the 15 parts (`dicom_archive_v2.tar.gz.aa`–`.ao`, 42.2 GB) are fetched one by one and each is checked against
  the SHA-256 that OSF reports (`scripts/replication/brixia_fetch.sh`). Part `aj` failed its check once and is re-fetched.
- The metadata files match OSF's SHA-256.
- **Count:** `metadata_global_v2.csv` lists **4,695 images** (2,351 patients). The changelog says v2 removed 8 erroneous DICOMs
  from v1's 4,703 (= P3's 4,644 + 59). The author's figure of 4,707 matches neither release.
- **P3's rule (Brixia score > 0) gives exactly P3's 4,644 images**, and also P3's "11 patients under 20".

**Rebuilt P3 training cohort** (`scripts/replication/p3_select.py`, seed 20261116): 4,644 images per class; 3,483 train and
1,161 validation per class (P3: 3,483 train), split by patient so no patient is in both.
- normal = CheXpert "No Finding", one image per patient, from 14,000 eligible patients;
- pneumonia = CheXpert "Pneumonia" = 1 (uncertain counts as negative), 4,644 drawn from 4,674 eligible
  (P3: 4,644 from 3,899 patients; eligible pool: 3,927 patients, 74.4% AP, age 62.5 ± 18.6 vs P3's 74.3%, 62.3 ± 18.7);
- all: frontal AP/PA, age ≥ 18.

**Rebuilt P3 test** (`scripts/replication/p3_test_select.py`):
- 406 normals = all Montgomery (80) + Shenzhen (326) normals, exactly P3's count. 36 of them are under 18, so P3's general
  statement about removing minors did not apply to these.
- 1,319 NIH pneumonia; all 10 figshare NIH test images are included.
- 2,501 BIMCV COVID-19 (fetching).
- The figshare Montgomery/Shenzhen examples end in `_1` (TB-positive), so they come from P3's TB experiment, not this test.

**Preprocessing** (`scripts/replication/p3_prepare.py`, P3 Supp. Note 6.1): grayscale → histogram equalization → 224 px
(test images padded to square first) → 3 channels in [0, 1].
- BIMCV PNGs are 16-bit; 16 of 60 checked are flagged MONOCHROME1 in their JSON, but all display with dark lungs, i.e. are
  already stored in display orientation, so no PNG is inverted. BrixIA DICOMs flagged MONOCHROME1 are inverted.
- Low-contrast 16-bit images are equalized at full bit depth.
- **Deviation:** P3's lung segmentation targets came from a lung U-Net of an earlier study that is not published. We use the
  TorchXRayVision PSPNet (lungs, p > 0.5), as in our own pipeline.
- **Mark position:** P3 says only "upper left corner"; we place the 22-px mark with a 4-px margin.

## P1 full code diff: 32f7b7d (2020-10-01, current at arXiv v2) vs the ported HEAD (2026-11-16)

`git diff 32f7b7d HEAD` over the whole repository, notebooks excluded. Every change on the training path:

| file | change | behaviour |
|---|---|---|
| `activmask/datasets/xray.py` | split code (`81ecbd3`): training cells drawn at 2× size and validation cells reversed at 32f7b7d; halved and not reversed at HEAD | **differs; restored to 32f7b7d** |
| `activmask/training.py` | data loaders: batch 16 hard-coded at 32f7b7d; from the config at HEAD (16); `device='cuda'` replaced by `cfg['device']`; formatting | same (batch 16 in both; device line restored by our port patch) |
| `activmask/utils/configuration.py` | default `batch_size` 32 → 16 | none (overridden by hard-coded 16 at 32f7b7d and by the configs at HEAD) |
| `activmask/models/loss.py` | adds `get_grad_saliency` | none (called only by `results.py`, plotting) |
| `activmask/models/resnet.py` | trailing whitespace | none |
| `config/xray-seeds/base_xray.yml` | `batch_size` 64 → 16; `n_iter: 50` added | none (32f7b7d ignored the config batch size and used 16; `n_iter` is used only by hyperparameter search) |
| `config/rsna-seeds/base_rsna.yml` | `n_iter: 50` added | none |
| method configs used (Standard, Masked, ActDiff, both tasks) | unchanged | — |
| `run_seeds.sh` | experiment list and paths; seed list unchanged | none |

Losses (ActDiff, masking), mask loading, augmentation (none in either version), optimiser (Adam, no scheduler), model
selection (lowest validation loss after the 5-epoch burn-in, test scored at that epoch) and evaluation are identical.
**Only `xray.py` changes behaviour; it is the only file restored.**

**A5 re-confirmed under the restored code** (cardiomegaly, seed 1234): the test split built with `mask_all` off and on has
the same 2,776 images, sites and labels in the same order; only the pixels outside the circle are zeroed (50.3% of pixels
non-zero vs 100%). The Standard and Masked configs still differ only in `test.mask_all`. Scoring the Standard model with test
masking is therefore exactly their Masked procedure.

**Gate (author, 2026-11-16).** After the first corrected seed (1234) of Standard and ActDiff on both tasks,
`scripts/replication/p1_gate.py` compares their test AUROC with the paper ± 2 SD:
- Standard: cardiomegaly 0.70 ± 0.05, RSNA 0.20 ± 0.03;
- ActDiff: cardiomegaly 0.73 ± 0.03, RSNA 0.68 ± 0.05.

If any is outside, P1 pauses for the author (the queue holds seeds 3232 and 3221 until `GATE_OK`).

- **Cardiomegaly test: 2,776 images vs the paper's 2,770.** The test is the remainder after the int-truncated training and
  validation draws: 4 × 2,772 − 5,542 − 2,770 = 2,776. Our pool has one more image per cell than the paper's sizes imply
  (2,772 vs 2,771; PadChest-224 has 160,845 images). The paper's 2,770 equals its validation size, i.e. the nominal 25%,
  not the remainder.
- **Gate decision rule (author, recorded 2026-11-16, before any seed-1234 result):**
  - inside paper ± 2 SD: continue;
  - outside that band by ≤ 0.03: continue all seeds, and judge on the registered 3-seed-mean rule;
  - further outside for any of the four: pause P1 and report.
  - Implemented in `scripts/replication/p1_gate.py`.
- **BIMCV fetch check (19:57):** progressing, not hung. 83 of 531 archives (435 GB) in 8.4 h, 14.3 MB/s, ~6 min per archive,
  no SHA1 failures, 397 images kept. The remaining ~2.4 TB take about 46 h.
- **P3 size-matched test variant (A9).** Whole patients drawn with seed 20261117:
  - 1,515 BIMCV COVID-19 images from 1,000 patients (P3: 1,145);
  - 1,295 NIH pneumonia images from 923 patients (P3: 941);
  - 406 normals, unchanged.
  - It contains 7 of the 10 BIMCV and all 10 NIH figshare test images; the full variant contains all 20.
  - P3's own counts sum to 3,216 but it states n = 3,126.
  - Whether either variant carries verdicts depends on `COVIDIsNet.pt` scoring within 0.03 of 0.773 on it. This check runs
    once BIMCV is complete (~46 h).
- **P3 test normals (A10):** 370 adults (Montgomery 67, Shenzhen 303), matching P3's Table 4 row sums. P3-compatible age
  reading: first number in the clinical reading = years. It keeps CHNCXR_0018 ("male 42") and counts CHNCXR_0202
  ("female 64days", an infant) as an adult, as P3's 370 implies. Variants: primary 2,501 + 1,319 + 370; size-matched
  1,515 + 1,295 + 370 = 3,180.
- **P3 marks, corrected before any P3 run (smoke tests only).** P3's main-text heatmap figure (PDF page 9, top row, the two
  marked X-rays at ~210 px) shows:
  - the mark box flush with the upper-left corner, starting at pixel 0;
  - box size 0.10–0.11 of the side, as the text says;
  - the **triangle as a right triangle filling the corner** (vertices top-left, top-right, bottom-left);
  - the circle inscribed in the same box.

  Our first guess (4-px margin, upright triangle) is replaced by this.
- **Mark survival under P3's augmentation** (rotation ±40° about the centre, translation ±28 px; 2,000 draws):
  - corner mark: 23% of the mark kept on average; fully kept in 8.7% of draws; fully lost in 58.2%;
  - mark well inside the lungs: always kept.

  P3 says augmentation "can sometimes remove the shapes"; in its design the shortcut is visible in under half of the training
  views. Reported next to every P3 corner result.
- **P3 caches:** training/validation cache built (13,932 images; mean lung fraction 0.322; visual check of masks and orientation
  done). TorchXRayVision PSPNet weights match the SHA-256 in `docs/ENVIRONMENT.md` (`019b167e…cadc044fc`).
- **P3 trainer smoke tests:** DenseNet and ISNet (corner / inside, all / 90-10) run end to end. The 90/10 rule marks 89.5–90.3%
  per class. Inside placement reaches ≥ 0.95 overlap for 99.94% of images; the rest are reported per image.

## Engineering notes (2026-11-17)

- **P2 Vanilla crash and restart.** The first P2 Vanilla run (seed 0, started 2026-11-16 19:22) **crashed at 22:49 with
  CUDA out-of-memory at epoch 34**. Our ad-hoc P3 cache build and smoke tests took GPU memory beside three training jobs.
  - The status given at 03:40 ("P2 at epoch 34, ~15 min/epoch") was wrong: it read the last line of the crashed run's log.
  - The run (no checkpoint yet; their trainer stores every 100 epochs) was restarted from scratch on 2026-11-17 03:44 with the
    same config and seed. The crashed run is kept in `runs/vanilla_resnet50d_seed0_crashed_oom_2026-11-16/` and not used.
  - From now on no GPU job is started outside the memory-aware launchers.
- **P2 speed profile (not loader-bound; no caching needed).**
  - P2's DataLoader alone (their dataset, transforms and augmentation; 8 workers; batch 12; no model) delivers **343 img/s**.
  - Training ran at 5.7–6.4 min/epoch (~55–60 img/s) beside two P1 jobs, with GPU utilisation 100% and CPU load ~7 of 96 cores.
  - So P2 is GPU-bound because of sharing, not loader-bound. The author's suggested preprocessing cache was **not**
    implemented, since it would not change the speed. After the restart, with less crowding, P2 trains at ~134 img/s
    (~2.5 min per training epoch).
- **P2 priority.** After the P1 gate, while P2 trains, the P1 queue starts a run only when P2 has no more than one other job
  beside it (`queue_viviano.sh`). P3 runs follow the same rule and wait for that slot.

## Queue, GPU-hours and run order (2026-11-17 09:10)

**Pairing measurements** (scheduling rule 2):
- ISNet beside P2: ISNet 37.0 min/epoch vs 14.7 alone (0.38), P2 72.8 vs 137 img/s alone (0.53); together 0.91 of one GPU,
  so **ISNet runs alone**.
- ISNet's last full epoch alone: 14.7 min, so a 48-epoch run takes ~11.8 h.
- P2 alone: 137 img/s, ~2.8 min/epoch with its evaluation every 5 epochs.
- P1 runs: ~5 h each beside one other job (seed-1234 runs).

**Long jobs one at a time** (author, 2026-11-17). ISNet results cannot be scored before BIMCV completes, so P2 goes first:
1. P2 Vanilla to epoch 300 (~10 h; checkpoint at epoch 100; nothing risky beside it before then);
2. P2 RR-ClArC (activations + λ × layer grid, ~4 h, estimate);
3. P3 ISNet chain alone: corner (resumes after epoch 2), inside (P3 design), inside 90/10 (~35 h).

ISNet was stopped cleanly after its epoch-2 checkpoint; P2 was paused (SIGSTOP) for one solo ISNet epoch and resumed.

**Short jobs beside the long ones only if a measurement shows a gain:** P1 kept runs (4), P1 location test (2), P4 timing,
then P4 runs (7), and our arms' feature passes. Otherwise they run after the long chain.

| item | GPU-hours left |
|---|---|
| P2 Vanilla + RR-ClArC | ~14 |
| P3 ISNet × 3 | ~35 |
| P1 kept + location test | ~30 |
| P4 (6 runs + location test; not yet timed) | ~27 |
| our arms (feature passes) | ~5 |
| **total** | **~111** |

- Calendar: ~2.5 days if every short job pairs at a measured gain; ~4.5 days if none does.
- GPU wall-clock used so far: ~22 h, including 8 superseded P1 runs (A7) and the crashed P2 run.
- BIMCV: 209 of 531 archives; ~30 h left; it gates only P3 scoring.
- Blockers: none.

## Updated estimate and run order after A14 (2026-11-17 ~09:45)

| item | runs | basis | GPU-hours left |
|---|---|---|---|
| P2 Vanilla (epoch ~85 → 300) | 1 | measured 2.8 min/epoch alone | ~10 |
| P2 RR-ClArC (activations + λ × layer grid) | 6 | estimate | ~4 |
| P3 ISNet chain (corner resumes after epoch 2; inside; inside 90/10) | 3 | measured 14.7 min/epoch alone | ~35 |
| P1 location test (Standard, tube inside / outside; Masked scored from it) | 2 | ~5 h each beside one job (measured) | ~10 |
| P4 Binary-ERM + Segmentation-ERM, splits 1–3 | 6 | estimate (paper ~2 min/epoch, 150 epochs); timing pending | ~20 |
| our arms (feature passes: P2, P3, P4, P1 location) | — | estimate | ~5 |
| **total** | | | **~84** |

- Phase 5 (thyroid, Trap A + B, 3 folds) is not included; it is costed by its timing test before its pre-registration.
- Calendar: long chain (P2 → RR-ClArC → ISNet × 3) ≈ 49 h. The short jobs (~35 h) fit beside it only if a measured pairing
  helps (ISNet + P2 did not: 0.91). Expect **~2.2 days if they pair, ~3.5 days if they run after**.
- Next measurements: P4 Segmentation-ERM memory/speed at batch 16 and P1 ResNet-18 beside P2. Both start after P2's
  epoch-100 checkpoint.

## Pairing measurement: P1 location run + P2 Vanilla (2026-11-17 13:30)
- P1 location run (cardiomegaly Standard, tube inside): 108 s/epoch alone (P2 paused 2 epochs with SIGSTOP; P2 has its
  epoch-100 checkpoint), 190 s/epoch paired, so 0.57 of its solo speed.
- P2: 153 img/s alone, 133 paired, so 0.87.
- **Total 1.44 > 1: pairing kept.** Short jobs run one at a time beside the long job (`scripts/replication/short_queue.sh`:
  location inside → outside → P4 Binary-ERM splits 1–3 → P4 Segmentation-ERM splits 1–3 with `--grad_ckpt`), never beside
  ISNet.
- P1 location test: placements feasible inside 99.92% (9 images keep their best placement) and outside 100%; ROI (lungs ∪
  heart) covers 36.2% of the image; tube rendering checked visually.

### 2026-11-17 20:10 UTC — P2 chain failure and fix
The first pass of `p2_chain.sh` (19:47–19:54) failed at the class-0 activation step: a Python 3.11 port bug in their
`isic_attacked.py` (`random.seed` with a numpy int; see `third_party/pahde/SOURCE.md`). All 6 RR-ClArC fine-tunes then
failed for want of the CAV file, and the chain, which did not check its steps, started the P3 ISNet chain. That chain was
still waiting for an empty GPU (no ISNet job had started) and was stopped by PID. Fixes: the seed call (`int(idx)`, the
same seed), and `p2_chain.sh` now stops at the first failed step and hands over to P3 only when all 6 RR-ClArC score
files exist. No P2 result existed or was seen; nothing was re-tuned.

### 2026-11-17 22:20 UTC — P1 location test: outside Standard kept a pre-shortcut epoch (checked; not a bug)
Outside, Standard (their model, their selection) shows no shortcut: correlated / reversed / clean AUROC 0.901 / 0.894 /
0.898 (inside 0.943 / 0.514 / 0.866). Checks:
- the tube reaches the network at the outside position: rendered through the same dataset class at 128 px, 200 training
  images with a tube each have 89–120 changed pixels (mean 109, as inside), 0.01% of them inside lungs ∪ heart; presence
  90.5% of positives / 9.6% of negatives in training, as registered; visually checked (scratch only, nothing saved);
- **cause: their model selection** (`training.py`: `burn_in=5`, then the lowest validation loss; P1's validation is
  reversed, 10/90). Inside, the shortcut appears at epoch 5 (reversed-val loss 0.73 → 2.26) and the kept epoch is 11 (val
  loss 1.005, val AUROC ≈ 0.50). Outside, the shortcut appears only at epoch 11 (0.45 → 0.65; late epochs val AUROC
  ≈ 0.70–0.75, weaker than inside), and the kept epoch is 8 (val loss 0.414, val AUROC 0.902), i.e. before it was learned.
Consequence for the registered analysis (unchanged, reported as is): at the outside position the Standard model has no
shortcut for Masked to remove, so Masked's crossover (+0.032 [+0.023, +0.041]) comes from Masked costing less outside
(−0.013) than inside (−0.045), not from Masked removing an outside shortcut. Their `last_model_1234.pth.tar` (epoch 100)
is on disk for both positions; any use of it would be a post-hoc sensitivity analysis and needs the author's decision.

### 2026-11-18 03:30 UTC — P4 Binary-ERM, 3 splits (gate, Binary half)
Robust AUROC on the 772-image trap test (their split k, best epoch by validation AUROC, test scored once per run):
split 1 0.3548, split 2 0.3614, split 3 0.3735 → **mean 0.363 ± 0.010 (SD over splits)**. Paper 0.359 ± 0.016
(range ±2 SD 0.327–0.391); notebook 0.310. Gate (A13 addendum, paper OR notebook; it widens the accepted range to
0.28–0.39 for Binary-ERM, stated openly): **passed on the paper's own value** (within ±2 SD), so the widening was not needed
for Binary-ERM. Segmentation-ERM (×3, `--grad_ckpt`) runs next; the P4 gate is decided when both are in.
Per-image scores: `results/replication/saab/scores/erm_split{1,2,3}.csv.gz`.

### 2026-11-18 06:10 UTC — pairing measured: P2 RR-ClArC fine-tune + P4 Seg-ERM (`--grad_ckpt`)
Measured with short SIGSTOP/SIGCONT pauses (no job killed or restarted): P2 fine-tune 11 img/s paired vs 48 alone (0.23,
over 90 s with Seg-ERM paused); Seg-ERM 12.1 img/s paired vs 13.6 alone (0.89, one full epoch, epoch 58, with P2 paused
166 s). Together **1.12** of one GPU, so the pairing is kept (rule: pair only if total throughput rises), although it
slows P2's remaining fine-tunes about 4× (≈ 23 min/epoch instead of ≈ 6) and so delays the ISNet chain by several hours.
Seg-ERM epochs 57 and 58 include the pauses (their logged img/s are not training-speed measurements).

### 2026-11-18 06:25 UTC — scheduling note (author's decision): critical path first
Rule 4 (longest chains first) overrides rule 2 (pair when total throughput rises) here: P2 feeds the critical path
(P2 → ISNet chain, ~35 h); P4 does not. Hence:
1. After the 4th RR-ClArC model is scored, P4 Seg-ERM split 1 is stopped cleanly right after its next epoch checkpoint
   (`checkpoint.pt` is written atomically each epoch; it resumes from it with model, optimiser, best score and log).
2. The last two RR-ClArC fine-tunes run alone; `p2_chain.sh` then starts the ISNet chain (alone).
3. **ISNet + Seg-ERM cannot be paired on this GPU** (RTX 5060 Ti, 16.3 GB): ISNet peaks at 10.4 GiB allocated (≈ 11.5 GB
   per process), Seg-ERM with activation checkpointing at 6.3 GiB (≈ 7.3 GB per process), ≈ 18.6 GB together; the
   launcher would not start Seg-ERM beside ISNet (it needs 8 GB free) and forcing it would risk an out-of-memory crash
   of ISNet. So the pairing was not measured, and P4 runs after the ISNet chain: `short_queue.sh` now also waits while
   `$WTSS_DATA/replication/P4_HOLD` exists, and `p3_chain.sh` removes that file when the chain is done. (There are no
   gaps inside the ISNet chain: its three runs follow each other.)
4. The A17 post-hoc scoring (~2 min of GPU per position) may take a slot at the boundary after the 4th fine-tune.
- 06:19:33: P4 Seg-ERM split 1 stopped (SIGTERM, by PID) right after its epoch-66 checkpoint (`checkpoint.pt`, epoch 66
  verified). Two orphaned DataLoader workers (ppid 1) still held its 7.3 GB; killed by verified PID at 06:20, GPU freed.
  It resumes from epoch 67 when `P4_HOLD` is removed.

### 2026-11-18 08:40 UTC — P2 gate: RR-ClArC NOT reproduced (0.338 vs 0.76); candidate cause found; waiting for the author
Selection on validation (section 3) chose λ = 10, last_conv (val mean 0.597). Test, scored once (1 seed, image-only CIs):
RR-ClArC clean 0.840 [0.826, 0.854], biased **0.338 [0.320, 0.356]**; Vanilla clean 0.839, biased 0.311. Paper: Vanilla
0.87 / 0.28, RR-ClArC 0.86 / **0.76**. |0.338 − 0.76| > 0.03 → **not reproduced**. Our Vanilla matches the paper within
~0.03 on both tests, so the biased test is built as theirs; the gap is in the correction step.
Diagnosis (no rerun, nothing tuned):
- Their `utils/training_utils.py:13-16` (`get_optimizer`) loads `optimizer_state_dict` from `ckpt_path`. Our Vanilla
  checkpoint (their trainer, `model_training/train_model.py:21,96`) stores Adam with `lr = 1e-05` (their config: lr 0.001,
  milestones 150, 250, 300 epochs). The loaded state overrides the configured `lr: 0.001` of the RR-ClArC configs, and
  their MultiStepLR keeps it, so **all 6 fine-tunes ran at lr 1e-5** (100× below the configured rate).
- Consistent with this: the logged RR-ClArC term (`train_auxloss`) does not fall over 10 epochs and is identical step for
  step between λ = 10 and λ = 50 (Adam is invariant to the loss scale, so λ barely changes the path at this rate).
- Their configs point to `…/resnet50d_p_artifact0.2_p_backdoor0_adam_lr0.001_last.pth`, a renamed copy; their code only
  writes checkpoints with optimizer state, so whether the paper's file carried it (and the paper's fine-tune ran at 1e-5
  or 1e-3) cannot be determined from the repository.
- Their `Freeze` callback is not used by `start_model_correction.py` (all parameters train).
Results: `results/replication/pahde/p2_reproduction.{md,csv}`, `rrclarc_selection.csv`; per-image scores in
`results/replication/pahde/scores/`.

### 2026-11-18 09:10 UTC — P2: what the paper states about the RR-ClArC fine-tune (arXiv 2501.13818, PDF pages)
Read from the rendered pages (pages 24, 31, 43 checked as images):
- **Learning rate**, p. 31 ("Hyperparameters"): "As learning rate, we use the learning rate used during training (see
  Tab. 4) divided by 10." Tab. 4 (p. 24), ISIC2019 (attacked), ResNet50: "Adam, LR 0.001, 300 epochs" → **1e-4**. Neither
  the configured 1e-3 nor the 1e-5 that their code path loads from the Vanilla checkpoint.
- **λ grid**, p. 31: "we test values λ ∈ {1¹, 5·1¹, 1², 5·1², ..., 5·1⁹} for RRR, and λ ∈ {1¹, 5·1¹, 1², 5·1², ..., 5·1¹²}
  for RR-ClArC" (the base is printed as 1; Tab. 12 shows it is 10). **Chosen λ**, Tab. 12 (p. 43), RR-ClArC, ISIC2019,
  ResNet: **5·10⁹** (VGG 10¹⁰, ViT 10¹).
- Their released generator (`config_generator_mitigation_isic_attacked.py:92-105`, single commit 4990898) lists the
  same grid but with every value from 10² to 5·10¹² commented out, leaving λ ∈ {10, 50}. Our pre-registration (section 3)
  took {10, 50} from that generator and wrongly says the paper leaves λ open; Tab. 12 states it.
- Layer and number of fine-tune epochs are not stated in the paper (their configs: last 3 blocks, 10 epochs).

### 2026-11-18 10:30 UTC — utilisation check and pairing in the P2 window
At 10:11 the GPU was 17–56% busy (65 of 180 W, 2.4 of 16.3 GB) with only the A18 fine-tune running; its main process is
bound to one CPU core (~90%; their trainer runs with `detect_anomaly=True`, kept as theirs). CPU load ~12 of 96 cores.
So the A18-independent GPU steps of `p2_window.sh` (lesion U-Net, ROIs, DINOv2 features for arms b, c) were started beside
it as `scripts/replication/p2_sidecar.sh` (same commands and outputs; the window skips finished steps). Measured pairing:
A18 fine-tune 2.60 it/s paired vs 4.1 alone (0.63); U-Net ~121 s/epoch paired vs ~85 s alone (~0.70; epoch timings
straddle the end of fine-tune 1, so approximate) → **~1.33 of one GPU, kept**. The pairing of the remaining sidecar
steps (U-Net prediction, DINOv2 extraction) with fine-tunes 2–3 is measured the same way when they overlap.

### 2026-11-18 10:56 UTC — test numbers of one unselected A18 config seen (recorded for transparency)
`p2_score.py` prints the accuracy of every set it scores, so the window log shows the test accuracy of A18 config
λ = 5·10⁹ / identity_1 (clean 0.842, biased 0.360), which I read while checking progress, before the layer selection.
Selection uses validation only (section 3, A18) and is not affected; nothing will be changed in response. From now on the
progress checks read only the "scored" lines of the window log.

### 2026-11-18 13:30 UTC — A18 gate: RR-ClArC NOT reproduced at the paper's settings (0.524 vs 0.76); waiting for the author
- Window: the 3 A18 fine-tunes and their scores finished; the sidecar finished U-Net (held-out Dice 0.942), lesion ROIs
  (HAM 10,015 / ISIC 2018 772 / U-Net 14,544; 7 empty U-Net masks) and the DINOv2 features for arms b, c. The window's A18
  selection step then crashed (12:55; my parser read λ as an integer and the layer by "lr0.001_"), so by design the
  window restarted ISNet and exited before extracting arm a' features. Fixed (a79572f; section-3 selection unchanged)
  and run on CPU at 13:25.
- Selection on validation (mean of clean and biased accuracy): identity_2 (val 0.843 / 0.530, mean 0.687) over last_conv
  (0.609) and identity_1 (0.600). **Test, scored once: clean 0.839 [0.825, 0.853], biased 0.524 [0.505, 0.543]**; Vanilla
  0.839 / 0.311. |0.524 − 0.76| > 0.03 → **not reproduced** (gate failed). The paper's settings raise biased accuracy from
  0.338 (section-3 runs) to 0.524, but not to 0.76.
- Arm a' (A19) still needs its features from the selected network (GPU, ~5 min); ISNet is running again. Arms b, c heads
  run on CPU now.

### 2026-11-18 13:40 UTC — P2 decision (author)
RR-ClArC on P2's own test is **"not reproduced" (0.524 vs 0.76)** as the final registered result. The head-to-heads vs
RR-ClArC carry no verdict; arms a, a', b, c are reported descriptively against our A18 RR-ClArC and the paper's numbers.
No further post-hoc tuning of RR-ClArC. A short email to Pahde et al. asking for the exact Table 1 configuration is
drafted in `docs/EMAIL_DRAFT_PAHDE_RRCLARC.md` (the author sends it). The decisive RR-ClArC comparison moves to phase 5
(thyroid Trap A), with RR-ClArC ported at the paper's stated settings (`docs/REPLICATION_PLAN.md`, phase 5 sketch).

### 2026-11-18 14:25 UTC — arm d (EfficientNet-B0) + ISNet pairing measured on P2: kept
Arm d P2 training 467 img/s paired vs 690 alone (0.68; one full epoch during a 56-s SIGSTOP of ISNet); ISNet epoch 11
22.9 min-equivalent paired vs 14.7 alone (0.64; from its logged duration minus the pause and its solo part) → **1.32 of
one GPU, kept** (A20: an arm-d job may pair with ISNet only if total throughput rises). Memory: beside ISNet 3.7 GB are
free; EfficientNet-B0 peaks at 2.8 GiB at 224 px (fits), not at 512 px (P4 runs without ISNet). Two failed starts
before any training step, fixed in the trainer: a launcher request above the free memory (4,000 MiB), and DataLoader
workers exceeding the container's task limit (2,816) — now one thread per worker.

### 2026-11-18 14:45 UTC — BIMCV complete; P3 test rebuilt
BIMCV fetch finished (531 / 531 archives). `p3_test_select.py` (A9/A10): full variant normal 370 (Montgomery 67 +
Shenzhen 303), pneumonia 1,319 (NIH, 938 patients), COVID-19 2,501 (BIMCV, 1,631 patients); **size-matched variant 370 /
1,295 / 1,515 = 3,180, P3's Table 4 row sums**; every image on disk. Next: `p3_prepare.py --which test` (queued on the
GPU launcher), then the ISNet runs' `--score_only` and the COVIDIsNet.pt equivalence check (A9).

### 2026-11-18 14:35 UTC — phase-5 cohorts re-downloaded and verified
`scripts/data/fetch_phase5.sh` (TN3K Google Drive archive, MMOTU and SEE-AI via the Kaggle API with the token in the
environment only, figshare 27645021 contamination masks) and `scripts/data/verify_phase5.py`: **all four match
`docs/DATA_MANIFEST.md` exactly** (files / bytes / listing SHA-256): TN3K 6,995 / 375,692,454; MMOTU 6,565 / 169,894,610;
SEE-AI 36,968 / 1,262,400,157; capsule contamination masks 29,403 / 532,520,931. (Tools added: bsdtar, kaggle, six.)

### 2026-11-18 15:00 UTC — arm d adapters for P1, P1 location, P3, P4 (before any of their results)
- P1 (cardiomegaly 128 px, RSNA 224 px): their configs apply no training augmentation, so arm d has none either; images
  and ROIs as arm b. P1 location test: arm b's rendering at 128 px; arm d's predictions also enter the A16 table.
- P3 90/10 inside: exactly the data of the ISNet run `isnet_inside_9010_seed0` (mark presence keyed by data seed 0),
  P3's joint augmentation, PSPNet lungs as ROI; validation macro-F1.
- P4 split 1: 512-px cache, their albumentations pipeline (albumentations 1.3.1 added to our environment without
  dependencies, the version of P4's environment), lesion ROI. **Artifact-flag rule (A13 item 8) read as "at least 3
  training images in each of the four artifact × diagnosis groups"**, the reading under which its own example (patches,
  2 melanomas) is skipped: flags used dark_corner, hair, gel_border, gel_bubble, ruler, ink, vasc; patches skipped. The
  flags' mask_cmc directions are combined into one subspace (mode chosen on validation).
- EfficientNet-B0 at 512 px does not fit beside ISNet; P4's arm-d run needs a GPU without ISNet (memory at batch 64 to be
  checked first).

### 2026-11-18 — P2: pretrained-weights discrepancy between paper and code (recorded; P2 decision unchanged)
P2's paper (arXiv 2501.13818, p. 22, A.6) names the ResNet50 starting weights `resnet50d.a3_in1k` (timm); their released
code (`models/resnet_timm.py:28`) loads `resnet50d.a1_in1k`, and our P2 runs (their code) therefore started from a1. Under
the paper-wins rule a3 is the stated recipe. P2 is closed as "not reproduced" by the author's decision (2026-11-18, no
further tuning); this is reported next to the P2 result. Phase 5 uses a3 (author).

### 2026-11-18 — P3 pre-trained weights folder re-listed (author asked)
The P3 README's Google Drive folder (1hIJp4c11R65db2K7EN4rYQ7dcr7ce72t), listed without download: `COVIDIsNet.pt` (COVID-19
ISNet without synthetic bias, Table 2), `FacesSyntheticBiasISNet.ckpt`, `ISNetDogsSyntheticBias.ckpt` (synthetic-bias
models for faces and dogs, Table 1), `ISNetChexpert.ckpt`, `TBISNet.ckpt`. **No synthetic-bias (corner-mark) COVID-19
model is released**, so the corner benchmark ISNet must be trained (chain unchanged; A22 not recorded).

### 2026-11-18 15:20 UTC — A9 equivalence check: COVIDIsNet.pt fails even on P3's training sources → suspected input/loading mismatch; waiting for the author
Run on CPU (inference only; on the GPU beside ISNet their LRP DenseNet ran out of memory). On our rebuilt test:
primary macro-F1 **0.223**, size-matched **0.260** (target 0.773); nearly every image predicted "pneumonia" (COVID-19:
2,352 of 2,501; normal: 334 of 370). Diagnostic (not part of A9): on a 900-image sample of our P3 cohort **validation**
split — BrixIA COVID-19 and CheXpert, P3's own training sources, through our identical preprocessing — the model also
predicts output 1 for 91% of images and never output 2 (macro-F1 0.281). So the failure is not about our test: the
released model does not receive the input it was trained on (preprocessing mismatch) or does not load correctly (a
pickled module under torch 2.x). The check is **not interpretable as an equivalence test**; no P3 verdict depends on it
until this is resolved. `results/replication/isnet/covidisnet_check.md` is kept as run.

### 2026-11-18 15:55 UTC — A9 diagnosis (author's order and limits): no bug found → P3 stays descriptive as registered
Each step checked on P3's training-source validation images (BrixIA, CheXpert; our cohort cache), never on the test:
1. **Strict load:** all 1,144 keys of `COVIDIsNet.pt` match a fresh `IsDense121Lightning(classes=3)` built from their
   code (`backup/ISNetLightning.py:13-92`); no missing or unexpected keys; logits identical (max difference 0.0).
   BatchNorm statistics are trained (num_batches_tracked 22,890; running_var mean 32.9).
2. **eval():** no module in training mode after `eval()`.
3. **Output:** with `heat=False`, `forward` returns `self.DenseNet(x)`, the classifier logits (`ISNetLightning.py:82-92`).
   Their LRP DenseNet and the standard torchvision DenseNet-121 with the same weights give the same predictions (max
   logit difference 0.33, LRP ε; same in-distribution macro-F1 0.272).
4. **Input (Supp. Note 6.1, p. 32 of the Supplementary Information, MOESM1):** grayscale → histogram equalization → [0, 1]
   → 224 × 224 → 3 channels (test images padded to square). Our cache follows this order. The first BatchNorm confirms
   the scale: conv0 output on our [0, 1] inputs has mean 2.06 / variance 7.2 vs norm0's running 1.72 / 8.9 (variance
   ratio 0.82; with 0–255 inputs it would be 53,000×). Cached images inspected: correct polarity, equalized.
5. **Class order (Supp. Note 6.2, p. 33):** normal 0, pneumonia 1, COVID-19 2 — ours.
Still, on P3's own training sources the model predicts "pneumonia" for 95% of BrixIA COVID-19 and 73% of CheXpert
normal images and never outputs COVID-19 (in-distribution macro-F1 0.27–0.28). No difference between their stated
pipeline and ours was found in text or code, so **nothing is changed and the test is not re-scored**; the A9 check stays
as run (0.223 / 0.260), **P3 is descriptive only as registered**, and the ISNet chain is unaffected. Untestable from text
or code: which CheXpert release P3 used (we use CheXpert-v1.0-small) and their exact equalization implementation.

### 2026-11-19 07:15 UTC — machine 1: P4, P5 and phase 5 moved to machine 2 (author)
A second machine (RTX 5080, "machine 2") takes over **P4** (Seg-ERM splits 1–3, arm a, the P4 gate), **P5** (Reveal to
Revise) and **phase 5**. Removed from machine 1 at 07:14 UTC, nothing deleted:
- `short_queue.sh` stopped by PID (it was only waiting on `P4_HOLD` to run Seg-ERM splits 1–3; no GPU job). The P4
  `seg_split1` checkpoint (`$WTSS_DATA/replication/saab/runs/seg_split1/`, epoch-66 checkpoint) is kept on disk.
- P5: the checkpoint / paper download loop and its waiter stopped by PID; the partial download
  (`$WTSS_DATA/replication/r2r/checkpoints/checkpoint_resnet18_isic_last.pth.part`) and the fetched code
  (`/workspace/third_party/r2r/`, `third_party/r2r/SOURCE.md`) are kept.
- Phase 5: nothing was queued on machine 1 (the trainer had not been written); the rebuilt thyroid cohort stays on disk.
Machine 1 now runs only: the P3 ISNet chain (`isnet_inside_all_seed0`, then `isnet_inside_9010_seed0`), their scoring on
the rebuilt test, the A24 in-distribution evaluations, the P3 arm feature passes (arms b, c, d on the A24 validation
images), and the exploratory track (CPU, validation only). Machine 1 never edits files machine 2 writes (P4, P5, phase-5
results and their STATUS entries).

### 2026-11-18 12:35 UTC (machine 2's clock) — machine 2: set up, data verified, P4 Seg-ERM split 1 running, P5 step 1 failed its check
- **Hardware (measured):** RTX 5070 Ti 16 GB (not an RTX 5080 as noted above), driver 595.91.07 / CUDA 13.2; Xeon E5-2680 v4,
  28 logical cores; 62 GB RAM; 288 GB container disk, **no persistent volume** (A27). Environments: `docs/ENVIRONMENT_MACHINE2.md`.
- **Data, checked against `docs/DATA_MANIFEST.md`:** ISIC 2019 (P5) 25,335 files / 9,803,958,095 B, listing hash **match**;
  TN3K (phase 5) 6,995 / 375,692,454 B, **match**; ISIC 2018 Task 1/2 (P4) 2,594 images + 2,594 Task 1 masks, complete; its
  listing differs from the thesis machine's by 2 files / 4 B outside the images (ISIC's `LICENSE.txt` / `ATTRIBUTION.txt`;
  ISIC moved the files on 2026-10-28). Bissoto trap sets at f93c865; SE-ResNeXt50 weights from the Internet Archive, SHA-256
  `a260b3a40f82dfe3…` (as `third_party/saab/SOURCE.md`). Thyroid cohort rebuilt on CPU (`scan_thyroid.py`, `prepare_thyroid.py`):
  3,493 images, Trap A 1,150, Trap B 326, marker-free 1,801, largest pHash group 3 — **identical to
  `docs/PREREGISTRATION_THYROID_TRAPS.md`**.
- **P4 Segmentation-ERM split 1** (A27: from scratch, machine 1's checkpoint unused), same command (`--grad_ckpt`, batch 16,
  fp32) via `scripts/replication/m2_p4_queue.sh` (splits 1 → 2 → 3): **1.15 min/epoch (26 img/s, peak 6.3 GiB)**, ≈ 2.9 h per
  split; all three expected ≈ 21:15 UTC. Arm a: `scripts/replication/p4_arm_a.py` (features per split on GPU, heads on CPU).
- **P5 step 1: Vanilla NOT within 0.03 of Table 1** (`results/replication/r2r/step1.md`): attacked ResNet-18, test accuracy
  original 0.838 / poisoned 0.552 (paper 0.895 / 0.356), macro-F1 0.772 / 0.622 (paper 0.821 / 0.390). The clean gap (−0.06)
  involves no artifact (the original checkpoint also gives 0.831); the poisoned gap has a candidate cause (Pillow ≥ 10.1
  renders the text artifact with a different default font). Per the author's rule, **P5 is dropped**; no amendment A26, no
  correction run. Waiting for the author on whether the font point warrants anything (any re-score of the test would be a
  second scoring and needs a dated amendment first).

### 2026-11-18 12:40 UTC (machine 2's clock) — machine 2: P4 Seg-ERM splits 1 + 2 paired, measured, kept
Split 1 alone: 26.3 img/s (1.14 min/epoch, epochs 0–1). Splits 1 and 2 together: 16.2 + 16.2 = **32.3 img/s (1.23 of one
GPU)**, 1.87 min/epoch each, 14.9 of 16.3 GB used → kept (rule 2). Three at once would need ≈ 22 GB, so no. Queue replaced by
`scripts/replication/m2_p4_pair.sh` (at most two splits at once; same command; split 3 starts when one finishes; the first
queue was stopped by PID before it launched anything else). New ETA: splits 1 and 2 ≈ 17:10 UTC, split 3 ≈ 20:00 UTC (it
pairs with the phase-5 timing test if that is measured faster). Split 1's epoch 2 straddles the pairing start.

### 2026-11-18 ~14:00 UTC (machine 2's clock) — machine 2: P6 (RR-ClArC, AAAI 2024) step 1 reproduced; SpurBreast dropped; A28
- **New candidates searched (author's request).** Kept: **P6 = Dreyer, Pahde et al., AAAI 2024** (RR-ClArC), as a secondary
  paper (A28). Dropped: **SpurBreast (MICCAI 2025)** — data on Zenodo 17128791 (field-strength split checked: 9,562 / 3,576 /
  6,788 images, as its Table 1), but no method to beat (dataset paper; baselines ResNet-50 / ViT-B test accuracy ≈ 0.52–0.55)
  and its strong spurious splits are **perfectly confounded** (every training positive 1.5 T, every negative 3 T; or every
  positive facing up), so no within-class artifact difference exists for mask_cmc.
- **P6 step 1 (released Vanilla checkpoints, their code, test scored once): reproduced for all three architectures**
  (`results/replication/rrclarc/step1.md`): clean / biased accuracy VGG-16 0.764 / 0.349 (paper 0.762 / 0.349), ResNet-18
  0.816 / 0.557 (0.818 / 0.568), EfficientNet-B0 0.839 / 0.625 (0.842 / 0.629). Run on CPU (the first GPU attempt ran out
  of memory beside the two P4 runs before any image was scored; the P4 runs were unaffected). Checkpoint SHA-256: VGG-16
  `428fcfc4…`, ResNet-18 `5d14a30d…`, EfficientNet-B0 `88b1259d…` (Drive files dated 2023-08-17). Their LSB trigger is
  deterministic bit-writing (no font, no RNG), so the P5 font issue cannot arise.
- **Discrepancies recorded (A28):** the paper's chosen RR-ClArC λ (Table A.3: 5·10⁸ / 10¹¹ / 5·10⁹) lie outside their
  released generator's grid (≤ 10⁷); the paper's p-bias 8.0 / 12.5 / 8.0 % vs 10 % in their correction configs. The paper
  wins. RR-ClArC fine-tunes (3, 10 epochs each) run when GPU memory frees (after P4 splits 1–2).

### 2026-11-18 (machine 2's clock) — machine 2: renumbering (A29): RR-ClArC → P7; EPVT/PLDG → P6, step 1 started
No P7 fine-tune had started (nothing stopped). Priority P4 > P6 > P7. P6 step 1: read EPVT (arXiv 2304.01508) and PLDG (arXiv
2401.03002), get their processed data, reproduce ERM and EPVT once; then report and wait. Its registration will be A30.
MaskMedPaint (ML4H 2024 findings; code `github.com/QixuanJin99/generative_validation` @ 40048f7) was examined and not taken:
its ISIC dataset is read from a prebuilt folder on the authors' server that the repository does not build, and its training
set is perfectly confounded (only melanoma-with-marking and benign-without-marking groups), so mask_cmc cannot be estimated.

### 2026-11-18 ~14:00 UTC (machine 2's clock) — machine 2: P6 (EPVT / PLDG) step 1, items 1–2 done; runs queued after P4
- Papers read; code pinned; data rebuilt (their Drive archive is 404): `third_party/epvt/SOURCE.md`. Bias-0 split-1 counts equal
  the paper's domain counts, validation and test sizes; training 12,343 vs 12,360.
- **Their trap-set results exist only as figures** (EPVT Fig. 3a, PLDG Fig. 8); digitised (±≈0.5): bias 1 ERM 65.3, EPVT 74.6;
  bias 0 ERM 92.3, EPVT 91.0. No spread shown although the text says five seeds.
- **PLDG skin path:** runnable from `train_latent_epvt.py` despite the commented README command (details in SOURCE.md).
- Scheduling (author: P4 first; pair only if measured not to slow P4): P4 splits 1 + 2 already share the GPU (14.9 of 16.3 GB);
  a ViT-B/16 at batch 130 in fp32 needs ≈ 12 GB, which does not fit beside one Seg-ERM run (≈ 8.5 GB free), so the 12 P6 runs
  (ERM and EPVT × 6 bias levels, split 1, README hparams) start after P4 split 3 (`scripts/replication/m2_p6_queue.sh`; ETA
  start ≈ 20:15 UTC). Nothing of ours is scored before the report and A30.
- **Correction (author, before any P6 result):** the paper's protocol, not the README's 1501 steps: 100 epochs, early stopping
  patience 22 on `val_bias` ROC-AUC, paper augmentation (= their code's). "Epoch" in their code's unit (one pass over the
  smallest domain's in-split): bias 1 → 1,001 steps. Recorded in `third_party/epvt/SOURCE.md`. Queue restarted for bias 1 only
  (ERM, then EPVT), timed; it still waits for P4 split 3.

### 2026-11-18 ~16:30 UTC (machine 2's clock) — machine 2: phase-5 trainer and toy-batch unit tests (CPU, nice 19)
- `src/wtss/experiments/phase5.py` (library), `scripts/phase5/p5_run.py` (one network per trap / fold), `tests/test_phase5.py`
  (**10 passed**, CPU, untrained weights, 64-px toy batches): joint image/ROI augmentation stays aligned; dataset draws are
  reproducible; early stopping (patience 5, ties and NaN never count, max-epoch stop); RR-ClArC aux loss equals the manual
  `l2_mean` formula and vanishes for gradients orthogonal to the CAV; their `compute_cav('signal')` (imported unchanged)
  points at the artifact channel; ResNet-50d with Pahde's hook names, one RR-ClArC step (all parameters receive gradients);
  ISNet training step and classification-only prediction on a toy batch; EfficientNet-B0 1,280-d features; `fit` in fp32
  with resume; arm-d head (mask_cmc + A21 grid, chosen on val_clean AUROC) removes the marker direction.
- Environment: our venv + pytorch_lightning 2.6.6 / torchmetrics 1.9.0 (the pahde env's versions; their ISNet module imports
  Lightning; torch kept at 2.11.0+cu128). ISNet requires `cudnn.deterministic=True` (their code refuses otherwise).
- Driver smoke test on CPU (1 batch, Trap A fold 1): arm d and the ResNet-50d ERM base run end to end. Thyroid environments
  (seed 42; n / malignant / marker) e.g. Trap A fold 1: train 957/214/267, val_clean 370/82/185, test_rev 357/166/189;
  Trap B fold 1: train 827/74/142, val_clean 106/24/53, test_rev 217/166/63.
- Not yet fixed (for `docs/PREREGISTRATION_PHASE5.md`, after the GPU timing test): ISNet initialisation (P3: random; others
  ImageNet), each network's optimiser and batch (current defaults: ERM base / RR-ClArC AdamW 1e-4 / 1e-5 head ×10 batch 48;
  ISNet SGD 1e-3 m 0.99 batch 16; arm d Adam 1e-4 batch 64), constant learning rate under early stopping.

### 2026-11-18 17:13 UTC (machine 2's clock) — machine 2: P4 Seg-ERM split 1 scored; arm a and arm-d comparison (descriptive)
- **Seg-ERM split 1 (machine 2, from scratch, A27): robust AUROC 0.609** on the 772-image trap test (best validation Dice 0.841
  at epoch 125 of 150, the registered selection; test scored once). Paper 0.727 ± 0.016 (3 seeds), notebook 0.772 (split 1).
  Gate (A13 + addendum) needs the 3-split mean; splits 2 and 3 pending (split 3 started 17:11, alone, ≈ 2.9 h).
- **Arm a split 1** (mask_cmc on this Seg-ERM's encoder, lesion-masked images): 0.528, Δ vs Seg-ERM −0.081 [−0.140, −0.022];
  C = 10 is the highest edge of the registered grid {0.01, 0.1, 1, 10}.
- **Arm d split 1** (machine 1's EfficientNet-B0 + mask_cmc scores) vs Seg-ERM split 1, same 772 images: 0.666 vs 0.609,
  Δ +0.057 [+0.003, +0.111] (image-only bootstrap, one network each).
- All descriptive until the P4 gate. No other epoch of Seg-ERM was scored on the test.

### 2026-11-18 21:35 UTC (machine 2's clock) — machine 2: P4 gate FAILED; reproduction-gap diagnosis (author's order; CPU, validation data only, no test variant scored)
**Gate (A13 + addendum):** Seg-ERM splits 1–3 robust AUROC 0.609 / 0.611 / 0.676 (best validation Dice at epochs 125 / 138 / 148
of 150), **mean 0.632 ± 0.038** vs paper 0.727 (|Δ| 0.095) and notebook 0.772 (|Δ| 0.140) → **fail**; Binary-ERM 0.363 → pass.
**P4 stays descriptive** unless the fix below changes the gate. Arm a (descriptive): 0.528 / 0.481 / 0.445 (mean 0.485 ± 0.042),
below Seg-ERM on every split, C at the top edge (10) each time. Arm d vs Seg-ERM split 1: +0.057 [+0.003, +0.111].
`results/replication/saab/p4_gate.md`, `p4_arm_a.md`; scores in `results/replication/saab/scores/`.

**Differences found, with sources** (`/workspace/third_party/saab` @ 1739cf2; history 9e06f97 2022-11-18 … 1739cf2 2023-02-25):
1. **Model selection (not a port error).** `src/task.py`: `ModelCheckpoint(monitor="valid_dice", mode="max",
   every_n_train_steps=5)`, Lightning default `save_top_k=1`, no early stopping; `cfg/config.yaml` `epochs: 100` in every commit.
   So their final model is the best-validation-Dice checkpoint, as ours (theirs holds the weights up to 5 steps after the
   validated epoch; ours the validated weights — recorded in SOURCE.md). **The notebook's Seg-ERM file is
   `…/method_seg/lr_0.0005/wd_0/dropout_0/seed_1/domino/maf4yoer/checkpoints/epoch=1-step=115.ckpt`:** step 115 is the first
   5-step save after the epoch-0 validation (113 steps per epoch at 1,822 / 16). With `save_top_k=1` and a validation set that
   lies inside training (Dice rises with training: ours 0.80 at epoch 0 → 0.84–0.87 at the best epoch), a 100-epoch run would
   have replaced it; the run that produced it most likely ended after about two epochs. Its save path lacks the
   `dataset_<source>/` level that `get_save_dir` gained in 1739cf2, so it was trained with the code of f5839e9 / a48c338
   (2022-11-21 / 2022-12-02). The ERM files the notebook scores are also early (`epoch=2-step=230`, 1739cf2; the earlier
   notebook version, a48c338, scored `epoch=7-step=795` → 0.2641 and `epoch=1-step=115` → its printed ERM line). **All notebook
   checkpoints postdate the paper (MLHC, August 2022), so none of them can be a Table 1 model.**
2. **Protocol vs Table 1 (not a port error).** The repository history starts 2022-11-18, after the paper, so it cannot show
   the Table 1 code (A13 item 6). Every commit: Adam lr 5e-4, wd 0, batch 16, plain cross-entropy (no class weights), 100
   epochs, ISIC loader without augmentation (`train_transform.json` is used only by `cxr.py`). Paper Appendix A.1 (p. 20,
   re-read): Adam, lr 0.0005, augmentation (affine, elastic, brightness, contrast), checkpoint on validation Dice, 150 epochs.
   We follow A.1 (A13). Input 512 × 512, their ISIC mean/std, segmentation target `(label + 1) × mask`, image score = mean over
   pixels of the class-2 softmax (`src/utils.score`) — identical in our port.
3. **Image resizing — PORT ERROR.** Their `img_loader` (`src/data/isic.py`) reads with `imageio`, then
   `transforms.Resize([512, 512])` on a PIL image (PIL bilinear, **antialiased** when shrinking); the mask goes through the same
   loader and `> 0.5`. Our cache (`p4_train.cache`) used `cv2.resize(INTER_LINEAR)` (**no antialiasing**) for images and masks
   (`> 127`). On the 259 validation images (width 576–6,748 px, median 3,008): mean |PIL − cv2| per image median 1.02 grey
   levels (90th percentile 2.44, max 5.50; widths > 3,000 px: median 1.47). Small, but it is not what their code does →
   amendment A31, fix, retrain, re-score once.
4. **Data (no difference).** Their notebook builds 2,853 rows and scores **772** test images (printed output), from Bissoto's
   `trap-sets/isic_annotated_test1.csv`, which has 772 rows in every commit of that repository (single data commit 841cc17,
   2020); ours are the same files (2,853 rows; 772 test, 618 benign / 154 melanoma). The paper's "513" (p. 12) matches no file
   or split: recorded as a paper inconsistency.

### 2026-11-18 ~21:50 UTC (machine 2's clock) — machine 2: P6 out-of-memory fixed (activation checkpointing, identical gradients); A31 reruns first
- P6 ERM bias 1 failed twice before training: out of memory at 20:11 (started the second split 3 ended, beside P4 arm-a's
  3.9 GB feature pass — a launch race) and at 21:33 alone (their batch 130 in fp32 needs > 15.4 GiB). Fix: activation
  checkpointing of the ViT-B/16 blocks (`EPVT_GRAD_CKPT=1`), checked identical (loss |Δ| 0; all 152 gradient tensors identical);
  peak 4.25 GiB (ERM) / 4.78 GiB (EPVT). Details in `third_party/epvt/SOURCE.md`; P6 queue now requires an exclusive GPU.
- Order (P4 keeps priority; changes A31's scheduling line, which assumed P6 bias 1 would already be running): A31 PIL cache
  (CPU, building) → Seg-ERM reruns `seg_pil_split1–3` (two paired, then the third) with arm a and the gate after each →
  P6 bias 1 (ERM, EPVT) → timing report. Expected: reruns ≈ 05:30 UTC on 19 Nov, P6 bias 1 ≈ 06:30.
- **Pairing decision (author, 21:40):** P6 bias 1 (ERM, then EPVT) starts beside A31 rerun split 3 (≈ 6.3 + ≈ 5 GB) once
  splits 1–2 are scored. `scripts/replication/m2_pair_measure.sh` measures it: split-3 min/epoch paired (2 fully paired
  epochs) vs alone (1.14, measured earlier on machine 2), P6 s/epoch paired vs alone (split 3 paused by SIGSTOP/SIGCONT for
  3 P6 epochs). Kept only if the total exceeds 1.0 of one GPU; otherwise P6 is stopped by PID and re-queued after split 3.
  The measurement will be recorded here.

### 2026-11-18 ~22:00 UTC (machine 2's clock) — candidate search (author's request while P4 reruns): one new primary candidate
- **Bissoto, Barata, Valle, Avila, "Artifact-based domain generalization of skin lesion models", ISIC Workshop @ ECCV 2022**
  (arXiv 2208.09756; code `github.com/alceubissoto/artifact-generalization-skin` @ 54b20fb, already cloned for P6). Same ISIC 2019
  trap sets (bias 0–1, 10 splits) and artifact labels as P6; ResNet-50, up to 100 epochs, patience 22, 10 runs, standard error.
  **Table 3 (bias 1, ROC AUC): ERM 0.58, GroupDRO on artifact environments 0.68, full pipeline (GroupDRO + NoiseCrop) 0.74.**
  NoiseCrop = test-time ROI censoring (lesion convex hull, background replaced by uniform noise, crop/rescale) — an ROI method
  whose failure mode for in-lesion artifacts (hair, gel bubbles) is the thesis question. Exact training commands in `run.sh`.
  Gaps: the masks NoiseCrop reads (`/deconstructing-bias-skin-lesion/isic2019-seg-299/*.png`, DeepLabv3+ for images without
  ground truth) are not released (that repository holds CSVs only); `data/skin_dataset.py` says the images were "already cropped
  and centered" (preprocessing not in the repository); workshop venue.
- Examined and not taken: SCORE (arXiv 2605.13340), SpurCon (arXiv 2608.17598), Learning To Focus (arXiv 2607.10851) — preprints,
  no venue, no code; Le et al. DFR on ISIC (IJCAI XAI workshop 2023) — no code found; Boland et al. (MELBA 2025) — no code;
  MIMM-X (MICCAI FAIMI workshop 2025) — CheXpert/MRI, no localized artifact.
- **Bissoto 2022 rejected by the author ("not good enough").** Further search, none taken:
  - SimBA (Stanley et al., MICCAI 2023 / JAMIA 2024 / eBioMedicine 2025; code `github.com/estanley16/SimBA` @ 2f7da7a; JAMIA data
    doi:10.5683/SP3/0A1IA0, 13.7 GB, CC BY-NC 4.0): synthetic 3D brain MRI, bias in 70 % / 30 % of the classes, "near" (left
    putamen, adjacent to the left-insula disease region) vs "far" (right postcentral gyrus). Its best method (reweighing)
    already brings the subgroup disparity to ≈ 0 (relative ΔTPR / ΔFPR 0.44 % / … near, −0.80 ± 0.64 % / −0.32 ± 2.13 % far;
    5 seeds), so nothing is left to beat, and "near" is adjacent, not inside, the disease region. Useful as a controlled
    test bed (its generator can place a bias inside the disease region), not as a target.
  - Yan et al., CVPR 2023 (ConfDerm, concept rewriting): no code or data release (project page 404; not in the authors' repos).
  - SilverLining (WACV 2026): no code, datasets not named. Tsue et al. 2023 (mammography view markers): private data.
  - Kvasir green-box overlay, endoscopy, pathology pen marks: no published mitigation benchmark with code found.
- Search round 4 (author: keep searching), none taken:
  - Friedrich, Stammer, Schramowski, Kersting, "A typology for exploring the mitigation of shortcut behaviour", Nature Machine
    Intelligence 2023 (arXiv 2203.03668; code ml-research/A-Typology-to-Explore-the-Mitigation-of-Shortcut-Behavior, Zenodo
    10.5281/zenodo.6781501): ISIC 2019 with real colour patches; RRR, RRR-G, RBR, CDEP, HINT, CE with annotation masks. On
    ISIC19 the paper reports **no accuracy improvement** (all ≈ 88 %, test still confounded; Table 2a) and its headline is
    an explanation score (wr), not a shortcut-broken test → no performance target.
  - MAGI (ICCV 2023): LIDC-IDRI / MSD nodule classification with explanation supervision; no shortcut or OOD test.
  - RES (KDD 2022, code YuyangGao/RES): gender and scene datasets only (not medical).
  - Fundus: Bressler et al. 2023 (BJO; laser-scar detection, not mitigation); Fernandes et al. 2025 (arXiv, age/sex attributes).
  - Chest X-ray lung-mask attention papers: arXiv preprints or no shortcut test (2301.10687, 2202.10971, 2512.23089).
- Friedrich et al. (NMI 2023) "in its own way" examined and **skipped by the author**: their ISIC19 metrics are test
  accuracy (no XIL method beats Vanilla 88.4 ± 0.5; patch images removed from the test) and the wr explanation score (best
  RRR-G 0.9 % GradCAM); the patches lie beside the lesion (outside the ROI); data built through the retired ISIC archive API +
  CDEP patch masks. Cited as related work only (mask-based XIL lowers attention on the artifact without accuracy gains).

### 2026-11-18 ~23:10 UTC (machine 2's clock) — new direction found: the ISIC trap-set benchmark lineage (DISC ICML 2023 → CBDebug ICLR 2026)
Author's request: "another direction in the same problem that more papers tackle; what 2026 is doing; who is near to us".
- **One shared test with a published lineage, the P4 data:** Bissoto's 2020 ISIC 2018 trap sets (5 splits; train 1,822 /
  val 259 / test 772; 7 artifacts incl. hair and gel bubbles that "could co-exist and cover the skin lesion region" — DISC
  p. 6). Papers reporting on it: Bissoto CVPRW 2020; **Saab MLHC 2022** (P4; Seg-ERM 0.727); **DISC, Wu et al., ICML 2023**
  (code github.com/Wuyxin/DISC @ d9865a8; ResNet-50; Table 1 ISIC average AUROC over the train-test splits, 3 runs:
  ERM 36.4 ± 0.7, GroupDRO 36.4, JTT 33.8, IRM 45.5, LISA 38.0, + 7 more, **DISC 55.1 ± 2.3**); **CBDebug, Enouen &
  Galhotra, ICLR 2026** (code github.com/ericenouen/cbdebug; post-hoc CBM on a frozen BioMedCLIP backbone, linear layer
  retrained after removing expert-flagged spurious concepts; Table 2 ISIC AUROC: original 39.3 ± 3.7, **CBDebug 58.0 ± 11.6**);
  a further OpenReview submission follows DISC's ISIC setup (not retrieved).
- **Data identity checked:** DISC's released ISIC archive (Google Drive 1Os34EapIAJM34DrwZMw2rRRJij3HAUDV, 11.2 GB): trap-set
  CSVs for splits 2–5 byte-identical to Bissoto f93c865; split 1 identical image IDs, labels and artifact flags (only the
  file extension in `image` differs, .jpg vs .png); images = ISIC 2018 Task 1/2 (2,594). Our P4 Binary-ERM (0.363, splits
  1–3) already matches DISC's ERM (36.4).
- **Near to us (novelty, cite):** SCER (Park et al., ICLR 2026; code MLAI-Yonsei/SCER): "Spurious Direction = average of
  differences in mean embeddings across domains within the same class" — the same within-class mean-difference construction
  as our δ_y, used to regularise classifier weights (not projected out); vision benchmarks only (Waterbirds, CelebA,
  MetaShift, ColorMNIST). CBDebug (above): frozen foundation backbone + linear retraining after concept removal, the same
  regime as our arm b. Neither addresses where the artifact lies relative to the ROI.
- Descriptive only (nothing registered): our P4 arms on splits 1–3 of the same trap sets — arm b (DINOv2 + mask_cmc)
  0.658 ± 0.003, arm c (U-MtE) 0.605 ± 0.006, arm d (split 1) 0.666 — are above DISC's 55.1 and CBDebug's 58.0, but on 3 of
  their 5 splits and not yet with their protocol; any claim needs a registered comparison (P8) first.

### 2026-11-19 18:30 UTC — machine 1: P3 scope reduced (author)
Reason: P3 is not reproduced (A9 check 0.223 / 0.260; even P3's released model fails on our pipeline), and on P3 our arms
cannot change any verdict. P3's role in the paper is now only to show, descriptively, that ISNet fails inside the ROI;
our arms on P3 are reported as a limitation. Only registered runs that are nearly done are finished:
- **Kept:** ISNet run 3 (90/10 inside): finish, score once on the rebuilt test, A24 evaluation. Run 2's A24 evaluation
  (done, committed 0f11749). Arm c (U-MtE) on the corner benchmark (registered under A12): one feature pass + CPU head,
  scored once. The final P3 table (ISNet corner vs inside vs 90/10 inside beside our registered arms b, c, d where they
  apply; rebuilt test primary, A24 secondary; descriptive, A9 caveat).
- **Stopped / not started:** the lung-masked DINOv2 reference row on the corner benchmark; any further P3 diagnosis,
  re-scoring, extra variants or arms; P3-specific exploratory checks (the one-vs-rest check). **The A24 feature passes
  for our arms b, c, d were not run** (not on the kept list); the arms are reported on the rebuilt test only, marked so.
- Then machine 1's GPU moves to the exploratory track: candidate (b), up to ~6 GPU-h, developed on the P1 location-inside
  validation only.

### 2026-11-18 ~23:30 UTC (machine 2's clock) — P8 = DISC (ICML 2023) set up; CBDebug (ICLR 2026) checked against R1–R8 (paper and code only, no runs)
**P8 (author's decision):** DISC, step 1 = reproduce their ERM and DISC on the ISIC trap sets with their code
(`third_party/disc/SOURCE.md`). Environment and concept bank being prepared; runs queue after P6 bias 1. P8 keeps priority over
CBDebug.
**CBDebug, Enouen & Galhotra, "Debugging Concept Bottleneck Models through Removal and Retraining":**
- Venue: **ICLR 2026** (proceedings PDF header "Published as a conference paper at ICLR 2026", proceedings.iclr.cc).
- Medical experiment ("frozen BioMedCLIP 39.3 → 58.0 ± 11.6"): Table 2, Post-hoc CBM with **automated feedback**; ISIC benign vs
  malignant "following the setup from DISC (Wu et al., 2023) with five different splits" (App. A, p. 15) — i.e. Bissoto's 2020
  trap sets (the P4/P8 data); spurious attributes dark corners, hair, gel borders, gel bubbles, ruler, ink, patches (localized;
  vary within each class in these trap sets); backbone **BioMedCLIP** (frozen; linear layer retrained); concept bank: DISC's
  texture concepts + the eight ISIC concepts of Yuksekgonul et al. (2023); γ = 5; metric **test AUROC** (27 groups); "three
  random seeds per dataset … averaged across seeds" — how the five splits enter the number is not stated. The large spread
  (± 11.6) is noted.
- **Code: the release does not cover the medical experiment.** github.com/ericenouen/cbdebug @ 3808f71 (2025-09-28) has data
  loaders and concept sets for Waterbirds, MetaShift and CelebA only (`pcbm/data_utils`, `pcbm/concepts/*.txt`); no ISIC
  loader, no ISIC concept list, no BioMedCLIP backbone; `pcbm/models/derma_models.py` is the PCBM upstream HAM10000 model.
- Against our requirements: R1 ✓ classification; R2 ✓ ICLR 2026; R3 ✓; **R4 ✗ for ISIC** (code covers the non-medical datasets
  only); R5 ✓ (DISC's ISIC data, checked identical to ours); R6 ✓ localized artifacts, some over the lesion; R7 ✓ AUROC on the
  reversed trap tests (but seeds/splits aggregation unclear, ± 11.6); R8 ✓ (artifacts present in both classes in training).
  Fails only R4; reproducing it would mean re-implementing the ISIC pieces (BioMedCLIP Post-hoc CBM, the concept set) from the
  paper. Not started.
- **P8 checks before A32 (2026-11-19):** images: 2,594 / 2,594 identical MD5 between DISC's archive and our ISIC 2018 copy;
  trap sets identical (see above). Aggregation: their code gives one AUROC per run (test AUROC at the best-validation-AUROC
  epoch, `run_expt.py:138–142`, AUROC of the class-1 softmax on that set's 772 test images, `utils/loss.py:194`); the paper averages
  per-set AUROC across the splits "as standard in Bissoto et al. (2020)" (5 fixed splits, mean ± SD) — no aggregation script
  is released. A32 registered.

### 2026-11-19 08:50 UTC (machine 2's clock) — A31 P4 gate FAILED again; pairing not kept; P6 EPVT NOT reproduced at bias 1; P8 started
- **Pairing measurement (P4 split 3 + P6 ERM bias 1, 02:31–02:42):** split 3 2.08 min/epoch paired vs 1.14 alone (0.55);
  P6 9.0 s/epoch paired vs 4.0 alone (0.44, split 3 paused by SIGSTOP/SIGCONT); **total 0.993 ≤ 1.0 → not kept**; P6 was
  stopped by PID and re-queued after split 3 (`/workspace/logs/pair_measure.json`); the stopped run is kept as
  `paired_stopped_erm_bias1_1_0242` and is not a result.
- **P4 A31 reruns (PIL-resized cache):** Seg-ERM robust AUROC 0.626 / 0.673 / 0.641 (best validation Dice at epochs 146 / 135 /
  131), **mean 0.647 ± 0.024** vs 0.727 (|Δ| 0.080) / notebook 0.772 → **gate FAILED; P4 stays descriptive** (A13). The port
  fix moved the mean from 0.632 to 0.647. `results/replication/saab/p4_gate_seg_pil.md`, `p4_arm_a_seg_pil.md`.
- **P6 step 1, bias 1, split 1 (paper protocol, A29 + decisions):** their ERM **0.690** test AUROC (digitised target 0.653,
  |Δ| 0.037 > 0.03); **EPVT 0.654** (target 0.746, |Δ| 0.092) → **EPVT not reproduced; ERM also outside 0.03.** Note EPVT ≤ ERM
  here (paper: EPVT +9.4). Timing: ERM 26 min (early stop at code-epoch 76 of 100), EPVT 43 min (67 epochs, early stop);
  all 6 levels × 2 would be ≈ 7 GPU-h. Scoring bug fixed before the EPVT score: the checkpoint's `model_num_domains` (4)
  does not match the 5 trained domain prompts; the scorer now builds the model with 5 domains as their `test_epvt.py` does
  (the first scoring attempt failed at loading, before any prediction). Per the author's rule P6 stops here and is reported.
- **P8 (DISC, A32) started 08:47** (ERM trap set 1); queue: ERM + DISC on trap sets 1–5.

### 2026-11-19 ~09:10 UTC (machine 2's clock) — P6 bounded diagnosis, step 1 (CPU; their code, configs, paper; no test scored)
a. **Domains — no error.** `train_epvt.py:283–286` builds the algorithm with `len(dataset)` = 5 domains and trains on all five
   environments (`train_loaders` over every `in_split`, lines 158–163); `--test_env 0` only fills `domain_mapping` (lines 303–306,
   never used afterwards) and the saved field `model_num_domains = len(dataset) − len(test_envs)` = 4 (line 321) — the source of
   the scorer mismatch (fixed in the scorer, no prediction affected). Environments are the sorted folder names C, D, G, H, R
   (`datasets.py:185–186`); our mapping clean/dark corner/gel bubble/hair/ruler → C/D/G/H/R; the order only permutes domain
   indices.
b. **Epoch unit — PORT ERROR (ours).** Their trainer uses each whole domain folder: `in_splits.append((env, in_weights))`
   (lines 129–143) with **no DomainBed 20 % holdout**, and `steps_per_epoch = min(len(env)/batch_size)` over those full domains
   (line 310). Our step count assumed an 80 % in-split: bias 1 → int(280/26) = 10 steps/epoch (1,001 steps), but their code gives
   **int(350/26) = 13 → 100 epochs = 1,301 steps**, and patience 22 counts 13-step epochs. Confirmation from their own README:
   bias 0 → int(672/26) = 25 steps/epoch and their `--steps 1501` = 60 × 25 + 1, i.e. **exactly the paper's 60 epochs for the
   OOD experiment**. Reading A (their code's unit) stands; the holdout was our mistake. → A33.
c. **Backbone / preprocessing / data — no port error.** torchvision `vit_b_16(pretrained=True)` = ImageNet-1k V1 weights
   (`vit_b_16-c867db91.pth`, unchanged since torchvision 0.13, 2022); `im21k` False, `attention_dropout` 0, feature dropout 0.1,
   `class_balanced` True — their defaults, used as is. Train/val/test transforms are their code. Images: ISIC 2019 (manifest
   match), plain-id symlink mirror. **12,343 vs 12,360 training images:** the released `train_bias_0_1.csv` has 12,343 rows
   (all annotated; val 2,057 and test 6,199 match the paper exactly); the paper's 12,360 matches no released file (17 images,
   0.14 %); recorded, unexplained.

### 2026-11-19 ~09:30 UTC (machine 2's clock) — P6 second error-check round (CPU) + pairing P6 A33 with P8 (author)
- **Correction to A33's description:** their code already defined epochs in its own unit for the pre-A33 runs —
  `steps_per_epoch = int(350/26) = 13` over the whole domains — so **patience 22 was already counted in the right unit**; only the
  total was short (1,001 steps = 77 code-epochs instead of 100). ERM logged 78 epochs and **stopped at the step cap** (no early
  stop); **EPVT early-stopped at epoch 67** (patience 22; "Best Val roc_auc 0.925") **before the cap**, so the A33 fix changes
  ERM (23 more epochs) but cannot change EPVT apart from GPU nondeterminism. A33's fix and rerun stand as registered.
- **Second round, settings as their scripts printed them (out.txt of both pre-A33 runs) vs README and paper:** lr 5e-6,
  lr_classifier 5e-5, weight_decay 1e-2 (paper), wd_classifier 1e-5, batch 26 × 5 domains = 130 (paper), prompt_dim 10 (EPVT
  paper), lr_prompt 1e-3 / lr_project 1e-4 / wd_project 1e-5 (their registry defaults; not stated in the paper), class_balanced,
  data_augmentation (their transform = paper's list), resnet_dropout 0.1 on the ViT features (their default), im21k False
  (ImageNet-1k ViT-B/16), seed 0, hparams_seed 0, val = `val_bias_1_1` (2,059), test = `test_bias_1_1` (6,181) — **no further
  difference found.** Bissoto's "already cropped and centered" note (`artifact-generalization-skin/data/skin_dataset.py`) does not
  apply: EPVT's `grouping.py` copies the raw ISIC 2019 training images. Remaining unexplained: 12,343 vs 12,360 training images.
- **Pairing (author):** P6 A33 (ERM then EPVT at bias 1, once) starts beside P8's DISC on trap set 1, measured with
  `scripts/replication/m2_pair_p6p8.sh` (DISC alone 8 epochs first; P6 alone while DISC is paused; kept only if total > 1.0,
  else P6 reruns after P8). P8 ERM alone was measured at 9.6 s/epoch (epochs 47–55, trap set 1). Limitation: the running P8 queue
  starts each new job only on an otherwise empty GPU, so P8's next job waits for P6 to finish. After P6, two P8 trap sets are
  paired the same way.
- **Incident 09:06–09:08 (my scripts; no result affected, ~20 GPU-min lost):** the new P8 pairing watcher treated the pre-A33
  P6 result files (not yet renamed) as "A33 done", stopped the old sequential P8 queue's shells and started two lanes; the lanes'
  running-job check required a space after `--seed k`, but the ERM command line ends with `--seed 1`, so lane a started a
  **second ERM on trap set 1 into the same output folder** while the original was at epoch ≈ 90, truncating its `test.csv` /
  `train.csv`. Both ERM-1 processes, both lanes and the watcher were stopped by PID (with DataLoader workers); the folder is kept
  as `corrupted_duplicate_start_0907_trapset_id=1` and is not used. Fixes: running-job match `--seed k( |$)` (tested on strings),
  watcher waits for `preA33_epvt_bias1_1` + new result files. P8 restarted 09:08 with one lane (ERM trap set 1 from scratch);
  the P6 + P8 pairing measurement still runs on DISC trap set 1; two P8 lanes start only after the P6 A33 rerun.

### 2026-11-19 ~10:20 UTC (machine 2's clock) — machine 2: P8 stalled 09:25–10:20 (two torch-2 port errors in DISC); A35
- **09:25 DISC trap set 1 failed at start:** `torch.load` of their pickled ERM model (`WeightsUnpickler … torchvision.models.resnet.ResNet`):
  torch ≥ 2.6 loads weights-only by default. Fix: `weights_only=False` at all five `torch.load` sites (torch 1.11 behaviour).
  My queue did not alert; the GPU sat idle until I checked at ~10:05.
- **10:10 restart hung** in DISC's concept-image preprocessing (0/58, all processes at 0 % CPU on a futex): their code forks a
  `Pool` after CUDA initialisation, which deadlocks under torch 2.11. Fix: the same `process_func` over the same chunks,
  sequentially (identical tensors, same order). ~3 min per DISC start. Both ports: `third_party/disc/torch2_port.patch`.
- Failed output folders kept, renamed `failed_torchload_0925_…_x`, `failed_poolhang_1010_…_x`; not used. ERM trap set 1 (done)
  unaffected. Restarted ~10:13; the P6 A33 + P8 pairing measurement (`m2_pair_p6p8.sh`, still waiting) starts on DISC trap set 1
  as planned. ~55 GPU-min lost.
- **Found while diagnosing — recorded as A35 before any P8 score:** DISC's ISIC loader hard-codes `id_val=True`: validation =
  20 % of each trap set's test CSV (154 images), test = the other 80 % (618), train = the full train CSV (1,822); A32's
  "val 259 / test 772" was wrong for P8. Gate unchanged (the published numbers come from this code); our arms use the same
  train / val / test images.

### 2026-11-19 ~10:50 UTC (machine 2's clock) — machine 2: third DISC port, DISC trains; P6 A33 paired; A35 addendum; watchdog
- **10:26 third torch-2 failure** (first DISC epoch): `loader_iter.next()` was removed in torch 2 → `next(loader_iter)` at all four
  sites (plus `np.int` → `int` in the unused cut-mix code), after a scan of the whole DISC tree for torch-1.11-era calls (only
  `pretrained=True` remains: deprecated but still loads the same ImageNet V1 weights; checked). Restarted ~10:27; trains.
  Folder of the failed start kept as `failed_iternext_1026_…_x`.
- **DISC speed:** 96.6 s/epoch alone (trap set 1, measured by `m2_pair_p6p8.sh`), ~10× ERM: each epoch retrains the concept
  bank (58 CAVs) and computes concept scores on 3 clusters, mostly on CPU (GPU utilisation 0–40 %). ≈ 2.7 h per DISC trap
  set alone. The P6 A33 rerun (ERM then EPVT, bias 1) started beside it at 10:46; the pairing measurement follows.
- **A35 addendum (author):** our P8 arms under both selection rules — primary DISC's 154 / 618 split, secondary the registered
  259-image Bissoto validation (inside the 1,822 training images in all five trap sets — checked, as A13 noted), both scored
  once on the 618; the write-up states that DISC's protocol selects on test-distribution images.
- **Watchdog (author):** `scripts/replication/m2_watchdog.sh` (running since 10:48): a queued job exiting non-zero is logged here
  and, for P8, retried once (incomplete folder moved aside); GPU idle > 10 min while jobs are queued is logged and either the P8
  lane is restarted (none alive) or a possible hang is alerted. Tested in dry mode on fake logs (retry once, second failure
  alert-only, P6 failure alert-only, idle-with-live-process alert). Its lines go under "Watchdog log" below.

### Watchdog log (machine 2, automatic; `scripts/replication/m2_watchdog.sh`)
