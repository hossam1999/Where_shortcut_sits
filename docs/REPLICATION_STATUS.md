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
