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
