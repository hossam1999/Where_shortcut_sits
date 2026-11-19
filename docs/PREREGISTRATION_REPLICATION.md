# Pre-registration: replicate three papers, compare our method on their tests, then the location test

Committed 2026-11-16, **before any of our methods (arms a, b, c) is run on any test set** of these papers. Phase-0 facts
(page and table numbers) are in `docs/REPLICATION_STATUS.md`; the plan is in `docs/REPLICATION_PLAN.md`. Anything decided
after a result is seen is labelled **exploratory** in every table and text. Amendments are allowed only if they are
stricter, or if they are forced by a fact we could not know now (data access, code that does not run). Each amendment is
dated and committed before the runs it affects.

## 0. Common rules

- **Same data and metric.** Every head-to-head number uses the paper's splits, images, backbone and metric. Their method
  and ours are scored on the identical test images (inner join on image id; the number of dropped images is reported).
- **Selection on validation only.** Every setting of ours (and every setting of theirs the paper leaves open) is chosen on
  the paper's validation split. The test set is scored once per setting. No setting is changed after a test result is seen.
- **Seeds.** P1: 10 training seeds (0–9). P2: 5 (0–4). P3: 5 (0–4), or 3 (0–2) if one training run of the slowest P3
  method takes more than 10 GPU hours on this machine. The P3 seed count is decided from a timing run of 1 seed and written
  into this file (as a dated amendment) **before** the full P3 runs. No seed is dropped. A failed run is rerun with the same seed;
  if it still fails, it is reported as failed.
- **Data split seeds are the papers' own** (P1 `np.random.seed(1234)` in their dataset code; P2 `np.random.default_rng(0)`;
  P3: the split we rebuild, fixed before any run, seed 20261116). Training seeds change only initialisation, data order and
  augmentation.
- **Predictions** (image_id, seed, method, test set, scores) are saved for every run under `$WTSS_DATA/replication/<paper>/`
  (never committed); summaries go to `results/replication/<paper>/`.
- **Software deviation (approved 2026-11-16).** Their code runs with torch/lightning/captum raised to CUDA 12.8 builds,
  because the pinned versions cannot run on this Blackwell GPU. Every code change is a `.patch` in `third_party/<paper>/`.

## 1. "Reproduced"

Our mean over seeds of the paper's own method, on the paper's metric and test, against the paper's number.

| paper | target (paper's own method) | reproduced if |
|---|---|---|
| P1 cardiomegaly | Masked 0.77 ± 0.09 AUROC (Table 1, with SPC) | \|ours − 0.77\| ≤ 0.09 (1 SD) |
| P1 pneumonia | ActDiff 0.68 ± 0.05 AUROC (Table 2, with VPC) | \|ours − 0.68\| ≤ 0.05 |
| P2 | RR-ClArC 0.76 accuracy, biased test (Table 1, ISIC 2019) | \|ours − 0.76\| ≤ 0.03 |
| P3 | ISNet 0.775 macro-F1 on the deceiving-bias test (Table 1, COVID-19) | \|ours − 0.775\| ≤ 0.03 |

- P3 reports ± 0.008, but this is a Bayesian SD over test images for one trained model, not a spread over seeds. We treat
  P3 as "no seed spread reported" and use 0.03. P2 reports no spread.
- Every other method of each paper is reported against its paper number with the same rule (P1: its own SD; P2, P3: 0.03).
  These are descriptive; only the targets above gate a verdict.
- **P3 test identity.** The 1,515 BIMCV COVID-19 images are not published as a list, so the P3 test is **rebuilt by P3's rule,
  not the identical image list** (`docs/REPLICATION_STATUS.md`). P3 verdicts are allowed only on this rebuilt test, and only if
  ISNet is reproduced on it. The proxy test (held-out BrixIA + CheXpert, run while BIMCV is not available) carries **no
  verdict**: it is labelled "not P3's test" everywhere.
- **If a paper's target is not reproduced**, we report the gap with numbers, and no "beaten" claim is made for that paper. The
  head-to-head table is still reported, labelled "against our (non-reproducing) run of their method".

## 2. "Beaten"

For each comparison below, "theirs" is **our run** of the paper's best method (same seeds, same test images), not the
paper's printed number.

Our arm beats theirs if both hold:
1. our mean over seeds is higher than theirs, on the paper's metric;
2. the paired difference (ours − theirs) has a 95% CI above 0, and its Holm-adjusted p < 0.05.

The CI and p come from a **crossed seed × image bootstrap** with 10,000 replicates, as in `src/wtss/stats_crossed.py`. Each
replicate resamples seeds (multinomial) and draws one Poisson(1) weight per test image, shared by every seed and both
methods. Seed i of ours is paired with seed i of theirs. Each replicate recomputes the metric per seed from the weighted
images (AUROC; accuracy; macro-F1) and averages the per-seed differences. The CI is the 2.5–97.5 percentiles. The p-value is
p = min(1, 2·min(P*(Δ ≤ 0), P*(Δ ≥ 0))). The same estimator is extended to accuracy and macro-F1 (unit-tested against the
AUROC version) before any run.

**Holm family: all 10 head-to-head comparisons across the three papers**, adjusted together at α = 0.05.

| # | paper / task | theirs (best in paper) | ours | metric |
|---|---|---|---|---|
| 1–3 | P1 cardiomegaly | Masked | a, b, c | AUROC, reversed test |
| 4–6 | P1 pneumonia | ActDiff | a, b, c | AUROC, reversed test |
| 7–9 | P2 | RR-ClArC | a, b, c | accuracy, biased test |
| 10 | P3 (original benchmark) | ISNet | c only | macro-F1, deceiving-bias test |

On P3's original benchmark, every training image carries its class's mark (artifact = label), so the within-class
difference does not exist and mask_cmc cannot be estimated. Arms a and b are therefore not run there; only U-MtE (arm c) is.

## 3. Our arms (fixed now)

The **artifact label** each arm uses at training time only:
- P1: hospital source (NIH = 0, PadChest = 1) for cardiomegaly; view (AP/PA) for pneumonia.
- P2: microscope inserted (yes/no).
- P3: corner mark present (90/10 variant and location test only).

None of our arms uses anything at test time except the image (and its ROI mask where the arm masks).

**a) mask_cmc on their backbone (same-backbone comparison).** For each seed, take their trained network:
- P1: the Masked ResNet-18 (both tasks), with its own input pipeline, so images are masked by their mask;
- P2: the Vanilla ResNet-50 (`resnet50d`), unmasked input, since that is what it was trained on and P2 has no masked method;
- P3: the U-Net lungs-only DenseNet-121 (90/10 variant and location test only).

Then:
1. Extract the penultimate features (global-average-pooled: 512 / 2048 / 1024-d) for train, validation and test.
2. Project out the within-class artifact mean differences: `cmc_directions` → `Projector` (`scripts/round8/candidates.py`),
   extended to K classes on 2026-11-16 with tests in `tests/test_round8.py`. The mode, pooled or per_class, is chosen on
   validation.
3. Refit the linear layer on the training set. The refit is a logistic regression, binary or multinomial; class-balanced
   weights; C ∈ {0.01, 0.1, 1, 10}; chosen on validation.

In P2, only class MEL has images with and without the artifact in training, so there is exactly one δ and the two modes coincide.

**b) Our frozen-encoder pipeline.**
- Encoder: DINOv2 ViT-B/14 @ 518 px, frozen, CLS token, L2-normalised.
- Input: ROI-masked images (P1 and P3 images are upsampled from their 224-px copies).
- ROI per task:
  - P1 cardiomegaly: PSPNet lungs ∪ heart (TorchXRayVision `chestx_det`, p > 0.5). Cardiomegaly evidence is the heart, which
    a lung-only mask removes.
  - P1 pneumonia: PSPNet lungs.
  - P2: lesion mask (HAM10000 manual; U-Net for BCN/MSK, as in E13).
  - P3: PSPNet lungs.
- Head: mask_cmc, with the same mode and C grid and validation rule as arm a.

**c) U-MtE** (`src/wtss/umte.py`), label-free:
- `UMtE("dino518").fit(train_images, train_rois)` with the same ROI as arm b;
- no artifact labels and no artifact examples;
- head: class-balanced logistic regression on `transform` features, C on validation.

**Validation rule per paper** (the same for their open settings and ours):
- **P1**: the paper's validation split. As in their code, it carries the reversed site/view link. The selection metric is
  AUROC. Their models select their epoch and settings on this split too.
- **P2**: the paper's validation split, scored twice: clean, and biased (microscope in every image, as the biased test). The
  selection metric is the mean of the two accuracies. It chooses λ ∈ {10, 50} and the layer (last 3 ResNet blocks) for RRR and
  RR-ClArC, which the paper leaves open, and our C and mode.
- **P3**: the paper's validation split with its marks (biased validation, as in the paper). The selection metric is macro-F1.

## 4. Metrics reported for every method

| paper | head-to-head metric | also |
|---|---|---|
| P1 | AUROC, reversed test | "clean": within-site (or within-view) AUROC on the reversed test, mean of the two strata, where the shortcut cannot help |
| P2 | accuracy, biased test | macro one-vs-rest AUROC (biased test); clean-test accuracy and macro AUROC |
| P3 | macro-F1, deceiving test | macro-F1 on biased and standard tests; macro one-vs-rest AUROC on all three; standard (mark-free) test = clean |

Mean ± SD over seeds is reported for every metric. A method that gains robustness while losing more than 0.03 in clean
performance against the standard model is flagged in the table ("buys robustness with diagnosis").

## 5. Phase 4: location test (the thesis question inside their setups)

In each setup the data, splits, backbones and methods stay identical. The same synthetic artifact (shape, size and opacity
fixed per setup) is placed at three positions relative to the ROI:
- outside: overlap ≤ 0.02;
- border: overlap in [0.40, 0.60];
- inside: overlap ≥ 0.95.

Placement uses `find_best_placements_for_mask` (`src/wtss/synthetic.py`) with targets 0, 0.5, 1.0. Only images where all
three positions are feasible are kept, so every position uses the same images. The number excluded and their label mix are
reported.

**Correlation.**
- Binary tasks: training 90/10 (artifact in 90% of positives, 10% of negatives). Tests: correlated 90/10, reversed 10/90, and
  clean (no artifact).
- Multi-class tasks: an attacked class (P2: MEL; P3: COVID-19) has the artifact in 90%, the other classes in 10%. Reversed:
  10% / 90%.
- Presence is set per image by `presence_vector` (stable hash of image id, label, seed and environment), independent of position.

| setup | base data | ROI | artifact | methods |
|---|---|---|---|---|
| P1 | cardiomegaly, P1's **No-SPC** split (site independent of label, so the tube is the only induced shortcut) | PSPNet lungs ∪ heart | radio-opaque tube `draw_tube`, 43 × 6 px at 224 px (= the 100 × 12 px at 518 px of our chest sweep, `TUBE_GEOMETRY`) | Standard, Masked (their circle), GradMask, ActDiff, RRR (masks for these = the ROI), ours a, b, c |
| P2 | ISIC 2019, P2's split | lesion mask | P2's operation (pixels set to black), as a **filled black disc** of diameter 0.12 × image width. The vignette itself cannot be moved inside a lesion; the disc keeps its colour and operation. | Vanilla, RRR, RR-ClArC, ours a, b, c |
| P3 | P3's data and split (proxy until BIMCV), marks moved | PSPNet lungs | P3's white mark (triangle; 10% of the image side), single mark type for the attacked class | DenseNet, U-Net lungs-only, ISNet, ours a, b, c |

P3 also runs a **90/10 variant of its original corner benchmark**: marks in the upper-left corner as in P3, but present
in 90% of the attacked class and 10% of the others, so that mask_cmc applies.

**Reported per position:**
- reversed-test AUROC (binary; attacked class one-vs-rest for P2/P3) and the paper's metric;
- each method's gain over standard training (reversed AUROC, paired, crossed bootstrap);
- the **crossover** = gain outside − gain inside, with a 95% CI;
- clean AUROC.

**Registered expectations** (tested with the crossed bootstrap, Holm over methods × setups within this phase):
- E-L1: for methods that act only outside the ROI (Masked, ISNet, U-Net lungs-only, RRR/GradMask with the ROI mask), the
  crossover is > 0 (CI above 0), and the gain inside is ≤ 0 or its CI includes 0.
- E-L2: for mask_cmc (arms a and b), the gain inside is > 0 (CI above 0) in every setup.
- E-L3: for U-MtE (arm c), the gain inside is > 0 (CI above 0) in every setup.

Each expectation is reported as supported, not supported, or contradicted, per setup, whatever the outcome.

## 6. What we will and will not claim

- "Beats X on P# by Δ [CI]" only if section 2 holds for that comparison, after Holm, and the paper's target is reproduced.
- Proxy-test (P3 before BIMCV) and exploratory results never support a "beats" or "reproduced" statement.
- A comparison that does not meet the rule is reported as "not beaten", with its numbers.

## Amendments

**A1 (2026-11-16, before any run; a fact from P1's code, found after the first commit).** P1's runner
(`scripts/run_seeds.sh`) uses the seeds **1234, 3232, 3221, 9856, 1290, 1987, 1111, 6400, 8888, 0451** (the last is
passed as the integer 451). `training.py` and `configuration.setup_dataset` pass this seed to the dataset, so **each P1 seed
draws its own balanced cohort and train/valid/test split** as well as its own initialisation. P1 therefore uses these 10
seeds, not 0–9, and the split is not fixed at 1234. Consequences:
- arms a, b and c are fitted and scored separately per seed, on that seed's split;
- the pairing of ours with theirs stays within seed (same split, same test images);
- in the crossed bootstrap, test images are shared across seeds only where the splits overlap. The estimator handles this
  already (one Poisson weight per image id).

Section 0's statement about "P1 `np.random.seed(1234)`" is replaced by this paragraph.

**A2 (2026-11-16, before any phase-4 run; reason: compute cost).** The full plan needs ~1,500–2,000 GPU-hours on this
machine (one 16 GB RTX 5060 Ti), about two-thirds of it in phase 4. By the author's decision, phases 2–3 stay exactly
as registered (all seeds, all methods, the papers' tests). Phase 4 (section 5) is reduced as follows:
- **Positions:** inside (overlap ≥ 0.95) and outside (overlap ≤ 0.02) only. The border position is dropped. The
  crossover (gain outside − gain inside) is unchanged. Images must be feasible at both remaining positions.
- **Seeds:** 3 per arm (P1: the first three of P1's list, 1234, 3232, 3221; P2 and P3: 0, 1, 2).
- **Methods per setup:**
  - standard training;
  - the paper's main method: P1 Masked; P2 RR-ClArC; P3 ISNet;
  - the paper's best other method among those we run, fixed from the papers' own tables and not from our results:
    - P1 GradMask (cardiomegaly Table 1: GradMask 0.74 > ActDiff 0.73 > RRR 0.57; the P1 location setup is cardiomegaly);
    - P2 RRR (the only other method of P2 that we run);
    - P3 U-Net lungs-only (Table 1 deceiving test 0.618; the best non-ISNet model there);
  - ours: arm a (mask_cmc on their backbone) and arm c (U-MtE). Arm b and the other paper methods are dropped from phase 4.
- **Unchanged:** the tests (correlated, reversed, clean), metrics, the crossed bootstrap, Holm within phase 4, and
  expectations E-L1 to E-L3 (applied to the methods that remain).
- After P1's phases 2–3 finish, real GPU-hours per run and a new total estimate are reported to the author. Work stops for
  the author's decision if the remaining total exceeds ~4 weeks.

**A3 (2026-11-16, before any affected run; reason: compute; author's decision).**
1. **Seeds.**
   - P2, P3 and all of phase 4: **1 training seed** (seed 0). P2 and P3 each report a single trained model, so this matches
     their protocol.
   - P1: **3 seeds**, the first three of P1's list (1234, 3232, 3221). P1's claim is a 10-seed mean with an SD up to 0.09, so one
     seed cannot test it. This replaces section 0's 10 seeds for P1's phases 2–3. Phase 4 uses 1 seed for every setup,
     P1 included (P1: seed 1234), which replaces A2's 3 seeds. (Wording corrected 2026-11-16, before any phase-4 run.)
   - Runs with seed 0 (P2, P3) and seed 1234 (P1) go first, so single-seed results exist early.
2. **Statistics with 1 seed** (P2, P3, phase 4): the same paired bootstrap, over test images only (one model run per method,
   the same images, 10,000 replicates, Poisson weight per image). **Training-seed variance is not included**, and every such
   table states it. "Beaten" there means "beaten for this training run". The Holm family and the "beaten" rule are otherwise
   unchanged.
3. **P1 verdicts** use the 3-seed mean, the registered "reproduced" rule (within the paper's SD) and the crossed seed × image
   bootstrap over the 3 seeds, with the note that the paper used 10 seeds.
4. The P3 seed-count rule in section 0 (5, or 3 if > 10 GPU-hours) is replaced by item 1.

**A4 (2026-11-16, before any affected run; reason: scope, the supervisor's actual request; author's decision).**
1. **Methods re-run in phases 2–3:** each paper's best method plus the standard baseline (to validate the setup).
   - P1: Standard, Masked, ActDiff, on both tasks, 3 seeds (A3).
   - P2: Vanilla and RR-ClArC, 1 seed.
   - P3: DenseNet and ISNet (item 2).
   - All other baselines (P1 GradMask, RRR, Adversarial; P2 RRR, P-ClArC, rP-ClArC; P3 U-Net lungs-only, RRR, GAIN, …) are
     **cited from the papers, not re-run**.
   - The exploratory P1 "Masked as described" runs are dropped as well.
   - The "reproduced" targets and the 10 head-to-head comparisons (section 2: ours vs P1 Masked, P1 ActDiff, P2 RR-ClArC,
     P3 ISNet) are unchanged.
2. **P3 original benchmark.** The released `COVIDIsNet.pt` (P3 README, Google Drive) is a pickled `IsDense` object with no
   metadata. Its name, unlike `FacesSyntheticBiasISNet.ckpt` and `ISNetDogsSyntheticBias.ckpt`, marks it as the COVID-19 model
   *without* synthetic marks (Table 2). It is therefore not the Table 1 model. We **train 1 ISNet and 1 DenseNet** on the
   synthetic-mark benchmark (seed 0). `COVIDIsNet.pt` is scored on the rebuilt o.o.d. test against Table 2 only, as a check that
   the rebuilt test matches P3's.
3. **Phase 4 only in two setups** (1 seed each); **no phase 4 for P2**.
   - **P1 cardiomegaly:** inside vs outside the lungs ∪ heart ROI. Methods: Standard, Masked, ActDiff, plus ours (arms a and c).
     This replaces A2's GradMask with ActDiff.
   - **P3:** P3's marks moved **inside the lungs**, compared with P3's original corner (outside) runs. Methods: DenseNet, ISNet,
     ours.
     - In P3's own design every training image carries its class's mark (artifact = label), so mask_cmc (arm a) cannot be
       estimated there; only U-MtE (arm c) applies.
     - **Whether the inside run keeps P3's design (all images marked, directly comparable with P3's Table 1) or uses the
       registered 90/10 presence (so that mask_cmc applies) is to be fixed by the author before the P3 phase-4 runs.**
   - The P3 90/10 corner variant and the border position are dropped.
4. **Our arms reuse trained networks and saved features.** Arm a uses each paper's re-run network: P1 Masked; P2 Vanilla;
   P3 the DenseNet, because the U-Net pipeline is not re-run (the registered U-Net lungs-only backbone is replaced by the
   DenseNet). Arms b and c use frozen DINOv2 features, extracted once per image set.
5. Order: P1 and P2 first (shortest first), then P3 once its data is available.

**A5 (2026-11-16, before any affected run; author's decisions).**
1. **P1 cardiomegaly "Masked" from the Standard run.** With their `load_config`, the loaded configs of
   `xray-seeds_resnet.yml` (Standard) and `xray-seeds_resnet_maskall.yml` (Masked) differ only in
   `dataset.test.XRayDataset.mask_all` (None vs True) and `experiment_name` (the output folder name). Every training key is
   identical (Adam lr 0.009992, same data, same seed; validation unmasked in both, so the same epoch is selected).
   - Cardiomegaly Masked is therefore **the Standard run's saved best model, scored on the test split built by their own
     dataset code with `mask_all=True`**. This saves 3 runs.
   - Their GPU training is not bit-deterministic, so a separate Masked run would differ from Standard only by GPU noise; using
     the same model makes the Masked − Standard comparison paired exactly.
   - **RSNA pneumonia is not shortened:** its Masked config uses lr 0.01 against Standard's 0.009996, so it is trained
     separately as written.
2. **P1 arm a input: ROI-masked images on every split** (train, validation, test), passed through their trained Masked
   network. This is our method's definition. Their literal pipeline (training features unmasked, test features masked) is
   reported as a side note only.
3. **P3 phase 4 (fixes the open point of A4 item 3): P3's own design**, with every image carrying its class's mark, moved
   inside the lungs. The comparison is P3's original corner (outside) result against our inside run. Only U-MtE (arm c)
   applies, because with artifact = label there is no within-class contrast for mask_cmc. A 90/10 inside+outside variant is
   added only if the measured ISNet time leaves room; that is decided after the ISNet timing test and recorded as a further
   amendment.

**A6 (2026-11-16, before any P3 result; author's decision).** Without this, mask_cmc is never compared with ISNet: P3's own
design has artifact = label, so mask_cmc cannot run there (A4, A5). Added: a **P3 90/10 variant at two positions**.
- **Design:** P3's data, split and test sets (the rebuilt o.o.d. test, plus the proxy until BIMCV is complete). P3's three
  class marks (triangle COVID-19, square normal, circle pneumonia; 10% of the image side).
- **Presence:** a mark is present on 90% of training and validation images of every class and absent from the other 10%,
  set per image by `presence_vector` (stable hash of image id, label, seed 0, environment), independent of position. A
  present mark is always the image's own class's mark, so the mark predicts the label, and mask_cmc can estimate within-class
  differences from the 10% of unmarked images.
- **Tests, as in P3:**
  - biased: the own-class mark on every test image;
  - standard: no mark;
  - deceiving: P3's swapped marks (circle on normal, triangle on pneumonia, square on COVID-19).
- **Positions:**
  - outside: P3's upper-left corner (overlap with the lungs ≤ 0.02);
  - inside: fully inside the lungs (overlap ≥ 0.95), placed with `find_best_placements_for_mask` on the P3 lung masks.
  - The same images are used at both positions.
- **Methods (seed 0):** DenseNet, ISNet, and ours: arm a (mask_cmc on their DenseNet trained in that condition, features of
  lung-masked images on every split, as in A5 item 2), arm b (DINOv2 + mask_cmc), arm c (U-MtE).
- **Metrics and rule:** macro-F1 on the deceiving test (P3's metric), plus macro one-vs-rest AUROC and the standard
  (clean) test. "Beaten" uses the image-only paired bootstrap of A3 (training-seed variance not included). These
  comparisons form their own Holm family (6 = arms a, b, c × 2 positions, each against ISNet), separate from section 2's 10.
- **Cost:** 2 ISNet (~11.3 h each) + 2 DenseNet (~3 h each), about +29 GPU-hours. New total for all remaining work: ~125 GPU-h.

**A7 (2026-11-16; port bug fix, not tuning).** The P1 port first used the repository's HEAD split code
(`activmask/datasets/xray.py` as changed in `81ecbd3`, 2021-02-03). That code was written after the published numbers:
arXiv v2, 2020-10-06, already has Tables 1–2 and N_train = 5,542. The file is restored from `32f7b7d` (2020-10-01), the
commit current at arXiv v2. It reproduces the paper's split sizes (cardiomegaly 5,542 / 2,770 / 2,776; RSNA
2,694 / 1,346 / 1,352) and its reversed validation and test (10/90). A full diff found no other behavioural change on the
training path (`docs/REPLICATION_STATUS.md`).
- **Superseded runs** (HEAD split code; archived in `$WTSS_DATA/replication/viviano/runs_superseded_head_splits/`, not
  used): RSNA Standard 1234, Masked 1234, ActDiff 1234, Standard 3232, Masked 3232 and ActDiff 3232 (last two stopped
  early); cardiomegaly Standard 1234 and ActDiff 1234.
- **Results seen before the fix:** RSNA seed 1234 Standard 0.355 and ActDiff 0.308 (their test AUROC at their selected
  epoch). This is stated here because these numbers were seen before the fix was made. The fix is a correction of our
  port to the authors' code that produced the published numbers; no setting was chosen to change a result.
- After the fix, P1 restarts from scratch with the registered seeds, methods and protocol, behind the author's gate.

**A8 (2026-11-16, before any corrected P1 result; author's decision).** A gate is placed after seed 1234 of P1 Standard and
ActDiff on both tasks; seeds 3232 and 3221 start only after it.
- **Bands:** paper ± 2 SD (Standard: cardiomegaly 0.70 ± 0.05, RSNA 0.20 ± 0.03; ActDiff: cardiomegaly 0.73 ± 0.03, RSNA
  0.68 ± 0.05).
- **Rule:** a result inside its band, or outside it by ≤ 0.03, passes, and all seeds then continue. If any result is further
  outside, P1 pauses for the author.
- The gate only decides whether to continue. The "reproduced" verdict stays the registered 3-seed-mean rule (section 1, A3).

**A9 (2026-11-16, before any P3 scoring; author's decision).** The rebuilt P3 test has 2,501 BIMCV COVID-19 images (1,631
patients) and 1,319 NIH pneumonia images (938 patients), against P3's 1,515 (1,145) and 1,295 (941). P3's class counts
(406 + 1,295 + 1,515 = 3,216) also disagree with its stated n = 3,126 (Supp. Note 10).
- *Settled by P3's main-text Table 4* (test confusion matrices, o.o.d. COVID-19): every one of the 8 models has row sums
  **370 normal / 1,295 pneumonia / 1,515 COVID-19, n = 3,180**. Recomputing per-class F1 from Table 4 reproduces Table 2
  (DenseNet-121 0.444 / 0.434 / 0.760, macro 0.546, exactly).
- So both "n = 3,126" and "406 normal" are wrong as descriptions of the evaluated test. 406 is the Montgomery + Shenzhen
  normal count *before* the age filter: 406 − 370 = 36, exactly the number of normals under 18 in our check. P3 therefore did
  remove minors from the normals, as its text says.
- (Note added 2026-11-16 at the author's request; the variants below are not changed by it.)
1. **Size-matched variant.** From our rebuilt sets, whole patients are drawn so that the class mix matches P3's counts:
   1,515 COVID-19 images from the 2,501 and 1,295 pneumonia images from the 1,319; the 406 normals are unchanged.
   - Rule: patients are taken in a random order (seed 20261117, numpy `default_rng`); a patient is added only if all of their
     images still fit within the target; drawing stops at the target or when no remaining patient fits. The count reached
     is reported.
   - Both variants are reported everywhere. The primary test stays as registered: the full rebuilt set, 2,501 + 1,319 + 406.
2. **Equivalence check with P3's released model.** `COVIDIsNet.pt` (P3 Table 2: macro-F1 0.773 on this o.o.d. test) is
   scored on both variants with P3's preprocessing.
   - If its macro-F1 is within 0.03 of 0.773 on a variant, the P3 head-to-head on that variant may carry "reproduced" /
     "beaten" verdicts (section 2 / A3 rules).
   - On a variant where it is not, the P3 head-to-head is descriptive only.
   - The P3 phase-4 comparisons (inside vs outside, A5/A6) are unaffected, since they compare our own runs on the same images.
3. The NIH pneumonia part follows the same rule. The 1,319 vs 1,295 gap is stated next to every P3 result.

**A10 (2026-11-16, before any P3 scoring; author's decision).** The 36 under-18 normals are dropped from both P3 test
variants, which then have **370 normals**, matching P3's evaluated test.
- **Evidence:**
  - P3 main-text Table 4 (test confusion matrices, o.o.d. COVID-19): the row sums are 370 normal / 1,295 pneumonia /
    1,515 COVID-19 in all 8 models.
  - Per-class F1 recomputed from Table 4 reproduces Table 2 (DenseNet-121 0.444 / 0.434 / 0.760, macro 0.546).
  - 406 − 370 = 36, the number of Montgomery + Shenzhen normals under 18.
- **Age rule:** the age is read from each image's clinical reading, taking the first number as years whatever its unit.
  - This is the only reading that gives 370: Montgomery 67 of 80, Shenzhen 303 of 326.
  - It keeps CHNCXR_0018 ("male 42", no unit).
  - It also counts one infant as an adult: CHNCXR_0202 ("female 64days"). Reading units literally would give 369. We follow
    P3's evaluated count and flag this image.
- **Final variants:**
  - primary: 2,501 COVID-19 + 1,319 pneumonia + 370 normal = 4,190;
  - size-matched: 1,515 + 1,295 + 370 = **3,180**, P3's evaluated total.
- **Equivalence check (A9) extended:** besides "macro-F1 within 0.03 of 0.773", `COVIDIsNet.pt`'s confusion matrix on each
  variant is reported next to P3's Table 4 ISNet matrix (normal 210 / 157 / 3; pneumonia 81 / 1,210 / 4; COVID-19
  93 / 156 / 1,266) and its Table 2 per-class F1 (0.555 / 0.858 / 0.907). The verdict gate stays the registered macro-F1 rule.

**A11 (2026-11-17, before any arm-a result; author's decision; evidence below).** P1's RSNA "Masked" path makes the
masked test image depend on the label.
- **Evidence (their code, as restored at `32f7b7d`; TorchXRayVision 0.0.18, the version they pin):**
  - `torchxrayvision/datasets.py` `RSNA_Pneumonia_Dataset.get_mask_dict` (lines 492–515) starts each mask as
    `np.zeros([size, size])` and sets only the pneumonia boxes to 1. Negatives have no box.
  - `activmask/datasets/xray.py` `JointXRayRSNADataset.__getitem__` takes `seg = sample["pathology_masks"][0]` (line 650)
    and, with `mask_all`, applies `img *= seg` (lines 669–671).
  - The Masked config sets `mask_all` on the test split (A5/YAML note).
  - Checked on seed 1234's test split: **all 200 negatives inspected have a 0-pixel mask and become the identical constant
    image** (one distinct value, 0.0). Positives keep only their boxes (3,396 of 50,176 pixels on average).
  - Example images (4 negatives, 4 positives): `$WTSS_DATA/replication/viviano/figures/rsna_masked_test_examples.png`,
    not committed (patient images).
- **Consequences:**
  1. **Arm a on RSNA** uses label-free PSPNet lung ROIs on every split (the same ROI as arm b), passed through their trained
     Masked network. This replaces "their mask" for RSNA only.
  2. **Their RSNA "Masked" result** (paper 0.56 ± 0.16; ours, seed 1234, 0.759) is reported as **"label-dependent at test
     time"** and is **not** the RSNA head-to-head target. The RSNA target becomes their best method whose test input does
     not depend on the label: **ActDiff (0.68 ± 0.05)**. The Holm family keeps its size: comparisons 4–6 of section 2 are
     against ActDiff, as already registered for RSNA pneumonia.
  3. **Cardiomegaly is unaffected:** the circle mask does not depend on the label, and Masked (0.77 ± 0.09) stays its target.

**A12 (2026-11-17; lean scope; author's decision; before any affected result).** Only each paper's best method and the
runs a "beat it" comparison needs are run (`docs/REPLICATION_PLAN.md`, "Lean scope"). Finished runs are kept and reported;
cancelled runs that never started are simply not run. Results already seen when this was recorded: the P1 seed-1234 gate
values (cardiomegaly Standard 0.627, RSNA Standard 0.186, RSNA ActDiff 0.708) and the P1 cardiomegaly seed-1234 arms
(arm b 0.869, arm c 0.593 test AUROC). Neither changes under A12.
- **P1 keep:** cardiomegaly Standard seeds 3232 and 3221 (Masked is scored from them, A5); RSNA ActDiff seeds 3232 and 3221.
  **Cancelled:** RSNA Standard and Masked seeds 3232/3221; cardiomegaly ActDiff (all seeds; the seed-1234 run in progress
  is stopped); ActDiff in the P1 location test.
- **P1 gate (A8)** checks only kept runs: cardiomegaly Standard, RSNA Standard (seed 1234, done) and RSNA ActDiff.
- **P1 location test (phase 4):** Standard with the tube inside vs outside the lungs + heart ROI, seed 1234 (2 runs);
  Masked is scored from them (A5); ours on top.
- **Our arm a sits on each paper's best network:** P1 cardiomegaly, the Standard/Masked model (unchanged); **P1 RSNA,
  the ActDiff model**, with the label-free PSPNet lung ROI on every split (replaces A11 item 1's Masked network); P2, the
  Vanilla network (RR-ClArC's base); P3, no network of theirs (DINOv2 + mask_cmc and U-MtE only, since the DenseNet runs
  are cancelled).
- **P2 keep:** Vanilla and RR-ClArC. P2 reports no λ or layer for RR-ClArC (Phase 0), so their grid (λ ∈ {10, 50} ×
  last 3 ResNet blocks) is run with the registered validation rule (section 3). **Cancelled:** RRR.
- **P3 keep:** ISNet on the original corner benchmark (1 run); ISNet with the marks inside the lungs in P3's design (all
  marked) and in the 90/10 variant (2 runs). **Cancelled:** DenseNet (all) and the 90/10 outside run; P3's Table 1 is
  the outside reference. Only U-MtE runs on the P3-design test (artifact = label); mask_cmc (arm b) and U-MtE run on the
  90/10 variant. A6's Holm family becomes 2 comparisons (arms b, c vs ISNet, 90/10 inside).
- Section 2's family of 10 is unchanged.

**A13 (2026-11-17; P4 Saab et al., MLHC 2022, ISIC only; author's decision; before any P4 result).**
1. **Scope:** ISIC only. The chest X-ray part is dropped (no SIIM access, unknown validation/test membership, no training
   tube labels); our own environment covers chest tubes via NIH.
2. **Data:** ISIC 2018 Task 1/2 training images and Task 1 lesion masks, joined to Bissoto's artifact and melanoma labels
   (github.com/alceubissoto/debiasing-skin @ f93c865, `trap-sets/` and `artefacts-annotation/isic_bias.csv`).
3. **Test:** the public release's 772-image test per split (Saab's `notebooks/isic_evaluation.ipynb` scores 772; the PDF's
   513 is reported as an inconsistency). We use the split triplet(s) their code/notebook uses and record which.
4. **Equivalence gate:** "reproduced"/"beaten" verdicts only if our reproduced Segmentation-ERM is within 0.03 of 0.727 **and**
   Binary-ERM within 0.03 of 0.359 robust AUROC (3-seed means). Otherwise P4 is descriptive only.
5. **Split as their code does:** 1,822 training images, including the 259 validation IDs. The validation-in-train overlap and
   any lesion or near-duplicate overlap between train and test are reported (thesis rules), without changing their split.
6. **Protocol:** their Appendix A.1 (150 epochs, Adam, the described augmentations; selection by validation AUROC for
   Binary-ERM and by validation Dice for the SE-ResNeXt50 U-Net), unless git history shows that the checked-in code (100
   epochs, no augmentation) produced Table 1. The choice is recorded. Port changes go in `third_party/saab/SOURCE.md` and a
   `.patch`.
7. **Runs:** Binary-ERM (reproduction check only) and Segmentation-ERM (their best method), 3 seeds each. Memory and speed
   are measured on this GPU first; if the paper's batch does not fit, the author is asked before gradient accumulation is used.
8. **Our arms, no new training:**
   - (a) features of their trained Segmentation-ERM encoder on lesion-masked images, mask_cmc, then a linear head. Every
     artifact flag that has all four artifact × diagnosis groups in training is used (e.g. hair); flags lacking a group
     (e.g. patches, with only 2 melanomas) are skipped. This rule is fixed now.
   - (b) DINOv2 + mask_cmc.
   - (c) U-MtE.

   ROI = ISIC 2018 Task 1 lesion masks (they exist for every image and do not depend on the label).
9. **P4 statistics:** robust AUROC (P4's metric), 3 seeds, crossed seed × image bootstrap; P4's head-to-heads (arms a, b, c vs
   Segmentation-ERM) form their own Holm family of 3.
10. **Location test:** one fixed artifact moved inside vs outside the lesion in their ISIC setup, Segmentation-ERM + ours,
   1 seed. Phase 5: a segmentation-supervised ERM arm on our ROI masks.
11. **Priority:** after the P1–P3 runs already queued; P4 fills free slots.

**A13 addendum (2026-11-17, before any P4 result; author's decision).** The equivalence gate (A13 item 4) gains a second
reference.
- **Reference values:** P4's evaluation notebook (`notebooks/isic_evaluation.ipynb`, saved outputs) scores its own
  checkpoints on the split-1 772-image trap test: Binary-ERM **0.310** (0.3098) and Segmentation-ERM **0.772** (0.7724).
  These checkpoints are not released as files: the notebook shows paths on the authors' machine, and the repository
  contains no weights. The reference values are therefore the notebook's printed scores (split 1, one model each), not
  checkpoints we re-score.
- **Gate:**
  - Binary-ERM passes if our 3-seed mean robust AUROC is within 0.03 of **either** the paper (0.359) **or** the notebook
    value (0.310);
  - Segmentation-ERM passes if it is within 0.03 of **either** the paper (0.727) **or** the notebook value (0.772);
  - verdicts require both to pass.
- **Reporting:** the paper, the notebook value and ours are reported side by side for both methods. For comparison with
  the notebook, our split-1 value is also shown on its own, since the notebook scores split 1 only.

**A14 (2026-11-17; only the experiments the claims need; author's decision; before any affected result).** Results seen when
this was recorded: P1 seed-1234 gate values and the P1 seed-1234 arms (cardiomegaly a/b/c; RSNA arm b 0.767). None
changes under A14.
1. **P1:** the remaining head-to-head seeds are cancelled (cardiomegaly Standard 3232/3221, RSNA ActDiff 3232/3221; none had
   started). Both P1 head-to-heads are **secondary** (global shortcut) and are reported as **"preliminary, 1 seed, no
   verdict"**. A3 item 3's 3-seed P1 verdict and section 2's comparisons 1–6 are therefore not computed; the Holm family
   of section 2 shrinks to P2 (3) + P3 (1) = 4 comparisons. **The P1 location test is kept** (Standard with the tube inside /
   outside; Masked scored from it, A5); it is P1's primary result.
2. **P4:** the P4 location test is cancelled; Saab's own overlap analysis (section 5.1) and phase 5 cover it.
3. **Phase 5:** one cohort, thyroid; Trap A, plus Trap B as the sanity check; 3 folds; arms = each paper's best method +
   ours. Other cohorts only if time remains after everything else, decided later as a new amendment.
4. Everything else is unchanged.

**A15 (2026-11-17, before any P1 location-test run; stricter clarification of section 5 / A12).** The P1 location test, in
P1's own pipeline (their code, Standard config, seed 1234):
- **Base data:** P1's balanced cardiomegaly split (No-SPC, `ratio = 0.5`: hospital independent of the label; their
  `JointDataset`, seed 1234), so the tube is the only induced shortcut. Images enter their network at 128 px, as coded.
- **Artifact:** our radio-opaque tube (`draw_tube` geometry, 43 × 6 px at 224 px = `TUBE_GEOMETRY`), drawn on the 224-px
  source image after their intensity normalisation and before their resize to 128 px. Intensities follow `draw_tube`
  (band: 0.35 × pixel + 0.65 × (pixel + 70/255 of the range); stripe at 245/255), mapped to their [−1024, 1024] scale.
- **Positions** (placement with `find_best_placements_for_mask` on the 224-px PSPNet lungs ∪ heart mask, p > 0.5):
  - inside: overlap with lungs ∪ heart ≥ 0.95;
  - **outside: overlap ≤ 0.02 with the union of lungs ∪ heart and P1's central circle** (their Masked mask: ellipse of
    radius 0.4 × side). "Outside" must also lie outside the mask of the method under test; otherwise Masked could never
    remove an "outside" tube and E-L1 could not be tested.

  Images where a position is infeasible are counted and reported, and keep their best placement (their split is not
  changed).
- **Presence** (stable hash of image, label, seed and environment): training 90% of positives / 10% of negatives; validation
  reversed (10/90), as P1's own design reverses validation; tests on P1's test split: correlated (90/10), reversed (10/90)
  and clean (none).
- **Methods:** Standard (their config), trained once per position (2 runs); Masked = the same model scored with test masking
  (A5); ours: arm a (mask_cmc on their network's circle-masked features, every split, artifact label = tube presence) and
  arm c (U-MtE, ROI = lungs ∪ heart).
- **Metric:** reversed-test AUROC (primary), each method's gain over Standard, the crossover (gain outside − gain inside) and
  clean AUROC (section 5). One seed: image-only bootstrap (A3).

**A16 (2026-11-17, before any arm result of the P1 location test; author's decision).** Results seen when this was recorded:
Standard and Masked, inside position (reversed AUROC 0.514 / 0.469; correlated 0.943 / 0.935; clean 0.866 / 0.837). None of
our arms has been computed for the location test.
- **All three of our arms** run at both positions of the P1 location test (A15):
  - a: mask_cmc on their trained Standard network's features of circle-masked images, every split;
  - b: DINOv2 ViT-B/14 @ 518 on lungs ∪ heart-masked images + mask_cmc (added back; A2/A12 had dropped it from phase 4);
  - c: U-MtE (ROI = lungs ∪ heart), label-free.
- For mask_cmc (arms a and b), the artifact label at training time is the pasted-tube indicator, known for every image.
  Mode and C are chosen on P1's validation split (reversed presence, A15) by AUROC.
- **Table:** reversed, correlated and clean AUROC per method and position; each method's gain over Standard; the crossover
  (gain outside − gain inside) with a 95% CI; and the paired difference vs Masked with a 95% CI. The bootstrap resamples
  test images only (one seed, A3); the test images are the same at both positions, so the crossover CI shares the image
  weights across positions. A column gives the share of tube pixels kept by P1's circle (inside 0.986, outside 0.000;
  `results/replication/viviano/location_tube_vs_circle.csv`).

**A17 (2026-11-18, POST HOC; author's decision; recorded before any epoch-100 model is scored).** Results seen when this
was recorded: the full registered P1 location table (A15/A16; `results/replication/viviano/location_1234.md`), including
the outside Standard result (correlated / reversed / clean 0.901 / 0.894 / 0.898). Their selection rule (`training.py`:
burn-in 5 epochs, then the lowest loss on P1's reversed validation split) kept epoch 11 inside and epoch 8 outside; outside,
the tube shortcut appeared only at epoch 11, so the kept outside model predates it (`docs/REPLICATION_STATUS.md`,
2026-11-17 22:20).
- **The registered table stays the primary result**, reported as is with this explanation; E-L1 to E-L3 are judged on it
  only.
- **Post-hoc sensitivity analysis, labelled "post hoc" on every number:** the same analysis on their saved
  `last_model_1234.pth.tar` (epoch 100, the model at the end of their training run) at **both** positions, so that the
  crossover compares models chosen by the same rule:
  - Standard (epoch 100) and Masked (= the epoch-100 model scored with test masking, A5);
  - arm a: mask_cmc on the epoch-100 network's features of circle-masked images (every split), mode and C chosen on P1's
    validation split by AUROC, exactly as registered for arm a;
  - arms b and c do not depend on their epoch selection; their registered predictions are reused unchanged.
- Same test images, presence, metrics, bootstrap (image-only, 10,000 replicates, shared weights across positions) and
  table layout as A16. No verdict and no Holm family membership: descriptive only. Output:
  `results/replication/viviano/location_1234_posthoc_last.{csv,md}`, scores
  `results/replication/viviano/scores/location_<position>_last_1234.csv.gz` and the arm-a file next to them.

**A18 (2026-11-18, after the P2 RR-ClArC result was seen; author's decision; standing rule "the paper wins over the code
unless git history shows the code produced the table").** Results seen when this was recorded: the six RR-ClArC
fine-tunes of section 3 (λ ∈ {10, 50} × 3 layers), selection λ = 10 / last_conv, test clean 0.840 / biased 0.338
(not reproduced; `results/replication/pahde/p2_reproduction.md`), and arm a (`arm_a.md`).
- **What was wrong.** Section 3 took λ ∈ {10, 50} from their config generator
  (`config_generator_mitigation_isic_attacked.py:92-105`), where the paper's other values are commented out, and says the
  paper leaves λ open. It does not: the paper tests λ ∈ {10¹, 5·10¹, …, 5·10¹²} (p. 31) and reports the chosen value,
  **λ = 5·10⁹ for RR-ClArC, ISIC2019, ResNet** (Tab. 12, p. 43). The paper also states the fine-tune learning rate:
  "the learning rate used during training (see Tab. 4) divided by 10" (p. 31), i.e. **1e-4** (Tab. 4, p. 24: Adam,
  0.001). Their code instead loads the Vanilla checkpoint's optimizer state, whose learning rate after their schedule is
  1e-5 (`utils/training_utils.py:13-16`). The repository has a single commit (4990898), so git history cannot show that
  the code produced the table; the paper wins.
- **Primary P2 RR-ClArC run (replaces section 3's grid for RR-ClArC):** our Vanilla's **weights only** (fresh Adam),
  **lr 1e-4, λ = 5·10⁹**, each of the 3 layers of their configs (identity_1, identity_2, last_conv), **10 epochs**
  (their configs; the paper is silent), all other settings from their configs. The layer is chosen on P2's validation
  split by the mean of clean and biased accuracy (section 3), and the gate (section 1, |biased − 0.76| ≤ 0.03) is applied
  to this run.
- **Reported alongside:** the six section-3 runs (their code path: λ ∈ {10, 50}, effective lr 1e-5 from the loaded
  optimizer state), with that explanation; not a verdict.
- Implementation: no file of theirs is edited. A copy of our Vanilla checkpoint with `model_state_dict` only makes their
  `get_optimizer` build a fresh Adam at the configured lr; the CAV activations of section 3 (same Vanilla) are reused.

**A19 (2026-11-18, before any A18 RR-ClArC model is scored; author's decision).** Results seen when this was recorded: the
section-3 RR-ClArC runs and arm a (as in A18); no A18 model has been scored. **Arm a'** = mask_cmc on top of the selected
A18 RR-ClArC network (their best model + our method):
- **identical to arm a except for the network**: the selected A18 RR-ClArC model (layer chosen on validation, A18) instead
  of our Vanilla. Input **unmasked**, as arm a (their dataset images as the network was trained: train "as trained",
  validation and test clean and biased); penultimate (global-average-pooled, 2048-d) features; the within-class artifact
  mean difference projected out (one δ, MEL; pooled and per_class coincide); class-balanced multinomial logistic
  regression refit on the training set; C ∈ {0.01, 0.1, 1, 10} chosen by the mean of clean and biased validation
  accuracy (section 3). Test scored once.
- **Head-to-head vs RR-ClArC** (the same selected A18 model) on P2's clean and biased tests: paired image-only bootstrap
  (10,000 replicates; one seed, A3), Δ = arm a' − RR-ClArC with its 95% CI. Also reported against our Vanilla.
- Note (write-up): arm a's selected C = 0.01 is the smallest value of the registered grid; any wider grid is post hoc
  only (sensitivity, no verdict). The same applies to arm a'.

**A20 (2026-11-18, before any arm-d run or result; author's decision; details in `docs/REPLICATION_PLAN.md`, "Arm d",
commit 112c75e).** **Disclosure:** arm d was added after, and because of, arm b's P2 clean result (clean 0.685 vs our
Vanilla 0.839; `results/replication/pahde/arm_b.md`): arm b keeps a frozen encoder, every competitor trains its network on
the task. Results seen when this was recorded: all registered P1 and P2 results and the P1 location test (registered and
A17 post hoc); no P3 or P4 head-to-head result.
- **Arm d = trained small encoder + mask_cmc.** torchvision ResNet-18, ImageNet weights, trained end-to-end on
  **ROI-masked images** (the same ROIs as arm b: P1 cardiomegaly and P1 location test lungs ∪ heart; P1 RSNA lungs;
  P2 lesion; P3 lungs; P4 lesion; phase 5 the cohort ROI), outside the ROI filled as in arm b.
- **One fixed recipe for every paper, nothing tuned on any test:** standard ERM on the paper's label task
  (cross-entropy; BCE where the paper is multi-label), the paper's input resolution and its own training augmentation,
  Adam lr 1e-4, batch 64, at most 50 epochs, early stopping on the paper's validation metric with patience 5 (the
  checkpoint with the best validation value is kept), seed 1234, mixed precision. Per paper: P1 128 px (their resize),
  validation AUROC; P2 224 px, their `isic_augmentation`, validation = mean of clean and biased accuracy (section 3);
  P3 224 px, their augmentation (P3 Supp. 5.1), validation macro-F1 on their biased validation; P4 512 px, their
  augmentation, validation AUROC; phase 5 as fixed in `docs/PREREGISTRATION_PHASE5.md`.
- **Head:** penultimate (512-d) features → mask_cmc with the same artifact labels and validation-chosen mode as arm a →
  linear head refit on the training set (class-balanced logistic regression, C ∈ {0.01, 0.1, 1, 10}, chosen by the
  paper's validation metric), on CPU. Test scored once.
- **Where:** the their-environment head-to-heads where mask_cmc applies (P1 cardiomegaly and RSNA, seed-1234 split;
  P2; P3 90/10 inside variant; P4 split 1), the P1 location test (inside and outside; A16 table layout), and phase 5
  (Trap A, 4 cohorts × 3 folds; Trap B thyroid), added to the phase-5 timing test and `PREREGISTRATION_PHASE5.md`.
- **Success rules:** the same as the other arms (sections 2, 5, 6); one seed, image-only bootstrap, stated in every
  table. Where a comparison has no verdict (P1 original tests A14, P2 A18 decision), arm d is descriptive too.
- **Scheduling:** after the registered critical-path runs; a ResNet-18 job may pair with ISNet only if the measured total
  throughput rises; checkpoint every epoch and resume.

**A21 (2026-11-18, before any arm-d result and before any phase-5 run; author's decision).** Results seen when this was
recorded: as in A20 (no arm-d run yet), plus the P2 arms a, a', b, c, which all chose C = 0.01, the smallest value of the
registered grid.
- **Arm d (A20), on P1–P4 and the P1 location test:** the mask_cmc / linear-head C grid is **{0.001, 0.003, 0.01, 0.03,
  0.1, 1, 10}** (replacing A20's {0.01, 0.1, 1, 10}), chosen on validation only; every table reports whether the
  chosen C is at an edge of this grid.
- **Phase 5:** the same widened grid for every arm that fits a mask_cmc or linear head, chosen on validation only, with the
  edge reported; fixed in `docs/PREREGISTRATION_PHASE5.md` before any phase-5 result.
- **Unchanged:** P2 arms a, a', b, c (and every other already-registered arm) keep the registered grid
  {0.01, 0.1, 1, 10}; their edge choice stays reported as is (any wider grid there is post hoc only).

**A20 correction (2026-11-18, before any arm-d run or result; author).** Arm d uses the resolution **their** network sees
for each task: P1 cardiomegaly 128 px (`JointDataset` default `new_size=128`), **P1 RSNA 224 px** (A20 wrongly said 128 for
all of P1; STATUS (d): RSNA images are built at 224 × 224), P1 location test 128 px (cardiomegaly pipeline), P2 224 px,
P3 224 px, P4 512 px (their loader resizes every image to 512 × 512 for Seg-ERM). **Phase 5:** arm d uses the same input
resolution, augmentation and early-stopping rule as the other phase-5 arms in `src/wtss/experiments/finetune.py`, so the
comparison stays identical; this is registered in `docs/PREREGISTRATION_PHASE5.md`, not as a separate choice (A20's own
recipe applies to P1–P4 and the P1 location test only).

**A20 encoder change (2026-11-18, before any arm-d run or result; author; plan commit b46bd26).** Arm d's encoder is
**torchvision EfficientNet-B0** (`efficientnet_b0`, IMAGENET1K_V1; 5.3 M parameters, 0.39 GFLOPs at 224 px), replacing
ResNet-18, so that it is smaller than every competitor in both parameters and compute (P3 DenseNet-121, 8.0 M, is the
smallest competitor by parameters). Penultimate features = the 1,280-d global-average-pooled output before the classifier.
Everything else in A20 (with its correction) and A21 is unchanged. Disclosure: a ResNet-18 P2 job had been launched at
13:46 UTC; it was stopped by PID while still waiting in the GPU launcher (empty log, no training step, no output) and its
directory was removed.

**A24 (2026-11-19, before any P3 inside-run result exists; author's decision; numbers A22–A23 are unused: A22 was proposed
and declined on 2026-11-18, not recorded).** Results seen when this was recorded: the A9 check (0.223 / 0.260), the corner
ISNet on the rebuilt test (0.427 / 0.385 / 0.387 primary), arms b, c, d on the P3 90/10 inside variant (rebuilt test). No
ISNet inside run has been scored.
- **Secondary, in-distribution evaluation** for every P3 location / 90-10 comparison (ISNet inside, ISNet 90/10 inside,
  arms b, c, d): the P3 **cohort-validation images** rendered in the biased / standard / deceiving conditions of each run's
  design (the run's mark position and presence rule; standard = no mark; deceiving = P3's swapped shapes), macro-F1, with
  the same image-only bootstrap as the primary results.
- **Labelled secondary everywhere:** these images were used for epoch and model selection (P3's validation split).
- The **rebuilt external test stays the registered primary test** and is reported unchanged.

**A27 (2026-11-18 12:20 UTC by machine 2's clock, before any machine-2 result; author's decision).** A second GPU machine
("machine 2") runs the P4 Segmentation-ERM runs, P5 and phase 5; machine 1 keeps P3 (the ISNet chain) and the exploratory
track. Results seen when this was recorded: everything in git at commit e562452 (P4 Binary-ERM splits 1–3, P4 arms b, c on
splits 1–3, P4 arm d split 1; no P4 Segmentation-ERM result).
- **Machine 2:** 1× NVIDIA GeForce RTX 5070 Ti, 16 GB (16,303 MiB), compute capability 12.0 (Blackwell), driver 595.91.07
  (max CUDA 13.2); Intel Xeon E5-2680 v4, 28 logical cores; 62 GB RAM; 288 GB container disk. **No persistent volume:**
  data, checkpoints and environments are lost if the instance is recycled or destroyed (only git survives).
- **Identical code, recipes and seeds:** the same repository commits and third-party commits (`third_party/*/SOURCE.md`),
  the same patches, torch 2.11.0+cu128 / torchvision 0.26.0 in every environment, the same scripts and command lines as
  on machine 1. Environments: ours (Python 3.12, `requirements.txt`), saab (Python 3.10; smp 0.3.4, albumentations 1.3.1),
  pahde (Python 3.11, their `requirements.txt` without `fast-slic`, which fails to build and is imported only by their
  SHAP/SpRAy analysis scripts, not by training, activation collection or RR-ClArC; ISNet's code runs in this
  environment, as on machine 1), r2r (Python 3.11, their `requirements.txt`). Exact package lists are committed with the
  first results (`docs/ENVIRONMENT_MACHINE2.md`).
- **P4 Segmentation-ERM split 1 restarts from scratch on machine 2.** Its epoch-66 checkpoint stays on machine 1 and is not
  used; splits 2 and 3 also run here. Different GPU and cuDNN kernels mean bitwise-different trajectories from machine 1;
  this is a hardware difference, not a recipe change. Every P4 Segmentation-ERM result, P4 arm a, the arm-d split-1
  paired comparison with Segmentation-ERM, and the P4 gate (A13 and addendum) come from machine 2.
- **Machine 1 keeps P3.** Machine 2 runs nothing for P3.
- Every STATUS entry from machine 2 is labelled "machine 2"; times are machine 2's UTC clock (it reads about 17 h behind
  the timestamps of machine 1's latest heartbeat commit, so cross-machine ordering is by git history, not by timestamp).

**A28 (2026-11-18 ~14:00 UTC by machine 2's clock; P6 added as a SECONDARY paper; author's decision; before any P6
correction run or arm result).** Results seen when this was recorded: P6 step 1 only — the released Vanilla ISIC
checkpoints scored with their code reproduce Table 1 for all three architectures (test accuracy, clean / biased: VGG-16
0.764 / 0.349 vs 0.762 / 0.349; ResNet-18 0.816 / 0.557 vs 0.818 / 0.568; EfficientNet-B0 0.839 / 0.625 vs 0.842 / 0.629;
`results/replication/rrclarc/step1.md`). P5 is dropped (step 1 failed); SpurBreast (MICCAI 2025) was examined and dropped
(dataset paper with no method to beat; its strong spurious splits are perfectly confounded with the label, so mask_cmc
cannot be estimated).
- **P6 = Dreyer, Pahde, Anders, Samek, Lapuschkin, "From Hope to Safety: Unlearning Biases of Deep Models via Gradient
  Penalization in Latent Space", AAAI 2024** (code `github.com/frederikpahde/rrclarc` @ a3e2d15; checkpoints from the Google
  Drive folder in its README). ISIC 2019, 8 classes, 80/10/10 split (`default_rng(0)`), synthetic **LSB** artifact (a text
  message written into the lowest 3 / 3 / 2 bits of every pixel) in a fraction of MEL images.
- **Secondary, like P1:** the LSB artifact is invisible and covers the whole image, so it is not a localized artifact and
  says nothing about inside vs outside the ROI. P6 shows only whether our constraint also removes a whole-image shortcut
  that a concept-level method targets. Its comparisons form their own Holm family, separate from the primary families.
- **Their best method: RR-ClArC.** Targets (Table 1, biased-test accuracy): VGG-16 0.570, ResNet-18 0.712,
  EfficientNet-B0 0.756 (clean 0.743 / 0.785 / 0.787). **Reproduced if |ours − published| ≤ 0.03 on biased accuracy**, per
  architecture; clean accuracy reported beside it.
- **Settings: the paper wins over the code** (standing rule). Paper Sec. 4.3 and App. A.3: fine-tune 10 epochs from the
  Vanilla checkpoint, cross-entropy, lr 1e-4, SGD (VGG-16, ResNet-18) / Adam (EfficientNet-B0), CAV at the last
  convolutional layer (`features.28` / `last_conv` / `last_conv`) with layers up to it frozen (their `Freeze` callback), gradient
  target = logits weighted by a random ±1 vector (their `all_logits_random`), `l2_mean`, `cavs_max`, `signal` direction, CAV on
  the training set; **λ as chosen in their Table A.3: VGG-16 5·10⁸, ResNet-18 10¹¹, EfficientNet-B0 5·10⁹.** Their released
  generator's grid stops at 10⁷ and does not contain these values (recorded discrepancy, as in P2); no grid is run.
  Artifact rate of the fine-tune data: the paper's p-bias (Table A.1) **8.0 % / 12.5 % / 8.0 %** of MEL (their correction
  configs say 10 %; recorded). One run per architecture (the paper states no seeds). Batch 64 (their local config).
- **Tests:** their clean and biased (trigger in every image of every class) test sets, each scored once; accuracy (paper's
  metric) and macro-F1.
- **Our arms** (artifact label at training time only = LSB inserted yes/no):
  - **a)** mask_cmc on their **Vanilla** network of each architecture: penultimate global-average-pooled features of the
    unmasked images as their network sees them, within-class artifact mean differences projected out (only MEL has both
    groups, so one δ), class-balanced multinomial logistic regression refit on the training set;
  - **a′)** the same on **our reproduced RR-ClArC** network of each architecture (A19 practice);
  - **b)** DINOv2 ViT-B/14, frozen, **at 224 px on the unmasked 224-px images** (224 = 16 × 14, so no resampling destroys the
    bit-level trigger before the encoder; P2's 518-px lesion-masked view would resample it away and test nothing), CLS token,
    L2-normalised, + the same mask_cmc head;
  - not run: arm c (U-MtE's generic overlays do not model a bit-level trigger) and arm d (their own EfficientNet-B0 Vanilla
    is already a trained small encoder; arm a on it fills that role).
  - Head grid: C ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 1, 10} (A21), chosen on validation; edge choices reported.
- **Validation rule:** their validation split scored clean and biased; selection metric = mean of the two accuracies (as P2,
  section 3). It chooses only our C (RR-ClArC has no open setting left).
- **"Beaten":** section 2 with one run per architecture (image-only paired bootstrap, 10,000 replicates; Holm over the
  P6 family = 3 arms × 3 architectures), on biased accuracy against **our** RR-ClArC of the same architecture; clean accuracy
  reported, with the registered clean-drop margin (0.01) flagged. If RR-ClArC is not reproduced for an architecture, that
  architecture's comparisons are descriptive (section 1).

**A29 (2026-11-18 by machine 2's clock; renumbering and priority only; author's decision; before any P7 correction run and
before any result of the new P6).** The paper registered in A28 (Dreyer, Pahde et al., RR-ClArC, AAAI 2024) is renamed
**P7**; everything in A28 stands for P7 unchanged. **P6** is now **EPVT (Yan et al., MICCAI 2023) with its journal version PLDG
(IEEE TMI 2025)**; its step 1 (read both papers, obtain their processed data, reproduce their ERM and EPVT once) runs now, and
its registration follows as **A30** (A28 is taken) only if EPVT reproduces. Priority on machine 2: P4 > P6 > P7. P6 runs
beside P4 only if a measured pairing does not slow P4; otherwise after P4 split 3. Results seen: as in A28, plus the P7
step-1 table (`results/replication/rrclarc/step1.md`, unchanged).

**A31 (2026-11-18 21:35 UTC by machine 2's clock; P4 port-error fix, not tuning; author's rule "fix a port error, re-score the
test once"; A30 stays reserved for P6).** Results seen when this was recorded: the P4 gate (Seg-ERM 0.609 / 0.611 / 0.676, mean
0.632; fail), arm a (0.485 mean), arm d split 1 vs Seg-ERM. Evidence: `docs/REPLICATION_STATUS.md` 21:35 entry, item 3.
- **Fix:** the 512-px cache is rebuilt as their `src/data/isic.py` does it: `imageio.imread` → `PIL.Image` →
  `torchvision.transforms.Resize([512, 512])` (PIL bilinear with antialiasing) for images; the same for Task 1 masks, then
  `ToTensor() > 0.5`. Stored in a new cache (`cache512_pil`); the cv2 cache and all runs on it are kept and reported as run.
- **Rerun:** Segmentation-ERM splits 1–3 with the unchanged registered protocol (A13 / Appendix A.1, `--grad_ckpt`, batch 16,
  fp32, 150 epochs, best validation Dice) on the fixed cache; **each test scored once**; the P4 gate (A13 + addendum) is
  decided on these three runs. Binary-ERM was trained on the same cv2 cache (machine 1): it passed the gate and is **not**
  rerun; this is stated next to the gate. Arm a is rerun on the fixed Seg-ERM encoders (same head, same grid). Arm d (machine
  1, cv2 cache) stays as run and is compared descriptively.
- **Not changed** (checked, not port errors): model selection, epochs, augmentation, learning rate, loss, scoring, test set.
- **Scheduling:** the three reruns follow the P6 bias-1 step-1 runs already in progress (≈ 7.6 GPU-h: two splits paired, then
  the third); P6 continues after them (P4 keeps priority).

**A32 (2026-11-19 by machine 2's clock; P8 = DISC registered; author's decision; before any P8 run and before any of our arms
is scored on trap sets 4 or 5).**
- **Paper:** Wu, Yuksekgonul, Zhang, Zou, "Discover and Cure: Concept-aware Mitigation of Spurious Correlation", ICML 2023
  (code github.com/Wuyxin/DISC @ d9865a8; `third_party/disc/SOURCE.md`). Benchmark: Bissoto et al. (2020) ISIC 2018 trap sets
  1–5 (train 1,822 / val 259 / test 772 each; artifact–label correlations reversed in each test).
- **Data identity (checked before registration):** DISC's released ISIC archive = our P4 data image for image (2,594 / 2,594
  JPEG files with identical MD5; trap-set CSVs 2–5 byte-identical, set 1 identical image ids, labels and artifact flags,
  only `.jpg` vs `.png` in `image`).
- **Disclosure — results seen before this amendment:** on trap sets 1–3 of this benchmark, P4 Binary-ERM (0.355 / 0.361 / 0.374),
  P4 Segmentation-ERM (cv2 cache: 0.609 / 0.611 / 0.676) and our arms: a (on Seg-ERM, 0.528 / 0.481 / 0.445), b (DINOv2 +
  mask_cmc, mean 0.658), c (U-MtE, mean 0.605), d (EfficientNet-B0 + mask_cmc, set 1 only, 0.666); and the published DISC
  (55.1) and CBDebug (58.0) numbers. Nothing on sets 4–5 has been seen. **Our arms are re-run UNCHANGED** — the registered P4
  settings (A13 item 8 flag rule: flags with ≥ 3 training images in each artifact × diagnosis group; lesion ROI = ISIC 2018
  Task 1 masks; outside = mean colour; validation-only selection of mode and C) — under DISC's protocol on **all five trap sets,
  including 4 and 5**, on the A31 (PIL-resized) 512-px cache:
  - **b** DINOv2 ViT-B/14 @ 518, lesion-masked, mask_cmc, C ∈ {0.01, 0.1, 1, 10} (as registered for b);
  - **d** EfficientNet-B0 on lesion-masked 512-px images (A20 recipe with its correction, A21 grid), retrained per trap set;
  - **a** mask_cmc on the reproduced **DISC** ResNet-50 of each trap set (penultimate 2,048-d features of lesion-masked images
    through their eval transform; same head and grid as P4 arm a);
  - **c** U-MtE (label-free), as P4.
- **Their protocol:** `scripts/isic.sh` as released (ERM: ResNet-50, lr 1e-3, batch 16, wd 1e-5, 100 epochs; DISC: lr 5e-4,
  ERM initialisation, color + texture concepts from their released concept bank, 3 clusters); `--seed` = trap-set id; per run
  the test AUROC at the epoch with the highest validation AUROC (their `run_expt.py`). **Aggregation: the mean over trap sets
  1–5 of the per-set AUROC** (DISC p. 6: "average AUROC … across the train-test splits, as standard in Bissoto et al. (2020)";
  Bissoto 2020 Table 3: 5 fixed splits, mean ± SD). One run per trap set; DISC's "repeated three times" cannot be mapped
  onto their code (seed = trap-set id) and is recorded as an ambiguity.
- **Gate:** reproduced if |ERM − 0.364| ≤ 0.03 **and** |DISC − 0.551| ≤ 0.03 (5-set means). Verdicts only if both pass;
  otherwise P8 is descriptive.
- **Head-to-head:** arms **b and d (primary)**, a and c (secondary) vs **our reproduced DISC**, on DISC's metric (5-set mean
  AUROC), paired on the same test images of each set. **Image-only bootstrap**: 10,000 replicates, one Poisson(1) weight per
  image id shared across the trap sets in which it appears and across both methods; trap sets fixed (not resampled); per-set
  AUROC recomputed and averaged. "Beaten" = section 2 (higher mean, 95 % CI of the paired difference above 0, Holm-adjusted
  p < 0.05); Holm family = the 4 P8 comparisons. Per-set results reported beside the mean.
- **Published rows (descriptive, not reproduced):** DISC Table 1 baselines (ERM 36.4 … LISA 38.0) and **CBDebug (Enouen &
  Galhotra, ICLR 2026) 58.0 ± 11.6** on the same benchmark, labelled "published number only — not reproduced: released code
  has no ISIC experiment (R4)"; no claim is made against it.

**A30 (machine 1, 2026-11-20, before any re-score; port bug in our P3 arm code, recorded like A7; author's correctness
check).** The author asked for a correctness check because arm c on the P3 corner benchmark was below chance on validation
(macro-F1 0.284). **Bug:** `scripts/replication/armd_train.py`, `P3.dataset()` built P3's `Data` with the **full**
cohort image and lung-mask arrays but the split's ids, labels and marks; P3's own trainer (`p3_train.py`) replaces the arrays
with the split's subset after building `Data`, our adapter did not. Training and validation images were therefore
misaligned with their labels (verified: 201 of 201 sampled training and 204 of 205 validation images differed from their
own cohort image; the test sets were aligned because there the subset is the whole array). **Affected (all our P3 arms):**
arm d on the 90/10 inside variant (registered result and the post-hoc per_class row), arms b and c on the 90/10 inside
variant, arm c on the corner benchmark. **Not affected:** every ISNet run (P3's own code), the A9 check, the A24 ISNet
evaluations, all non-P3 results. **Fix:** pass the split's subset arrays (`gray[idx]`, `roi[idx]`), with an alignment
assertion. **Re-scoring:** arm c on the corner benchmark is re-extracted and re-scored once (authorised); the other affected
results are marked invalid and wait for the author's decision. The invalid results files are kept, labelled.
**A33 (2026-11-19 by machine 2's clock; P6 port-error fix, not tuning; author's diagnosis rule; recorded before any rerun; A30 is now machine 1's P3 entry, so a P6 registration, if ever made, takes the next free number).**
Results seen: P6 bias 1 split 1 with the wrong step count: ERM 0.690, EPVT 0.654 (targets 0.653 / 0.746); P8 not yet gated.
- **Error:** our P6 step count assumed DomainBed's 20 % per-domain holdout; their trainer trains on each whole domain and
  defines an epoch as `int(min domain size / 26)` steps (`train_epvt.py:129–163, 310`). Evidence that this is their meaning:
  their README's 1,501 steps at bias 0 = 60 epochs × 25 + 1 (the paper's 60 epochs). Domains, backbone, preprocessing and data:
  no port error (STATUS 09:10, items a and c).
- **Fix:** 100 epochs = 100 × int(min |domain| / 26) + 1 steps (bias 1, split 1: 1,301), early stopping patience 22 in the same
  unit, everything else unchanged (A29 decisions; activation checkpointing; README hyper-parameters).
- **Rerun:** ERM and EPVT at bias 1, split 1, seed 0, **once**, after P8's gate; same scorer (5 domains, as their test script).
  Gate unchanged (within 0.03 of the digitised 65.3 / 74.6). If EPVT still fails, P6 is final "not reproduced" (no further
  reruns; the 5-seed run applies only if no port error had been found) and an email to the authors is drafted.
