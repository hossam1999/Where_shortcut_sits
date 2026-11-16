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
