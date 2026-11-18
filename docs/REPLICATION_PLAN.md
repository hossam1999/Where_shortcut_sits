# Replication plan: three papers to reproduce and beat

Agreed after the November 2026 supervisor meeting. Each paper is compared **on its own test, metric and data** (a fair,
like-for-like number). Then all of their methods and ours are run on our inside-vs-outside location test.

| # | Paper | Venue | Their test and metric | Number(s) to beat |
|---|---|---|---|---|
| 1 | Viviano et al., *Saliency is a possible red herring when diagnosing poor generalization* | ICLR 2021 (main track) | Cardiomegaly: chest X-ray (PadChest + NIH, 224 px), hospital source linked to the label 90/10 in training, reversed at validation and test; central **circular** mask. Pneumonia: **RSNA Pneumonia**, AP/PA view linked 90/10; mask = pneumonia boxes. ROC-AUROC, 10 seeds, ResNet-18 at 224 px | Cardiomegaly (Table 1): Masked **0.77 ± 0.09** (standard 0.70 ± 0.05, GradMask 0.74 ± 0.03, ActDiff 0.73 ± 0.03, Adversarial 0.68 ± 0.07, RRR 0.57 ± 0.09). Pneumonia, AP/PA view (Table 2): ActDiff **0.68 ± 0.05** (masked 0.56 ± 0.16, Adversarial 0.62 ± 0.10, GradMask 0.49 ± 0.06, RRR 0.21 ± 0.04, standard 0.20 ± 0.03) |
| 2 | Pahde et al., *Ensuring medical AI safety* | Machine Learning 114:206, 2025 | ISIC 2019, 8 classes, synthetic "microscope" vignette (black outside a centred circle) in 20% of melanomas. "Biased" test = artifact in all classes. **Accuracy**, ResNet-50 (timm `resnet50d.a3_in1k`), one run, no spread reported | Table 1, biased: RR-ClArC **0.76** (RRR 0.51, vanilla 0.28); clean: RR-ClArC 0.86, RRR 0.84, vanilla 0.87 |
| 3 | Bassi et al., *Improving deep neural network generalization and robustness to background bias via LRP optimization* (ISNet) | Nature Communications 15:291, 2024 | COVID-19 / pneumonia / normal chest X-rays. A class mark in the upper-left corner (outside the lungs) in training. Matched, cyclically reversed and mark-free tests. **Macro-F1**, DenseNet-121 | Table 1 (biased / standard / deceiving, all ± ~0.01 Bayesian SD over test images, one model each): ISNet **0.775** on all three; lungs-only U-Net 0.618 / 0.619 / 0.618; plain DenseNet 0.775 / 0.434 / 0.195. The test is the o.o.d. set (BIMCV COVID-19+ + ChestX-ray14 pneumonia + Montgomery/Shenzhen). External tuberculosis test, AUROC: ISNet 0.809 |

## How we compete
- **How our method applies:** our class-conditional constraint needs an image-level label for the shortcut during training only. That label is:
  - the hospital or the view (Viviano);
  - the inserted artifact (Pahde);
  - the corner mark (ISNet).
- **The label-free variant (U-MtE)** is reported beside it.
- **Report in their metric.** AUROC for Viviano, accuracy for Pahde, macro-F1 for ISNet, plus AUROC for all three. Their metric is used for the head-to-head number; ours is extra.

## Honest expectations
- **Viviano:** the masked spread is large (± 0.09), so we need ≥ 10 seeds and confidence intervals. The pneumonia view task is the best showcase, because the view changes the whole chest, the lungs included.
- **Pahde:** RR-ClArC is the closest method to ours. The difference to stress: theirs removes the artifact direction for all images; ours removes it within each class, after masking, with nothing needed at test time.
- **ISNet:** on its *original* test the mark is outside the lungs, and ISNet is already unaffected (0.775 everywhere). We can at best match it there. The real comparison is **their benchmark with the mark moved inside the lungs**: same data, code and metric, where a penalty outside the mask cannot help. The external tuberculosis AUROC (0.809) is a secondary target.

## Lean scope: only each paper's best method, only necessary runs (decided 2026-11-17; amendment A12)
Every run must feed a "beat it" comparison. If no comparison needs a run, the run is dropped. Runs already finished
are kept and reported; runs not yet started are cancelled.

| Paper | Best method (the one we beat) | Their env.: runs still needed | Location test: runs still needed | Our env. (phase 5) |
|---|---|---|---|---|
| P1 Viviano, cardiomegaly | **Masked**, 0.77 (scored from the Standard model, A5) | Standard seeds 3232, 3221 (Masked is scored from them at no cost) | Standard with the tube inside / outside the lungs + heart ROI: **2 runs**, Masked scored from them | Masked is already our `mask` arm: **no new runs** |
| P1 Viviano, pneumonia | **ActDiff**, 0.68 (Masked leaks the label, A11) | ActDiff seeds 3232, 3221 | none | ActDiff arm, Trap A only |
| P2 Pahde | **RR-ClArC**, 0.76 accuracy | Vanilla (RR-ClArC starts from it) + RR-ClArC fine-tunes. Use the paper's reported config; run their grid only if the paper does not give one | none | RR-ClArC arm, Trap A |
| P3 ISNet | **ISNet**, 0.775 macro-F1 | ISNet, original corner benchmark: **1 run** | ISNet with marks inside the lungs: (a) P3's design, (b) 90/10 so that mask_cmc applies: **2 runs** | ISNet arm, Trap A |
| P4 Saab (MLHC 2022), **ISIC only** (A13) | **Segmentation-ERM**, robust AUROC 0.727 ± 0.016 (3 seeds); binary ERM 0.359 ± 0.016 as the reproduction check | Port of their code (Appendix A.1 protocol); Binary-ERM + Segmentation-ERM, 3 seeds, on the public 772-image test (Saab's own notebook scores 772; the PDF's 513 is reported as an inconsistency); verdicts only if both reproduce within 0.03 | One fixed artifact moved inside vs outside the lesion, Segmentation-ERM + ours, 1 seed | Segmentation-supervised arm on our ROI masks, Trap A |

**Cancelled:**
- P4: the chest X-ray part (no SIIM access, unknown val/test membership, no training tube labels; our NIH cohort covers chest tubes).
- P1: RSNA Standard and RSNA Masked seeds 3232/3221; cardiomegaly ActDiff (all seeds); ActDiff in the cardiomegaly
  location test.
- P2: the RRR fine-tunes.
- P3: the DenseNet baseline and the 90/10 "outside" run. P3's own Table 1 is the outside reference.

**P1 gate:** checks only the runs we keep (cardiomegaly Standard, RSNA ActDiff, and RSNA Standard seed 1234, which is
already done).

**Our arms:** each runs on top of the paper's best network:
- P1 cardiomegaly: the Standard/Masked model;
- P1 pneumonia: the ActDiff model;
- P2: the Vanilla network (RR-ClArC's base);
- P3: no network of theirs is needed (DINOv2 + mask_cmc, U-MtE).

**Arm d: trained small encoder + mask_cmc (added 2026-11-18, amendment A20; registered before any arm-d result).**
- **Why:** every competitor trains its whole network on the task data; arm b keeps DINOv2 frozen and loses clean
  accuracy where the task needs task-specific features (P2: clean 0.685 vs Vanilla 0.839). Arm d answers whether the
  constraint still works when our encoder is trained like theirs. Disclosed as motivated by arm b's P2 clean result.
- **Encoder: EfficientNet-B0, ImageNet weights (torchvision `efficientnet_b0`, IMAGENET1K_V1), 5.3 M parameters,
  0.39 GFLOPs at 224 px** (changed from ResNet-18 on 2026-11-18, before any arm-d run). It is smaller than every
  competitor in both parameters and compute: P1 ResNet-18 11.7 M / 1.8 GFLOPs, P3 DenseNet-121 8.0 M / 2.9 GFLOPs
  (the smallest competitor by parameters), P2 ResNet-50 25.6 M / 4.1 GFLOPs, P4 SE-ResNeXt50 U-Net > 25 M. ResNet-18
  was dropped because it is larger than DenseNet-121 in parameters. The smallest DINOv2 (ViT-S/14, 22 M) is larger
  than both, so it is not used. Penultimate features = the 1,280-d global-average-pooled output before the classifier.
- **Resolution:** the resolution their network sees for each task (P1 cardiomegaly 128 px as coded, P1 RSNA 224 px,
  P2 224, P3 224, P4 as their code feeds Seg-ERM); in phase 5, the same input resolution, augmentation and
  early-stopping rule as the other phase-5 arms in `finetune.py`.
- **Training (one fixed recipe for every paper, nothing tuned on a test):** ROI-masked inputs (the same ROIs as arm b),
  the paper's label task, standard ERM (cross-entropy, or BCE where the paper is multi-label), the
  paper's own augmentation, Adam lr 1e-4, batch 64, up to 50 epochs, early stopping on the paper's validation metric
  (patience 5), seed 1234, mixed precision. Then penultimate features → mask_cmc (same artifact labels and mode
  selection as arm a; C ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 1, 10} chosen on validation, A21) → linear head
  refit on the training set, on CPU.
- **Where:** each head-to-head on the paper's own test where mask_cmc applies (P1 cardiomegaly and RSNA, P2, P3 90/10
  inside variant, P4 split 1), the P1 location test (tube inside and outside), and phase 5 (Trap A, 4 cohorts, 3 folds;
  Trap B thyroid). Same success rules as the other arms; 1 seed, image-only bootstrap, said in every table.
- **Priority:** after the registered runs on the critical path; EfficientNet-B0 jobs are the "small job" that may pair with
  ISNet (rule 3), only if measured throughput rises.

**Phase 5, lean:**
- their best method vs ours, **Trap A only** (where the claim is made);
- the 4 registered cohorts, 3 folds;
- **Trap B for thyroid only**, as a sanity check that their method works when the artifact is outside;
- standard and masked results come from the existing thesis runs (no re-running).

### Running it smartly
1. **GPU only for training.** Feature extraction is batched once per trained network and cached. Every head fit
   (mask_cmc, U-MtE heads, bootstraps) runs on CPU, in parallel across the 96 cores, never on the GPU.
2. **Pack by measured throughput, not by job count.** Before each pairing, measure images/s for 1 vs 2 jobs. Keep the
   pairing only if total throughput rises. Three jobs at once slowed P2 by about 2.5×.
3. **Pair heavy with light.** ISNet (10.4 GB) runs only beside one small job (P1 ResNet-18 or a P2 fine-tune).
4. **Critical path first.** The longest chains start first (P3 ISNet ×3, P2 Vanilla). Short jobs fill the gaps.
5. **Checkpoint every few epochs and resume after a crash.** The P2 crash lost 34 epochs.
6. **Reuse, do not recompute:** cached DINOv2 features, the thesis's saved predictions, their released weights where
   valid.
7. **Phase 5 only:** fp32 (changed from mixed precision on 2026-11-18, author: LRP is precision-sensitive and the
   networks are small) and the same early-stopping rule for every network (patience 5, at most 48 epochs). Both are
   registered before the first run and applied identically, so the comparison stays fair. Their environment keeps their
   exact recipes.

## The principle: beat each paper in two environments (decided 2026-11-17)
Every target must be beaten twice. Both are required for a claim of "better than paper X".
1. **Their environment:** their data, their splits, their test design and their metric. This is phase 2
   (reproduce) and phase 3 (head-to-head). Phase 4 is a bridge between the two environments: their data, with the
   artifact moved inside the ROI.
2. **Our environment:** our data and our test design, run on our cohorts with their methods. This is **phase 5**.
   - **Our data:** ISIC 2019 hair / ruler traps, thyroid and ovary calipers, capsule debris, NIH chest tubes.
   - **Our test design:** correlated, reversed and clean tests; Trap A (artifact inside the ROI) vs Trap B (outside);
     the transplant; the unaltered test sets.

### Phase 5, minimal scope (author, 2026-11-18; replaces the sketch below where they differ)
- **Cohort:** thyroid only (TN3K, real calipers). **Trap A and Trap B, 3 folds each.**
- **Their methods:** RR-ClArC (P2) and ISNet's LRP penalty outside our ROI (P3).
  RR-ClArC: λ ∈ {5·10⁸, 5·10⁹, 5·10¹⁰} chosen on validation, lr = training lr / 10, fresh optimizer, layer and epochs per
  their configs (A18 practice).
- **Ours:** arm d (EfficientNet-B0 trained on ROI-masked images + mask_cmc, A20/A21; the same resolution, augmentation
  and early-stopping rule as the other phase-5 arms), plus the thesis's arms on the same folds: DINOv2 + mask_cmc refit
  on CPU with the A21 C grid, and U-MtE. ERM and mask from the thesis's runs.
- **Dropped, and why:** ActDiff (P1), Segmentation-ERM (P4), and the ovary, capsule and ISIC-hair cohorts. P1 and P4 are
  compared with us in their own environments only. RR-ClArC and ISNet are the two methods that could plausibly work when
  the artifact lies inside the ROI (RR-ClArC acts on a concept direction, not on location; ISNet is the strongest
  location-based method), so they carry phase 5; one cohort with real in-ROI markers keeps the cost bounded.
- **Decisions (author, 2026-11-18):**
  - RR-ClArC base: **ResNet-50 (their architecture, timm `resnet50d.a3_in1k`, as P2's paper states on p. 22)**, an ERM base
    trained under the phase-5 rule below; their layer names unchanged. Reason: their chosen λ depends strongly on the
    architecture (ResNet 5·10⁹, VGG 10¹⁰, ViT 10¹), so on ConvNeXt the grid {5·10⁸, 5·10⁹, 5·10¹⁰} would have no basis;
    every competitor then runs on its published architecture (ISNet on DenseNet-121). The thesis's ConvNeXt-Tiny ERM
    stays as the plain baseline row.
  - **Epoch rule, identical for every phase-5 network** (RR-ClArC's ResNet-50 ERM base, ISNet, arm d's encoder): early
    stopping on validation (patience 5), at most 48 epochs (P3's count), so ISNet is not under-trained. RR-ClArC's own
    fine-tune keeps their 10 epochs. Our DINOv2 arms are unchanged (frozen encoder, CPU heads).
  - **Precision: fp32 for every phase-5 network** (no mixed precision: LRP is precision-sensitive and the networks are
    small).
  - ERM checkpoints of the thesis do not exist (never saved); the thyroid cohort is rebuilt on CPU (approved).
  - **One selection criterion for every tuned choice in phase 5, identical for their methods and ours: AUROC on the
    thesis's `val_clean`** (the inner validation, 20% of the training groups, with the marker present in 50% of each
    class, so it carries no label information; `src/wtss/data/isic2019_spec.py` RATES). This is the criterion the
    thesis's own arms used (`wtss.heads.select_C`, called with `val_clean` in `src/wtss/experiments/spec_traps.py`; the
    decision threshold is also set on `val_clean`). It decides early stopping of every network (RR-ClArC's ResNet-50 ERM
    base, ISNet, arm d's encoder), RR-ClArC's λ and layer, and mask_cmc's mode and C (A21 grid).
- About 18 networks plus the RR-ClArC λ fine-tunes. A timing test on Trap A fold 1 with all three methods comes first,
  then `docs/PREREGISTRATION_PHASE5.md` and the updated GPU-hour estimate for the author's approval.

### Phase 5 sketch (pre-register before any phase-5 run; start after P1–P3 phases 2–3)
- **Where:** our end-to-end pipeline, `src/wtss/experiments/finetune.py`. It already runs `erm`, `mask`, `balanced`,
  `mte_post`, `umte_ft` on our traps with ResNet-50 and folds. Their methods are added as new arms, ported from
  their code with the same loss definitions and their published hyperparameters, and no tuning on our test sets.
  - **P1 Viviano:** Masked (already = our `mask` arm), ActDiff, and GradMask if it is cheap. The ROI is our lesion,
    nodule or organ mask.
  - **P2 Pahde:** RR-ClArC, with the artifact concept direction learned from our image-level artifact labels.
    **Decided 2026-11-18 (author):** phase 5 now carries the decisive RR-ClArC comparison (thyroid Trap A, both
    methods run by us under identical conditions), because P2's own result was not reproduced (A18: 0.524 vs 0.76).
    RR-ClArC is ported with **the paper's stated settings**, not the released code's defaults:
    - λ from the paper's RR-ClArC grid {10¹, 5·10¹, …, 5·10¹²} (p. 31) as far as it applies to our cohort, selected on
      validation; the ISIC2019 ResNet value 5·10⁹ (Tab. 12) is in it. The exact grid is fixed in the phase-5
      pre-registration after the timing test;
    - learning rate = our ERM training learning rate / 10 (p. 31);
    - fine-tuning from the trained ERM network's **weights only, with a fresh optimizer** (not the decayed optimizer
      state their code loads, A18);
    - the layer(s) and number of epochs are not stated in the paper: their configs' last 3 ResNet blocks and 10 epochs,
      the layer chosen on validation as in A18;
    - CAV settings as in their configs (`cavs_max`, `signal`, `all_logits_random`), CAV computed on the training set.
  - **P3 ISNet:** LRP relevance penalty outside our ROI masks. It is the most expensive arm (LRP per step).
  - **P4 Saab:** segmentation-supervised ERM (their best method), with our ROI masks as the segmentation target.
  - **Ours:** mask_cmc on the fine-tuned features, U-MtE (`mte_post`, `umte_ft`).
- **Scope, to keep cost bounded:**
  - cohorts: thyroid, ovary, capsule and ISIC 2019 hair (the four registered real-trap cohorts);
  - traps: Trap A and Trap B;
  - 3 folds and 1 seed per fold;
  - tests: correlated, reversed and clean, plus the unaltered test sets where they exist.
- **Success rule:** the same as phase 3. Reversed-test AUROC is primary; the paired difference vs each paper's method
  must have a 95% CI above 0 (crossed bootstrap over folds × images); Holm across all phase-5 comparisons. Clean
  AUROC must not drop by more than the registered margin (0.01).
- **Expected message:** methods that act by location (Masked, ActDiff, ISNet, and Saab's segmentation supervision)
  help in Trap B but not in Trap A. Ours helps in both. RR-ClArC is the strongest competitor in Trap A, and the
  comparison with it is the decisive one; after A18 it is carried by phase 5 (thyroid Trap A), not by P2's own test.
- **C grid (A21, 2026-11-18):** every arm that fits a mask_cmc or linear head uses C ∈ {0.001, 0.003, 0.01, 0.03, 0.1,
  1, 10}, chosen on validation only; tables report whether the choice hits an edge. Arm d (A20) is in phase 5 and in the
  timing test.
- **Cost:** not yet estimated. A timing test on one cohort (thyroid, Trap A, 1 fold, every arm) comes first.

## How to frame each paper's result (decided 2026-11-17)
**P1 Viviano is the odd one out.** Its shortcut is *global*, not a localized artifact:
- the hospital the X-ray came from (cardiomegaly);
- the X-ray view, AP or PA (pneumonia).

Such a signal is spread over the whole image, inside and outside the mask at once. So "inside vs outside the ROI",
the thesis question, does not apply to it.
- **P1 head-to-head on their original test → SECONDARY.** It only shows that our method also copes with a global
  shortcut. Label it "global shortcut (hospital / AP-PA view), not a localized artifact".
- **P1 phase 4 → the P1 result that addresses the thesis.** It is the location test: a localized radio-opaque tube
  placed inside vs outside the lungs + heart ROI, in their pipeline, with their methods (Standard, Masked, ActDiff)
  against ours.
- **Why P1 stays in the plan anyway:**
  1. its limitation section names our problem (confounders inside the mask are not removed);
  2. it uses the same reversed-test design (90/10 → 10/90, AUROC);
  3. its methods are the "fix it by location" family that the thesis predicts fails for in-ROI artifacts.
- **P1 pneumonia "Masked" is label-dependent at test time** (A11). The mask is the pneumonia boxes, so every negative
  becomes one constant image. It is reported as a finding about the paper, not as a target. The pneumonia target is
  ActDiff (0.68).

**P2 Pahde and P3 ISNet → PRIMARY head-to-heads.** Both use localized artifacts (an inserted vignette; corner marks).

**P3 phase 4 (marks inside the lungs) → the strongest location result**, because ISNet's own design assumes the mark
lies outside the lungs.

## Then: our location test with every method
On the same images, the artifact goes fully outside, on the border and fully inside the ROI, each with correlated, reversed and clean tests. Methods compared:
- standard model and masking;
- RRR, GradMask, ActDiff (Viviano);
- RR-ClArC (Pahde);
- ISNet (Bassi);
- our constraint and U-MtE.

## Verified on 2026-11-16 (details, page and table numbers: `docs/REPLICATION_STATUS.md`)
Answers: Viviano's cardiomegaly mask is a central circle; splits come from a fixed seed (no files); the 224-px PadChest is what they used. Pahde's artifact is a centred vignette (outside the lesion); splits are deterministic (rng seed 0). ISNet's repo has no COVID-19 pipeline; marks are white shapes, 10% of the image, upper-left corner; the test set needs BIMCV COVID-19+.

### Original open questions
1. **Viviano:** is the mask a central circle or a lung mask? The exact limitation quote. Whether the split files are in the repo.
2. **PadChest access:** whether the downsized TorchXRayVision copy is enough, to avoid the ~1 TB download.
3. **Pahde:** where the inserted artifact sits relative to the lesion. Whether the split files are released.
4. **ISNet:** whether the code still runs. Which COVID datasets are still downloadable. The exact mark positions.

## BIMCV COVID-19+ access for P3 (found 2026-11-15)
- **The problem:** IEEE DataPort needs a subscription for the Extension entries.
- **Free alternative:** the dataset owners' own mirror at the Barcelona Supercomputing Center, linked from
  github.com/BIMCV-CSUSP/BIMCV-COVID-19 (`BIMCV-COVID19+/Readme.md`):
  - Share: `https://b2drop.bsc.es/index.php/s/BIMCV-COVID19-cIter_1_2_3` (iterations 1+2+3, a superset of what P3
    could have used). It is a Nextcloud public share; the owners recommend WebDAV for bulk downloads.
  - The data is split into **571 separate .tgz archives, each with a `*.tar-tvf.txt` listing of its contents**, plus
    `covid19_posi_metadata.tar.gz`, `covid19_posi_sessions_tsv.tar.gz`, `covid19_posi_derivative.tar.gz` and
    `sha1sum.txt` / `sha1sum-b.txt`. So we can read the listings, then download **only the archives that hold the
    images P3 used**, not the whole dataset.
- **Licence:** the BIMCV-COVID19 Research Use Agreement. Free for academic research; no sale; cite the dataset; do
  not redistribute the images.
- **Not checked yet:** the mirror's links could not be reached from the analysis session (network policy). The GPU
  machine must check that they still work.
- **The image list:** the P3 repo (github.com/PedroRASB/ISNet) has no list of the 1,515 BIMCV images and no data
  scripts; its README points to the paper. Get the selection rule from the paper's Methods and supplement, or ask the
  authors for the image IDs.
- **Pre-trained weights:** P3's trained ISNet weights for COVID-19 are on Google Drive (link in the P3 README). Use
  them to check our reproduction.
- **Not a substitute:** the Kaggle "COVID-19 Radiography Database" contains ~2,473 BIMCV images, but renamed and
  processed, so they cannot be matched to P3's test.
