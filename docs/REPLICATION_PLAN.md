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

## The principle: beat each paper in two environments (decided 2026-11-17)
Every target must be beaten twice. Both are required for a claim of "better than paper X".
1. **Their environment:** their data, their splits, their test design and their metric. This is phase 2
   (reproduce) and phase 3 (head-to-head). Phase 4 is a bridge between the two environments: their data, with the
   artifact moved inside the ROI.
2. **Our environment:** our data and our test design, run on our cohorts with their methods. This is **phase 5**.
   - **Our data:** ISIC 2019 hair / ruler traps, thyroid and ovary calipers, capsule debris, NIH chest tubes.
   - **Our test design:** correlated, reversed and clean tests; Trap A (artifact inside the ROI) vs Trap B (outside);
     the transplant; the unaltered test sets.

### Phase 5 sketch (pre-register before any phase-5 run; start after P1–P3 phases 2–3)
- **Where:** our end-to-end pipeline, `src/wtss/experiments/finetune.py`. It already runs `erm`, `mask`, `balanced`,
  `mte_post`, `umte_ft` on our traps with ResNet-50 and folds. Their methods are added as new arms, ported from
  their code with the same loss definitions and their published hyperparameters, and no tuning on our test sets.
  - **P1 Viviano:** Masked (already = our `mask` arm), ActDiff, and GradMask if it is cheap. The ROI is our lesion,
    nodule or organ mask.
  - **P2 Pahde:** RR-ClArC, with the artifact concept direction learned from our image-level artifact labels.
  - **P3 ISNet:** LRP relevance penalty outside our ROI masks. It is the most expensive arm (LRP per step).
  - **Saab (if adopted as P4):** segmentation-supervised ERM, with our ROI masks as the segmentation target.
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
  comparison with it is the decisive one.
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
