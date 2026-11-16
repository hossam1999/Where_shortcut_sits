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
