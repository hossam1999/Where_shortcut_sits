# Replication plan: three papers to reproduce and beat

Agreed after the November 2026 supervisor meeting. Each paper is compared **on its own test, metric and data** (a fair,
like-for-like number). Then all of their methods and ours are run on our inside-vs-outside location test.

| # | Paper | Venue | Their test and metric | Number(s) to beat |
|---|---|---|---|---|
| 1 | Viviano et al., *Saliency is a possible red herring when diagnosing poor generalization* | ICLR 2021 (main track) | Chest X-ray (PadChest + NIH). Hospital source linked to the label 90/10 in training, reversed at test. ROC-AUROC, 10 seeds, ResNet-18 at 224 px | Cardiomegaly: Masked **0.77 ± 0.09** (standard 0.70, GradMask 0.74, ActDiff 0.73, RRR 0.57). Pneumonia, AP/PA view: ActDiff **0.68 ± 0.05** (masked 0.56, standard 0.20) |
| 2 | Pahde et al., *Ensuring medical AI safety* | Machine Learning 114:206, 2025 | ISIC 2019, 8 classes, synthetic artifact in 20% of melanomas. "Biased" test = artifact in all classes. **Accuracy**, ResNet-50 | RR-ClArC **0.76** (RRR 0.51, vanilla 0.28); clean 0.86 |
| 3 | Bassi et al., *Improving deep neural network generalization and robustness to background bias via LRP optimization* (ISNet) | Nature Communications 15:291, 2024 | COVID-19 / pneumonia / normal chest X-rays. A class mark in the upper-left corner (outside the lungs) in training. Matched, cyclically reversed and mark-free tests. **Macro-F1**, DenseNet-121 | ISNet **0.775** on all three tests; lungs-only U-Net 0.618 on all three; plain DenseNet 0.775 / 0.434 (clean) / 0.195 (reversed). External tuberculosis test, AUROC: ISNet 0.809 |

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

## Still to verify (needs arxiv / OpenReview / GitHub access or the GPU machine)
1. **Viviano:** is the mask a central circle or a lung mask? The exact limitation quote. Whether the split files are in the repo.
2. **PadChest access:** whether the downsized TorchXRayVision copy is enough, to avoid the ~1 TB download.
3. **Pahde:** where the inserted artifact sits relative to the lesion. Whether the split files are released.
4. **ISNet:** whether the code still runs. Which COVID datasets are still downloadable. The exact mark positions.
