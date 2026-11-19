# Draft email to Saab et al. (not sent; the author sends it)

**Sent 2026-11-19 by the author to ksaab@stanford.edu, Cc dlrubin@stanford.edu, chrismre@stanford.edu (signed "Hossam Elkady"). Answer, when it arrives, is recorded in docs/REPLICATION_STATUS.md.**

**To:** Khaled Saab <ksaab@stanford.edu> (first author)
**Cc (suggested):** Daniel Rubin <dlrubin@stanford.edu>, Christopher Ré <chrismre@stanford.edu> (senior authors)
*Addresses exactly as printed on p. 1 of the MLHC 2022 paper (PMLR 182:1–25). The paper does not mark a corresponding author;
it prints an address for every author: Khaled Saab ksaab@stanford.edu, Sarah Hooper smhooper@stanford.edu, Mayee Chen
mfchen@stanford.edu, Michael Zhang mzhang@cs.stanford.edu, Daniel Rubin dlrubin@stanford.edu, Christopher Ré
chrismre@stanford.edu. Stanford addresses from 2022 may have changed since.*
**Subject:** Question about the ISIC Segmentation-ERM result (Table 1), "Reducing Reliance on Spurious Features … with Spatial Specificity"

Dear Dr. Saab and co-authors,

I am an MSc student working on shortcut learning in medical imaging, specifically on how the position of an artifact
relative to the region of interest decides whether region-based fixes work. Your paper is central to this, and I am
reproducing its ISIC result with your public code (github.com/khaledsaab/spatial_specificity, commit 1739cf2) and Bissoto et
al.'s trap sets. Thank you for releasing it.

Binary ERM reproduces closely: robust AUROC 0.363 ± 0.009 over trap sets 1–3 (Table 1: 0.359 ± 0.016). Segmentation-ERM does
not: following Appendix A.1 (SE-ResNeXt50 U-Net, ImageNet weights, Adam lr 5e-4, the listed augmentations, cross-entropy,
150 epochs, checkpoint on validation Dice), we obtain 0.626 / 0.673 / 0.641 on trap sets 1–3 (mean 0.647 ± 0.024), versus
0.727 ± 0.016 in Table 1. The selected checkpoints come from late epochs (131–146), as the validation Dice keeps rising.

Looking for the difference, we noticed that the repository history starts in November 2022, after the paper, and that the
checked-in configuration (`cfg/config.yaml`: 100 epochs) and ISIC loader (no augmentation) differ from Appendix A.1. The
evaluation notebook scores a Segmentation-ERM checkpoint named `epoch=1-step=115.ckpt` (robust AUROC 0.772), i.e. a model
from about one epoch of training. Could you help us with:

1. which code version and configuration produced the ISIC Segmentation-ERM numbers in Table 1 (number of epochs, whether
   augmentation was used for ISIC, learning rate, batch size);
2. how the final model was selected (best validation Dice over all epochs, a fixed epoch, or early stopping), and whether the
   three seeds correspond to trap sets 1–3;
3. which test set Table 1 uses: the paper mentions 513 test images (p. 12), while the trap-set test files and your notebook
   contain 772;
4. if possible, the Table 1 Segmentation-ERM checkpoints.

We will report your answer, cite your paper and code, and are happy to share our results with you before submission.

Thank you very much for your time.

Kind regards,
[Name]
[MSc programme, university, supervisor]
