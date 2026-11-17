# Saab et al., Reducing Reliance on Spurious Features in Medical Image Classification with Spatial Specificity (MLHC 2022)

- Paper: PMLR v182, `saab22a` (https://proceedings.mlr.press/v182/saab22a/saab22a.pdf; the Stanford link in the README now
  returns a web page). Numbers: Table 1, p. 13 (ISIC robust AUROC: Binary-ERM 0.359 ± 0.016, Segmentation-ERM
  0.727 ± 0.016, 3 seeds). Protocol: Appendix A.1, p. 20. Quotes checked: section 5.1, p. 13 (largest spatial overlap:
  hair > gel bubbles ~50% > rulers ~20% > dark corners) and p. 15 ("struggles to overcome spurious features that have high
  overlap with the abnormal region").
- Repository: https://github.com/khaledsaab/spatial_specificity, commit `1739cf288253df216d1ce5f4697ea697e8bb9c86`
  (2023-02-25). Local clone `/workspace/third_party/saab` (git-ignored).
- Labels and splits: https://github.com/alceubissoto/debiasing-skin, commit `f93c8654f50c581544f32ad6b1450cb925a176ff`
  (`trap-sets/isic_annotated_{train,val,test}{1..5}.csv`; `artefacts-annotation/isic_bias.csv`).
- Images and masks: ISIC 2018 Task 1/2 training images and Task 1 ground truth (ISIC challenge S3).
- Upstream environment: `requirements.txt` (unversioned: pytorch, torchvision, hydra-core, meerkat-ml==0.2.0, imageio,
  dosma, pytorch-lightning, wandb, segmentation-models-pytorch, albumentations). Their notebook ran Python 3.9.
- Environment used here: `/workspace/envs/saab` (Python 3.10, torch 2.11.0+cu128, torchvision 0.26.0,
  segmentation-models-pytorch 0.3.4, albumentations 1.3.1). The cu128 build is forced by the Blackwell GPU (approved
  deviation, as for P1–P3).

## Facts established from their code and data (2026-11-17)

- **Seed = split.** `train.py` calls `build_isic_dp(seed=cfg.train.seed)`, which reads trap set `k = seed`
  ("seeds for isic need to be in [1,5]"), and `seed_everything(seed)`. Three seeds are three trap-set triplets: 1, 2, 3.
- **Split sizes** (every trap set): train 1,822, validation 259, test 772. **All 259 validation images are also in
  training** (5/5 splits); training and test share no image id; the union is the 2,594 ISIC 2018 Task 1 images. Their
  code trains on the 1,822 (validation included) and selects on the 259, as registered (A13 item 5).
- **Test = 772 images.** Their `notebooks/isic_evaluation.ipynb` scores 772 (split 1); the PDF says 513 (p. 12), which is
  reported as an inconsistency.
- **Robust AUROC** = AUROC on the trap test (their notebook; the paper: "the robust and overall AUROC are equivalent for the
  ISIC dataset", p. 12). The notebook's own checkpoints give Binary-ERM 0.310 (lr 5e-4, wd 0.01, epoch 2) and
  Segmentation-ERM 0.772 (lr 5e-4, wd 0, epoch 1), split 1.
- **Image score of Segmentation-ERM:** the mean over pixels of the class-2 (melanoma lesion) softmax (`src/utils.py`,
  `score`); segmentation target `(label + 1) * lesion mask` (`src/data/isic.py`, `seg_loader`).
- **Git history starts 2022-11-18, after the paper**, so it cannot show which code produced Table 1. **Protocol:
  Appendix A.1** (A13 item 6): Binary-ERM ResNet-50, lr 1e-5, checkpoint on validation AUROC; Segmentation-ERM U-Net with
  an SE-ResNeXt50 (32x4d) encoder (ImageNet), lr 5e-4, augmentations (affine, elastic, brightness, contrast), checkpoint
  on validation Dice; cross-entropy, Adam, 150 epochs. Not stated in A.1, taken from the config: batch 16, weight decay 0.
  The checked-in config (100 epochs) and the ISIC loader (no augmentation) are not used.

## Port changes (all in `scripts/replication/p4_train.py`; no edit to their files, so no `.patch` is needed)

Their training stack (Lightning 1.x `validation_epoch_end`, TorchMetrics `compute_on_step` and `dice_score`, Meerkat 0.2
DataPanels, Hydra) does not install with CUDA 12.8 torch. A plain PyTorch loop drives their components:
- **Models:** Binary-ERM = torchvision ResNet-50 with ImageNet V1 weights and their `nn.Sequential(Dropout(0), Linear(2048, 2))`
  head (their `ResNet` wrapper loads the same V1 weights through the removed `model_urls`); Segmentation-ERM =
  `smp.Unet("se_resnext50_32x4d", encoder_weights="imagenet", classes=3)` with the segmentation head split off as `fc`,
  as in `task.py`.
- **Data:** their `isic.py` resizes every image to 512 × 512 (bilinear) and normalises with their ISIC mean/std; the same
  is done once into a cached uint8 array. Masks: Task 1 ground truth, resized to 512, thresholded at 0.5.
- **Augmentation** (Segmentation-ERM, training only): their `src/data/train_transform.json` (albumentations 0.3.1
  serialisation) loads unchanged in albumentations 1.3.1. Its final `Resize(512)` and single-channel CXR `Normalize` are
  dropped, because the images are already 512 and are normalised with their ISIC RGB statistics. Appendix A.1 names affine,
  elastic, brightness and contrast; the file also flips, gamma-jitters and grid/optical-distorts.
- **Metrics:** validation AUROC (scikit-learn, equivalent to `torchmetrics.AUROC` for binary), their
  `compute_dice_score` copied verbatim (overlap = pixels non-zero in both label maps; total = sum of label values).
- **Checkpointing:** best model on the registered validation metric, plus a full checkpoint every epoch for resume
  (scheduling rule 5). Their Lightning `ModelCheckpoint(every_n_train_steps=5)` also keeps the best by the same metric.
