# Pre-registration — end-to-end fine-tuning check (thyroid calipers; capsule debris if time allows)

Committed before any fine-tuned model is trained. Purpose: test whether the frozen-backbone findings hold when
the whole network is trained.
- Traps: the thyroid (and capsule) spec traps of docs/PREREGISTRATION_THYROID_TRAPS.md / _CAPSULE_TRAPS.md;
  seed 42, folds 0–4 (5 bootstrap clusters; image-level paired bootstrap within folds).
- Model: ResNet-50 (ImageNet, timm), 224 px, AdamW, one-cycle LR 1e-4 (head ×10), 8 epochs, flips/rot90
  augmentation, fixed epochs (no early stopping); threshold on clean validation.
- Arms: erm; mask (train+test on ROI-masked images); balanced; mask_balanced; mte_ft (masked images + generic
  overlay inserted then masked; loss = BCE + ‖normalise(g(x)) − normalise(g(x⁺))‖² on the penultimate embedding —
  the end-to-end analogue of U-MtE; no artifact label or mask).
- Claims (reversed AUROC, 95 % CI): **F1** crossover — (mask − ERM)_B > (mask − ERM)_A (reported per trap);
  **F2** mte_ft − mask > 0 in Trap A; **F3** mask_balanced − balanced > 0 in Trap A. Reported either way.

## Results — thyroid, ResNet-50 fine-tuned end-to-end (results/finetune/thyroid/resnet50; 5 folds)
Trap A reversed / corr (clean): ERM 0.431 / 0.744 (0.587); mask 0.536 / 0.827 (0.691); balanced 0.599;
mask_balanced 0.747 / 0.677 (0.721); **mte_ft 0.679 / 0.703 (0.698)**.
- **F1 not supported**: crossover (mask − ERM)_B − (mask − ERM)_A = +0.057 [−0.021, +0.130] (direction as
  predicted: +0.161 in B vs +0.105 in A; CI includes 0 with 5 folds).
- **F2 supported**: mte_ft − mask (Trap A) = +0.143 [+0.099, +0.188]; no gaming (corr 0.703 ≥ rev 0.679); clean
  +0.007 vs mask. Trap B: −0.034 [−0.062, −0.008].
- **F3 supported**: mask_balanced − balanced (Trap A) = +0.147 [+0.104, +0.192].
min(rev, corr), Trap A: mte_ft 0.679 ≈ mask_balanced 0.677 > mask 0.536 > balanced 0.599 — the label-free
end-to-end U-MtE matches the best label-using arm.

## Results — capsule, ResNet-50 fine-tuned (results/finetune/capsule/resnet50; 5 folds)
Trap A reversed / corr (clean): ERM 0.495 / 0.900 (0.719); mask **0.211** / 0.915 (0.596); balanced 0.670;
mask_balanced 0.687 / 0.780 (0.754); mte_ft 0.190 / 0.904 (0.567).
- **F1 supported (strongly)**: crossover = +0.584 [+0.469, +0.691]; masking HARMS in-ROI end-to-end
  (−0.284 [−0.347, −0.226]) and helps out-of-ROI (+0.301 [+0.225, +0.380]).
- **F2 not supported**: mte_ft − mask (A) −0.021 [−0.039, −0.003]; mte_ft also lowers clean AUROC (debris ≈ fibrin
  regime, as in the frozen analysis).
- F3 not supported: mask_balanced − balanced (A) +0.017 [−0.070, +0.105].

## Amendment (before training): ovarian ultrasound (MMOTU) fine-tuning with the same design and claims F1–F3.

## Results — ovary, ResNet-50 fine-tuned (results/finetune/ovary/resnet50; 5 folds; small cohort → wide CIs)
Trap A reversed / corr (clean): ERM 0.457 / 0.671 (0.576); mask 0.556 / 0.570 (0.578); balanced 0.602;
mask_balanced 0.593; **mte_ft 0.649 / 0.743 (0.687)** — best reversed, correlated and clean AUROC.
F1 not supported (crossover +0.136 [−0.085, +0.329]); F2 not supported at 95 % (mte_ft − mask +0.094
[−0.025, +0.217]; mte_ft − ERM +0.192 [+0.103, +0.278]); F3 not supported (−0.009). Underpowered (1,202 images).

## Amendment (before training): ViT-S/16 (timm vit_small_patch16_224.augreg_in21k_ft_in1k), lr 3e-5, thyroid, same
design and claims F1–F3 (transformer replication of the ResNet-50 check).

## Results — thyroid, ViT-S/16 fine-tuned (results/finetune/thyroid/vit_small_patch16_224.augreg_in21k_ft_in1k)
Trap A reversed / corr (clean): ERM 0.377 / 0.860 (0.640); mask 0.465 / 0.868 (0.688); balanced 0.693;
mask_balanced 0.710 / 0.788 (0.760); mte_ft 0.475 / 0.878 (0.695).
- **F1 supported**: crossover +0.168 [+0.074, +0.265] (mask − ERM: +0.088 in-ROI vs +0.256 out-of-ROI).
- **F2 not supported**: mte_ft − mask (A) +0.009 [−0.021, +0.040] (ResNet-50: +0.143) — the end-to-end invariance
  analogue does not transfer to ViT-S at this learning rate / epoch budget.
- F3 not supported: mask_balanced − balanced (A) +0.017 [−0.023, +0.055].
