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
