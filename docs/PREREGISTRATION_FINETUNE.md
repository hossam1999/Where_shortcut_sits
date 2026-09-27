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
