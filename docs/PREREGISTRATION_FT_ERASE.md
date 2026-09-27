# Pre-registration — fine-tune, then erase (U-MtE on an end-to-end fine-tuned network)

Committed before running. Motivation (external review): the end-to-end evidence for U-MtE is weaker than the
frozen-feature evidence (an invariance-penalty analogue works on thyroid with ResNet-50 but not with ViT-S or on
capsule). U-MtE is a representation-level method; the natural end-to-end use is to apply it to the representation of
a network that was fine-tuned on masked images.

## Design (`scripts/run_finetune_spec.py --arms mte_post --traps trapA`)
Same traps, fold seeds, ResNet-50, 8 epochs and training as the existing `mask` arm (results/finetune/*/resnet50).
After fine-tuning: penultimate features of masked training images with and without a generic overlay → difference
subspace (90 % held-out energy, as in the frozen setting) → erase → logistic head refitted on the training set with
C chosen on clean validation (`mte_post`). Control `mask_post`: identical body and head fitting without erasure.
Cohorts: thyroid first (the distinct-overlay regime), then ovary and capsule if time allows; Trap A (in-ROI) only.

## Claims (reversed AUROC, Trap A, 95 % CI over 5 folds)
- **F1** (primary) mte_post − mask_post > 0 on thyroid (the erasure step helps after fine-tuning).
- **F2** mte_post − mask > 0 on thyroid (vs the fine-tuned masked network's own head).
- Ovary / capsule: reported with CIs; capsule is expected to fail without disease protection (debris ≈ fibrin).
Reported whichever way it goes.

## Results (results/finetune/thyroid/resnet50/paired_deltas.csv; thyroid, Trap A, 5 folds)
Deviation: the first launch failed because half-precision penultimate features overflowed (NaN); features are now
extracted in fp32 (no other change). Ovary and capsule were not run (time budget).

| contrast | reversed AUROC Δ [95 % CI] |
|---|---|
| F1 mte_post − mask_post (erasure step alone) | +0.004 [+0.000, +0.012] |
| F2 mte_post − mask (vs the network's own head) | −0.027 [−0.053, −0.001] |

**F1 essentially null, F2 not supported.** Applied after fine-tuning, the generic-overlay subspace no longer contains
the direction along which the fine-tuned network encodes the caliper: once the network has been trained on the
correlated data, the shortcut feature is task-specific, and overlays that resemble no particular artifact do not move
the representation along it. U-MtE is therefore a method for frozen (foundation-model) representations; the
end-to-end route that works is the invariance penalty during training (thyroid ResNet-50 +0.143), not post-hoc
erasure. This supports the framing of U-MtE as a representation-level intervention for frozen encoders.
