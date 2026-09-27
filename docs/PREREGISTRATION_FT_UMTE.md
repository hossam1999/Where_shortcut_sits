# Pre-registration — erase-while-fine-tuning (U-MtE for fully fine-tuned networks, any architecture)

Committed before running. Motivation: post-hoc U-MtE on a fine-tuned ResNet-50 was null (+0.004;
PREREGISTRATION_FT_ERASE.md) because unconstrained fine-tuning on the correlated data builds a task-specific caliper
feature that generic overlays do not move. A remedy for fully fine-tuned networks must act *during* training.

## Method (`umte_ft`, src/wtss/experiments/finetune.py)
Masked training image x and the same image with a generic overlay x⁺ (same library, half inside the ROI) in every
batch. Let g be the penultimate features of the network being trained.
1. Running overlay-shift subspace: C ← 0.98 C + 0.02 E[(g(x⁺) − g(x))(g(x⁺) − g(x))ᵀ] (detached); every 20 steps U =
   top eigenvectors of C covering 90 % of its energy (≤ 64), μ = running mean of g(x).
2. Head sees the erased features: logit = h(g(x) − U Uᵀ (g(x) − μ)); the final U is used at test time.
3. Plus the invariance penalty of `mte_ft` (λ = 1): ‖ĝ(x) − ĝ(x⁺)‖², ĝ = L2-normalised features.
No artifact example, artifact mask or artifact label; only the ROI mask; architecture-agnostic.
Secondary arm `cons_ft`: prediction consistency, BCE + λ (logit(x) − logit(x⁺))².

## Design
Thyroid Trap A, 5 folds, same training as the existing arms: ResNet-50 (lr 1e-4) and ViT-S/16 (lr 3e-5), 8 epochs.
`run_finetune_spec.py --cohort thyroid --arms umte_ft --traps trapA [--arch vit_small_patch16_224.augreg_in21k_ft_in1k --lr 3e-5]`.

## Claims (reversed AUROC, Trap A, 95 % CI over folds)
- **G1** umte_ft − mask > 0 for ResNet-50 and for ViT-S (success = both).
- **G2** umte_ft − mte_ft reported (does erasure add to invariance?); cons_ft − mask reported if run.
Clean and correlated AUROC reported alongside (a gain that only flips the shortcut is not a success).
Reported whichever way it goes.

## Amendment (before running it) — combined arm
Observed so far: umte_ft works for ResNet-50 (+0.160 [+0.114, +0.207]) but is not significant for ViT-S (+0.061
[−0.005, +0.141]); cons_ft is null for ResNet-50 (+0.007 [−0.014, +0.029]) but significant for ViT-S (+0.062
[+0.031, +0.087]). Pre-registered combined arm **umte_cons_ft** = projection + feature invariance + prediction
consistency (λ = 1 each; no tuning). **G3**: umte_cons_ft − mask > 0 for both ResNet-50 and ViT-S (thyroid, Trap A).
Also running: umte_ft on ovary (ResNet-50), reported with CI.

## Amendment 2 (before running) — other cohorts
umte_cons_ft on ovary and capsule (ResNet-50, Trap A, 5 folds), reported with CIs. Capsule debris resembles erosion
fibrin; without disease protection the frozen U-MtE failed there, so failure is the expected outcome.
