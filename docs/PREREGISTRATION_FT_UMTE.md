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

## Results (thyroid, Trap A, 5 folds; results/finetune/thyroid/*/paired_deltas.csv)
| arm − mask (reversed AUROC) | ResNet-50 | ViT-S/16 |
|---|---|---|
| mte_ft (invariance only; earlier) | +0.143 [+0.099, +0.188] | +0.009 [−0.021, +0.040] |
| umte_ft (projection + invariance) | **+0.160 [+0.114, +0.207]** | +0.061 [−0.005, +0.141] |
| cons_ft (prediction consistency) | +0.007 [−0.014, +0.029] | **+0.062 [+0.031, +0.087]** |
| **umte_cons_ft (all three)** | **+0.157 [+0.114, +0.205]** | **+0.143 [+0.103, +0.187]** |

AUROC (clean / correlated / reversed): ResNet-50 mask 0.691 / 0.827 / 0.536; umte_cons_ft 0.688 / 0.661 / 0.693.
ViT-S mask 0.688 / 0.868 / 0.465; umte_cons_ft 0.712 / 0.798 / 0.608.
- **G1 partial** (ResNet-50 yes, ViT-S not significant). **G3 supported**: the combined arm improves on masking for
  both architectures with no artifact annotation, and each component alone works for only one architecture.
- On ResNet-50 the correlated AUROC falls below the reversed one (0.661 vs 0.693): mild over-correction (flag), its
  min(rev, corr) 0.661 is close to the label-using mask_balanced (0.677). On ViT-S there is no flip (0.798 vs 0.608)
  and clean AUROC improves (0.712 vs 0.688), but the shortcut is only partly removed.
- Interpretation: post-hoc erasure fails because unconstrained fine-tuning builds a task-specific shortcut feature;
  estimating the overlay subspace *during* training on the network's own features, and requiring overlay-invariant
  features and predictions, stops that feature from forming.

## Status of Amendment 2
First attempt not run: at 15:28 UTC the GPU became unavailable inside the container (`nvidia-smi: Input/output error`, host-side
driver problem; both jobs failed at CUDA initialisation before training). Commands to run when a GPU is available:
`python scripts/run_finetune_spec.py --cohort ovary --arms umte_cons_ft --traps trapA` and the same with
`--cohort capsule`. umte_ft on ovary (ResNet-50, run before the failure): +0.107 [−0.003, +0.229].

Recovery (15:37 UTC): the kernel module and /dev/nvidia* were intact; only the host-mounted `libcuda.so.610.43.02`
was unreadable. The identical userspace libraries were extracted from NVIDIA's official 610.43.02 package into
/root/data/cache/nvidia_driver/lib and the runs were relaunched with `LD_LIBRARY_PATH` pointing there (same driver
version, same GPU; no other change). Results below.

## Results of Amendment 2 (ResNet-50, Trap A, 5 folds)
| cohort | umte_cons_ft − mask | AUROC clean / corr / rev (umte_cons_ft) | mask |
|---|---|---|---|
| ovary | +0.094 [−0.011, +0.210] | 0.687 / 0.726 / 0.650 | 0.578 / 0.570 / 0.556 |
| capsule | −0.022 [−0.046, +0.001] | 0.555 / 0.879 / 0.190 | 0.596 / 0.915 / 0.211 |
- **Ovary: positive, not significant** (the 1,202-image cohort is underpowered for fine-tuning: every ovary contrast
  has a CI of about ±0.11). It improves every environment over masking and has the best min(rev, corr) of the
  annotation-free arms together with umte_ft (0.650 / 0.663 vs mask 0.556 and the label-using mask_balanced 0.551).
- **Capsule: fails, as pre-registered.** Fine-tuned masking collapses on in-ROI debris (reversed 0.211 vs ERM 0.495);
  debris resembles erosion fibrin, so an overlay-driven method cannot separate it from disease (theory: large ρ).
  Only label-based balancing helps there (mask_balanced reversed 0.687).
- Overall: erase-while-fine-tuning works across architectures (ResNet-50, ViT-S) for distinct-overlay artifacts
  (thyroid significant, ovary positive), and — like frozen U-MtE without protection — not for pathology-like artifacts.

## Amendment 3 (before running) — ovary power extension
The ovary result (+0.094 [−0.011, +0.210], 5 clusters) is underpowered. Added: spec split seeds 123 and 456, folds 0–4
(10 more clusters; ids 10·seed + fold), arms mask and umte_cons_ft only, same training. **G4**: umte_cons_ft − mask > 0
on ovary Trap A with all 15 clusters. Reported whichever way it goes.

### Amendment 3 result (run 2026-09-27 on the rebuilt data; results/finetune/ovary/resnet50_power_g4)
umte_cons_ft − mask on ovary Trap A with 15 clusters: +0.045 [−0.022, +0.112]. **G4 not supported.** (The fine-tuned
ResNet-50 barely learns the ovary task — see docs/PREREGISTRATION_REVIEW2.md, amendment 1 — so fine-tuning contrasts on
this cohort remain uninformative.)
