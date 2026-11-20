# Pre-registration — Phase 5 (thyroid TN3K calipers, Trap A / Trap B): their methods vs our arms, one rule for all

**Status: DRAFT for the author's approval (machine 2, 2026-11-20). No phase-5 network has been trained beyond the timing
test (30 batches each, outputs tagged `timing`, never scored). Nothing in this document has been run on a test set.**

Basis: `docs/REPLICATION_PLAN.md` "Phase 5, minimal scope" (author, 2026-11-18), A18 (RR-ClArC λ), A20/A21 (arm d, C grid),
A31 (fp32). Code: `src/wtss/experiments/phase5.py`, `scripts/phase5/p5_run.py` (10 unit tests pass, `tests/test_phase5.py`).

## 1. Data and folds (fixed)
Thyroid TN3K with real calipers (the thesis cohort, rebuilt on machine 2 from the thesis code), the thesis spec traps
(`build_spec_envs`, seed 42, pHash leakage groups), **Trap A (calipers inside the nodule ROI) and Trap B (outside), folds
0–2**, images from the 518-px cache, network input 224 px.

| trap | fold | train_corr | val_clean | test_corr | test_rev | clean |
|---|---|---|---|---|---|---|
| A | 0 / 1 / 2 | 961 / 957 / 961 | 390 / 370 / 378 | 298 / 297 / 298 | 357 / 357 / 356 | 460 |
| B | 0 / 1 / 2 | 817 / 827 / 818 | 104 / 106 / 110 | 255 | 217 / 217 / 216 | 130 / 132 / 130 |

`val_clean` = the thesis's inner validation (20 % of the training groups, marker in 50 % of each class: no label
information). Every test environment is scored once per network.

## 2. Methods (their recipes on their published architectures; ours as registered)
| network | architecture / init | optimiser, lr, batch | epochs / stopping |
|---|---|---|---|
| ERM base (RR-ClArC's) | timm `resnet50d.a3_in1k` (P2 paper p. 22), ImageNet | AdamW lr 1e-4 (head ×10), wd 1e-4, batch 48 (thesis fine-tune optimiser) | early stopping on val_clean AUROC, patience 5, ≤ 48 |
| **RR-ClArC** (P2) | the fold's ERM base, all parameters train (A18) | same optimiser type, lr = base / 10, fresh state, batch 48 | 10 epochs, last kept (their configs) |
| **ISNet** (P3) | DenseNet-121 LRP (their `IsDense121Lightning`, heat = True, P3's P/d/E/A/B), penalty outside the nodule ROI | SGD lr 1e-3, momentum 0.99, grad-norm clip 1 (P3), batch 16 | early stopping, patience 5, ≤ 48 (P3's 48) |
| **arm d** (ours, A20/A21) | torchvision EfficientNet-B0, ImageNet, trained on ROI-masked images | Adam lr 1e-4, batch 64 | early stopping, patience 5, ≤ 48 |
| DINOv2 + mask_cmc, U-MtE (ours) | frozen DINOv2 ViT-B/14 @ 518, CPU heads | — | — |
| ERM (ConvNeXt-T), mask | the thesis's own results on the same folds (plain baseline rows) | — | — |

Common rules: **fp32** everywhere; **constant learning rate** (no scheduler in any of their configs or ours); 2-class
softmax / unweighted cross-entropy; score = p(malignant); augmentation = the thesis fine-tune augmentation (random
horizontal flip + random rot90, applied jointly to the ROI); seed = fold.

**RR-ClArC grid:** λ ∈ {5·10⁸, **5·10⁹** (their ISIC ResNet value, P2 Tab. 12, A18), 5·10¹⁰} × layer ∈ {identity_1, identity_2,
last_conv} (their three ResNet layers); CAV from `cavs_max` activations of the fold's training images, direction `signal`,
target `all_logits_random`, `l2_mean` (their code, imported unchanged). 9 fine-tunes per trap × fold.
**Heads (arm d, DINOv2 + mask_cmc, U-MtE):** mask_cmc mode and C on the A21 grid {0.001, 0.003, 0.01, 0.03, 0.1, 1, 10}, edge
reported.

**One selection criterion for every tuned choice (theirs and ours): AUROC on `val_clean`** — early stopping of every network,
RR-ClArC's (λ, layer), mask_cmc's mode and C.

## 3. Outcomes and comparisons (proposed; author to confirm)
- **Primary metric:** AUROC on `test_rev` (artifact–label correlation reversed), per trap; **secondary:** AUROC on `clean`
  and on `test_corr`.
- **Statistic:** mean over the 3 folds of the paired difference (ours − theirs), crossed fold × image bootstrap
  (`wtss.replication_stats.paired_bootstrap`, 10,000 replicates, one Poisson(1) weight per image shared by folds and methods).
- **Comparisons (Trap A primary, Trap B sanity check):** arm d vs RR-ClArC, arm d vs ISNet, DINOv2 + mask_cmc vs RR-ClArC,
  DINOv2 + mask_cmc vs ISNet → **Holm family of 4 on Trap A**; U-MtE vs both and every Trap B comparison reported
  descriptively.
- **Success ("beaten"):** the registered rule (section 2 of `PREREGISTRATION_REPLICATION.md`): higher mean, 95 % CI of the
  paired difference above 0, Holm-adjusted p < 0.05 — on `test_rev`, **with the clean-AUROC drop ≤ 0.01** (the thesis's
  phase-3 rule).
- **Expected direction (the thesis claim):** location-based ISNet helps in Trap B and not in Trap A; concept-level RR-ClArC
  is the real competitor in Trap A.

## 4. Timing test (Trap A fold 1, 30 training batches, 2026-11-20 12:30 UTC, RTX 5070 Ti, alone) and GPU budget
| network | img/s (train) | peak GPU | ≈ s/epoch (train + val) | per run | × 6 trap-folds |
|---|---|---|---|---|---|
| ERM ResNet-50d | 244 | 4.5 GB | ~5 | ≤ 4 min | ≤ 0.4 h |
| RR-ClArC fine-tune | ≈ ERM | ≈ 4.5 GB | ~6 | ~1 min + CAV | 54 fine-tunes ≈ 1–1.5 h |
| ISNet, from scratch | 15.6 | 13.8 GB | ~65 | ≤ 52 min | ≤ 5.2 h |
| ISNet, ImageNet init | 18.8 | 13.8 GB | ~55 | ≤ 44 min | ≤ 4.4 h |
| arm d (EfficientNet-B0) | 244 | 5.4 GB | ~5 | ≤ 4 min | ≤ 0.4 h |

**Total ≤ ~7 GPU-h** (early stopping usually ends sooner); ISNet runs alone (13.8 GB). CPU heads (DINOv2 features at 518
for ~2,600 images, mask_cmc, U-MtE, bootstrap) ≈ 1 h.

## 5. Open decisions for the author (recommendation first)
1. **ISNet initialisation.** Their class defaults to `pretrained=False` and P3 ran from scratch; every other phase-5 network is
   ImageNet-initialised and the training sets are small (~820–960 images). **Recommend ImageNet init** (a fair competitor,
   and 15 % faster); alternative: from scratch, as P3.
2. **Optimisers.** **Recommend each method's own recipe** (table above: ERM base / RR-ClArC AdamW, ISNet SGD + clip, arm d
   Adam); alternative: one common optimiser for all.
3. **Constant learning rate** for all networks (none of the recipes schedules it). **Recommend: confirm.**
4. **Comparisons and Holm family** as in section 3. **Recommend: confirm**, or name the family.
5. **RR-ClArC grid** 3 λ × 3 layers selected on val_clean (A18 practice). **Recommend: confirm.**

On approval: run order ERM bases → RR-ClArC grid → arm d → ISNet (alone) → CPU heads → comparisons; every result pushed as
it exists; each test environment scored once per network.
