# P6 (EPVT) fidelity audit — machine 2, 2026-11-19 (CPU only; nothing scored on any test)

Author's instruction (2026-11-19, before the A34 5-seed run): confirm, element by element, that our pipeline replicates EPVT's.
Evidence only from their papers, code, configs, git history and their own logs. Loop: list differences → fix port errors
(amendment first) → re-audit; stop when a full pass finds nothing new.

Sources: EPVT code `github.com/SiyuanYan1/EPVT @ 8d46593` (local `/workspace/third_party/epvt`, our changes = `git diff`,
`third_party/epvt/paths_port.patch`); PLDG code `@ 1dd9999`; EPVT paper arXiv 2304.01508 v3 (`/workspace/data/lit/epvt.txt`);
our A33 runs `$WTSS_DATA/epvt/runs/{erm,epvt}_bias1_1/` (their trainer's own printout).

## 0. Their own logs
- `EPVT/train_output/out.txt`, `err.txt`: **one crashed PACS ERM debug run** (DoPrompt-main, `KeyError: 'lr_classifier'` at
  `algorithms.py:134`), no skin run, no validation curve. Usable facts: torch 1.12.1 / torchvision 0.13.1 / Python 3.8.15;
  `vit_b_16(pretrained=True)` = `ViT_B_16_Weights.IMAGENET1K_V1`; eval transform `Resize((224,224))` + ImageNet normalisation.
- `PLDG/train_output/out.txt`, `err.txt`, `results.txt`: crashed diabetic-retinopathy runs (dataset DR, `train_dr_prompt.py`;
  DataLoader interrupt, then `KeyError: 'lr_classifier'` for CORAL); `results.txt` = "Total records: 0" (also
  `domainbed/results/results.txt`). No skin run, no validation curve.
- **Conclusion:** no line-by-line comparison with their training is possible; there is nothing to compare our validation curve
  with. The audit therefore rests on code, configs and paper.

## 1. Checklist (pass 1; pass 2 re-checked every row — no new difference)

| # | item | theirs | ours (A33 runs) | source | match? |
|---|---|---|---|---|---|
| a1 | domains and order | folders named by artifact column, read as `sorted(os.scandir)` → clean, dark_corner, gel_bubble, hair, ruler | folders C, D, G, H, R (same sorted order); printed `['C','D','G','H','R']` | `datasets.py:185–186`; `data_proc/grouping.py`; our `p6_epvt_group.py` | ✓ |
| a2 | domain assignment rule | max over {dark_corner, hair, gel_bubble, ruler}; clean = 1 if 0 < max ≤ 0.65; argmax → folder | same code (paths only; `.loc` instead of chained assignment, same effect) | `grouping.py`; `p6_epvt_group.py` | ✓ |
| a3 | which images | trap-set train CSV of the split (README: `train_bias_0_1.csv`) | `train_bias_1_1.csv` (A29: bias 1, split 1) | `grouping.py:5`; A29 | ✓ (split k not stated in paper; k = 1) |
| a4 | per-domain counts, bias 1 split 1 | not published for bias 1 | C 3,197 (2,921 ben / 276 mel), D 1,419 (558/861), G 1,451 (925/526), H 5,942 (5,586/356), R 350 (119/231) | our folders | n/a |
| a5 | dataset sizes | paper Table 3: 12,360 / 2,057 / 6,199 (the bias-0 / OOD setting) | bias 1 split 1: train **12,359** (= CSV exactly, no image in two domains), val 2,059, test 6,181; no overlaps | `train/val/test_bias_1_1.csv` | ✓ (bias-0 counts matched exactly, SOURCE.md; paper 12,360 vs 12,359/12,343 unexplained, 1–17 images) |
| a6 | class labels | ImageFolder classes sorted: ben = 0, mel = 1; CSV `label` 1 = melanoma | folder class = CSV label for all 12,359 | `datasets.py:241–245` | ✓ |
| a7 | class balance / sampling | `class_balanced` True (registry default): per-domain `WeightedRandomSampler`, replacement, 26 per domain per step | same code, printed `class_balanced: True` | `hparams_registry.py`; `train_epvt.py:137`; `fast_data_loader.py:16–41` | ✓ |
| a8 | images | ISIC 2019 training input, full resolution | official ISIC 2019 challenge files (S3), 25,331, full resolution; plain-id symlink mirror (ISIC names 2,074 files `*_downsampled`) | `DATA_MANIFEST.md`; SOURCE.md | ✓ |
| a9 | domains used for training | all 5: the `test_envs` filter is commented out | all 5 (`--test_envs 0` as README; no effect) | `train_epvt.py:152–169` | ✓ |
| b1 | train preprocessing | PIL `.convert('RGB')` (ImageFolder), no prior resize; augmentation below | same code | `datasets.py:206–214` | ✓ |
| b2 | val/test preprocessing (trap sets) | `MelanomaDataset`: PIL RGB, `Resize((224,224))`, ToTensor, ImageNet mean/std | same class | `datasets.py:418–447` | ✓ |
| b3 | test-time transform, OOD sets | `test_epvt.py`/`test_erm.py`: **training augmentations at test** (random flips, crop, rotation, hue), single pass | not used (P6 gate is the trap set) | `test_epvt.py` | n/a for P6; note for any OOD comparison |
| c1 | train augmentation | HFlip, VFlip, RandomResizedCrop(224, 0.75–1), RandomRotation(45), ColorJitter(hue 0.1), ImageNet norm | same code (`data_augmentation: True`) | `datasets.py:206–214`; paper p. 7 "random flip, crop, rotation, and color jitter" | ✓ |
| d1 | backbone | torchvision `vit_b_16(pretrained=True)` → IMAGENET1K_V1 (`vit_b_16-c867db91.pth`), `heads` = Identity, attention_dropout 0 | same file `vit_b_16-c867db91.pth` | `networks.py:230`; their out.txt warning | ✓ |
| d2 | feature dropout | `resnet_dropout` 0.1 on the 768-d feature | same | `networks.py:222, 242` | ✓ |
| d3 | classifier | Linear(768, 2) (`nonlinear_classifier` False) | same | `networks.py:263–272` | ✓ |
| d4 | prompts | prompt_dim 10; shared prompt (10 × 768); per-domain u (5 × 10), v (5 × 768); domain prompt k = shared ⊙ (u_k ⊗ v_k) (low-rank multiplicative); appended after the patch/class tokens by a pre-hook on `encoder.dropout` | same code; printed `prompt_dim: 10` | `algorithms.py:2013–2077, 1990–2011`; paper p. 4–5 | ✓ |
| d5 | domain prompt generator (adapter) | `Project` MLP 768 → 3072 → 5·10, softmax over domains, BCE to the domain one-hot (λ = 1) | same | `algorithms.py:2033, 2128–2150` | ✓ |
| d6 | **domain mixup** | λ ~ Beta(0.2, 0.2) per step; **x_i, x_j, y_i, y_j each from a separate `random_pairs_of_minibatches` call (4 independent domain permutations)** → in 98.8 % of steps the labels belong to other domains than the mixed images | same code (we run theirs) | `algorithms.py:2101–2117`; check below | ✓ code — **✗ vs paper Eq. 3** (labels "corresponding" to x_i, x_j, p. 5) → **decision needed (section 3)** |
| d7 | test-time prediction | `predict`: frozen pass → adapter weights → prompted pass; eval mode | same code | `algorithms.py:2172–2175` | ✓ |
| e1 | hidden defaults (registry, not on the command line) | vit_base_16 True, im21k False, resnet_dropout 0.1, attention_dropout 0, data_augmentation True, class_balanced True, lambda 1.0, lr_prompt 1e-3, lr_project 1e-4, wd_project 1e-5, weight_decay 1e-2 | identical, as printed by our runs | `hparams_registry.py` | ✓ |
| e2 | command-line hparams | `{"lr": 5e-6, "lr_classifier": 5e-5, "batch_size": 26, "wd_classifier": 1e-5, "prompt_dim": 10}` (ERM: without prompt_dim) | identical strings | `run.sh`, README | ✓ |
| f1 | optimisers | EPVT: AdamW featurizer (lr 5e-6, wd 1e-2) + classifier (5e-5, 1e-5); AdamW prompts (1e-3, wd 1e-5 hard-coded); AdamW adapter (1e-4, 1e-5). ERM: AdamW featurizer (5e-6, wd 1e-2), classifier (5e-5, 1e-5) | same code; torch AdamW defaults (betas 0.9/0.999, eps 1e-8) unchanged 1.12 → 2.11 | `algorithms.py:116–146, 2036–2053` | ✓ |
| f2 | scheduler, warm-up, clipping, AMP | none, none, none, fp32 | same | code | ✓ |
| f3 | batch | 26 per domain × 5 = 130 | same | paper p. 7 "batch size is 130" | ✓ |
| f4 | epoch unit, steps | `steps_per_epoch = int(min |domain| / 26)`; README 1,501 steps at bias 0 = 60 × 25 + 1 | 13 per epoch; 1,301 = 100 × 13 + 1 (A33) | `train_epvt.py:307, 328–330`; paper p. 7 "100 epochs for trap set debiasing" | ✓ (A33) |
| f5 | seeds, determinism | seed 0 (random, numpy, torch), cudnn deterministic, benchmark off; DataLoader workers as DomainBed | same code; seed 0 | `train_epvt.py:96–102` | ✓ (bit-identity across torch 1.12 / 2.11 and GPUs not possible) |
| f6 | activation checkpointing (our engineering port) | — | ViT blocks recomputed in backward | `networks.py` (our diff) | ✓ **verified for the full EPVT update** (below): identical losses and gradients |
| f7 | `--ood_vis True` (README EPVT command) | evaluates 4 OOD sets every epoch (no effect on weights; consumes the torch RNG when the OOD DataLoaders iterate) | False (OOD sets not downloaded) | `train_epvt.py:383–428` | ~ RNG stream only; no systematic effect |
| f8 | wandb | logging only | disabled (`WANDB_MODE=disabled`) | — | ✓ |
| g1 | validation, selection | every epoch: ROC-AUC on `val_bias` (MelanomaDataset), `val_roc >= best` saves `<exp>.pkl` | same code; EPVT best val 0.925, stopped at epoch 67 | `train_epvt.py:439–448` | ✓ |
| g2 | early stopping | code: patience 200; paper: 22 | 22 (author: paper wins) | `train_epvt.py:327`; paper p. 7 | ✓ (A29/A33) |
| g3 | scored checkpoint | the best-validation `<exp>.pkl` | `epvt.pkl` / `erm.pkl` | — | ✓ |
| g4 | trap-set test script | **EPVT: none released** (`test_epvt.py` scores only the OOD sets; PLDG's in-domain block is commented out); ERM: `test_erm.py` in-domain block = MelanomaDataset + `eval_indomain` | `p6_epvt_score.py`: `test_erm.py`'s in-domain block, applied to both | `test_erm.py:191–210`; `test_epvt.py`; PLDG `test_prompt10.py:204–213` | ~ the only released in-domain scorer; EPVT's own is not public (ambiguity; our docstring overstated it — corrected) |
| g5 | model construction at test | `test_epvt.py`: 5 domains, registry defaults + prompt_dim 10, `model_dict` | 5 domains, the checkpoint's own hparams (= same values) | `test_epvt.py` | ✓ |
| g6 | ROC-AUC | `roc_auc_score(y, softmax(logits)[:, 1])` | same function | `metrics.py:71` | ✓ |
| g7 | prompt averaging at test | none (adapter-weighted prompt per image) | same | `algorithms.py:2172` | ✓ |
| — | environment | Python 3.8, torch 1.12.1, torchvision 0.13.1, PIL 9.4, RTX 3090 24 GB | Python 3.10, torch 2.11 + cu128, torchvision 0.26, PIL 12.3, RTX 5070 Ti 16 GB (Blackwell needs CUDA ≥ 12.8) | out.txt | ~ unavoidable; transforms on PIL images unchanged between these versions |

## 2. Checks run in this audit
- **Activation checkpointing, full EPVT update (CPU; `scripts/replication/p6_ckpt_check_epvt.py`):** real bias-1 images, 2 per
  domain × 5; same seeds; optimiser steps disabled: losses loss_dp 0.67748702 / loss_a 0.63699675 / loss_w 0.50225544 in both;
  **159 gradient tensors identical (max relative difference 0)**. (The earlier check covered ERM only.)
- **Mixup pairing quirk:** with x = y = domain id, `torch.cat([xi …])` and `torch.cat([yi …])` from their code come from different
  domain orders in **988 / 1,000** steps.
- Data: folders = `train_bias_1_1.csv` image for image and label for label; no image in two domains; no train/val/test overlap.

## 3. Differences found
**Port errors: none** in either pass (the A33 epoch-length error was already fixed; activation checkpointing now verified for
EPVT). Nothing to amend as a port fix.

**Paper vs code (their released code; needs the author's decision before A34):**
1. **Domain mixup labels (d6).** Paper Eq. 3: L = λ·CE(G(x_mix), y_i) + (1−λ)·CE(G(x_mix), y_j) with y_i, y_j "the
   corresponding labels" of x_i, x_j. Their code draws x_i, x_j, y_i, y_j from four independent random pairings, so the mixup
   loss mostly uses other images' labels (DomainBed's own Mixup uses one pairing). ERM is unaffected. Under the standing rule
   "paper wins over code", the paper-faithful fix is one `random_pairs_of_minibatches` call per step for all four tensors
   (one line). Either choice is defensible: (A) keep the released code (what A33 ran); (B) follow Eq. 3 — then all five A34
   EPVT seeds are new runs (seed 0 of A33 does not count for B); ERM seeds unchanged.
2. **No released EPVT trap-set scorer (g4)**; we use `test_erm.py`'s in-domain block — the only one released.
3. **OOD test-time augmentation (b3)**: matters only for a future OOD-table comparison, not for the P6 gate.

**Not differences:** `--ood_vis` (RNG stream only), library versions (environment), 12,359 vs 12,360 (different setting / 1 image).
