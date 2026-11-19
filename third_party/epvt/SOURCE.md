# P6 — EPVT (Yan et al., MICCAI 2023) and PLDG (Yan et al., IEEE TMI 44(1):348–360, 2025)

Renumbered by A29 (RR-ClArC is P7). Step 1 only until the author approves A30.

## Code (pinned)
- EPVT: https://github.com/SiyuanYan1/EPVT @ `8d46593a091ad41c92508e1569a108953cacfb31` (2024-02-14). Local `/workspace/third_party/epvt`.
- PLDG: https://github.com/SiyuanYan1/PLDG @ `1dd999924e72e98b277fcbbd145e69db07993e13` (2025-10-17). Local `/workspace/third_party/pldg`.
- Trap sets / artifact annotations: https://github.com/alceubissoto/artifact-generalization-skin @
  `54b20fbcd53524616f64bb5b9237eebc2cb270b8` (`trap_sets/{train,val,test}_bias_{0,0.3,0.5,0.7,0.9,1}_{1..10}.csv`,
  `isic_inferred_wocarcinoma.csv`). Local `/workspace/third_party/bissoto_ag`.
- Upstream environment: `requirements.txt` (torch 1.12.1, timm 0.4.12, Python 3.8). Ours: `/workspace/envs/epvt`, Python 3.10,
  torch 2.11.0+cu128 (Blackwell), **timm 0.4.12 as theirs**, pandas 1.5.3, numpy < 2, scikit-learn 1.2.1, TorchSnooper 0.8.

## Papers (read 2026-11-18: arXiv 2304.01508 v3 and 2401.03002 v1, full text)
- **Backbone:** ViT-Base/16, ImageNet (their code: torchvision `vit_b_16` weights `vit_b_16-c867db91.pth`), 224 px.
- **Trap-set debiasing (the P6 target):** Bissoto's ISIC 2019 trap sets, bias 0, 0.3, 0.5, 0.7, 0.9, 1 (1 = artifact–label
  correlation reversed between train and test); ERM vs EPVT (EPVT paper **Fig. 3a, p. 8**; PLDG paper **Fig. 8, p. 9**, same
  ERM/EPVT curves plus PLDG). **Figures only, no table**; metric ROC-AUC; the text says ROC-AUC "with five random seeds" for
  all experiments but the figure shows no spread. Which of Bissoto's 10 splits per bias is not stated (we use split 1).
  Digitised from the 300-dpi rendered pages (gridlines 100/90/80/70/60, ±≈0.5 AUC; agrees with the printed differences):

  | bias | 0 | 0.3 | 0.5 | 0.7 | 0.9 | 1 |
  |---|---|---|---|---|---|---|
  | ERM | 92.3 | 91.6 | 85.5 | 82.0 | 76.2 | 65.3 |
  | EPVT | 91.0 | 90.8 | 88.4 | 83.9 | 79.2 | 74.6 |
  | PLDG − ERM (PLDG Fig. 8 labels) | −0.28 | −0.82 | +2.47 | +0.26 | +2.43 | +6.13 |

  Best method on the trap sets: **EPVT** (PLDG text: "EPVT exhibits greater robustness than our PLDG").
- **OOD datasets** (train on ISIC 2019 bias-0 split 1, five artifact domains; test Derm7pt-dermoscopic 872, Derm7pt-clinical 839,
  PAD-UFES-20 531, PH2 200): EPVT Table 1 (p. 7) ERM 81.24 / 71.61 / 82.62 / 83.06, avg 79.63; **EPVT 83.69 / 73.96 / 86.67 /
  91.91, avg 84.11** (± over 5 seeds, 1.4–1.6). PLDG Table I re-reports: ERM avg 78.15, EPVT 84.43, PLDG 83.68.
- **Training data:** ISIC 2019 per Bissoto's split; domains from Bissoto's inferred artifact annotations: dark corner 2,351, hair
  4,884, gel bubble 1,640, ruler 672, clean 2,796 (EPVT p. 6); train 12,360 / val 2,057 / test 6,199 (EPVT Table 3).
- **Model selection:** "training-domain validation set" (DomainBed); in their code: the epoch with the highest ROC-AUC on Bissoto's
  `val_bias_*` set (checked every epoch), early stopping patience 200 epochs in code (paper: 22).
- **Decision (author, 2026-11-18, before any P6 result; replaces the earlier "1501 steps" instruction): the paper's protocol
  wins over the README** (standing rule; the repositories' history starts after the papers, so it cannot show which code made
  the figures). Trap sets: **100 epochs, early stopping patience 22 epochs on `val_bias` ROC-AUC** (their selection rule), the
  paper's augmentation (their code's training transform already is: horizontal/vertical flip, RandomResizedCrop(224, 0.75–1),
  rotation 45°, colour jitter (hue 0.1) — no change). **"Epoch" is read in their code's unit**: `steps_per_epoch =
  int(min over the five domains of |in-split| / batch 26)` (`train_epvt.py:310`), which their early-stopping counter also uses;
  DomainBed holds out 20 % of each domain (unused). So 100 epochs = 100 × steps_per_epoch + 1 steps (the +1 final evaluation as
  in their 1,501 = 75 × 20 + 1): **bias 1 split 1: 10 steps/epoch → 1,001 steps (≈ 13 passes over the 9,889 training
  images); bias 0: 20 → 2,001 steps**. The alternative reading (one pass over all training images per epoch, ≈ 76 steps) would
  give ≈ 7,600 steps; it is not used. **Confirmed by the author (2026-11-18, before any P6 run): reading A.** Reason: the paper
  does not define "epoch"; their code does, their patience counter uses it, and their README's 1,501 steps equals 75 × 20 + 1
  in that unit, so this is the authors' own meaning (not a paper-vs-code conflict). DomainBed's 20 % per-domain holdout stays
  unused, as theirs. Patience: `EPVT_PATIENCE=22` (their code: 200). Seed 0, one run per method and bias
  (step 1); bias 1 first and timed; if 6 levels × 2 methods would exceed ≈ 10 GPU-h, the author is asked before continuing.
- **Hyper-parameters:** paper: AdamW, lr 5e-6, wd 1e-2, batch 130, prompt length 10 (EPVT) / 4 (PLDG skin), 60 epochs (OOD) /
  100 epochs (trap sets), early stopping 22. **README (superseded by the decision above for steps / patience; the rest is used):** `--steps 1501`, batch 26 per domain × 5
  domains = 130 per step (≈ 16 epochs of 12,343), lr 5e-6, lr_classifier 5e-5, wd_classifier 1e-5, prompt_dim 10. The paper's
  100 epochs for the trap sets vs the README's 1501 steps is recorded as a discrepancy.

## Data (rebuilt; their processed archive is gone)
- Their Google Drive archive (`12SoMs_44jD4mRT6JEyIfdjBa4Fw07i2m`, EPVT README) returns **404** (2026-11-18). Rebuilt with their
  `data_proc/grouping.py` logic unchanged (`scripts/replication/p6_epvt_group.py`; paths, trap-set CSV, symlinks, domain-folder
  letters C/D/G/H/R as their SKIN class reads).
- ISIC 2019 images: ours (listing hash = `docs/DATA_MANIFEST.md`). Bissoto's CSVs use plain ids; ISIC names some files
  `*_downsampled.jpg`, so a symlink mirror with the suffix stripped is used (`$WTSS_DATA/epvt/isic2019_plain/`, 25,331 files).
- **Counts, bias 0 split 1** (their OOD setting): train 12,343 (paper 12,360: −17), domains dark corner 2,351, hair 4,884, gel
  bubble 1,640, ruler 672, clean 2,796 (**identical to the paper**), val 2,057 (identical), test 6,199 (identical).
  Bias 1 split 1: train 12,359 (clean 3,197, dark corner 1,419, gel bubble 1,451, hair 5,942, ruler 350), val 2,059, test 6,181.
  Per-run counts in `$WTSS_DATA/epvt/bias<b>_1/counts.json`. Every image of the bias-0/1 train, val and test sets resolves.

## Port (`third_party/epvt/paths_port.patch`)
Hard-coded server paths in `train_erm.py`, `train_epvt.py`, `test_erm.py`, `test_epvt.py` → environment variables
`EPVT_EVAL_ROOT` (Bissoto trap-set CSVs), `EPVT_IMFOLDER` (ISIC images), `EPVT_ROOT`, `EPVT_SPLIT` (trap set `<bias>_<split>`;
default `0_1` = theirs). No other change. In-domain scoring: `scripts/replication/p6_epvt_score.py` (their `MelanomaDataset`
and `misc.eval_indomain`, without the OOD block of their test scripts). CPU smoke test (2 steps) of `train_erm.py`: runs end to end. `train_epvt.py` hard-codes CUDA (`algorithms.py` `.cuda()`, `device="cuda"`): on CPU it loads the data and stops at the first training step; it is first tested on the GPU when P4 frees it. Their `wandb` logging runs with `WANDB_MODE=disabled`.

## PLDG skin path (author's question, 2026-11-18)
- The README's skin command is inside an HTML comment, but **`domainbed/scripts/train_latent_epvt.py` (algorithm `Latent_EPVT`,
  dataset `DG_Dataset`) is present and is the skin trainer**: domain folders `clean/dark_corner/gel_bubble/hair/ruler` under
  `ISIC2019_train/`, val/test from `val_bias_0_1.csv` / `test_bias_0_1.csv`, clustering into 4 pseudo-domains at epochs 0 and 10.
  Runnable after the same path port (not yet done) and a folder-name mapping (full names instead of EPVT's letters).
- Quirks recorded: the commented command passes `--use_domain_labels False`, which argparse `type=bool` turns into **True**, but
  `Latent_EPVT.update` hard-codes `domain_labels=False` and uses the cluster labels, so the method is label-free as described; its
  training set is built with `split='val'`, so **no training augmentation** is applied in the released code (paper: flips, crop,
  rotation, colour jitter); 5,900 steps, one batch of 130; their validation/test `imfolder` points at the grouped training folder.

## Activation checkpointing (machine 2, 2026-11-18; engineering, not a recipe change)
Their batch (26 per domain × 5 = 130) in fp32 does not fit a 16 GB GPU: the first ERM run stopped with CUDA out-of-memory
(15.4 GiB in use; their papers used RTX 3090s, 24 GB). With `EPVT_GRAD_CKPT=1`, each torchvision ViT-B/16 encoder block's
`forward` is wrapped in `torch.utils.checkpoint` (non-reentrant) in place (`domainbed/networks.py`, `_checkpoint_blocks`;
parameter names unchanged). The blocks have dropout 0 and LayerNorm only (no BatchNorm), so nothing differs mathematically.
Check (`scripts/replication/p6_ckpt_check.py`; 8 real bias-1 training images, identical initial weights, deterministic
algorithms): loss 0.74716187 plain vs 0.74716187 checkpointed (|Δ| 0); gradients of all 152 parameter tensors identical
(max relative difference 0). Peak memory of one training step at batch 26 × 5: ERM 4.25 GiB, EPVT 4.78 GiB (of 15.47).
The first OOM'd attempts (20:11 beside P4's arm-a feature pass; 21:33 alone) left no checkpoint and are not results.

## A36 — domain mixup per paper Eq. 3 (author, 2026-11-19; before any A34 run)
Their `forward_prompt` (`algorithms.py:2106–2109` @ 8d46593) draws x_i, x_j, y_i, y_j from four separate
`random_pairs_of_minibatches` calls; the paper (p. 5, Eq. 3) trains x_mix against its own pair's labels. Changed to one pairing
per step (`mixup_eq3.patch`); verification in `docs/P6_EPVT_AUDIT.md` §4. The A33 runs used the released version.
