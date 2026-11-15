# Prompt for Claude Code on the GPU machine: replicate and beat three papers

## What the machine needs before you start
- **Hardware:** 1 GPU (A100/H100 80 GB recommended; 24 GB works for papers 1–2), 16+ CPU cores, 64+ GB RAM, a **1 TB
  NVMe SSD**.
- **Keys:** a GitHub key, and a Kaggle token (already used before).
- **Dataset access to request early (approvals take days):**
  - **CheXpert** (Stanford AIMI account; the *small* version is enough);
  - **BrixIA COVID-19** (register at brixia.github.io; the download link arrives by email);
  - **BIMCV COVID-19+** only if the supervisor wants ISNet's external test (389 GB, skipped by default).
- **Tools:** Academic Torrents needs no account, but the machine needs `aria2c` (or another torrent client) for the
  224-px PadChest.

The full plan, the numbers to beat and the open questions are in `docs/REPLICATION_PLAN.md`.

```
You are working on my MSc thesis repo "Where the Shortcut Sits" on a new GPU machine. The thesis shows that ROI
masking removes an image-artifact shortcut only when the artifact lies OUTSIDE the ROI; inside it the shortcut survives.
We have our own fix (a class-conditional feature constraint after masking, "mask_cmc", plus a label-free variant,
U-MtE). Goal now: replicate THREE published papers on their own tests, then add our method and compare on the SAME
test, then run all methods on our inside-vs-outside location test. Read docs/REPLICATION_PLAN.md first.

THE THREE PAPERS (numbers to beat, each on its own test and in its own metric)
  P1 Viviano et al., "Saliency is a possible red herring when diagnosing poor generalization", ICLR 2021.
     Chest X-ray, PadChest + NIH; hospital source linked to the label 90/10 in training, reversed at test.
     ROC-AUROC, ResNet-18 at 224 px, 10 seeds. Cardiomegaly: Masked 0.77±0.09 (standard 0.70, GradMask 0.74,
     ActDiff 0.73, RRR 0.57). Pneumonia (AP/PA view shortcut): ActDiff 0.68±0.05 (masked 0.56, standard 0.20).
  P2 Pahde et al., "Ensuring medical AI safety...", Machine Learning 114:206 (2025).
     ISIC 2019, 8 classes, synthetic artifact in 20% of melanomas; "biased" test = artifact in all classes.
     ACCURACY, ResNet-50. RR-ClArC 0.76 (RRR 0.51, vanilla 0.28); clean 0.86.
  P3 Bassi et al., ISNet, Nature Communications 15:291 (2024).
     COVID-19 / pneumonia / normal chest X-rays; class mark in the upper-left corner (outside the lungs) in training;
     matched / cyclically reversed / mark-free tests. MACRO-F1, DenseNet-121. ISNet 0.775 on all three tests;
     lungs-only U-Net 0.618 on all three; plain DenseNet 0.775 / 0.434 / 0.195.
  Every number above comes from a third-party summary: CHECK EACH ONE against the paper (table/figure) in phase 0.

GROUND RULES (non-negotiable)
- Pre-register before results: phase 1 writes docs/PREREGISTRATION_REPLICATION.md (what we run, seeds, metrics,
  what counts as "reproduced" and as "beaten", the exact test sets). Commit and push it BEFORE running any of our
  methods on any test set. Anything decided after seeing results is labelled exploratory.
- Same splits, same images, same backbone, same metric as the paper for every head-to-head number. Our settings are
  chosen on validation data only; the test set is scored once per setting.
- Report honestly: if a paper cannot be reproduced, say so with numbers and do NOT claim to beat it. If our method
  does not win, report that. Never tune on the test set, never drop seeds, never change a test to make numbers fit.
- Never commit data, images, checkpoints, features or predictions. Commit code, configs, small CSV/MD summaries.
- Commit often; push every commit to branch claude/festive-bohr-95evo5 AND to stage-reports. End commit messages
  with my usual co-author lines if configured.
- If something blocks you (data access, broken code, a number that cannot be checked), STOP that paper, write what
  is blocked in docs/REPLICATION_STATUS.md, push, and continue with the next paper.

SETUP
1. git clone git@github.com:hossam1999/Where_shortcut_sits.git && cd Where_shortcut_sits
   git checkout claude/festive-bohr-95evo5
2. Our code: Python 3.12 venv, pip install -r requirements.txt (docs/ENVIRONMENT.md).
   export WTSS_DATA=<1TB disk>/data WTSS_CACHE=$WTSS_DATA/cache PYTHONPATH=$PWD/src
   Each paper's own code gets its OWN environment (conda or Docker) with the versions its repo pins; record the repo
   URL, commit hash and environment file in third_party/<paper>/SOURCE.md. Clone third-party code under
   third_party/ (git-ignored except SOURCE.md and our patches as .patch files).
3. Check `nvidia-smi`, free disk (>= 800 GB), and write docs/REPLICATION_STATUS.md with the machine specs.

PHASE 0 - VERIFY BEFORE BUILDING (write all answers to docs/REPLICATION_STATUS.md, push)
4. For each paper, open the paper PDF and its official code (use the link in the paper's code-availability section;
   do not trust third-party mirrors). Record, with page/table numbers:
   P1: is the mask a central CIRCLE or a lung mask? exact limitation sentence about "confounding variables that exist
       within the mask" (quote + page); are the split files in the repo; which PadChest resolution and preprocessing
       (is the TorchXRayVision 224-px PadChest from Academic Torrents enough?); exact numbers in Table 1 / Fig. 7.
   P2: where the inserted artifact is placed relative to the lesion (fixed corner? random?); split files released?;
       exact Table 1 numbers; how RR-ClArC and RRR are configured (hyperparameters).
   P3: does the code still run; which datasets build the COVID benchmark (BrixIA, CheXpert, ...); exact mark
       shapes, size and position; exact Table 1 numbers; how the U-Net lungs-only pipeline is built.
   If a number differs from the list above, the PAPER wins: update docs/REPLICATION_PLAN.md.

DATA (1 TB budget; do not download what is not needed)
5. Reuse our pipeline for what we already use (docs/DATA.md, Makefile `data` targets): ISIC 2019 (+ lesion masks),
   NIH ChestX-ray14 (~42 GB). New: PadChest 224 px from Academic Torrents (images-224.tar 12.8 GB, the "Fixed
   cropping" version, for TorchXRayVision) unless phase 0 shows P1 needs full resolution (then STOP and ask: ~1 TB);
   CheXpert small (10.8 GB); BrixIA (link from me). Skip BIMCV COVID-19+ unless I say so. Verify file counts against
   each source and add every new dataset to docs/DATA_MANIFEST.md (scripts/verify/data_manifest.py).
   Lung (and heart) masks: TorchXRayVision PSPNet, as in src/wtss/data/cxr.py.

PHASE 1 - PRE-REGISTRATION (commit before running our methods)
6. Write docs/PREREGISTRATION_REPLICATION.md. Must contain, per paper:
   - "Reproduced" = our mean of the paper's own method within the paper's reported spread (P1: ±1 SD; P2/P3: if no
     spread is reported, within 0.03 absolute). Same seeds count as the paper (P1 10; P2 5; P3 5, or 3 if one run
     takes > 10 GPU hours - decide from a timing run and write it down BEFORE the full runs).
   - "Beaten" = our method's mean is higher AND the paired difference (ours - theirs, same seeds and test images,
     crossed seed x image bootstrap, 10,000 replicates, as in src/wtss/stats_crossed.py) has a 95% CI above 0.
     Holm correction over the head-to-head comparisons of all three papers.
   - Our arms (fixed now, not after results):
     a) mask_cmc on THEIR backbone: take their trained network (P1: the "Masked" model; P2: the Vanilla ResNet-50,
        since P2 has no ROI-masked method of its own; P3: the U-Net lungs-only DenseNet), extract penultimate
        features, project out the within-class artifact mean differences
        (scripts/round8/candidates.py: cmc_directions / fit_cmc, mode chosen on validation from {pooled, per_class}),
        refit the linear layer on the training set. This is the fair same-backbone comparison.
     b) our frozen-encoder pipeline: DINOv2 ViT-B/14 (518 px) on ROI-masked images + mask_cmc head.
     c) U-MtE (src/wtss/umte.py), label-free.
     The "artifact label" each method uses at TRAINING time only: P1 hospital source (cardiomegaly) or AP/PA view
     (pneumonia); P2 the inserted artifact; P3 the corner mark.
     Note for P3: every training image carries its class's mark (artifact = label), so within-class differences
     cannot be estimated and mask_cmc does not apply to the ORIGINAL P3 benchmark: say so, run only U-MtE there.
     cmc_directions is written for 2 classes: extend it to K classes (one delta per class that has both groups;
     pooled = their mean, per_class = their span), with unit tests in tests/, before any run.
   - Metrics: each paper's own metric for the head-to-head number, plus ROC-AUROC (one-vs-rest macro for P2/P3)
     for all, plus clean-test performance (a fix must not buy robustness by losing diagnosis).

PHASE 2 - REPRODUCE (their code, their settings)
7. P1 first (cheapest), then P2, then P3. For each: timing run on 1 seed, then all seeds. Save per-image predictions
   (image_id, seed, method, test set, scores) to $WTSS_DATA/replication/<paper>/ and a summary CSV to
   results/replication/<paper>/reproduce.csv. Compare with the paper; write "reproduced: yes/no, by how much".
   P1 methods: Standard, Masked, GradMask, ActDiff, RRR. P2: Vanilla, RRR, RR-ClArC. P3: DenseNet, U-Net lungs-only,
   ISNet.

PHASE 3 - HEAD-TO-HEAD ON THEIR TEST
8. Run our arms a), b), c) on exactly the same splits and test sets. Table per paper: their methods vs ours, their
   metric + AUROC + clean, mean ± SD over seeds, paired difference with 95% CI vs their best method, Holm p.
   results/replication/<paper>/head_to_head.csv and .md.

PHASE 4 - LOCATION EXTENSION (the thesis question inside their setups)
9. In each setup, keep everything identical but place the SAME synthetic artifact (same shape, size, opacity) at
   three positions relative to the ROI: fully outside (overlap 0), on the border (~0.5), fully inside (>= 0.95).
   Use our placement code (src/wtss/synthetic.py: find_best_placements_for_mask) and our tube/ruler drawing.
   Correlation 90/10 in training, correlated / reversed (10/90) / clean tests, as in our traps.
   P1: ROI must contain the disease evidence: for cardiomegaly use the lung + HEART mask from PSPNet (a lung-only mask
       removes the heart); artifact = a 100x12 px radio-opaque tube as in our chest sweep.
   P2: ROI = lesion mask (ISIC 2019); artifact = their inserted artifact.
   P3: ROI = lungs; artifact = their mark moved from the corner into the lungs / onto the border; ALSO a 90/10 variant
       so that mask_cmc applies.
   Run ALL methods (theirs + ours) at all three positions. Report reversed-test AUROC per position, the crossover
   (outside - inside gain of each method over standard training) and clean AUROC. Expected (registered): methods that
   only act outside the ROI (masking, ISNet, RRR/GradMask with an ROI mask) lose their benefit inside; ours should not.

PHASE 5 - WRITE-UP
10. docs/REPLICATION_RESULTS.md, in plain English: per paper, reproduced or not; head-to-head table; location table;
    what we can and cannot claim (e.g. "beats Masked on P1 cardiomegaly by X [CI]" only if the registered rule holds).
    Run pytest -q tests. Commit summaries only; push to both branches.
11. Finish with a short summary for me: per paper one line (reproduced? beaten? by how much), the location result,
    GPU hours used, and anything blocked or still to decide.
```
