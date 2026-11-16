# Full prompt for Claude Code on the GPU machine (replication programme, from 2026-11-17)

Paste everything inside the code block. It works for a fresh session or one that is already running: the machine
reads the repo state first.

```
You are continuing the replication programme of my MSc thesis "Where the Shortcut Sits" on this GPU machine.
Read first, in this order: docs/REPLICATION_PLAN.md (the plan; it is authoritative), docs/PREREGISTRATION_REPLICATION.md
(amendments A1-A11), docs/REPLICATION_STATUS.md, docs/REPLICATION_RESULTS.md. Check the running jobs and queues
before touching anything; never kill or restart a running job unless it is a cancelled run under A12 below.

GOAL
The thesis shows ROI masking removes an artifact shortcut only when the artifact lies OUTSIDE the ROI. Our fixes:
mask_cmc (class-conditional constraint after masking; scripts/round8/candidates.py, K-class version) and U-MtE
(label-free; src/wtss/umte.py). We must beat FOUR papers, each with ONLY its best method, in TWO environments:
  (1) THEIR environment: their data, splits, test design and metric (reproduce, then head-to-head);
  (2) OUR environment: our cohorts and trap design (phase 5).
Plus a location test inside their setups (phase 4).

TARGETS (best method only; numbers to beat; verify any not yet verified against the paper)
  P1 Viviano et al., ICLR 2021. Cardiomegaly: Masked 0.77 +- 0.09 AUROC (scored from the Standard model, A5).
     Pneumonia (RSNA): ActDiff 0.68 +- 0.05 (Masked leaks the label at test, A11: reported as a finding, not a
     target). P1's original-test head-to-head is SECONDARY (global shortcut); its location test is primary.
  P2 Pahde et al., Machine Learning 2025. RR-ClArC: 0.76 accuracy on the ISIC 2019 biased test (clean 0.86).
  P3 Bassi et al., Nature Communications 2024 (ISNet). ISNet 0.775 macro-F1 on the corner-mark benchmark.
     Test rebuilt per A9/A10 (verdicts only if COVIDIsNet.pt is within 0.03 of 0.773 macro-F1 on that variant).
  P4 Saab et al., "Reducing Reliance on Spurious Features in Medical Image Classification with Spatial
     Specificity", MLHC 2022 (PMLR). Best method in their Table 1 (expected: segmentation-supervised ERM, robust
     AUROC 0.820 +- 0.039 CXR / 0.727 +- 0.016 ISIC, 3 seeds; binary ERM 0.455 / 0.359). Key quotes to verify:
     sec. 5.1 p. 13 (larger gaps for artifacts with more lesion overlap: hair > gel bubbles ~50% > rulers ~20% >
     dark corners) and p. 15 ("struggles to overcome spurious features that have high overlap with the abnormal
     region"; chest-tube tip crossing the pneumothorax).

STEP 1 - Record amendment A12 (lean scope) NOW, before any affected result. From docs/REPLICATION_PLAN.md
"Lean scope":
  - P1 keep: cardiomegaly Standard seeds 3232/3221 (Masked scored from them); RSNA ActDiff seeds 3232/3221.
    Cancel: RSNA Standard/Masked seeds 3232/3221, cardiomegaly ActDiff (all seeds), ActDiff in P1 phase 4.
    P1 phase 4: Standard with the tube inside vs outside the lungs+heart ROI (2 runs); Masked scored from them.
    Gate only on kept runs. Our arm a sits on the best method's network (cardiomegaly Standard/Masked; RSNA ActDiff).
  - P2 keep: Vanilla + RR-ClArC with the paper's reported config (their grid only if none is reported). Cancel RRR.
  - P3 keep: ISNet original (1 run); ISNet with marks inside the lungs, P3 design + 90/10 variant (2 runs).
    Cancel DenseNet and the 90/10 outside run (P3 Table 1 is the outside reference). Only U-MtE on the P3-design
    test (artifact = label); mask_cmc on the 90/10 variant.
  - Finished runs stay and are reported; cancelled runs that never started are simply not run.

STEP 2 - P4 Saab feasibility (no training yet). Report in REPLICATION_STATUS.md: official code (link, commit, does it
run on cu128), datasets and access (ISIC: which release and which artifact labels; CXR: which dataset, e.g.
CANDID-PTX/SIIM, and its access terms), whether splits and overlap/artifact labels are released or rebuildable,
exact Table 1 numbers and the best method, GPU-hour estimate. Then record amendment A13 (P4 added: best method only,
3 seeds, ISIC first, CXR only if its data are reachable; P4 phase 4 = one fixed artifact moved inside vs outside
their ROI on ISIC; phase-5 arm = segmentation-supervised ERM on our ROI masks). Do not start P4 training until A13
is pushed. If CXR data need a request, tell me what to request.

STEP 3 - Run, in critical-path order:
  a) P3 ISNet chain (3 runs) and P2 Vanilla first (longest); P1 kept seeds and P1 phase 4 fill gaps; P4 after A13.
  b) Our arms on each paper's best network as soon as it exists (features once; heads on CPU).
  c) Head-to-heads in their metric + AUROC (+ clean). Preliminary results labelled "preliminary" until all
     registered seeds/runs exist; never change anything because of a preliminary number.
  d) P3 equivalence check (COVIDIsNet.pt on both test variants) as soon as BIMCV completes.

STEP 4 - Phase 5 (our environment), after P1-P4 head-to-heads are done or running smoothly:
  - Port the best methods as arms of src/wtss/experiments/finetune.py: ActDiff (P1; Masked = our existing `mask`
    arm, no re-run), RR-ClArC (P2, concept direction from our image-level artifact labels), ISNet LRP penalty
    outside our ROI (P3), segmentation-supervised ERM on our ROI masks (P4). Their losses and published
    hyperparameters; nothing tuned on our tests. Unit-test each loss on a toy batch.
  - Scope: Trap A for thyroid, ovary, capsule, ISIC 2019 hair; Trap B for thyroid only; 3 folds; tests correlated,
    reversed, clean (+ unaltered test sets where they exist). ERM/mask from the existing thesis runs.
  - Mixed precision and one early-stopping rule for every arm, identical for all.
  - Timing test (thyroid Trap A, 1 fold, all arms) -> write docs/PREREGISTRATION_PHASE5.md (scope, success rule:
    reversed-test AUROC primary, paired diff vs each paper's method with 95% crossed-bootstrap CI above 0, Holm over
    phase-5 comparisons, clean AUROC must not drop by more than 0.01) -> push -> WAIT for my approval -> run.

SCHEDULING RULES (mandatory)
  1. GPU only for training/feature extraction. Every head fit, mask_cmc, U-MtE head, bootstrap and table on CPU,
     parallel across cores.
  2. Pair jobs by MEASURED total throughput (images/s with 1 vs 2 jobs); keep a pairing only if it raises the total.
     Never 3 GPU jobs at once unless measured faster.
  3. ISNet (~10.4 GB) only beside one small job.
  4. Longest chains first; short jobs fill gaps.
  5. Checkpoint every few epochs; every launcher resumes from the last checkpoint after a crash.
  6. Reuse: cached DINOv2 features, the thesis's saved predictions, released weights where valid. Never recompute
     what exists.
  7. All GPU work through the memory-checking launchers; no ad-hoc GPU jobs; no pattern kills.

RULES THAT NEVER CHANGE
  - Their environment: their exact recipes, splits, images and metric. Settings chosen on validation only; each
    test scored once. Any change is a dated amendment BEFORE the affected result.
  - With 1 seed, the bootstrap resamples images only, and every table says so.
  - Honest reporting: "not reproduced" if it is not; no verdict if a rule says none; nothing dropped or re-tuned
    after seeing results. Port bugs are fixed and recorded (like A7); tuning is not allowed.
  - Never commit data, images with patient pixels, checkpoints, features, predictions or private links.
  - Commit often; push every commit to claude/festive-bohr-95evo5 AND stage-reports (rebase on the remote first).

REPORTING
  - Keep REPLICATION_STATUS.md current (queue, GPU-hours used/left, blockers). Results go in REPLICATION_RESULTS.md:
    per paper, 1) reproduced? 2) beaten in their environment? 3) location test, 4) beaten in our environment.
  - Stop and ask me only for: data access I must provide, a gate failure, a result that contradicts a registered
    expectation in a way that might mean a bug, or approval of PREREGISTRATION_PHASE5.md.
  - After STEP 1-2, send me: the new GPU-hour estimate, the run order, and the P4 feasibility summary.
```
