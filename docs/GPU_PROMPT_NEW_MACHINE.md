# Prompt for the replacement GPU machine (old machine died 2026-11-18)

Paste everything inside the code block into Claude Code on the new machine.

```
You are resuming the replication programme of my MSc thesis "Where the Shortcut Sits" on a NEW machine. The previous
machine died; its disk (data, checkpoints, caches) is gone. Everything committed to git is safe.

SETUP
1. git clone git@github.com:hossam1999/Where_shortcut_sits.git && cd Where_shortcut_sits &&
   git checkout claude/festive-bohr-95evo5
2. Read, in order: docs/REPLICATION_RESUME.md, docs/REPLICATION_PLAN.md, docs/PREREGISTRATION_REPLICATION.md,
   docs/REPLICATION_STATUS.md (latest entries), docs/REPLICATION_RESULTS.md.
3. Record amendment A23 in PREREGISTRATION_REPLICATION.md BEFORE any new result: machine change (old GPU, date of
   failure, new GPU model, driver, CUDA, torch versions); which runs were lost mid-way and restart from scratch
   (ISNet isnet_corner_all_seed0 at epoch 29/48; P4 seg_split1 at epoch 66/150); every result already in git stays as
   it is; runs after the switch run on the new hardware with identical code, recipes and seeds.
4. Rebuild only the environments still needed: our venv (requirements.txt), isnet (P3), saab (P4), pahde (needed for
   the phase-5 RR-ClArC port). P1 and P2 are FINISHED: do not rebuild their data or environments.
   Re-apply third_party/<paper>/*.patch to fresh clones at the commits in third_party/<paper>/SOURCE.md.
5. Persistence (new rule): put data and outputs on the persistent volume, and copy every checkpoint off the machine
   after each save if I give you a storage target (rclone remote); never commit checkpoints, data or private links.
   Update docs/REPLICATION_RESUME.md with the new layout.

DATA (only what remaining work needs; verify counts/hashes against docs/DATA_MANIFEST.md and STATUS)
  - P3: its training sources (BrixIA — I will give you the private link, keep it only in a secrets file outside git;
    CheXpert small; the other P3 sources listed in STATUS) and the BIMCV test images via scripts/replication/bimcv_fetch.sh
    + bimcv_select.py (stream each archive, keep only the selected PNG/JSON, delete the archive). BIMCV is ~2.8 TB of
    download: START IT FIRST, it runs on the network while the GPU trains (ISNet training does not need BIMCV; only
    scoring does).
  - P4: ISIC 2018 Task 1/2 + Bissoto trap sets/labels (small).
  - Phase 5: TN3K (thyroid), rebuild the cohort on CPU (scan_thyroid.py -> prepare_thyroid.py), DINOv2 feature caches.
  - Not needed: PadChest, NIH, RSNA, ISIC 2019, HAM10000, ovary, capsule.
  Tell me at once which datasets need my credentials (BrixIA link, Kaggle token, Stanford AIMI for CheXpert).

RUN ORDER (critical path first; GPU rules from the plan still apply, re-measure pairings on this GPU)
  1. P3 ISNet: isnet_corner_all_seed0, isnet_inside_all_seed0, isnet_inside_9010_seed0, from scratch, exact P3 recipe.
     If the machine has 2+ GPUs, run them in parallel, one per GPU.
  2. P4 Seg-ERM splits 1-3 (exact registered protocol, grad checkpointing as verified), then the P4 gate (A13) and the
     P4 arm-d comparison with Seg-ERM split 1 (see RESUME "When P4 Seg-ERM split 1 is scored").
  3. Phase 5 (thyroid only; RR-ClArC on ResNet-50d a3, ISNet on DenseNet-121, arm d EfficientNet-B0; fp32; early stopping
     patience 5, max 48 epochs; one selection criterion for everything): trainer + toy-batch unit tests -> timing test on
     Trap A fold 1 -> docs/PREREGISTRATION_PHASE5.md -> push -> WAIT for my approval -> run.
  4. Scoring as data arrives: ISNet runs on the rebuilt P3 test once BIMCV is complete (corner run reported beside 0.775
     as descriptive evidence of test equivalence, not a gate); P3 head-to-heads with arms b, c, d (descriptive, A9).

RULES (unchanged): their environment = their exact recipes; settings chosen on validation only; each test scored once;
honest reporting; any change is a dated amendment before the affected result; commit often and push every commit to
claude/festive-bohr-95evo5 AND stage-reports; keep REPLICATION_RESUME.md and STATUS current (heartbeat every 2 h).

FIRST REPORT TO ME: GPU/CPU/RAM/disk/network speed measured on this machine; BIMCV download speed and ETA; the GPU-hour
estimate for steps 1-3 on this hardware; which credentials you need from me.
```
