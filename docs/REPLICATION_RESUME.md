# How to resume the replication (read this first in a new session)

> **2026-11-19 (author): two machines.** Machine 1 (this one, RTX 5060 Ti) runs only the P3 ISNet chain, its scoring, the
> A24 in-distribution evaluations, the P3 arm feature passes, and the exploratory track. **P4, P5 and phase 5 run on
> machine 2**; the P4, P5 and phase-5 notes below are kept for reference and are not machine 1's work.

Kept current on every state change and by a heartbeat every 2 h (`scripts/replication/heartbeat.sh`; auto block at the
end). Authoritative plan: `docs/REPLICATION_PLAN.md`. Rules and amendments A1–A14: `docs/PREREGISTRATION_REPLICATION.md`.
Findings and deviations: `docs/REPLICATION_STATUS.md`. Results: `docs/REPLICATION_RESULTS.md`.

## Machine layout
- Repo `/root/Where_shortcut_sits` (container disk, NOT persistent); data and outputs under `$WTSS_DATA=/workspace/data`
  (1.1 TB host volume, persistent). Third-party clones `/workspace/third_party/{viviano,pahde,isnet,saab,bissoto}`; envs
  `/workspace/envs/{viviano,pahde,saab,tools}`; our env `/workspace/venv-wtss`.
- If the repo is lost: `git clone git@github.com:hossam1999/Where_shortcut_sits.git && git checkout claude/festive-bohr-95evo5`,
  then re-apply `third_party/<paper>/*.patch` to the clones (P1: `git -C /workspace/third_party/viviano apply …/cu128_port.patch`;
  P2: same with `pahde`). Private links (BrixIA) live only in `/workspace/secrets/` (never in git).
- GPU rules (mandatory): all GPU work through `scripts/replication/gpu_wait.sh` (memory + at most one other job); never kill
  by pattern; stop a job by PID **and its DataLoader workers** (`pgrep -P <pid>`), else they keep GPU memory.

## Machine 2 (A27; updated 2026-11-18 12:35 UTC, machine 2's clock)
RTX 5070 Ti 16 GB, 28 cores, 62 GB RAM, 288 GB disk, **no persistent volume** (re-download data if recycled; only git
survives). Repo `/root/Where_shortcut_sits`; `WTSS_DATA=/workspace/data`; third-party clones `/workspace/third_party/{saab,bissoto,pahde,isnet,r2r}`
(pahde patched with `cu128_port.patch`); envs `/workspace/venv-wtss`, `/workspace/envs/{saab,pahde,r2r}` (`docs/ENVIRONMENT_MACHINE2.md`).
| job | state | resume |
|---|---|---|
| P4 Seg-ERM splits 1–3 (`m2_p4_pair.sh`, two at a time, 1.23× measured) | splits 1 + 2 running, 1.87 min/epoch each; ETA 1+2 ≈ 17:10, 3 ≈ 20:00 UTC | `nohup scripts/replication/m2_p4_pair.sh > /workspace/logs/m2_p4_pair.log 2>&1 &` (resumes from each `checkpoint.pt`) |
| P4 arm a | after each split: `saab` env `p4_arm_a.py --stage features --split k`, then our env `p4_arm_a.py --stage heads` | — |
| P4 arm-d split-1 comparison | needs arm d's features from machine 1 (not in git); arm d's test scores are in git (`results/replication/armd/scores/p4_split1.csv.gz`), so the paired comparison with Seg-ERM split 1 is computed from scores | — |
| P5 step 1 | **done, check failed** (0.838/0.552 vs 0.895/0.356); P5 dropped pending the author | — |
| Phase 5 | thyroid cohort rebuilt; trainer + unit tests next, then timing test, `PREREGISTRATION_PHASE5.md`, wait | — |

## Done (results in git)
- Phase 0–1, amendments A1–A14 (pushed).
- **P1 Viviano, seed 1234 (preliminary, 1 seed, no verdict — A14):**
  - cardiomegaly: Standard 0.627, Masked 0.829 (A5), arms a 0.866, b 0.869, c 0.593 —
    `results/replication/viviano/preliminary_xray_1234.md`, scores `results/replication/viviano/scores/xray_*_1234.csv.gz`;
  - RSNA: Standard 0.186, Masked 0.759 (label-dependent, A11), ActDiff 0.708, arms a 0.610, b 0.767, c 0.378 —
    `results/replication/viviano/preliminary_rsna_1234.md`, scores `results/replication/viviano/scores/rsna_*_1234.csv.gz`.
  - Gate passed on kept runs (`$WTSS_DATA/replication/viviano/GATE_OK`).
- P3 data (cohort, test rebuilt per A9/A10), P4 port (`third_party/saab/SOURCE.md`); P4 data: ISIC 2018 unzipped
  (2,594 images + masks) and the 512-px cache built (`$WTSS_DATA/replication/saab/cache512/`).

## Running (updated 2026-11-17 20:15 UTC)
| job | progress | checkpoint on disk | resume command |
|---|---|---|---|
| P2 Vanilla ResNet-50d, seed 0 | **finished 19:26** (test clean 0.839, biased 0.311); scored → `$WTSS_DATA/replication/pahde/scores/vanilla_seed0.csv.gz` (+ features) | `runs/vanilla_resnet50d_seed0/checkpoints/config/checkpoint_resnet50d_last.pth` | — |
| `p2_chain.sh` (2nd pass, started 20:06 after the port-bug fix; see STATUS 20:10) | waiting for a GPU slot → class-0 activations → 6 RR-ClArC fine-tunes → scores | per step (skips finished steps; stops at the first failure) | `nohup scripts/replication/p2_chain.sh > /workspace/logs/p2_chain.log 2>&1 &` |
| P1 location test, tube outside (Standard, seed 1234) | started 18:05 via short queue, epoch ~50 / 100 at 20:05 | best model only (`$WTSS_DATA/replication/viviano/location/outside/results/`) | restart the short queue (re-runs a position whose `stats_1234.pkl` is missing) |
| P1 location arms (A16), `$WTSS_DATA/replication/viviano/location/run_arms.sh` | inside: a, b done, c (U-MtE) running; outside after its scores exist; table rebuilt after each | `…/location/<pos>/arms/preds_<arm>.csv.gz`, DINOv2 features cached per split | `nohup $WTSS_DATA/replication/viviano/location/run_arms.sh > $WTSS_DATA/replication/viviano/location/arms.log 2>&1 &` (skips a position whose `summary.json` exists) |
| Short-job queue (`short_queue.sh`) | outside train → outside score → P4 ERM ×3 → P4 Seg ×3 | per job | `nohup scripts/replication/short_queue.sh > /workspace/logs/short_queue.log 2>&1 &` |
| BIMCV fetch | see heartbeat | `$WTSS_DATA/bimcv/done.txt` | `WTSS_DATA=/workspace/data nohup scripts/replication/bimcv_fetch.sh &` |

## P2 window (2026-11-18, A18; `scripts/replication/p2_window.sh`, log `/workspace/logs/p2_window.log`)
Stops the ISNet chain at its next epoch checkpoint, then alone on the GPU: A18 RR-ClArC ×3 (+ scores in
`$WTSS_DATA/replication/pahde/scores_a18/`), lesion U-Net train + predict, lesion ROIs, DINOv2 features for P2 arms b, c;
restarts the ISNet chain (also on failure), then CPU: `p2_select.py --set a18`, `p2_arms.py --arm bc --stage heads`.
Restart: `nohup scripts/replication/p2_window.sh > /workspace/logs/p2_window.log 2>&1 &` (every step skips finished
work; an incomplete A18 fine-tune directory is moved aside and rerun). If it stopped with FAILED, ISNet has been restarted.

## When the ISNet corner run finishes (author, 2026-11-18)
`p3_train.py` scores it on the rebuilt test automatically (`runs/isnet_corner_all_seed0/preds_{biased,standard,deceiving}.csv.gz`).
Report its macro-F1 on both A9 variants beside P3's 0.775 (deceiving test, Table 1), labelled **descriptive supporting
evidence of test equivalence, not a gate** (P3 stays descriptive: A9 check 0.223 / 0.260).

## When P4 Seg-ERM split 1 is scored (author, 2026-11-18)
Re-run `armd_train.py --setting p4_split1 --stage heads` (CPU): `heads_p4` adds the paired comparison with our Seg-ERM
split 1 automatically once `saab/runs/seg_split1/preds_test.csv` exists (arm d 0.666 vs Binary-ERM 0.355 so far).
Descriptive until the P4 gate (A13).

## P4 arm a (author, 2026-11-19)
As soon as each Seg-ERM split k exists: arm a = mask_cmc on THEIR Seg-ERM encoder's features of lesion-masked images
(A13 item 8a; flags with >= 3 images per group), head as registered; report vs Seg-ERM alone (paired, same 772 test images)
together with the P4 gate (paper or notebook, A13 addendum).

## Chains (how they hand over)
- `p2_chain.sh` starts `p3_chain.sh` (ISNet alone; flag `$WTSS_DATA/replication/ISNET_CHAIN_ACTIVE`) **only** when all 6
  RR-ClArC score files exist. ISNet run 1 `isnet_corner_all_seed0` resumes after epoch 2.
- `short_queue.sh` pauses while the ISNet flag **or `$WTSS_DATA/replication/P4_HOLD`** exists (2026-11-18: P4 waits for
  the whole ISNet chain; `p3_chain.sh` removes the hold at its end). P4 Seg-ERM split 1 resumes from its `checkpoint.pt`.
- `short_queue.sh` pauses while the ISNet flag exists. If `p2_chain.sh` stops with FAILED, fix, record, and restart it.
- Never edit a running bash script (bash reads it as it executes); stop it by PID, edit, restart.
- **Before restarting `p2_chain.sh`:** their Lightning callback writes `last.ckpt` after *every* epoch, and the chain skips
  a fine-tune whose `last.ckpt` exists. A run is complete only if `last.ckpt` has `epoch == 9` (10 epochs, 16,890 steps):
  `/workspace/envs/pahde/bin/python -c "import torch,sys; print(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['epoch'])" <dir>/last.ckpt`.
  Move an incomplete run's directory to `checkpoints_corrected_incomplete/` (and delete its score file if one exists)
  before restarting; their code cannot resume a fine-tune, so it restarts from our Vanilla (record it in STATUS).

## Queued next (run order, A12–A14; long jobs one at a time, short jobs beside them only if a measured pairing helps)
1. P2 Vanilla — done.
2. P2 RR-ClArC via `p2_chain.sh` (running). Afterwards on CPU: selection among the 6 (mean of clean + biased validation
   accuracy, section 3), then P2 arms a/b/c and the head-to-head.
3. P3 ISNet chain alone: `nohup scripts/replication/p3_chain.sh &` — run 1 `isnet_corner_all_seed0` resumes after epoch 2
   (`$WTSS_DATA/replication/isnet/runs/isnet_corner_all_seed0/checkpoint.pt`), then `isnet_inside_all_seed0`,
   `isnet_inside_9010_seed0`. Score after BIMCV completes: `p3_test_select.py`, `p3_prepare.py --which test`, then
   `p3_train.py --name <run> --score_only …` (same args). Equivalence check (A9/A10): score `COVIDIsNet.pt` on both variants.
4. Short jobs via `scripts/replication/short_queue.sh` (pairing P1 + P2 measured at 1.44, kept; never beside ISNet): P1 location
   test (A15; `p1_location.py train|score --position inside|outside`), then P4 Binary-ERM and Segmentation-ERM (`--grad_ckpt`,
   batch 16, fp32; checkpointing verified identical, see `third_party/saab/SOURCE.md`) splits 1–3
   (`gpu_wait.sh 9000 /workspace/envs/saab/bin/python scripts/replication/p4_train.py --method erm|seg --split k`; resumes
   from `$WTSS_DATA/replication/saab/runs/<method>_split<k>/checkpoint.pt`); our arms' feature passes (P2, P3, P4).
5. Phase 5 (thyroid only, A14): write `docs/PREREGISTRATION_PHASE5.md` after a timing test, push, **wait for the author's approval**.

## Cancelled (never started unless noted)
- A12: RSNA Standard/Masked seeds 3232/3221; cardiomegaly ActDiff all seeds (seed 1234 stopped mid-run, archived in
  `runs_cancelled_A12_*`); P2 RRR; P3 DenseNet and the 90/10 outside run.
- A14: cardiomegaly Standard 3232/3221 and RSNA ActDiff 3232/3221 (P1 head-to-heads stay preliminary, 1 seed); P4 location test;
  phase-5 cohorts other than thyroid.
- A7: 8 P1 runs on the post-publication split code (archived, not used).

## Open decisions for the author
- None right now. Next stop points: P4 batch-16 memory check; phase-5 pre-registration approval; any gate failure.

## Downloads
- BIMCV: see the table above (≈ 6 min per archive).
- ISIC 2018 Task 1/2 (P4): done and unzipped.

<!-- HEARTBEAT START -->
### Heartbeat 2026-11-20 04:07 UTC (auto-generated)

GPU jobs (PID, MiB, command):
```
548152  1394 MiB       10:15 /workspace/venv-wtss/bin/python -u scripts/replication/p3_arms_bc.py --stage features
550815  1306 MiB       01:09 /workspace/venv-wtss/bin/python -u scripts/newmethod/adv_train.py --round 2 --alpha 0.1 --stage train
```
- P2 Vanilla: Epoch 300: of 300; checkpoints: checkpoint_resnet50d_100.pth checkpoint_resnet50d_200.pth checkpoint_resnet50d_300.pth checkpoint_resnet50d_last.pth 
- P3 isnet_corner_all_seed0: last epoch 47 of 47
- P3 isnet_inside_9010_seed0: last epoch 47 of 47
- P3 isnet_inside_all_seed0: last epoch 47 of 47
- P4 erm_split1: last epoch 149 of 149
- P4 erm_split2: last epoch 149 of 149
- P4 erm_split3: last epoch 149 of 149
- P4 seg_split1: last epoch 66 of 149
- P4 timing_erm_b16: last epoch 0 of 149
- P4 timing_seg_b4: last epoch 0 of 149
- P4 timing_seg_b8: last epoch 0 of 149
- BIMCV: 531 of 531 archives (0 SHA1 failures)
- Disk: 293G used, 739G free
<!-- HEARTBEAT END -->
