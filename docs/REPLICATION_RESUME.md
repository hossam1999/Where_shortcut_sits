# How to resume the replication (read this first in a new session)

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

## Running (2026-11-17 09:40)
| job | PID | progress | checkpoint on disk | resume command |
|---|---|---|---|---|
| P2 Vanilla ResNet-50d, seed 0 | 130207 | epoch ~83 / 300 | none before epoch 100 (their trainer saves every 100) → `$WTSS_DATA/replication/pahde/runs/vanilla_resnet50d_seed0/checkpoints/<config>/checkpoint_resnet50d_100.pth` | if it dies after epoch 100: copy `runs/vanilla_resnet50d_seed0/config.yaml`, set `ckpt_path` to the latest checkpoint and **`num_epochs: 300 - (epoch+1)`** (their resume trains num_epochs MORE epochs), run with `scripts/replication/run_pahde.sh <that yaml> 0 vanilla_resnet50d_seed0_resume`. Record the resume next to the P2 result. Before epoch 100: restart from scratch with `run_pahde.sh …/resnet50d_p_artifact0.2_p_backdoor0_adam_lr0.001.yaml 0 vanilla_resnet50d_seed0`. |
| BIMCV fetch | 6956 | 224 / 531 archives | `$WTSS_DATA/bimcv/done.txt` | `WTSS_DATA=/workspace/data nohup scripts/replication/bimcv_fetch.sh &` (resumes; logs in `$WTSS_DATA/bimcv/fetch_log.tsv`) |

## Queued next (run order, A12–A14; long jobs one at a time, short jobs beside them only if a measured pairing helps)
1. P2 Vanilla → epoch 300.
2. P2 RR-ClArC: their `run_preprocessing` (activations) on our Vanilla, then their mitigation grid (λ ∈ {10, 50} × last 3
   ResNet blocks, 10 epochs each; selection: mean of clean + biased validation accuracy, section 3). Scripts to be written
   (uses their `experiments/mitigation_experiments/start_model_correction.py`; set `ckpt_path` to our Vanilla's last checkpoint).
3. P3 ISNet chain alone: `nohup scripts/replication/p3_chain.sh &` — run 1 `isnet_corner_all_seed0` resumes after epoch 2
   (`$WTSS_DATA/replication/isnet/runs/isnet_corner_all_seed0/checkpoint.pt`), then `isnet_inside_all_seed0`,
   `isnet_inside_9010_seed0`. Score after BIMCV completes: `p3_test_select.py`, `p3_prepare.py --which test`, then
   `p3_train.py --name <run> --score_only …` (same args). Equivalence check (A9/A10): score `COVIDIsNet.pt` on both variants.
4. Short jobs (beside a long job only if measured faster): P1 location test (Standard, tube inside / outside the lungs + heart
   ROI, seed 1234; script to be written; Masked scored from it, A5); P4 timing (memory check at batch 16 first — ask the
   author if it does not fit), then P4 Binary-ERM and Segmentation-ERM splits 1–3
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
### Heartbeat 2026-11-17 13:20 UTC (auto-generated)

GPU jobs (PID, MiB, command):
```
130207  1884 MiB    09:35:58 python -u -m model_training.start_training --config_file /workspace/data/replication/pahde/runs/vanilla_resnet50d_seed0/config.yaml
186052  1844 MiB       07:06 /workspace/envs/viviano/bin/python -u scripts/replication/p1_location.py train --position inside
```
- P2 Vanilla: Epoch 173: of 300; checkpoints: checkpoint_resnet50d_100.pth 
- P3 isnet_corner_all_seed0: last epoch 2 of 47
- P4 timing_erm_b16: last epoch 0 of 149
- P4 timing_seg_b4: last epoch 0 of 149
- P4 timing_seg_b8: last epoch 0 of 149
- BIMCV: 268 of 531 archives (0 SHA1 failures)
- Disk: 259G used, 773G free
<!-- HEARTBEAT END -->
