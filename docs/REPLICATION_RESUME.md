# How to resume the replication (read this first in a new session)

> **2026-11-20: machine 1 released without a backup — see the next section first.**
>
> **2026-11-19 (author): two machines.** Machine 1 (this one, RTX 5060 Ti) runs only the P3 ISNet chain, its scoring, the
> A24 in-distribution evaluations, the P3 arm feature passes, and the exploratory track. **P4, P5 and phase 5 run on
> machine 2**; the P4, P5 and phase-5 notes below are kept for reference and are not machine 1's work.

Kept current on every state change and by a heartbeat every 2 h (`scripts/replication/heartbeat.sh`; auto block at the
end). Authoritative plan: `docs/REPLICATION_PLAN.md`. Rules and amendments A1–A14: `docs/PREREGISTRATION_REPLICATION.md`.
Findings and deviations: `docs/REPLICATION_STATUS.md`. Results: `docs/REPLICATION_RESULTS.md`.

## Machine 1 (e9f997b4f379, RTX 5060 Ti) RELEASED 2026-11-20 — NO BACKUP (author's decision)

Machine 1 was released on **2026-11-20 without a Drive backup** (the rclone remote could not be set up on this machine).
**Lost:** the 43 trained models listed below (3.32 GB: P2 Vanilla + RR-ClArC, ISNet ×3, P1 runs, arm d encoders, P4
checkpoints), all data, image caches, ROI caches and features under `/workspace/data`. **In git:** every result table,
every per-image test score (`results/replication/<paper>/scores/`), the alignment audit, the exploratory-track results, and the
training-log summaries and P2 configs of the lost runs (`results/replication/train_logs/`). Nothing reported depends on the lost
files; they are needed only to re-run or extend a run. The machine-2 sections below are machine 2's and are unchanged.

### How to rebuild each item (GPU-hours: measured alone on the RTX 5060 Ti unless marked "estimate")
Order: environments and data first, then models. All GPU steps through `scripts/replication/gpu_wait.sh`.

| item | script / command | GPU-h |
|---|---|---|
| Environments | `/workspace/envs/{viviano,pahde,saab}`, `/workspace/venv-wtss`; clones in `/workspace/third_party/`, then apply `third_party/<paper>/*.patch` (see "Machine layout") | 0 |
| Data | P1: NIH, PadChest (`images-224`), RSNA (Kaggle, token via env var only); P2: ISIC 2019 + their attacked set; P3: P3's sources + BIMCV (`bimcv_fetch.sh`) + BrixIA (`brixia_fetch.sh`; the private link was only in `/workspace/secrets/` on machine 1 and is lost with it — re-obtain it from the author's own records, never commit it); P4: ISIC + Bissoto trap sets | 0 (download) |
| Caches | P3 `p3_prepare.py`; P2 `p2_render.py`, lesion U-Net + ROIs (`p2_window.sh` steps 2, CPU/GPU ~1 h estimate); P1 ROIs via `p1_seed_pipeline.sh`; P1 location `p1_location_prep.py`; arm d caches are built by `armd_train.py` (now with byte-exact alignment assertions) | ~1 (estimate) |
| P2 Vanilla ResNet-50d (300 epochs) | `scripts/replication/run_pahde.sh results/replication/train_logs/pahde/vanilla_resnet50d_seed0/config.yaml 0 vanilla_resnet50d_seed0` | ~14–16 (2.8 min/epoch alone) |
| P2 section-3 RR-ClArC grid (λ 10/50 × 3 layers) + scores | `scripts/replication/p2_chain.sh` | ~4 (estimate) |
| P2 A18 RR-ClArC (λ 5e9, lr 1e-4, 3 layers), selection, arm a′ features | `scripts/replication/p2_window.sh` (selected: identity_2; config in `train_logs/pahde/`) | ~2–3 (estimate) |
| P3 ISNet corner, inside, inside 90/10 (48 epochs each, alone) | `scripts/replication/p3_chain.sh` (after `p3_prepare.py`) | ~35 (14.7 min/epoch alone, ~11.8 each) |
| P1 cardiomegaly Standard, seed 1234 | `scripts/replication/run_viviano.sh xray resnet 1234`, then `p1_seed_pipeline.sh xray 1234` (scores + arms a–c) | ~2.3 |
| P1 RSNA ActDiff, seed 1234 | `scripts/replication/run_viviano.sh rsna resnet_actdiff 1234`, then `p1_seed_pipeline.sh rsna 1234` | ~2.5 |
| P1 location inside / outside (A15–A17) | `/workspace/envs/viviano/bin/python scripts/replication/p1_location.py train|score --position inside|outside`; arms `p1_location_arms.py` | ~3 each (108 s/epoch alone) |
| Arm d encoders (EfficientNet-B0, A20/A21) | `scripts/replication/armd_queue.sh p1_cardio p1_rsna p1_loc_inside p1_loc_outside p2 p3_9010_inside` (needs each setting's data; P3 needs P3's cache, not the ISNet model); P4: `armd_p4_window.sh` | ~1.5 in total (0.1–0.4 each) |
| P4 Binary-ERM splits 1–3; Seg-ERM split 1 (machine 1's) | `/workspace/envs/saab/bin/python scripts/replication/p4_train.py --method erm|seg --split k [--grad_ckpt]` — **P4 is machine 2's (A27)**; machine 2 runs its own checkpoints, so these are not needed by its queue | ~1.8 per ERM split; Seg 2.9 to epoch 66 |

Total to rebuild everything machine 1 trained: roughly **65–70 GPU-h** (alone), plus downloads.

### The 43 lost files (SHA-256, for comparing a rebuild; paths relative to `$WTSS_DATA/replication/`)
A rebuild will not be bit-identical (non-deterministic GPU kernels); compare the per-image scores in git instead. The
hashes identify exactly which files the reported results came from.

| # | paper | file | bytes | SHA-256 | what it is |
|---|---|---|---|---|---|
| 1 | P2 | `pahde/runs/vanilla_resnet50d_seed0/checkpoints/config/checkpoint_resnet50d_last.pth` | 283,025,083 | `d7926681c90a1bd48019897dd63dea7bfb807c21b16aaf53bb274cfe96ce6460` | P2 Vanilla ResNet50d, seed 0, their recipe (last epoch, full training state) |
| 2 | P2 | `pahde/runs/vanilla_resnet50d_seed0/vanilla_weights_only.pth` | 94,507,857 | `a59e3e9e6ccab8618182b001bd03d6174a7d2842f71f476f3446f83305f70a2d` | P2 Vanilla weights only (A18 starting point: weights only, fresh Adam) |
| 3 | P2 | `pahde/runs/vanilla_resnet50d_seed0/config.yaml` | 559 | `df5a1557d76f01bb7bd4ff603a86ff8d25b911b14c9b8f6bd7030c4bdbb3a445` | P2 Vanilla training config |
| 4 | P2 | `pahde/checkpoints_corrected_a18/resnet50d_RRClarc_lamb5e9_adam_lr0.0001_identity_2/last.ckpt` | 94,509,351 | `c876df14962217e4d9b1cd9a7b27bc7331037fb58adb00fbdb983c6362dc1393` | P2 A18 selected RR-ClArC (lambda 5e9, lr 1e-4, identity_2); network of arm a' |
| 5 | P2 | `pahde/mitigation_a18/resnet50d_RRClarc_lamb5e9_adam_lr0.0001_identity_2.yaml` | 789 | `3fed1e4f5b02256476ca5cb05dcfee395125ee9df15bf8123880aca7dcbdcabc` | P2 A18 selected RR-ClArC config |
| 6 | P2 | `pahde/checkpoints_corrected/resnet50d_RRClarc_lamb10_adam_lr0.001_last_conv/last.ckpt` | 94,509,351 | `d742cb11800ebc11778dda491290f7cf800ff4b19deb39cdd137d7d547cd27f3` | P2 section-3 selected RR-ClArC (lambda 10, last_conv; scored, superseded by A18) |
| 7 | P2 | `pahde/mitigation/resnet50d_RRClarc_lamb10_adam_lr0.001_last_conv.yaml` | 799 | `372d45d0fa696e294d53c9c2562fa0d649a318ea3d954d5520abec6281755ada` | P2 section-3 selected RR-ClArC config |
| 8 | P3 | `isnet/runs/isnet_corner_all_seed0/best.pt` | 28,484,922 | `452090feb27c12ae34084843ecfa65acc20374333e1e4f259bb6712d81c56847` | ISNet corner benchmark (A12), best-validation weights (scored) |
| 9 | P3 | `isnet/runs/isnet_corner_all_seed0/checkpoint.pt` | 56,475,059 | `0ee42ce8fc1b2b1611cd9af0918629ad600f961ea4b40562de3182fe95ab361a` | ISNet corner, final training checkpoint |
| 10 | P3 | `isnet/runs/isnet_inside_all_seed0/best.pt` | 28,484,922 | `f8a673998790ff962ea33ccacae7bd081b5e5c1e4c6a8c22ca96d68a61c190d1` | ISNet marks inside lungs, all images, best-validation weights (scored) |
| 11 | P3 | `isnet/runs/isnet_inside_all_seed0/checkpoint.pt` | 56,474,803 | `e256b2da1e264fa31e4414f6a2d80a3914ded797529b99c5b75164e96bef2f21` | ISNet inside all, final training checkpoint |
| 12 | P3 | `isnet/runs/isnet_inside_9010_seed0/best.pt` | 28,484,922 | `bc063b4a5c5301ebabd5b150bbc5d08d659c4b092840853293229e49c71f4b2e` | ISNet marks inside, 90/10 (A6), best-validation weights (scored) |
| 13 | P3 | `isnet/runs/isnet_inside_9010_seed0/checkpoint.pt` | 56,474,803 | `23aade074233abffc55c9739585461997e439a72f741f89fadb78d3197a4554a` | ISNet inside 90/10, final training checkpoint |
| 14 | P1 | `viviano/runs/xray_resnet_1234/results/xray-seeds_resnet/best_model_1234.pth.tar` | 44,769,995 | `ca00c3773f138670fc97fd9fe6ebc36624d1a51d13ba607316c161da2d7732a4` | P1 cardiomegaly Standard, seed 1234, best (scored; arm a network) |
| 15 | P1 | `viviano/runs/xray_resnet_1234/results/xray-seeds_resnet/last_model_1234.pth.tar` | 44,769,995 | `c352c8f5f6aeca541b438d32b3f83fddaf4e10826faebf7a439387a615486285` | P1 cardiomegaly Standard, seed 1234, last |
| 16 | P1 | `viviano/runs/rsna_resnet_actdiff_1234/results/rsna-seeds_resnet_actdiff/best_model_1234.pth.tar` | 44,786,709 | `9b7c085d58be5604a6a93d9909319a49b5e08eec8d1de16e53aeaa0950c0b79a` | P1 RSNA ActDiff, seed 1234, best (scored; arm a network) |
| 17 | P1 | `viviano/runs/rsna_resnet_actdiff_1234/results/rsna-seeds_resnet_actdiff/last_model_1234.pth.tar` | 44,774,421 | `fe465e6bd0a83ba9c9e2080e1a99007c9af0f84c5ba8c848987accab7132db94` | P1 RSNA ActDiff, seed 1234, last |
| 18 | P1 | `viviano/location/inside/results/xray-location_inside/best_model_1234.pth.tar` | 44,769,995 | `05adcc19669995c5949f56a2947bf0223b5cf23df50313b380a5992420023b53` | P1 location test (A15), tube inside, best |
| 19 | P1 | `viviano/location/inside/results/xray-location_inside/last_model_1234.pth.tar` | 44,769,995 | `afecf9204dde7cbe63c355a10ed40fb1fcd1f9df8e0cdfdf506a916b6b911c28` | P1 location test (A15), tube inside, last (epoch 100; A17 post hoc) |
| 20 | P1 | `viviano/location/outside/results/xray-location_outside/best_model_1234.pth.tar` | 44,769,995 | `34bdfea857e8eb15734e84d40b8578f1b209697625ce5c196d52dfdb3f12180d` | P1 location test (A15), tube outside, best |
| 21 | P1 | `viviano/location/outside/results/xray-location_outside/last_model_1234.pth.tar` | 44,769,995 | `c17dec7e05ef0d2b754164f0d3422f54773165f043d91b7ee4c341b83591ab93` | P1 location test (A15), tube outside, last |
| 22 | armd | `armd/p1_cardio/best.pt` | 16,319,717 | `6e0fbcd26b109d58a8af753dc69b49fd58718cdb50894286b32883941dbb19fe` | Arm d EfficientNet-B0, P1 cardiomegaly 128 px (A20), best-validation |
| 23 | armd | `armd/p1_cardio/checkpoint.pt` | 48,609,083 | `5d1e51fc9a0c8fff617942857343647dc0ad901609ad7fc63fbceb17728e83ae` | Arm d P1 cardiomegaly, final checkpoint |
| 24 | armd | `armd/p1_rsna/best.pt` | 16,319,717 | `0dbffc042937aed905e1fa3be2f3897c7914c34a05d36602de4f53c0509ad424` | Arm d P1 RSNA 224 px, best-validation |
| 25 | armd | `armd/p1_rsna/checkpoint.pt` | 48,608,443 | `747c3323a67445a50ae9f15aa5ab5444736e29b17e6edf8cfdf8b7585fedf59c` | Arm d P1 RSNA, final checkpoint |
| 26 | armd | `armd/p1_loc_inside/best.pt` | 16,319,717 | `b6d17ff84acb219a8e17d14cdef49230abff10dd307169922c214478c5bc1993` | Arm d P1 location inside, best-validation |
| 27 | armd | `armd/p1_loc_inside/checkpoint.pt` | 48,609,019 | `81bcab0b2b932b69eab999b34511248a65fc41c7acf6fe19c4a138fdac1de636` | Arm d P1 location inside, final checkpoint |
| 28 | armd | `armd/p1_loc_outside/best.pt` | 16,319,717 | `86ef4d3163dd5cc0865413bf8bd911ff39d9d0bf62b437eebac850bde4b42c05` | Arm d P1 location outside, best-validation |
| 29 | armd | `armd/p1_loc_outside/checkpoint.pt` | 48,608,251 | `ce10bbbfbdddd1d74b478a8f87e4b8638ace7ddc69a00da33805dcd6c1def209` | Arm d P1 location outside, final checkpoint |
| 30 | armd | `armd/p2/best.pt` | 16,350,437 | `2947942351532fd00ee9ce0de104031550f84e423ba970a5f91857321ee54ef7` | Arm d P2, best-validation |
| 31 | armd | `armd/p2/checkpoint.pt` | 48,702,011 | `76139609c05a07c735e251f58c1f6e6ef934d48f6143f3764dc615bab511e787` | Arm d P2, final checkpoint |
| 32 | armd | `armd/p3_9010_inside/best.pt` | 16,324,837 | `99aff8ef85ac49aa729d17d8bb075f1a4bf34fddd6be4e676f8575603375b692` | Arm d P3 90/10 inside (A30 re-run, valid), best-validation |
| 33 | armd | `armd/p3_9010_inside/checkpoint.pt` | 48,624,635 | `700c0418f39e287d48bb6b04da390a28ad49ca497b20f99754f0fda19a6aab61` | Arm d P3 90/10 inside (A30 re-run), final checkpoint |
| 34 | armd | `armd/p4_split1/best.pt` | 16,319,717 | `daea99d8779c51114dd2eec752e3d67b96e3df714b840e4033c107c1c0060cf5` | Arm d P4 split 1 (512 px), best-validation |
| 35 | armd | `armd/p4_split1/checkpoint.pt` | 48,610,171 | `384989ded98e3b46f8fbb432b07f75dc0408a45f9860d05fba71ea4e4273d3c5` | Arm d P4 split 1, final checkpoint |
| 36 | P4 | `saab/runs/erm_split1/best.pt` | 94,348,589 | `f14ef375dee74de40a995342aba39db2be6647c5ad220ed11069198f23e99033` | P4 ERM split 1 (machine 1), best-validation |
| 37 | P4 | `saab/runs/erm_split1/checkpoint.pt` | 282,616,180 | `22930380a8ff37080114b0b196b704a0deeee7f725831031ad7c5bdb2ff34a04` | P4 ERM split 1, final checkpoint |
| 38 | P4 | `saab/runs/erm_split2/best.pt` | 94,348,589 | `c890213b41160adb82b29f347c859442299eb8a1f1ca7ce41c7c3f150264dff0` | P4 ERM split 2, best-validation |
| 39 | P4 | `saab/runs/erm_split2/checkpoint.pt` | 282,616,180 | `bf8e2a3fef02bce546d95a7f25ca11d3a8dcee44988b8e61329eeb3892351362` | P4 ERM split 2, final checkpoint |
| 40 | P4 | `saab/runs/erm_split3/best.pt` | 94,348,589 | `9f7d4799b6dd8900c55b88f6ee49db1bc762c454d48ebd45f312cf024eae820e` | P4 ERM split 3, best-validation |
| 41 | P4 | `saab/runs/erm_split3/checkpoint.pt` | 282,616,180 | `bf4608365edf32ece9696d85f0f28681cdba570d8579d895432cdc0f6735ef37` | P4 ERM split 3, final checkpoint |
| 42 | P4 | `saab/runs/seg_split1/best.pt` | 138,512,769 | `09286504ea7c61394abe60b20188b5201d169ab07673ea0ba09b4ad1e589fe99` | P4 segmentation-model split 1, best-validation (stopped at epoch 66) |
| 43 | P4 | `saab/runs/seg_split1/checkpoint.pt` | 414,959,259 | `9d380b204c618b5b12887408734e78b5ea658101665a2405650d5a272b950055` | P4 segmentation-model split 1, epoch-66 checkpoint (resumable) |

## Machine layout
- Repo `/root/Where_shortcut_sits` (container disk, NOT persistent); data and outputs under `$WTSS_DATA=/workspace/data`
  (1.1 TB host volume, persistent). Third-party clones `/workspace/third_party/{viviano,pahde,isnet,saab,bissoto}`; envs
  `/workspace/envs/{viviano,pahde,saab,tools}`; our env `/workspace/venv-wtss`.
- If the repo is lost: `git clone git@github.com:hossam1999/Where_shortcut_sits.git && git checkout claude/festive-bohr-95evo5`,
  then re-apply `third_party/<paper>/*.patch` to the clones (P1: `git -C /workspace/third_party/viviano apply …/cu128_port.patch`;
  P2: same with `pahde`). Private links (BrixIA) live only in `/workspace/secrets/` (never in git).
- GPU rules (mandatory): all GPU work through `scripts/replication/gpu_wait.sh` (memory + at most one other job); never kill
  by pattern; stop a job by PID **and its DataLoader workers** (`pgrep -P <pid>`), else they keep GPU memory.

## Machine 2 (A27) — **machine 2 released 2026-11-21; all results in git; no checkpoints kept** (machine 1 released 2026-11-20)

**Machine 2, second instance (2026-11-21, A43; P1 location test only).** A new machine (RTX 5070 Ti 16 GB, 28 cores, no
persistent disk) rebuilt P1's data (Kaggle mirrors of NIH 224 and PadChest 224, verified against `docs/DATA_MANIFEST.md`)
and the viviano env (torch 2.11 + cu128; numpy 1.26.4, pandas 1.5.3, opencv-python-headless 4.10.0.84, **scikit-learn 1.3.2**
because their `training.py` uses `OneHotEncoder(sparse=False)`). Rebuild check of seed 1234's test split: exact match.
Seeds 3232 and 3221 × inside / outside trained 4 at once (117 epochs/h total vs 60 for 2 and 33 for 1); scored; arms a–d.
**Done 2026-11-21 ~14:30: formal 3-seed result in `docs/REPLICATION_RESULTS.md`** (E-L1 supported; E-L2, E-L3 not
supported). Re-run: `P1_SEED=<s> WTSS_DATA=/workspace/data scripts/replication/m2_p1_loc.sh` (per seed: list → prep →
train both positions → score → arms a–c → arm d), then `venv-wtss/bin/python scripts/replication/p1_location_table3.py`.
No checkpoints kept (non-persistent disk).
Every experiment is finished; final tables: `docs/REPLICATION_RESULTS.md`. Model checkpoints, feature caches and data lived only on
machine 2's non-persistent disk and are gone with it; everything needed to re-run is in the scripts and documents below.
History of the machine-2 queue (kept for the record):
RTX 5070 Ti 16 GB, 28 cores, 62 GB RAM, **no persistent volume**: every result is pushed the moment it exists by
`scripts/replication/m2_push_results.sh` (detached worktree `/root/wtss_push`; small result files only, never checkpoints).
Repo `/root/Where_shortcut_sits`; `WTSS_DATA=/workspace/data`; clones `/workspace/third_party/{saab,bissoto,pahde,isnet,r2r,epvt,disc}`
(port patches in `third_party/<name>/`); envs `/workspace/venv-wtss`, `/workspace/envs/{saab,pahde,r2r,epvt,disc}`
(`docs/ENVIRONMENT_MACHINE2.md`). Logs `/workspace/logs/`. **Queue (author, 2026-11-19):** P8 runs → P8 gate → our P8 arms (both
A35 selection rules) → P6 A34 (5 seeds, A36 mixup) → phase 5 (timing test → `PREREGISTRATION_PHASE5.md` → WAIT for approval →
run) → P7.
| job | state | resume |
|---|---|---|
| P4 Saab | **final: gate failed twice (0.632; A31 0.647 vs 0.727); descriptive**; `docs/EMAIL_DRAFT_SAAB.md` (not sent) | — |
| P5 | dropped (step-1 check failed) | — |
| P8 DISC (A32, A35) ERM+DISC trap sets 1–5 | ERM 1 done; DISC 1 restarted 10:13 after two torch-2 ports (`third_party/disc/torch2_port.patch`) | `LANE=a WTSS_DATA=/workspace/data nohup scripts/replication/m2_p8_lanes.sh >> /workspace/logs/m2_p8_queue.log 2>&1 &` (locks `$WTSS_DATA/disc/locks/<m>_<k>`; remove a stale lock first; a folder without 101 test rows is rerun — move it aside) |
| P6 A33 rerun (ERM+EPVT bias 1, once) | **done 12:10: ERM 0.662 (pass), EPVT 0.655 (miss)**; was: `m2_pair_p6p8.sh` measures alone/paired, keeps pairing only if > 1.0, else reruns after P8 (`m2_p6_a33.sh`) | `nohup scripts/replication/m2_pair_p6p8.sh > /workspace/logs/m2_pair_p6p8.log 2>&1 &` |
| P8 two-lane pairing | `m2_pair_p8p8.sh` waits for the A33 results, then measures lane b beside lane a | `nohup scripts/replication/m2_pair_p8p8.sh > /workspace/logs/m2_pair_p8p8.log 2>&1 &` |
| P8 third lane | `m2_pair_p8x3.sh` after two lanes are kept: kept if total > 1.0, CPU < 85 %, total > two-lane total | `nohup scripts/replication/m2_pair_p8x3.sh > /workspace/logs/m2_pair_p8x3.log 2>&1 &`; concurrency file `$WTSS_DATA/disc/max_other_jobs` |
| Result pusher | every 2 min: finished P8/P6 runs, pairing JSONs, and anything under `$WTSS_DATA/results_push/<repo path>` → commit + push both branches | `WTSS_DATA=/workspace/data nohup scripts/replication/m2_push_results.sh > /workspace/logs/m2_push.log 2>&1 &` |
| Watchdog | logs failures / idle GPU > 10 min to STATUS ("Watchdog log"), retries a P8 job once | `WTSS_DATA=/workspace/data nohup scripts/replication/m2_watchdog.sh > /workspace/logs/m2_watchdog.log 2>&1 &` |
| Post-P8 chain | waits for the P8 runs, then gate → arms → P6 A34 | `WTSS_DATA=/workspace/data nohup scripts/replication/m2_p8_post.sh > /workspace/logs/m2_p8_post.log 2>&1 &` (each `p8_arms.py` stage skips finished work) |
| P8 DONE (2026-11-20) | gate FAILED (ERM 0.507, DISC 0.489); arms descriptive, all four above our DISC (b 0.736, d 0.691) — `results/replication/disc/` | — |
| P8 gate + arms a–d | after all 10 runs; primary: DISC's 154/618 split; secondary: fit 1,563, select on Bissoto's 259, score 618 (A35); bootstrap + Holm (A32) | not written yet |
| P6 A39 FINAL (2026-11-21) | reproduced at 0.05 (bias 1 and 0.9); arm b beats EPVT at both levels (+0.086 / +0.031, Holm p 0.0008), arm a both, arm d bias 1 only; A42 clean flag: b/a/c below EPVT − 0.01, d within — `results/replication/epvt/arms/`; section C cancelled | — |
| P6 FINAL (2026-11-20) | not reproduced: ERM 0.668 ± 0.017 (pass), EPVT (A36) 0.696 ± 0.006 vs 0.746 (miss); email draft `docs/EMAIL_DRAFT_YAN_EPVT.md` | — |
| A34 P6 5 seeds (A36 mixup) | A33 missed (EPVT 0.655): ERM seeds 1–4 + EPVT seeds 0–4 with A36 (paper Eq. 3); after P8 gate + arms, before P7; audit `docs/P6_EPVT_AUDIT.md` | `nohup scripts/replication/m2_p6_5seeds.sh > /workspace/logs/m2_p6_5seeds.log 2>&1 &` (needs `third_party/epvt/mixup_eq3.patch` applied in `/workspace/third_party/epvt`) |
| Phase 5 | trainer + 10 unit tests done (`scripts/phase5/p5_run.py`); timing test + prereg next, then wait | — |
| P7 RR-ClArC (A28) | running: their preprocessing → correction → scoring per architecture (ResNet-18, EfficientNet-B0, VGG-16) | `WTSS_DATA=/workspace/data nohup scripts/replication/m2_p7.sh > /workspace/logs/m2_p7.log 2>&1 &` (skips finished steps; needs `third_party/rrclarc/py311_port.patch` applied) |
| P7 RR-ClArC (old row) | step 1 reproduced (`results/replication/rrclarc/step1.md`); fine-tunes fill GPU during approvals | — |

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
### Heartbeat 2026-11-20 20:32 UTC (auto-generated)

GPU jobs (PID, MiB, command):
```
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
