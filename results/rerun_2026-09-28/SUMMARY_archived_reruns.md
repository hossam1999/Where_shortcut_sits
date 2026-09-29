# Regeneration of the four archived analyses (A1 addendum) and release handoff

Rule: `docs/PREREGISTRATION_FINAL.md`, A1 and its addendum (2026-09-29). Same commands, configurations and seeds; per-image predictions saved in `results/rerun_2026-09-28/`; crossed seed × image bootstrap (10,000). Full old-versus-new table: `results/bootstrap_correction/archived_old_vs_new.{csv,md}`.

## Input checks (rebuilt inputs against the archive; nothing was adjusted or re-run to get closer)

| analysis | rebuilt input | archived | rebuilt |
|---|---|---|---|
| LaMa | images inpainted (thyroid / ovary) | archived log no longer on disk; = images with a non-empty archived artifact mask: 1,692 / 854 | 1,692 / 854 |
| LaMa | trap cell counts (thyroid, ovary) | counts.csv | identical |
| capsule template | inputs | archived cached features reused | — |
| chest drains | drain detector held-out precision at the registered threshold (out-of-fold, NEATX-labelled) | 0.95 | 0.9501 (1,704 of 3,543 selected; CV AUROC 0.9955) |
| chest drains | real-drain cohort size | 29,687 | 29,687 |
| chest drains, devices | lung masks: IoU against the archived masks still on disk (`cache/images/nih_ptx_518`, 4,076 shared images) | — | mean 1.0000, min 0.9998; 4,072 identical |
| chest devices | RANZCR-CLiP images linked to NIH | 28,786 (of 30,083) | 28,786 (of 30,083) |

## Old versus new

| analysis | rows | regenerated | verdict changed | point estimate moved > 0.03 |
|---|---|---|---|---|
| capsule template U-MtE | 32 | 32 | 1 | 4 |
| LaMa | 19 | 19 | 2 | 2 |
| chest drains | 32 | 32 | 10 | 0 |
| chest devices | 180 | 180 | 19 | 9 |

Verdict changes (CI excludes zero, old → new):

- capsule template U-MtE: `trapA|mte_balanced|clean|-erm` +0.007 [-0.018, +0.034] → -0.033 [-0.066, -0.002]
- LaMa: `thyroid|mte_protect-mask_lama` -0.009 [-0.045, +0.023] → +0.060 [+0.013, +0.105]
- LaMa: `thyroid|mte_protect-mask_lama (universal run)` -0.009 [-0.045, +0.023] → +0.060 [+0.013, +0.105]
- chest drains: `mte_protect|mask|clean` -0.003 [-0.009, +0.002] → -0.004 [-0.009, -0.001]
- chest drains: `jtt|erm|clean` -0.012 [-0.019, -0.005] → -0.012 [-0.024, +0.003]
- chest drains: `mask|erm|test_rev` +0.012 [+0.002, +0.023] → +0.005 [-0.013, +0.023]
- chest drains: `mte|mte_aug|test_rev` -0.010 [-0.016, -0.002] → -0.002 [-0.010, +0.008]
- chest drains: `mte_protect|mask|test_rev` -0.011 [-0.019, -0.003] → -0.003 [-0.013, +0.006]
- chest drains: `mte_balanced|balanced|test_rev` +0.044 [+0.025, +0.064] → +0.032 [-0.022, +0.079]
- chest drains: `mask_dfr|dfr|test_rev` +0.018 [+0.001, +0.034] → +0.023 [-0.004, +0.051]
- chest drains: `mask_dfr|dfr|clean` +0.017 [-0.003, +0.039] → +0.017 [+0.000, +0.034]
- chest drains: `mte|jtt|test_rev` -0.010 [-0.019, -0.000] → -0.005 [-0.021, +0.011]
- chest drains: `mte|jtt|clean` +0.016 [+0.004, +0.028] → +0.012 [-0.005, +0.029]
- chest devices: `trapA|mask|clean|-erm` +0.016 [+0.001, +0.030] → +0.020 [-0.002, +0.040]
- chest devices: `trapB|prevcal|clean|-erm` -0.022 [-0.041, -0.004] → -0.022 [-0.053, +0.007]
- chest devices: `trapA|prevcal|clean|-erm` -0.036 [-0.070, -0.002] → -0.025 [-0.090, +0.034]
- chest devices: `trapA|balanced|clean|-erm` +0.060 [+0.034, +0.086] → +0.029 [-0.010, +0.070]
- chest devices: `trapA|leace_paired|clean|-erm` +0.029 [+0.012, +0.047] → +0.023 [-0.003, +0.049]
- chest devices: `trapA|leace_unpaired|clean|-erm` +0.063 [+0.033, +0.094] → +0.016 [-0.037, +0.066]
- chest devices: `trapA|i2e|clean|-erm` -0.004 [-0.016, +0.009] → -0.026 [-0.057, -0.004]
- chest devices: `trapA|i2e_balanced|clean|-erm` +0.058 [+0.030, +0.085] → +0.034 [-0.009, +0.078]
- chest devices: `trapB|inpaint|clean|-erm` +0.003 [+0.000, +0.006] → +0.003 [-0.000, +0.007]
- chest devices: `trapB|leace_paired|clean|-erm` +0.004 [+0.000, +0.008] → +0.004 [-0.001, +0.009]
- chest devices: `trapA|mask|clean|-erm` +0.029 [+0.009, +0.049] → +0.017 [-0.004, +0.038]
- chest devices: `trapA|i2e|test_rev|-erm` +0.034 [+0.016, +0.052] → +0.021 [-0.002, +0.045]
- chest devices: `trapA|i2e_rank1|test_rev|-erm` +0.008 [+0.001, +0.017] → +0.008 [-0.001, +0.018]
- chest devices: `trapB|prevcal|clean|-erm` -0.024 [-0.049, -0.000] → -0.024 [-0.055, +0.006]
- chest devices: `trapB|i2e|clean|-erm` +0.005 [+0.000, +0.010] → +0.005 [-0.001, +0.011]
- chest devices: `trapA|inpaint|clean|-erm` +0.008 [-0.003, +0.019] → +0.022 [+0.007, +0.039]
- chest devices: `trapA|dfr|clean|-erm` +0.036 [+0.005, +0.066] → +0.002 [-0.034, +0.035]
- chest devices: `trapB|mask|clean|-erm` +0.009 [+0.001, +0.021] → +0.010 [-0.001, +0.023]
- chest devices: `raddino518|Infiltration|crossover` -0.032 [-0.051, -0.013] → -0.025 [-0.055, +0.005]

Device-matched RAD-DINO follow-up (`cxr_traps/raddino518_devmatched`, 0 rows): **not re-estimated (dropped for time)** by the author's decision on 2026-09-29. Its archived point estimates and intervals stand unchanged and remain without per-image predictions:

- (no archived rows found)

Not re-estimated (no regenerated value): none.

LaMa: the thyroid U-MtE-protect change is a reproduction difference of the protected arm (see the provenance note in `archived_old_vs_new.md`); under A1 the sentence that U-MtE-protect matches LaMa-then-mask on thyroid no longer holds and must be rewritten.

## Theory (A3)

The chest-radiograph cells enter only the retrospective comparison (`theory_with_cxr/`); the regenerated folder cited by the paper (`theory/`) and the pre-registered prospective test (84 cells, round 4) are unchanged. Registered decision rule: quantitatively predictive only if every T2 crossover sign agrees, r ≥ 0.7 for T1 and T2, and T1 MAE < reference (a).

```
without chest X-ray cells (regenerated, as cited): T1 n=136 r=0.914 MAE=0.052 (ref clean 0.237); T2 signs 9/9 r=0.999; all signs agree: True; verdict: quantitatively predictive
with chest X-ray cells (retrospective): T1 n=154 r=0.880 MAE=0.053 (ref clean 0.245); T2 signs 12/13 r=0.999; all signs agree: False; verdict: qualitative account
```

## Preserved material (Part C)

- C3 blinding keys: `audit_local/KEY_open_after_review.csv` **match** `audit/KEY_SHA256.txt`; the A4b key **matches** `results/round6/a4b_KEY_SHA256.txt`. `results/round6/rating/` holds 1 uncommitted file (contents not opened).
- C4 archives in `/root/release_archive/` (per-file hashes: `results/ARCHIVE_MANIFEST.csv`):

```
dd46aea53947614b8fdebaa913c31c40c06a8623f8130968ce7700a1ff6a3b01  wtss_outputs.tar
510b7bcfa2cd52cd4a9522048c0df8d27d879aeb712c4d5e7f143d3ec3104381  wtss_private_keys.tar
wtss_outputs.tar: 2280202240 bytes
wtss_private_keys.tar: 61440 bytes
```

Archive log (files, exclusions):

```
reused staged tar (447 files); appended 156
dd46aea53947614b8fdebaa913c31c40c06a8623f8130968ce7700a1ff6a3b01  wtss_outputs.tar
510b7bcfa2cd52cd4a9522048c0df8d27d879aeb712c4d5e7f143d3ec3104381  wtss_private_keys.tar
outputs: 603 files, 2280202240 bytes in 1 part(s); excluded: 0; private: 3 files
```
- Feature caches are not exported (5.5 GB under `$WTSS_CACHE/features`); rebuild commands: `docs/AFTER_RELEASE.md`.
- Trained weights in the archive: spec lesion U-Net, ISIC 2020 hair segmenter, capsule debris probe, drain detector (parameters exported from an identical refit that reproduces the saved scores to 1e-7). Not on disk and therefore not archived: the original ISIC 2018 lesion U-Net (`unet_resnet34_384.pt`) and fine-tuned network checkpoints (the fine-tuning code never saved them). The caliper detector is rule-based (no weights).

## Final checks (D1, D3)

```
$ make test
PYTHONPATH=src TQDM_DISABLE=1 python -m pytest -q tests
..................................................................       [100%]
66 passed in 7.25s
$ python scripts/verify/audit_numbers_final.py --docs all
3742 intervals and 7743 three-decimal numbers checked in all; 0 failed to trace
$ make verify
15                leace      0.5                 0.237         0.212         0.263                0.237        0.212        0.263      -0.000                       True
16                  dfr      0.5                 0.221         0.184         0.261                0.221        0.184        0.261      -0.000                       True
17             balanced      1.0                 0.281         0.249         0.315                0.281        0.248        0.315      -0.000                       True
18             groupdro      1.0                 0.197         0.168         0.226                0.197        0.168        0.226      -0.000                       True
19                leace      1.0                 0.294         0.265         0.325                0.294        0.265        0.325      -0.000                       True
20                  dfr      1.0                 0.279         0.240         0.319                0.279        0.240        0.320      -0.000                       True
$ secret scan of git log -p --all
history scanned: 54237943 characters
Kaggle API token (KGAT_...): 0 match(es)
Hugging Face token (hf_ + 30 chars): 0 match(es)
GitHub token (ghp_/gho_/ghs_/github_pat_): 0 match(es)
kaggle.json key field: 0 match(es)
private key block: 0 match(es)
assignment of a token value (token=/api_key=/password= followed by >= 20 chars): 0 match(es)
mentions of the word 'kaggle' (commands and documentation, not credentials): 51
```

## Download (author)

```bash
rsync -avP -e "ssh -p <PORT>" root@<HOST>:/root/release_archive/ ./wtss_release_archive/
cd wtss_release_archive && sha256sum -c SHA256SUMS
```

