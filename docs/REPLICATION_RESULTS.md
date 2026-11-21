# Replication results: P1 Viviano 2021, P2 Pahde 2025, P3 Bassi/ISNet 2024, P4 Saab 2022, P5 R2R, P6 Yan/EPVT 2023, P7 Dreyer/RR-ClArC 2024, P8 Wu/DISC 2023, and phase 5

Status: **final** (machine 2, 2026-11-21; machine 1 released 2026-11-20). Every number traces to a file under
`results/replication/`. Rules: `docs/PREREGISTRATION_REPLICATION.md` (amendments A1–A42) and `docs/PREREGISTRATION_PHASE5.md`.
Phase-0 checks, deviations and engineering notes: `docs/REPLICATION_STATUS.md`.

## Final summary (2026-11-21)

**Tiers.** **Tier 1** = the paper's best method was reproduced (gate passed), so "beaten / not beaten" verdicts apply
(section 2: higher mean, 95 % CI of the paired difference above 0, Holm-adjusted p < 0.05). **Tier 2** = not reproduced:
descriptive comparisons against **our run of their official code** under identical conditions; no claim against the published
number. **Confirmatory** = phase 5: our test design, their methods run by us under one pre-registered rule.
**Tolerance:** the reproduction tolerance was widened from 0.03 to 0.05 for every paper (P2–P8) on 2026-11-20, **after the first
gate results were known** (A38); unrounded gaps at both tolerances: `results/replication/GATES_TOLERANCE.md`.

| paper (artifact) | reproduced? (gap; tolerance 0.05) | tier | our best arm vs their method (our run) | verdict / reading |
|---|---|---|---|---|
| P1 Viviano, ICLR 2021 (global hospital / view shortcut) | gate passed on kept runs; 1 seed (A14) | preliminary | arm b: cardiomegaly +0.040 vs Masked, RSNA +0.059 vs ActDiff | **no verdict** (1 seed); secondary (not localized) |
| P1 location test (tube inside / outside the lungs + heart; A15–A17, A43) | — (our test in their pipeline) | **formal** (3 seeds; crossed bootstrap; Holm over 5) | arms a / b gain inside +0.195 / +0.204 (CIs reach −0.006) | **E-L1 supported (by the letter); E-L2 (a, b, d) and E-L3 not supported**; none contradicted |
| P2 Pahde, ML 2025 (inserted microscope) | **no** (RR-ClArC 0.524 vs 0.76; gap 0.236) | 2 | arm a′ 0.667 vs RR-ClArC 0.524 (+0.143) | descriptive |
| P3 Bassi, Nat. Commun. 2024 (marks) | **no** (A9: 0.223 / 0.260 vs 0.773) | 2 | see P3 section (inside-the-lungs controlled comparison) | descriptive |
| P4 Saab, MLHC 2022 (ISIC 2018 trap sets) | **no** (Seg-ERM 0.647 vs 0.727; gap 0.080) | 2 | arm b 0.670 vs Seg-ERM 0.647 (+0.023, n.s.) | descriptive: **no arm above Seg-ERM** |
| P5 R2R (LSB / attacked ISIC) | **no** (step-1 checkpoints: 0.838 / 0.552 vs 0.895 / 0.356) | — | not run (dropped) | — |
| **P6 Yan, MICCAI 2023 (EPVT; ISIC 2019 trap sets)** | **yes** at 0.05 (bias 1: EPVT gap 0.0497, within the ±0.005 digitisation uncertainty; bias 0.9: 0.017) | **1** | **arm b +0.086 (bias 1) / +0.031 (bias 0.9)**, Holm p 0.0008 | **beaten** at both levels (arm b; also arm a); arm d beaten at bias 1 only |
| P7 Dreyer, AAAI 2024 (RR-ClArC; invisible LSB trigger) | **yes** (ResNet-18 0.713 vs 0.712; EffNet-B0 0.728 vs 0.756; VGG-16 0.531 vs 0.570) | 1 | arm a 0.709 vs 0.713 (ResNet-18) | **not beaten** (secondary: whole-image artifact) |
| P8 Wu, ICML 2023 (DISC; ISIC 2018 trap sets) | **no** (ERM 0.507 vs 0.364; DISC 0.489 vs 0.551) | 2 | arm b 0.736 vs DISC 0.489 (+0.247 [+0.204, +0.290]) | descriptive: all four arms above our DISC run |
| **Phase 5 (thyroid TN3K calipers; Trap A inside the ROI)** | — | **confirmatory** | **DINOv2 + mask_cmc 0.716, arm d 0.690 vs ISNet 0.561, RR-ClArC 0.467** | **beaten: 4 / 4 registered comparisons** (clean condition met) |

**Caveats that travel with these labels.** P6's bias-1 pass is inside 0.05 by less than the digitisation uncertainty; P6
had **no registered clean condition** — the reported clean check (A42, `test_bias_0_1`) shows arm b 0.02–0.04 AUROC below
EPVT, arm d within 0.01. Phase 5's comparisons against ISNet have wide CIs (ISNet's Trap A fold 2 collapsed to 0.365).
P8: the published numbers may carry a shared-cache bug of DISC's released code (A37); P6: the released EPVT mixup pairing
differs from the paper (A36; we follow the paper). Several heads chose the lowest C of their registered grid (reported in each
section). New-method track: **cancelled** (author's rule after A39).

## How to read this report

| comparison | role | why |
|---|---|---|
| P2 head-to-head on its own biased test | **primary** | localized artifact (inserted "microscope" vignette) |
| P3 head-to-head on its own o.o.d. test | **primary** | localized artifact (corner marks) |
| P3 location test (marks moved inside the lungs, phase 4) | **primary; the strongest location result** | ISNet's own design assumes the mark lies outside the lungs |
| P1 location test (tube inside vs outside the lungs + heart ROI, phase 4) | **primary for P1** | it addresses the thesis question (where the shortcut sits) |
| P1 head-to-head on its original tests | **secondary**: *global shortcut (hospital / AP-PA view), not a localized artifact* | the signal is spread over the whole image, inside and outside the mask at once, so "inside vs outside the ROI" does not apply |

"Preliminary" tables (one seed of P1) carry no verdict. With one training seed (P2, P3, phase 4) the bootstrap resamples test
images only, so training-seed variance is not included and "beaten" means "beaten for this training run" (A3).

## P1 — Viviano et al., ICLR 2021 (secondary head-to-head; primary location test)

### Finding: P1's RSNA "Masked" test input depends on the label

In P1's RSNA pneumonia task, "Masked" multiplies each image by its pneumonia bounding boxes. Negatives have no box.

- **Code** (P1 at `32f7b7d`, TorchXRayVision 0.0.18 as pinned by P1):
  - `torchxrayvision/datasets.py`, `RSNA_Pneumonia_Dataset.get_mask_dict` (lines 492–515): every mask starts as
    `np.zeros([size, size])`, and only the boxes are set to 1.
  - `activmask/datasets/xray.py`, `JointXRayRSNADataset.__getitem__`: `seg = sample["pathology_masks"][0]` (line 650),
    then `img *= seg` when `mask_all` is set (lines 669–671).
  - The Masked config sets `mask_all` on the test split. Because of a YAML duplicate key it is set on the test split *only*;
    see `REPLICATION_STATUS.md`.
- **Statistic** (seed 1234 test split): 200 of the 676 test negatives were inspected, and **every one becomes the same
  constant image (pixel SD 0, value 0.0)**. Positives keep only their box contents (3,396 of 50,176 pixels on average).
  By the code above this holds for every negative.
- **Figure:** `figures/replication/p1_rsna_masked_test_masks.png` shows the masks applied (4 negatives: all zeros;
  4 positives: box masks). No X-ray pixels are committed; the masked X-ray examples stay under `$WTSS_DATA`.
- **Consequence** (A11): their RSNA Masked score (paper 0.56 ± 0.16) is reported as **label-dependent at test time** and is
  not a head-to-head target. The RSNA target is ActDiff (0.68 ± 0.05). Our arm a on RSNA uses label-free PSPNet lung ROIs.
  Cardiomegaly (circle mask, label-free) is unaffected.

### Reproduction (their code; 3 seeds: 1234, 3232, 3221)
*pending (gate after seed 1234; A8)*

### Head-to-head on P1's original tests (secondary: global shortcut (hospital / AP-PA view), not a localized artifact)
**Preliminary, 1 seed (1234), no verdict (A14).** Paired image-only bootstrap; training-seed variance not included. Full tables:
`results/replication/viviano/preliminary_xray_1234.md`, `preliminary_rsna_1234.md`; per-image test scores in
`results/replication/viviano/scores/`.

| task (target) | target AUROC | Standard | arm a | arm b | arm c (U-MtE) |
|---|---|---|---|---|---|
| cardiomegaly (Masked, A5) | 0.829 (paper 0.77 ± 0.09) | 0.627 (paper 0.70) | 0.866, Δ +0.037 [+0.031, +0.043] | 0.869, Δ +0.040 [+0.026, +0.054] | 0.593, Δ −0.236 [−0.256, −0.216] |
| RSNA pneumonia (ActDiff, A11) | 0.708 (paper 0.68 ± 0.05) | 0.186 (paper 0.20) | 0.610, Δ −0.099 [−0.132, −0.065] | 0.767, Δ +0.059 [+0.021, +0.098] | 0.378, Δ −0.330 [−0.375, −0.285] |

Notes:
- **Arm a on RSNA** (`a_mask_cmc_their_net`) is the registered arm a: mask_cmc on the ActDiff network's features of
  PSPNet-lung-masked images (A11 ROI, A12 network). Verified: the ActDiff network's final layer reproduces the saved logits of
  the feature pass to 5e-7, the Masked network's does not (1.7). The table's *diagnostic* row (`arma_lungs`) is the ActDiff
  network's own logits on those lung-masked images (0.689): lung masking alone barely changes ActDiff; it is not a method.
- **RSNA Masked** (0.759 in their definition, 0.840 as logit-difference AUROC) is label-dependent at test time (A11). The two
  definitions disagree because every negative gets the identical score, and ties are handled differently.
- **Clean (within-site / within-view) AUROC:** RSNA ActDiff 0.471 (below chance): its reversed-test AUROC comes mostly from the
  view, not the disease. Arm b keeps 0.763.
- U-MtE is designed for localized overlays inside the ROI and does not remove a global shortcut; it fails on both P1 tasks.

### Location test (primary for P1; phase 4; A15, A16; post-hoc sensitivity A17)

#### Formal result: 3 seeds (A43; machine 2, 2026-11-21) — **this is the registered P1 location result**
Seeds **1234** (machine 1, committed per-image scores; **different GPU and software build**; its results below were seen
before A43), **3232** and **3221** (machine 2: RTX 5070 Ti, torch 2.11 + cu128, scikit-learn 1.3.2 pinned for their code).
Before any training the rebuilt seed-1234 test split matched the committed files exactly (ids, order, labels, tube flags;
`results/replication/viviano/location_rebuild_check.md`). Their code and Standard config, their epoch-selection rule; Masked =
the Standard model with P1's circle mask (A5); arms a–d as registered (arm d via A20). Reversed-test AUROC as the 3-seed mean
± SD; gain and Δ = paired difference on the reversed test; crossover = gain outside − gain inside. **Crossed seed × image
bootstrap**, 10,000 replicates (one Poisson weight per test image, shared by seeds, methods and positions; seeds resampled).

| method | position | reversed (mean ± SD) | correlated | clean | gain over Standard [95% CI] | Δ vs Masked [95% CI] | crossover [95% CI] |
|---|---|---|---|---|---|---|---|
| Standard (theirs) | inside | 0.672 ± 0.187 | 0.933 | 0.885 | — | — | — |
| Standard (theirs) | outside | 0.913 ± 0.017 | 0.918 | 0.916 | — | — | — |
| Masked (theirs, A5) | inside | 0.615 ± 0.195 | 0.912 | 0.845 | −0.057 [−0.082, −0.039] | — | +0.040 [+0.018, +0.067] |
| Masked (theirs, A5) | outside | 0.896 ± 0.013 | 0.896 | 0.896 | −0.017 [−0.025, −0.010] | — | (same) |
| a: mask_cmc on their net | inside | 0.867 ± 0.010 | 0.899 | 0.889 | +0.195 [−0.006, +0.338] | +0.252 [+0.037, +0.384] | −0.195 [−0.324, −0.007] |
| a: mask_cmc on their net | outside | 0.913 ± 0.007 | 0.913 | 0.913 | +0.000 [−0.012, +0.013] | +0.017 [+0.009, +0.026] | (same) |
| b: DINOv2 + mask_cmc | inside | 0.876 ± 0.005 | 0.912 | 0.897 | +0.204 [−0.006, +0.359] | +0.261 [+0.038, +0.404] | −0.215 [−0.354, −0.015] |
| b: DINOv2 + mask_cmc | outside | 0.902 ± 0.002 | 0.902 | 0.902 | −0.011 [−0.025, +0.005] | +0.006 [−0.007, +0.020] | (same) |
| c: U-MtE | inside | 0.618 ± 0.021 | 0.963 | 0.890 | −0.054 [−0.257, +0.080] | +0.004 [−0.214, +0.131] | +0.055 [−0.068, +0.251] |
| c: U-MtE | outside | 0.915 ± 0.006 | 0.915 | 0.915 | +0.002 [−0.010, +0.014] | +0.019 [+0.009, +0.029] | (same) |
| d: EfficientNet-B0 + mask_cmc (A20) | inside | 0.754 ± 0.023 | 0.964 | 0.910 | +0.082 [−0.129, +0.218] | +0.139 [−0.086, +0.269] | −0.072 [−0.194, +0.127] |
| d: EfficientNet-B0 + mask_cmc (A20) | outside | 0.924 ± 0.004 | 0.924 | 0.924 | +0.010 [−0.003, +0.026] | +0.027 [+0.016, +0.040] | (same) |

**Registered expectations (section 5), Holm over the 5 tests of phase 4.** The family is P1's five: P3's location test
ended descriptive (A9 check failed) and has no E-L test, so the earlier note "the family also contains the P3 location test"
no longer applies.

| expectation | Δ [95% CI] | p | p Holm | verdict |
|---|---|---|---|---|
| E-L1 Masked: crossover > 0, gain inside ≤ 0 or CI includes 0 | +0.040 [+0.018, +0.067]; gain inside −0.057 | 0.0001 | 0.0005 | **supported** (by the letter; see below) |
| E-L2 arm a: gain inside > 0 | +0.195 [−0.006, +0.338] | 0.074 | 0.276 | **not supported** |
| E-L2 arm b: gain inside > 0 | +0.204 [−0.006, +0.359] | 0.069 | 0.276 | **not supported** |
| E-L2 arm d (A20): gain inside > 0 | +0.082 [−0.129, +0.218] | 0.52 | 1.00 | **not supported** |
| E-L3 arm c (U-MtE): gain inside > 0 | −0.054 [−0.257, +0.080] | 0.61 | 1.00 | **not supported** |

No expectation is contradicted. **Tier: formal (3 seeds, crossed bootstrap, Holm).** Sources:
`results/replication/viviano/location_3seed.{md,csv}`, `location_3seed_expectations.csv`; per-image scores
`results/replication/viviano/scores/location_{inside,outside}[_<arm>]_{1234,3232,3221}.csv.gz`; arm d per seed
`results/replication/armd/p1_loc_{inside,outside}_s{3232,3221}.md`.

**Per seed, reversed-test AUROC** (Standard: correlated / reversed / clean; the epoch their rule kept):

| | seed 1234 | seed 3232 | seed 3221 |
|---|---|---|---|
| Standard inside | 0.943 / 0.514 / 0.866 (epoch 11) | 0.887 / **0.879** / 0.884 (**epoch 5**) | 0.971 / 0.624 / 0.903 (epoch 15) |
| Standard outside | 0.901 / 0.894 / 0.898 (epoch 8) | 0.930 / 0.923 / 0.927 (epoch 12) | 0.923 / 0.923 / 0.923 (epoch 9) |
| Masked inside | 0.469 | 0.836 | 0.540 |
| a inside | 0.856 | 0.871 | 0.874 |
| b inside | 0.877 | 0.870 | 0.880 |
| c inside | 0.598 | 0.618 | 0.640 |
| d inside | 0.735 | 0.746 | 0.780 |

**Reading.** (1) The arms' own inside AUROC is stable across seeds (a 0.856–0.874, b 0.870–0.880). What varies is the
Standard reference. **At seed 3232 their selection rule kept epoch 5** (the end of burn-in), before the inside marker was
learned: the kept Standard model has no inside shortcut (correlated 0.887 ≈ reversed 0.879), so the arms gain ≈ 0 there
(a −0.008, b −0.009), against +0.34 / +0.25 (a) and +0.36 / +0.26 (b) at seeds 1234 / 3221. With 3 seeds resampled, this
spread puts the lower CI bound of arms a and b just below 0 (−0.006), so E-L2 is not supported as registered. Their
paired advantage over Masked inside is significant before multiplicity correction (a +0.252 [+0.037, +0.384], b +0.261
[+0.038, +0.404]) but was not a registered test. (2) At the outside position the kept Standard model has no shortcut at any
seed (correlated ≈ reversed; as found for seed 1234), so E-L1's crossover is "supported by the letter": it reflects Masked
costing less outside than inside, not Masked removing an outside shortcut. (3) U-MtE (arm c) does not remove an inside
marker (reversed 0.62, correlated 0.96). Seed 1234's one-seed "supported" for E-L3 (+0.084, image-only CI) does not hold up: arm c gains
+0.016 at seed 3221 and −0.261 at seed 3232 (where Standard has no inside shortcut). (4) A17's epoch-100 sensitivity analysis was not repeated for
the new seeds (A43).

#### Seed 1234 alone (machine 1; A16, registered one-seed table, superseded by the 3-seed result above)
**Design.** P1's own pipeline (their code, Standard config, seed 1234) on P1's balanced cardiomegaly split. The artifact
is **a synthetic bar-shaped radio-opaque marker (43 x 6 px at 224 px; straight, bright, with a central stripe), not a
realistic tube**. It is drawn on the 224-px image after their intensity normalisation and before their resize to 128 px.
Real chest drains look different (long, thin, curved, faint; 9 examples, marker vs NEATX drains, were sent to the author
on 2026-11-18 and are not committed because they contain patient pixels); realism is covered elsewhere in the thesis
(real device traps on NIH / RANZCR; transplant of real artifacts). The marker sits either inside the lungs ∪ heart ROI
(PSPNet mask; 99.95% of its pixels inside; 98.6% inside P1's circle) or outside both the ROI and P1's circle (0% inside
either). The outside placements lie near the image edge: below the lungs + heart near the bottom edge in 47.5% of test
images, above the lungs near the top edge in 38.2% (both: median 9–10 px from that edge at 224 px), beside the ROI in
14.3% (`results/replication/viviano/location_outside_regions.csv`). **Limitation:** because "outside" must avoid P1's
central circle (and lungs ∪ heart), outside markers necessarily lie near the image border (median 8 px from the nearest
edge at 224 px; 99.7% within 25 px of an edge) and inside markers near the centre (median 55 px; 2.6% within 25 px), so
position relative to the ROI is partly confounded with centre vs edge (`results/replication/viviano/location_edge_distance.csv`);
the thesis's transplant experiment (the same real artifact moved across a lesion boundary) addresses location without
this confound.
Presence: 90% of positives / 10% of negatives in training; validation reversed (P1's design); tests on P1's test split:
correlated (90/10), reversed (10/90), clean (none). Masked = the same Standard model scored with P1's circle mask (A5).

**Registered result (primary).** One seed: paired image-only bootstrap, 10,000 replicates; training-seed variance is not
included (A3). Gain and Δ are on the reversed test; crossover = gain outside − gain inside.

| position | method | kept by circle | kept by lungs + heart | correlated | reversed | clean | reversed gain vs Standard [95% CI] | Δ vs Masked (reversed) [95% CI] | crossover [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| inside | Standard (theirs) | 0.986 | 1.000 | 0.943 | 0.514 | 0.866 | +0.000  | +0.045 [+0.035, +0.054] | — |
| inside | Masked (theirs, A5) | 0.986 | 1.000 | 0.935 | 0.469 | 0.837 | -0.045 [-0.054, -0.035] | — | +0.032 [+0.023, +0.041] |
| inside | a: mask_cmc on their net | 0.986 | 1.000 | 0.893 | 0.856 | 0.884 | +0.342 [+0.324, +0.360] | +0.387 [+0.369, +0.405] | -0.328 [-0.347, -0.310] |
| inside | b: DINOv2 + mask_cmc | 0.986 | 1.000 | 0.904 | 0.877 | 0.895 | +0.364 [+0.343, +0.384] | +0.408 [+0.387, +0.430] | -0.359 [-0.379, -0.338] |
| inside | c: U-MtE | 0.986 | 1.000 | 0.960 | 0.598 | 0.881 | +0.084 [+0.063, +0.105] | +0.129 [+0.107, +0.150] | -0.070 [-0.092, -0.049] |
| outside | Standard (theirs) | 0.000 | 0.000 | 0.901 | 0.894 | 0.898 | +0.000  | +0.013 [+0.006, +0.019] | — |
| outside | Masked (theirs, A5) | 0.000 | 0.000 | 0.881 | 0.881 | 0.881 | -0.013 [-0.019, -0.006] | — | +0.032 [+0.023, +0.041] |
| outside | a: mask_cmc on their net | 0.000 | 0.000 | 0.908 | 0.908 | 0.908 | +0.014 [+0.007, +0.021] | +0.027 [+0.021, +0.033] | -0.328 [-0.347, -0.310] |
| outside | b: DINOv2 + mask_cmc | 0.000 | 0.000 | 0.899 | 0.899 | 0.899 | +0.005 [-0.004, +0.015] | +0.018 [+0.007, +0.029] | -0.359 [-0.379, -0.338] |
| outside | c: U-MtE | 0.000 | 0.000 | 0.908 | 0.908 | 0.908 | +0.014 [+0.004, +0.023] | +0.026 [+0.015, +0.037] | -0.070 [-0.092, -0.049] |

Source: `results/replication/viviano/location_1234.{md,csv}`; per-image scores in `results/replication/viviano/scores/`.

**Why the outside Standard model shows no shortcut (checked; not a bug).** The marker reaches the network at the outside
position (about 109 changed pixels per image at 128 px, as inside; presence as registered). Their selection rule
(`training.py`: burn-in 5 epochs, then the lowest loss on P1's reversed validation split) kept epoch 11 inside, after the
marker was learned (epoch 5), but **epoch 8 outside, before the marker was learned (epoch 11)**; late outside epochs reach
only reversed-validation AUROC ≈ 0.70–0.75, so the outside marker is also learned less strongly. At the outside position
there is therefore no shortcut in the kept model for any method to remove, and the outside rows are near Standard for
every method. Masked, arm a and arms b and c score identically on the three outside tests because their masks remove the
whole marker.

**Seed 1234 alone, expectations before Holm (superseded by the 3-seed analysis above; kept as seen before A43):**
- E-L1 (Masked): crossover +0.032 [+0.023, +0.041] > 0 and gain inside −0.045 ≤ 0 → *supported by the letter*, but not
  for the reason the expectation states: the outside Standard model has no shortcut, so the crossover reflects Masked
  costing less outside than inside, not Masked removing an outside shortcut.
- E-L2 (mask_cmc, arms a and b): gain inside +0.342 [+0.324, +0.360] and +0.364 [+0.343, +0.384] → supported.
- E-L3 (U-MtE, arm c): gain inside +0.084 [+0.063, +0.105] → supported; much smaller than mask_cmc, because the marker
  lies inside the region U-MtE keeps and U-MtE is label-free.

**Post-hoc sensitivity analysis (A17; every number POST HOC, descriptive, no verdict).** Their epoch-100 models
(`last_model_1234.pth.tar`) at both positions: Standard, Masked and arm a; arms b and c reused unchanged (their gains
change only because the Standard reference is now the epoch-100 model). Same test images and bootstrap as the registered
table. Recorded as A17 before any epoch-100 model was scored.

| position | method | kept by circle | kept by lungs + heart | correlated | reversed | clean | reversed gain vs Standard [95% CI] | Δ vs Masked (reversed) [95% CI] | crossover [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| POST HOC inside | Standard (theirs) | 0.986 | 1.000 | 0.966 | 0.468 | 0.897 | +0.000  | +0.042 [+0.025, +0.058] | — |
| POST HOC inside | Masked (theirs, A5) | 0.986 | 1.000 | 0.929 | 0.426 | 0.799 | -0.042 [-0.058, -0.025] | — | +0.211 [+0.193, +0.229] |
| POST HOC inside | a: mask_cmc on their net | 0.986 | 1.000 | 0.921 | 0.633 | 0.844 | +0.166 [+0.150, +0.182] | +0.208 [+0.198, +0.218] | +0.006 [-0.013, +0.025] |
| POST HOC inside | b: DINOv2 + mask_cmc | 0.986 | 1.000 | 0.904 | 0.877 | 0.895 | +0.410 [+0.390, +0.429] | +0.452 [+0.430, +0.473] | -0.225 [-0.242, -0.207] |
| POST HOC inside | c: U-MtE | 0.986 | 1.000 | 0.960 | 0.598 | 0.881 | +0.130 [+0.112, +0.149] | +0.172 [+0.149, +0.194] | +0.063 [+0.042, +0.085] |
| POST HOC outside | Standard (theirs) | 0.000 | 0.000 | 0.962 | 0.714 | 0.910 | +0.000  | -0.169 [-0.183, -0.156] | — |
| POST HOC outside | Masked (theirs, A5) | 0.000 | 0.000 | 0.884 | 0.884 | 0.884 | +0.169 [+0.156, +0.183] | — | +0.211 [+0.193, +0.229] |
| POST HOC outside | a: mask_cmc on their net | 0.000 | 0.000 | 0.886 | 0.886 | 0.886 | +0.172 [+0.157, +0.187] | +0.003 [-0.006, +0.011] | +0.006 [-0.013, +0.025] |
| POST HOC outside | b: DINOv2 + mask_cmc | 0.000 | 0.000 | 0.899 | 0.899 | 0.899 | +0.185 [+0.169, +0.201] | +0.016 [+0.004, +0.027] | -0.225 [-0.242, -0.207] |
| POST HOC outside | c: U-MtE | 0.000 | 0.000 | 0.908 | 0.908 | 0.908 | +0.193 [+0.177, +0.210] | +0.024 [+0.013, +0.036] | +0.063 [+0.042, +0.085] |

POST HOC reading (descriptive, no verdict): with both positions scored at the end of their training run, the outside
Standard model has learned the marker (correlated 0.962 vs reversed 0.714), Masked removes it outside (gain +0.169
[+0.156, +0.183]) and not inside (−0.042 [−0.058, −0.025]), and its crossover is +0.211 [+0.193, +0.229]: the location
pattern E-L1 describes, which the registered selection rule had hidden. Arm a (mask_cmc on the epoch-100 network) gains
inside (+0.166) and outside (+0.172), with no crossover (+0.006 [−0.013, +0.025]); its inside gain is half the
registered one (+0.342), as the epoch-100 network's features carry the marker more strongly. Arms b and c gain at both
positions. One seed; image-only bootstrap. Source: `results/replication/viviano/location_1234_posthoc_last.{md,csv}`.

## P2 — Pahde et al., Machine Learning 2025 (primary)

**Reproduction: not reproduced (final; author's decision 2026-11-18).** Target: RR-ClArC biased-test accuracy 0.76
(Table 1, ResNet-50, ISIC 2019). Primary run (A18, the paper's stated settings: λ = 5·10⁹ from Tab. 12, lr 1e-4 = training
lr / 10 from p. 31, our Vanilla's weights with a fresh Adam, 10 epochs, layer chosen on validation): **0.524 [0.505, 0.543]**,
clean 0.839. The section-3 runs (their code path: λ ∈ {10, 50} from their config generator, effective lr 1e-5 because
their `get_optimizer` loads the Vanilla's decayed optimizer state) gave 0.338. Our Vanilla matches the paper (clean 0.839
vs 0.87; biased 0.311 vs 0.28), so the biased test is built as theirs. The exact configuration behind Table 1 has been
requested from the authors (`docs/EMAIL_DRAFT_PAHDE_RRCLARC.md`). **No verdict** in any comparison with RR-ClArC
(section 6); the comparisons below are descriptive. The decisive RR-ClArC comparison is carried by phase 5 (thyroid Trap
A, both methods run by us at the paper's stated settings).

**Selection criterion.** RR-ClArC's layer (among 3) and our arms' C (among 4) are chosen by the same criterion: the mean of
clean and biased accuracy on P2's 2,533 validation images (accuracy of the argmax prediction). RR-ClArC is taken at the
end of its 10 epochs (`last.ckpt`, as in their evaluation); no method selects an epoch.

**All methods, P2's tests (seed 0; image-only bootstrap, 10,000 replicates; training-seed variance not included; Δ paired
on the same test images):**

| method | clean [95% CI] | biased [95% CI] | Δ biased vs our RR-ClArC [95% CI] | Δ biased vs Vanilla [95% CI] |
|---|---|---|---|---|
| Paper: Vanilla / RRR / RR-ClArC | 0.87 / 0.84 / 0.86 | 0.28 / 0.51 / 0.76 | — | — |
| Our Vanilla | 0.839 [0.825, 0.853] | 0.311 [0.293, 0.329] | — | — |
| Our RR-ClArC (A18, identity_2) | 0.839 [0.825, 0.853] | 0.524 [0.505, 0.543] | — | — |
| a: mask_cmc on our Vanilla | 0.822 [0.807, 0.836] | 0.691 [0.673, 0.709] | — | +0.380 [+0.360, +0.401] |
| a': mask_cmc on our RR-ClArC (A19) | 0.818 [0.803, 0.833] | 0.667 [0.648, 0.685] | +0.143 [+0.126, +0.161] | +0.356 [+0.335, +0.377] |
| b: DINOv2-518 on lesion-masked images + mask_cmc | 0.685 [0.666, 0.703] | 0.681 [0.663, 0.700] | — | +0.370 [+0.345, +0.397] |
| c: U-MtE (lesion ROI), label-free | 0.680 [0.662, 0.698] | 0.678 [0.660, 0.697] | — | +0.367 [+0.341, +0.394] |

Arm a' on clean: −0.021 [−0.033, −0.009] vs our RR-ClArC. Sources: `results/replication/pahde/{p2_reproduction_a18,
p2_reproduction, arm_a, arm_aprime, arm_b, arm_c}.md`; per-image test predictions in `results/replication/pahde/scores/`.

**Reading (descriptive).** On the network side (a, a'), projecting out the one artifact direction recovers most of the
biased accuracy at a small clean cost (−0.02). Lesion masking with a frozen DINOv2 encoder (b, c) removes the microscope
almost completely (clean ≈ biased) but costs about 0.155 clean accuracy against the fine-tuned ResNet on this 8-class
task. Arm a' is not better than arm a on the biased test.

**Notes on the C grid.** The selected C is the smallest value of the registered grid {0.01, 0.1, 1, 10} for arm a
(validation mean 0.772), arm a' (0.751) and arm c (0.694); arm b chose 0.01 as well (`arm_b.md`). The registered grid is
kept; any wider grid is post hoc only (sensitivity, no verdict). U-MtE kept k = 64 directions (its maximum, `max_k`).
Lesion ROIs: HAM10000 manual 10,015, ISIC 2018 manual 772, U-Net 14,544 (held-out Dice 0.942; 7 empty masks, kept).

## P3 — Bassi et al., Nature Communications 2024 (primary)
*pending (data ready; ISNet starts after the P1 gate; BIMCV test images fetching)*

**Framing decided 2026-11-18 (author).**
- **A9 equivalence check:** `COVIDIsNet.pt` scores macro-F1 0.223 (primary) / 0.260 (size-matched) against 0.773, and fails
  on P3's own training sources too; the diagnosis found no difference from P3's stated pipeline
  (`docs/REPLICATION_STATUS.md`, 2026-11-18). **P3 stays descriptive as registered**; the authors are asked
  (`docs/EMAIL_DRAFT_BASSI_ISNET.md`).
- **Corner-run ISNet:** its macro-F1 on the rebuilt tests is reported beside P3's 0.775 as **descriptive supporting
  evidence of test equivalence, not a gate**.
- **Inside-the-lungs comparison** (ISNet and our arms trained by us on identical data, A6): reported with 95% CIs as a
  **controlled, descriptive comparison**.



## P4 — Saab et al., MLHC 2022 (tier 2, descriptive; A13, A31, A38, A40)
**Not reproduced:** Binary-ERM 0.363 vs 0.359 (passes); Segmentation-ERM (A31 rerun) **0.647** vs 0.727 (gap 0.080; also misses
the notebook's 0.772) (`results/replication/saab/p4_gate_seg_pil.md`). Second-tier comparison vs our Seg-ERM run, trap sets 1–3,
772 test images each; image-only bootstrap shared across sets; Holm over 4 (`results/replication/saab/a40/comparison.md`):

| arm | 3-set mean | vs Seg-ERM 0.647: Δ [95 % CI] | p Holm |
|---|---|---|---|
| b DINOv2 + mask_cmc | 0.670 | +0.023 [−0.029, +0.076] | 0.88 |
| c U-MtE | 0.623 | −0.023 [−0.075, +0.029] | 0.88 |
| d EfficientNet-B0 + mask_cmc | 0.621 | −0.026 [−0.073, +0.022] | 0.88 |
| a mask_cmc on Seg-ERM's encoder | 0.552 | −0.095 [−0.141, −0.049] | 0.0004 |

Reading: Segmentation-ERM uses the lesion masks as training supervision; on these trap sets that is as good as our constraint.

## P5 — R2R (dropped)
Step 1 failed: the released checkpoints give 0.838 / 0.833 / 0.552 (original / clean / poisoned accuracy) vs 0.895 / 0.895 /
0.356 in the paper (`results/replication/r2r/step1.md`); not reproduced at 0.03 or 0.05; dropped before any arm was run.

## P6 — Yan et al., MICCAI 2023 (EPVT) / TMI 2025 (PLDG) (tier 1; A29, A33, A34, A36, A38, A39, A41, A42)
**Reproduced at 0.05** (A38). Bias 1 (5 seeds): ERM 0.668 vs 0.653, EPVT (A36: paper's Eq. 3 mixup) **0.696** vs 0.746 (gap
0.0497 — within the digitisation uncertainty); bias 0.9: ERM 0.762 vs 0.762, EPVT 0.775 vs 0.792 (`results/replication/epvt/
A34_gate.json`, `A39_bias09_gate.json`; audit `docs/P6_EPVT_AUDIT.md`). A39 comparison vs our EPVT (5 seeds), crossed seed ×
image bootstrap, Holm over 8, Holm-adjusted CIs (`results/replication/epvt/arms/comparison_holm_ci.md`):

| arm | bias 1: AUROC, Δ [Holm-adj. CI] | bias 0.9: AUROC, Δ [Holm-adj. CI] | beaten (1 / 0.9) | clean check A42 (bias 0 test; within EPVT − 0.01?) |
|---|---|---|---|---|
| **b DINOv2 + mask_cmc** | **0.782, +0.086 [+0.064, +0.107]** | **0.806, +0.031 [+0.009, +0.054]** | **yes / yes** | 0.889 / 0.889 vs EPVT 0.910 / 0.929: no |
| a mask_cmc on EPVT's features | 0.758, +0.061 [+0.039, +0.086] | 0.795, +0.021 [+0.003, +0.040] | yes / yes | 0.885 / 0.886: no |
| **d EfficientNet-B0 + mask_cmc** | 0.718, +0.022 [+0.003, +0.040] | 0.754, −0.021 [−0.039, −0.003] | yes / no | 0.924 / 0.932: yes |
| c U-MtE (label-free) | 0.625, −0.072 | 0.687, −0.088 | no / no | 0.870 / 0.877: no |

EPVT 0.696 / 0.775; ERM 0.668 / 0.762. The clean check is reported, not part of the verdict (A39 registered no clean
condition; A42). Lesion ROIs: HAM10000 manual, ISIC 2018 manual, lesion U-Net (rebuilt on machine 2, held-out Dice 0.942).

## P7 — Dreyer et al., AAAI 2024 (RR-ClArC; tier 1; secondary: invisible whole-image trigger; A28, A38)
**Reproduced** (biased-test accuracy): ResNet-18 0.713 vs 0.712, EfficientNet-B0 0.728 vs 0.756, VGG-16 0.531 vs 0.570 (passes at
0.05). Our arms vs our RR-ClArC, Holm over 9; a clean drop > 0.01 = not beaten (`results/replication/rrclarc/p7/arms_comparison.md`):
ResNet-18 arm a 0.709 (−0.004, n.s.), arm b 0.614; EfficientNet-B0 arm a 0.712 (−0.017), arm b 0.668; VGG-16 arm a 0.571
(+0.039, clean −0.015) and arm b 0.616 (+0.085, clean −0.078) → **not beaten** on any architecture. Arms a and a′ coincide (the
paper's `Freeze` leaves the pooled features unchanged).

## P8 — Wu et al., ICML 2023 (DISC; tier 2; A32, A35, A37, A38)
**Not reproduced:** ERM 5-set mean **0.507** vs 0.364, DISC **0.489** vs 0.551 (`results/replication/disc/gate.md`), with one
data root per trap set (A37: their released code's image cache is shared across trap sets; caches verified row by row). The
published numbers may carry that bug (pre-specified A37 reading). Second-tier comparison vs our DISC run (DISC's own 154 / 618
rows; image bootstrap shared across sets; Holm over 4; `results/replication/disc/arms/comparison.md`):

| arm | primary rule (DISC's 154 val) | secondary (held-out 259 val) |
|---|---|---|
| b DINOv2 + mask_cmc | **0.736** (+0.247 [+0.204, +0.290]) | 0.757 (+0.268) |
| d EfficientNet-B0 + mask_cmc | 0.691 (+0.201 [+0.154, +0.248]) | 0.648 (+0.159) |
| c U-MtE | 0.630 (+0.141) | 0.661 (+0.171) |
| a mask_cmc on DISC's features | 0.621 (+0.132) | 0.608 (+0.118) |

Our DISC 0.489, our ERM 0.507. Published row (descriptive): CBDebug (ICLR 2026) 0.580 ± 0.116 — not reproduced: no ISIC code.

## Phase 5 — thyroid TN3K calipers (confirmatory; `docs/PREREGISTRATION_PHASE5.md`, approved 2026-11-20)
72 networks (ERM ResNet-50d bases, arm d, RR-ClArC 3 λ × 3 layers chosen on val_clean, ISNet ImageNet init), folds 0–2, one
selection criterion (val_clean AUROC) (`results/replication/phase5/analysis/results.md`). test_rev AUROC (mean ± SD):

| method | Trap A (inside the ROI) | Trap B (outside) |
|---|---|---|
| **DINOv2 + mask_cmc** | **0.716 ± 0.027** | 0.789 |
| **arm d (EfficientNet-B0 + mask_cmc)** | **0.690 ± 0.076** | 0.746 |
| ISNet (P3) | 0.561 ± 0.172 | 0.673 |
| RR-ClArC (P2) | 0.467 ± 0.049 | 0.595 |
| U-MtE | 0.504 | 0.782 |
| ERM ResNet-50d | 0.440 | 0.546 |

Trap A, Holm family of 4, crossed fold × image bootstrap; "beaten" also needs a clean drop ≤ 0.01: arm d vs RR-ClArC +0.224
[+0.159, +0.292]; arm d vs ISNet +0.129 [+0.035, +0.239]; DINOv2 + mask_cmc vs RR-ClArC +0.249 [+0.192, +0.304]; DINOv2 +
mask_cmc vs ISNet +0.155 [+0.019, +0.316] — **all beaten**, clean AUROC higher in every case. ISNet improves when the marker is
outside (0.673) rather than inside (0.561), as the thesis predicts for location-based methods.
