# Replication results: Viviano 2021 (P1), Pahde 2025 (P2), Bassi/ISNet 2024 (P3)

Status: **in progress**, filled in as runs finish. Every number traces to a file under `results/replication/`. Rules:
`docs/PREREGISTRATION_REPLICATION.md` (with amendments A1–A11). Phase-0 checks, deviations and engineering notes:
`docs/REPLICATION_STATUS.md`.

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

**Registered expectations (section 5), before Holm (the family also contains the P3 location test, still pending):**
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
