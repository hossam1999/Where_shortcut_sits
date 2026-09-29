# Pre-registration — Round 9: three candidates from the development search, tested against ROI masking

Drafted 2026-09-29 at the end of the overnight search; **approved by the author on 2026-09-29 with the changes of §7** and committed before any data was generated with the seeds 9101–9505. **No round-9 confirmation data has been generated.** The
candidates were found by a search over **12 candidates** (plus 5 reference arms) on development data only
(`docs/ROUND9_SEARCH_LEDGER.md`, `results/round9_search/`); this selection is itself a source of optimism that the
confirmation below is meant to remove.

## 1. What the search does and does not show
- On the development proxy (29 components mirroring D1–D4), no candidate met every component. The three below met 28, 28
  and 27 with no loss and do not flip the shortcut. Round 8's mask_bal also scored 28/29 on the same proxy and then
  lost on ISIC 2019 → 2020 in confirmation; the proxy cannot see ISIC 2020.
- The reason to test mask_condadv first: it is the only arm whose all-pairs contrast with masking was ≥ −0.001 in every
  development cohort (positive on the three ISIC 2019 sources) — the proxy of the one cell that made round 8's best
  candidate fail.

## 2. Candidates (labels as in the ledger; none is new)
| order | candidate | what it is | label and closest prior work | needs (train / test) | inference cost |
|---|---|---|---|---|---|
| 1 | mask_condadv | MLP projection (768→256→128) of frozen masked features trained with the classification loss while an adversary predicts the automatic artifact label from the projection given the diagnosis, through gradient reversal λ ∈ {0.1, 1, 10}; linear head on the projection | adaptation of conditional adversarial debiasing: Zhang et al. 2018 (arXiv 1801.07593), Ganin et al. 2016 (arXiv 1505.07818), Kim et al. 2019 (arXiv 1812.10352) | artifact labels + ROI / image + ROI | masking + a small MLP |
| 2 | mask_irm | IRMv1 penalty with artifact × label groups as environments, λ ∈ {1, 10, 100}, linear head on masked features | published, applied here: Arjovsky et al. 2019 (arXiv 1907.02893) | as 1 | as masking |
| 3 | mask_vrex | V-REx penalty (variance of group risks), groups = artifact × label, β ∈ {1, 10, 100}, linear head on masked features | published, applied here: Krueger et al. 2021 (arXiv 2003.00688) | as 1 | as masking |
Code: `scripts/round9_search/methods.py` (frozen at the commit of this draft). Encoder: DINOv2-B/14 @ 518.

## 3. Selection (unchanged from round 8)
Per training set, among {masking head, the candidate's settings}: worst-group AUROC on `val_groups` with the
same-artifact guard (≥ masking's − 0.005); masking as fallback; C by validation AUROC. The grids above are frozen as in
development. **Held-out cohort ovary** uses the most frequent development choice (mask_condadv λ = 1, mask_irm λ = 100,
mask_vrex β = 100; `results/round9_search/choices_*.csv`), without looking at its validation data.

## 4. Data
- **Primary confirmation**: new trap resamples with the reserved seeds **9101, 9202, 9303, 9404, 9505** (thyroid,
  capsule, ovary held out, ISIC hair), never used before this registration.
- **Semi-confirmatory**: the natural test sets (thyroid split, capsule, BCN / HAM / MSK) and ISIC 2019 → 2020 with new
  validation seeds 9101–9505 — these test sets were already seen in rounds 1–8, and the round-8 result on them informed
  the choice of candidates.
- Ovary is held out; the verdict is also reported without it.

## 5. Criterion and multiplicity
Round 8's criterion (docs/PREREGISTRATION_ROUND8.md §7 with Amendment 1): D1 (6 NI, 0.01), D2 (4 hard-pair SUP + 4 Trap A
min(rev, corr) SUP), D3 (16 per-cell NI 0.03 + 4 pooled NI 0.01), D4 no flipping; intersection–union within a
candidate; **fixed sequence mask_condadv → mask_irm → mask_vrex**, each at one-sided 0.025, stop at the first failure;
failed components labelled loss / inconclusive. Crossed seed × image bootstrap, 10,000 replicates. Decision rule as round
8 §9. Stability: the seed-level SD of the Trap A reversed contrast is reported for every candidate.

## 6. Questions that were open in the draft (answered by the approval, §7)
- Whether to run it: the search did not find a candidate that is better than round 8's mask_bal on the development proxy;
  what it offers is one candidate (mask_condadv) with a different trade-off (no all-pairs cost, smaller Trap A gains).
- Whether the ISIC 2020 cell should stay semi-confirmatory for a candidate chosen partly because of it.
- Whether the MedSigLIP / ConvNeXt replication family of round 8 should be added (cheap; heads only).

## 7. Changes made by the author on approval (part of this registration)
1. **Descriptive references on the round-9 data** (not in the fixed sequence): round 8's **mask_cmc** and **mask_bal**
   with their frozen round-8 recipes (the round-8 grids and selection rule; for the held-out ovary the round-8 frozen
   settings mask_cmc = per_class, mask_bal = λ 1 from `results/round8/frozen_choice.json`), fitted on the same training
   sets and seeds, so every candidate's trade-off is compared with them on identical data. Their components are reported
   with the same criterion, without entering the sequence.
2. **ISIC 2019 → 2020 decomposition**: for every candidate and reference, the all-pairs contrast with masking on ISIC
   2020 is split exactly into its easy, hard and same-artifact parts (π_k × Δ_k, as `results/round8/pair_decomposition.csv`).
3. **Replication family R** (as round 8): Trap A min(rev, corr) SUP for mask_condadv, mask_irm and mask_vrex with
   MedSigLIP (thyroid, capsule, ovary) and ConvNeXt (thyroid, capsule); Holm within the 15 tests; secondary.
4. **This is the last remedy round**: no further search follows, whatever the result.

## 8. Implementation and order (fixed at registration)
- Code: `scripts/round9/confirm.py` (training sets from `scripts/round8/heads_run.build` with the seeds 9101–9505; the
  candidates' heads from `scripts/round9_search/methods.py` unchanged; references from `scripts/round8/candidates.py`),
  `scripts/round9/analyse.py` (round-8 components, intersection–union and fixed sequence from `scripts/round8/analyse.py`),
  `scripts/round9/run_all.sh`. Every confirmation step refuses to run unless this file is committed and unchanged.
- Smoke runs use the smoke seed 99991 on small subsets with validation images standing in for every test set; they
  never generate data with 9101–9505.
- Outputs: `results/round9/` (choices, components, verdicts, replication, decomposition, SUMMARY.md); predictions and
  features are never committed.
- Seed-level stability: the SD over seeds of the Trap A reversed contrast with masking, per cohort, for every arm.
