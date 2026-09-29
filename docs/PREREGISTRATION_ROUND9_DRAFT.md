# DRAFT - NOT A REGISTRATION UNTIL THE AUTHOR APPROVES IT

# Round 9 (draft): three candidates from the development search, tested against ROI masking

Written 2026-09-29 at the end of the overnight search. **No round-9 confirmation data has been generated.** The
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

## 6. To decide before approval
- Whether to run it: the search did not find a candidate that is better than round 8's mask_bal on the development proxy;
  what it offers is one candidate (mask_condadv) with a different trade-off (no all-pairs cost, smaller Trap A gains).
- Whether the ISIC 2020 cell should stay semi-confirmatory for a candidate chosen partly because of it.
- Whether the MedSigLIP / ConvNeXt replication family of round 8 should be added (cheap; heads only).
