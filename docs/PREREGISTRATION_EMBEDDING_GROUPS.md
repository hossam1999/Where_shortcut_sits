# Pre-registration — stricter leakage groups from image embeddings (thyroid, ovary, capsule)

Committed before any model is refitted with these groups. Motivation: TN3K/TNCD, MMOTU and SEE-AI publish no patient
or video identifiers, so the main analyses group near-duplicates by perceptual hash (and, for capsule, blocks of 100
consecutive frames). A reviewer asked for a stricter grouping. ISIC (lesion IDs) and chest drains (patient IDs) are
unaffected.

## Rule (fixed now; `scripts/data/embedding_groups.py`)
- Embedding = cached DINOv2 ViT-B/14 @518 CLS feature of the unmasked image, L2-normalised.
- Edge between two images if cosine similarity ≥ τ; groups = connected components of these edges **united with the
  existing pHash / frame-block groups** (so the new grouping is never looser than the old one).
- τ = the smallest value on the grid 0.99, 0.98, …, 0.80 for which the largest group holds ≤ 5 % of the cohort
  (the most aggressive threshold that still allows 5-fold grouped splits). Chosen without labels or results.
- Images without a cached embedding keep their existing group.

**Amendment (before any refit, after inspecting only the similarity distribution):** raw CLS cosine is dominated by
the shared ultrasound appearance (ovary: median nearest-neighbour similarity 0.99; at τ = 0.99 the largest group
already held 26 % of the cohort, so no grid value met the 5 % cap). Embeddings are therefore **mean-centred per cohort
before L2 normalisation**, for all three cohorts; the rest of the rule is unchanged.

## Re-analysis
Same traps, same seeds, same arms (erm, mask, balanced, U-MtE, U-MtE_balanced, U-MtE_protect, overlay augmentation;
generic overlays), only the grouping changes: `run_thyroid_traps.py --cohort {thyroid,ovary,capsule} --generic
--groups_csv ... --tag emb_groups`. Reported for each cohort: P1 crossover and the P2–P4 contrasts, next to the main
estimates. Claim: conclusions are robust if every estimate keeps its sign and its CI still excludes 0 wherever it did
before. Reported whichever way it goes.

## Results (results/leakage/embedding_groups.json, embedding_groups_compare.csv)
Groups: thyroid 3,455 → 3,081 (τ = 0.80, largest group 4.6 %); ovary 1,059 → 969 (τ = 0.80, 2.1 %); capsule 137 → 135
(τ = 0.93, 3.7 %). Trap cell counts are unchanged (only the split assignment changes).

| cohort | test | main [95 % CI] | embedding groups [95 % CI] | robust |
|---|---|---|---|---|
| Thyroid | P1 crossover | +0.230 [+0.204, +0.255] | +0.235 [+0.211, +0.259] | yes |
| Thyroid | P2 U-MtE_protect − mask | +0.217 [+0.174, +0.257] | +0.251 [+0.219, +0.279] | yes |
| Thyroid | P3 erase − augment | +0.207 [+0.182, +0.235] | +0.192 [+0.162, +0.218] | yes |
| Thyroid | P4 U-MtE_bal − balanced | +0.112 [+0.090, +0.134] | +0.152 [+0.138, +0.167] | yes |
| Ovary | P1 crossover | +0.170 [+0.116, +0.224] | +0.157 [+0.080, +0.235] | yes |
| Ovary | P2 | +0.043 [+0.026, +0.061] | +0.047 [+0.020, +0.072] | yes |
| Ovary | P3 | +0.035 [+0.010, +0.059] | +0.020 [−0.011, +0.053] | **no** |
| Ovary | P4 | +0.028 [+0.000, +0.057] | +0.039 [+0.015, +0.064] | yes |
| Capsule | P1 crossover | +0.368 [+0.340, +0.397] | +0.394 [+0.359, +0.430] | yes |
| Capsule | P2 | −0.010 [−0.021, −0.001] | −0.007 [−0.013, −0.001] | yes |
| Capsule | P3 | −0.130 [−0.147, −0.110] | −0.136 [−0.153, −0.122] | yes |
| Capsule | P4 | +0.082 [+0.064, +0.100] | +0.068 [+0.050, +0.085] | yes |

**11/12 robust.** The location law (P1) is unchanged in all three cohorts without patient IDs. The one change: ovary
erase − augment (P3) loses significance (+0.020 [−0.011, +0.053]), so "erase beats augment" is robust in thyroid and
ISIC hair (lesion IDs) but fragile in ovary. Embedding grouping is still a proxy for patient identity; it removes
visually near-identical images (same patient/session) but cannot rule out different-looking images of one patient.
