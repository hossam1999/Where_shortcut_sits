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

## Re-analysis
Same traps, same seeds, same arms (erm, mask, balanced, U-MtE, U-MtE_balanced, U-MtE_protect, overlay augmentation;
generic overlays), only the grouping changes: `run_thyroid_traps.py --cohort {thyroid,ovary,capsule} --generic
--groups_csv ... --tag emb_groups`. Reported for each cohort: P1 crossover and the P2–P4 contrasts, next to the main
estimates. Claim: conclusions are robust if every estimate keeps its sign and its CI still excludes 0 wherever it did
before. Reported whichever way it goes.
