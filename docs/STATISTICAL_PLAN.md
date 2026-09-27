# Statistical plan: primary vs secondary claims and multiplicity

## Estimation
Every comparison is a hierarchical paired bootstrap (10,000 replicates): clusters = training seeds (or folds for the
chest-drain and fine-tuning analyses), paired test images resampled within clusters, AUROC differences computed
within clusters and averaged. Two-sided bootstrap p-values (percentile method, floored at 1/10,000) are reported
alongside 95 % CIs (`wtss.stats.boot_p`).

## Primary family (16 tests; results/PRIMARY_CLAIMS.md)
Each test below was individually pre-registered before its data were analysed (docs/PREREGISTRATION_*.md). The
grouping into one *primary family* for multiplicity control was defined on 2026-09-27 after results were available,
to answer the question a reviewer will ask ("are the headline claims robust to multiple testing?"); we therefore
apply the conservative Holm correction over the whole family and report BH for reference.
- P1 location crossover, [mask − ERM]_out-of-ROI − [mask − ERM]_in-ROI > 0: ISIC hair, thyroid, capsule, ovary.
- P2 protected U-MtE − mask > 0 (in-ROI): ISIC hair, thyroid, capsule, ovary.
- P3 erase − augment (same generic overlays) > 0 (in-ROI): ISIC hair, thyroid, capsule, ovary.
- P4 U-MtE + balanced − balanced > 0 (in-ROI): ISIC hair, thyroid, capsule, ovary.
All on DINOv2 ViT-B/14 @518 (the backbone with every cohort); other backbones are replications.

## Secondary / exploratory
Everything else (other backbones, controlled sweeps, JTT, SPLINCE, DFR combinations, adaptive selection, SLAS,
fine-tuning, chest drains) is secondary: reported with unadjusted 95 % CIs and interpreted as replication or
exploration, not as confirmatory evidence.

## Robustness summary
Reversed-test AUROC is always read together with correlated-test and clean AUROC; min(reversed, correlated) is the
(post hoc) robustness summary, and arms with correlated < reversed − 0.02 are flagged as shortcut-flipping.
