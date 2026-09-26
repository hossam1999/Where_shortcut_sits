# Follow-up protocol: metadata-matched real-hair traps

Committed **after** the pre-registered ISIC 2019 trap results (DINOv2) and the post-hoc diagnostics
(`scripts/analysis/real_trap_diagnostics.py`), and **before** any matched-trap AUROC exists. It is a
follow-up, not a replacement: the pre-registered verdicts (C2 not supported on DINOv2) stand as reported.

## Motivation (from diagnostics, no outcome of this follow-up known)
- Removing hair pixels at test only closes 47% (Trap A) / 44% (Trap B) of the ERM clean–reversed gap: most of
  what the ERM head exploits in the real trap is not the hair pixels.
- Hair-present and hair-free images differ in sex (HAM: 42% vs 57% female), age (52 vs 58 y) and
  anatomical site (HAM lower extremity: 16% vs 42%), i.e. 'hair present' is also a patient/site proxy.
- The in-ROI harm mechanism (retention of a used artifact) presupposes that the shortcut is the artifact.

## Design change (only this)
Within each source, artifact-present and hair-free images are drawn from the same strata of
(anatomical site group × sex × age band [<40, 40–59, ≥60]); in each stratum the environments are built by
the same maximum-size subsampling with P(A|Y) = 0.9/0.1 (train, correlated test) and 0.1/0.9 (reversed).
Everything else (folds, QC, groups, arms, thresholds, bootstrap, source matching across traps) is unchanged.

## Questions (descriptive, 95% hierarchical CI)
- M1: pixel share of the ERM shortcut (hair removed at test only) is larger than in the unmatched trap.
- M2: mask − ERM on Trap A reversed AUROC (sign and CI); M3: mask − ERM on Trap B; M4: crossover.
- M5: I2E − ERM on Trap A, relative to the pixel-removal upper bound (oracle inpaint − ERM).
Outcomes are reported whichever way they go.
