# Related work & novelty audit (living document; last update 2026-09-27)

Honest map of what is and is not new, for the paper's positioning. Each row: closest prior work → our delta.

| Our component | Closest prior work | What is new here (claimed) | Risk |
|---|---|---|---|
| Location-dependent failure of ROI masking (in-ROI vs out-of-ROI artifacts), pre-registered traps, 3 modalities (dermoscopy hair, thyroid US markers, capsule debris) | Bissoto et al. 2020 (CVPRW, trap sets, "not so fast"), 2022 (artifact-based DG, NoiseCrop test-time masking); "Mask of truth" (arXiv 2412.04030, masking regions in medical images) | Overlap r as the controlled variable; crossover [mask−ERM]_B − [mask−ERM]_A > 0 in every modality; synthetic dose–response | Must cite Bissoto; check Mask-of-truth for overlap analysis |
| Generic procedural overlay library (artifact-agnostic) | Random erasing (Zhong 2020), occlusion augmentation (Fong 2019), MediAug | Used to **estimate a nuisance subspace**, not as augmentation; U7 ablation: same overlays as augmentation gain ≤ +0.015, erasure +0.21 (thyroid) / +0.045 (hair) | Reviewers may call it augmentation — U7 answers |
| Paired-difference subspace erasure (I2E) | LEACE (Belrose 2023), INLP (Ravfogel 2020), Chuang et al. 2023 (biased-prompt projection, VLM text side), Holstege et al. 2023 (joint subspace), Meta 2023 "Identifying and disentangling spurious features" | Concept subspace from **synthetic insertion pairs**: no concept labels, no real artifact examples; multi-rank, k chosen label-free | Core method novelty is moderate; strength is annotation-free + evidence |
| Mask-then-erase ordering (MtE / U-MtE) | none found combining ROI masking with residual in-ROI erasure | Division of labour: mask removes out-of-ROI, erasure targets in-ROI residue | Main method claim |
| Disease-protected erasure (U9, `insertion.protect`) | **SPLICE / SPLINCE** ("Preserving task-relevant information under linear concept removal", OpenReview 2025): oblique projection preserving covariance with the task label | Protection subspace estimated from artifact-free images; concept subspace label-free | **Not novel on its own** — present as an adaptation; cite SPLICE; consider SPLICE as a baseline where A labels exist |
| Pseudo-group balancing from overlay detector (U8) | JTT / group inference without labels (Liu 2021), DFR (Kirichenko 2023) | Group labels inferred from a detector trained only on synthetic overlays | Small gains so far (+0.04) |
| SLAS token-level suppression | Test-Time Selection (Bissoto 2023, keypoints), token pruning literature | Few-shot patch-token artifact localisation (5 images → AUROC 0.93) | Downstream gains small; secondary tool |

## Baselines still to add for the paper
- SPLICE (needs A labels) vs mte_protect.
- NoiseCrop-style test-time masking (ours ≈ mask arm; verify equivalence or add).
- JTT (two-stage error-set upweighting) as a label-free group baseline vs pbal.

Sources: arXiv 2302.00070; OpenReview cacf1140 (SPLICE); arXiv 2310.11991; arXiv 2306.12673; arXiv 2208.09756;
CVPRW 2020 Bissoto; arXiv 2412.04030; arXiv 2504.18983.
