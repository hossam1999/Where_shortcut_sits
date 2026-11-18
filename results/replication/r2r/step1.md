# P5 (Reveal to Revise, MICCAI 2023) step 1: released ISIC ResNet-18 checkpoints, Vanilla, scored with their code (machine 2, A27)

Their dataset, split (`default_rng(0)`, 80/10/10, n = 25,331 → test 2,533), synthetic artifact ("Clever Hans" text,
p_artifact = 0.1 of MEL, as the paper's Appendix A.2 and their correction configs) and metric code
(`utils/metrics.py`: accuracy, macro-F1), at f00ce06. No training. Each set scored once. Script:
`scripts/replication/p5_step1.py`; all rows in `step1.csv` (test and validation; `ch` = artifact as in training,
`attacked` = artifact in every image, `clean` = no artifact).

| checkpoint | test set | ours: accuracy | paper | Δ | ours: macro-F1 | paper | Δ |
|---|---|---|---|---|---|---|---|
| synthetic-attacked (`checkpoint_resnet18_isic_attacked.pth`) | original (artifact as in training, `ch`) | 0.838 | 0.895 | −0.057 | 0.772 | 0.821 | −0.049 |
| | original without artifact (`clean`) | 0.833 | 0.895 | −0.062 | 0.769 | 0.821 | −0.052 |
| | poisoned (artifact in all classes, `attacked`) | 0.552 | 0.356 | +0.196 | 0.622 | 0.390 | +0.232 |
| original (`checkpoint_resnet18_isic_last.pth`) | clean | 0.831 | — | | 0.792 | — | |

**Gate (|ours − paper| ≤ 0.03 on Vanilla): failed** on every Table 1 number (accuracy and F1, original and poisoned).

Two separate gaps:
1. **Clean / original accuracy is 0.06 below the paper (F1 0.05)**, with no artifact involved; the original (non-attacked)
   checkpoint gives the same level (0.831). So the checkpoints, as released, do not reach Table 1's clean numbers on the split
   their code builds.
2. **The poisoned drop is much smaller than the paper's** (0.552 vs 0.356). Candidate cause, not tested: their artifact
   draws the text with Pillow's default font (`ImageDraw.text` without a font, `utils/artificial_artifact.py`). Pillow
   ≥ 10.1 (October 2023) changed the default from the bitmap font to a FreeType font; the checkpoints predate it (paper 2023),
   our environment has Pillow 13.0.0, so the text we render differs in shape from what the attacked network learned. This
   cannot explain gap 1.

Port notes: torch 2.11 + cu128 (Blackwell); pandas pinned to 2.2.3 because their `metadata.agg(sum)` (`datasets/isic.py`)
fails under pandas 3; 8 DataLoader workers (their default 0; artifacts are seeded per index, so outputs are identical).
