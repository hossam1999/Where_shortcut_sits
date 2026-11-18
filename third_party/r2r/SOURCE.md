# P5 — Reveal to Revise (Pahde, Dreyer, Samek, Lapuschkin, MICCAI 2023; arXiv 2303.12641)

- Code: https://github.com/maxdreyer/Reveal2Revise, commit **f00ce0614947b48bb8320beeff64c700c6d94c09** (f00ce06).
  The machine's network kept dropping (2026-11-19), so the full archive could not be downloaded (codeload tarball cut at
  3.5 MB / 7.2 MB of ~291 MB uncompressed; its header confirmed the commit). The repository tree at that commit was listed
  through the GitHub API (1,521 files) and the 199 code / config files used here (`*.py`, `*.yaml`, `*.sh`, `*.md`, `*.txt`,
  `*.json`, plus `data/*.json|csv`; 376 KB) were fetched one by one from
  `raw.githubusercontent.com/maxdreyer/Reveal2Revise/f00ce06…/<path>` into `/workspace/third_party/r2r/` (0 failures).
  Not fetched: images under `static/` and `data/localized_artifacts/` (figures, not needed for step 1).
- Checkpoints (README: "All checkpoints can be found in the download scripts in the `scripts` folder"):
  `scripts/download_checkpoints_isic.sh:9` ResNet-18 original
  (`datacloud.hhi.fraunhofer.de/s/pijw3KFNdcZNi2e/download/checkpoint_resnet18_isic_last.pth`) and
  `scripts/download_checkpoints_isic_attacked.sh:7` ResNet-18 synthetic-attacked
  (`datacloud.hhi.fraunhofer.de/s/ZZibCgcN9r8sC3X/download/checkpoint_resnet18_isic_attacked.pth`).
- Split: `datasets/base_dataset.py:25-34` — `np.random.default_rng(0)`, validation 10%, test 10%, train the rest;
  `datasets/isic.py:101`. ISIC 2019 training images + `ISIC_2019_Training_GroundTruth.csv` (8 classes).
- Synthetic artifact: `datasets/isic_attacked.py` — text "Clever Hans" in black (`ch_text`), presence drawn once after
  `np.random.seed(0)` (`isic_attacked.py:47-49`), only for `attacked_classes = ['MEL']`.
- **Discrepancy (recorded):** the training config generator sets `p_artifact: 0.8`
  (`config_files/config_generator_training_isic_attacked.py:20`); the paper's Appendix A.2 states 10% of melanoma. Per the
  author (2026-11-19) we follow the paper and the checkpoint script for step 1; which rate the released attacked checkpoint
  was trained with is not stated in the code.

## Machine 2 (A27, 2026-11-18)
- Full `git clone` succeeded on machine 2; HEAD = f00ce0614947b48bb8320beeff64c700c6d94c09 (`/workspace/third_party/r2r`).
- Checkpoints downloaded from the two URLs above: `checkpoint_resnet18_isic_last.pth` SHA-256
  `3d653706b6216b6bf642542659e71dd16ec28670e180bfae71e85633db2d316a`, `checkpoint_resnet18_isic_attacked.pth` SHA-256
  `30aa74dc0aa5f1715c61f53927af5ed1e06c0204a6261ea486b1ac9d988d7213` (plain state dicts, 9 outputs).
- On the rate discrepancy: their `scripts/download_checkpoints_isic_attacked.sh:5` labels the attacked checkpoints
  "MEL 0.1 (train/val/test)", and their correction configs (`config_files/correcting_isic_attacked/local/*.yaml`) set
  `p_artifact: 0.1`; only the training generator says 0.8. Step 1 uses 0.1.
- Environment `/workspace/envs/r2r`: Python 3.11, torch 2.11.0+cu128, their `requirements.txt`; **pandas pinned to 2.2.3**
  (their `datasets/isic.py` calls `metadata.agg(sum)` over a frame with string columns; pandas 3 no longer maps the builtin
  `sum` to `Series.sum` and raises). Pillow 13.0.0: its default font is FreeType, not the bitmap font of Pillow < 10.1 that
  their `ImageDraw.text` call (no font argument) used in 2023 — the rendered "Clever Hans" text therefore differs from the
  training artifact (recorded as a candidate cause in `results/replication/r2r/step1.md`).
