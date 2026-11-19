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
