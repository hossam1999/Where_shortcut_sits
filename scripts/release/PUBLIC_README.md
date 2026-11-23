# Where the Shortcut Sits — code

Code for *Masking cannot remove what is inside the region of interest: location-dependent shortcut learning in medical
image classification* (Elkady, Ghanem and Hafez; submitted to BMC Medical Imaging).

This repository contains the code needed to rebuild the cohorts, run every experiment and recompute every statistic in
the paper, together with the pre-registration documents. It contains **no data, images, model weights, predictions or
result files**. All datasets are public and must be obtained from their original sources under their own licences.

## Contents

| Path | What it is |
|---|---|
| `src/wtss/` | The package: cohort construction, artifact detectors, ROI masking, trap environments, the class-conditional constraint (`mask_cmc`), U-MtE, baselines and the crossed seed × image bootstrap |
| `scripts/data/` | Dataset preparation (ISIC 2019/2020, TN3K, MMOTU, SEE-AI, chest radiograph cohorts) |
| `scripts/round4/` … `scripts/round9/`, `scripts/final/`, `scripts/analysis/` | The experiments, in the order they were registered |
| `scripts/replication/`, `scripts/phase5/` | The audit of eight published mitigation methods and the confirmatory comparison on thyroid calipers |
| `third_party/<paper>/` | For each audited paper: the upstream repository and commit (`SOURCE.md`) and the minimal patches we applied |
| `tests/` | Unit tests (`make test`) |
| `docs/PREREGISTRATION_*.md` | The pre-registrations, each with its dated amendments, unchanged from the versions committed before the corresponding results |
| `docs/` (other) | Statistical plan, data manifest, environment, artifact-detector audits, code audit of EPVT |

## Pre-registration and amendments

Every confirmatory analysis was registered before its results. Amendments are dated and appended, never edited in place.
One amendment changed a decision rule and is reported in the paper: the reproduction tolerance for the audited papers
was registered as 0.03 and widened to 0.05 for all papers after the first reproduction checks
(`docs/PREREGISTRATION_REPLICATION.md`, amendment A38).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
export WTSS_DATA=/path/to/datasets WTSS_CACHE=/path/to/cache
make test
```

Dataset locations and expected file counts: `docs/DATA.md`, `docs/DATA_MANIFEST.md`. Some encoders (e.g. MedSigLIP)
require a Hugging Face token in `HF_TOKEN`. The audited papers' code is cloned at the commits in
`third_party/<paper>/SOURCE.md` and patched with the `.patch` files there.

## Licence and citation

AUTHOR CONFIRMATION REQUIRED: licence (e.g. MIT) and citation entry once the paper is accepted.
