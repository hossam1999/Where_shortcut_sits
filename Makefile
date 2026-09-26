# Where the Shortcut Sits — full reproduction pipeline.
# All steps are idempotent (cached images/features are reused). Set WTSS_DATA to the data root.
PY := PYTHONPATH=src TQDM_DISABLE=1 python
export WTSS_DATA ?= /root/data

.PHONY: all data verify synthetic isic2019 cxr proposed paper test

all: data verify synthetic isic2019 cxr paper

# ---------------------------------------------------------------- data (≈75 GB download)
data:
	bash scripts/data/download_data.sh
	$(PY) scripts/data/prepare_isic2019.py --stage unzip
	$(PY) scripts/data/prepare_isic2019.py --stage groups --phash_threshold 2
	$(PY) scripts/data/train_lesion_unet.py --stage train --epochs 12
	$(PY) scripts/data/train_lesion_unet.py --stage predict
	$(PY) scripts/data/prepare_isic2019.py --stage cache
	$(PY) scripts/data/prepare_cxr.py --stage synthetic
	$(PY) scripts/data/prepare_cxr.py --stage detector
	$(PY) scripts/data/prepare_cxr.py --stage drain

# ---------------------------------------------------------------- verification of the pilot
verify:
	$(PY) scripts/verify/recompute_archived_cis.py
	$(PY) scripts/verify/compare_synthetic_to_archive.py

# ---------------------------------------------------------------- synthetic overlap (thesis §7-8)
ARMS_MAIN := erm mask inpaint balanced groupdro dfr leace inpaint_consistency inpaint_consistency_lam0
synthetic:
	$(PY) scripts/run_synthetic.py --cohort isic2018 --backbone dino518 --tag main --arms $(ARMS_MAIN)
	$(PY) scripts/run_synthetic.py --cohort isic2018 --backbone dino518 --tag proposed --arms erm --proposed
	$(PY) scripts/run_synthetic.py --cohort isic2018 --backbone dino224 --tag main --arms erm mask balanced dfr leace --proposed
	$(PY) scripts/run_synthetic.py --cohort isic2018 --backbone dermlip224 --tag main --arms erm mask balanced dfr leace --proposed
	$(PY) scripts/run_synthetic.py --cohort isic2018 --backbone dino518 --tag occlusion --phase occlusion --arms erm mask inpaint
	$(PY) scripts/run_synthetic.py --cohort isic2018 --backbone dino518 --tag dilation --overlaps 0.5 0.75 1.0 --arms erm dilate0 dilate10 dilate25 dilate50
	$(PY) scripts/run_synthetic.py --cohort isic2018 --backbone dino518 --artifact ruler_variable --tag stress --overlaps 0 0.5 1 --arms erm mask balanced leace --proposed
	$(PY) scripts/run_leakage_experiment.py

# ---------------------------------------------------------------- ISIC 2019 real hair (thesis §9)
isic2019:
	$(PY) scripts/run_traps.py --cohort isic2019 --counts_only
	$(PY) scripts/run_traps.py --cohort isic2019 --backbone dino518
	$(PY) scripts/run_traps.py --cohort isic2019 --backbone dermlip224

# ---------------------------------------------------------------- chest radiography (WP1)
cxr:
	$(PY) scripts/run_synthetic.py --cohort nih_ptx --backbone dino518 --artifact tube --tag main --arms erm mask balanced dfr leace --proposed
	$(PY) scripts/run_synthetic.py --cohort nih_ptx --backbone raddino518 --artifact tube --tag main --arms erm mask balanced dfr leace --proposed
	$(PY) scripts/run_traps.py --cohort nih_drain --backbone raddino518
	$(PY) scripts/run_traps.py --cohort nih_drain --backbone dino518

paper:
	$(PY) scripts/make_paper_tables.py

test:
	$(PY) -m pytest -q tests
