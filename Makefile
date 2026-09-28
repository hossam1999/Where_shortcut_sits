# Where the Shortcut Sits — full reproduction pipeline.
# All steps are idempotent (cached images/features are reused). Set WTSS_DATA to the data root.
PY := PYTHONPATH=src TQDM_DISABLE=1 python
export WTSS_DATA ?= /root/data

.PHONY: all data data_extra verify synthetic isic2019 cxr thyroid ovary capsule drain natural finetune review2 review3 \
        sensitivity lama baselines analysis paper test

all: data data_extra verify synthetic isic2019 cxr thyroid ovary capsule drain natural finetune sensitivity lama \
     baselines analysis paper

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

# ---------------------------------------------------------------- added datasets (thyroid, ovary, capsule, CXR devices)
data_extra:
	bash scripts/data/download_extra.sh
	$(PY) scripts/data/scan_thyroid.py
	$(PY) scripts/data/prepare_thyroid.py
	$(PY) scripts/data/prepare_ovary.py
	$(PY) scripts/data/prepare_capsule.py --stage probe
	$(PY) scripts/data/prepare_capsule.py --stage cohort
	$(PY) scripts/data/link_clip_nih.py
	$(PY) scripts/data/prepare_clip.py
	$(PY) scripts/data/prepare_isic2019.py --stage artifact_stats
	$(PY) scripts/data/native_hair_stats.py
	$(PY) scripts/data/train_unet_spec.py --stage train
	$(PY) scripts/data/train_unet_spec.py --stage predict
	$(PY) scripts/data/prepare_spec_cohort.py

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

# ---------------------------------------------------------------- ISIC 2019 real hair (author spec E13/E15)
GEN := erm mask balanced mask_balanced dfr mask_dfr mte mte_balanced mte_aug mte_protect mte_protect_balanced jtt mask_jtt umte_jtt
isic2019:
	$(PY) scripts/run_spec_e13.py --backbone dino518
	$(PY) scripts/run_spec_e13.py --backbone dermlip224
	$(PY) scripts/run_spec_e13.py --backbone dino518 --generic --tag spec_universal --save_val --arms $(GEN)
	$(PY) scripts/run_spec_e13.py --backbone dermlip224 --generic --tag spec_universal --arms erm mask balanced dfr mte mte_balanced mte_protect

# ---------------------------------------------------------------- thyroid / ovary / capsule (real traps + controlled sweeps)
thyroid:
	$(PY) scripts/run_thyroid_traps.py --backbone dino518
	$(PY) scripts/run_thyroid_traps.py --backbone medsiglip448
	$(PY) scripts/run_thyroid_traps.py --backbone dino518 --generic --tag universal --save_val --arms $(GEN)
	$(PY) scripts/run_thyroid_traps.py --backbone medsiglip448 --generic --tag universal --arms $(GEN)
	$(PY) scripts/run_thyroid_traps.py --backbone convnext384 --generic --tag universal --arms $(GEN)
	$(PY) scripts/run_synthetic.py --cohort thyroid --backbone dino518 --artifact caliper --tag main --arms erm mask inpaint balanced dfr leace --proposed
	$(PY) scripts/run_synthetic.py --cohort thyroid --backbone medsiglip448 --artifact caliper --tag main --arms erm mask inpaint balanced dfr leace --proposed
ovary:
	$(PY) scripts/run_thyroid_traps.py --cohort ovary --backbone dino518
	$(PY) scripts/run_thyroid_traps.py --cohort ovary --backbone dino518 --generic --tag universal --save_val --arms $(GEN)
	$(PY) scripts/run_synthetic.py --cohort ovary --backbone dino518 --artifact caliper --tag main --arms erm mask inpaint balanced dfr leace --proposed
capsule:
	$(PY) scripts/run_thyroid_traps.py --cohort capsule --backbone dino518
	$(PY) scripts/run_thyroid_traps.py --cohort capsule --backbone medsiglip448
	$(PY) scripts/run_thyroid_traps.py --cohort capsule --backbone dino518 --generic --tag universal --save_val --arms $(GEN)
	$(PY) scripts/run_thyroid_traps.py --cohort capsule --backbone medsiglip448 --generic --tag universal --arms $(GEN)
	$(PY) scripts/run_thyroid_traps.py --cohort capsule --backbone convnext384 --generic --tag universal --arms $(GEN)
	$(PY) scripts/run_synthetic.py --cohort capsule --backbone dino518 --artifact debris --tag main --arms erm mask inpaint balanced dfr leace --proposed
	$(PY) scripts/run_synthetic.py --cohort capsule --backbone medsiglip448 --artifact debris --tag main --arms erm mask inpaint balanced dfr leace --proposed
drain:
	$(PY) scripts/run_drain_spec.py --backbone raddino518 --save_val --arms $(GEN) splice mask_splice
	$(PY) scripts/run_drain_spec.py --backbone dino518
natural:
	$(PY) scripts/run_natural.py --cohort thyroid
	for s in HAM BCN MSK; do $(PY) scripts/run_natural.py --cohort isic_$$s; done
	$(PY) scripts/run_natural.py --cohort capsule
scale:   # DINOv2 ViT-S/14 and ViT-L/14 (docs/PREREGISTRATION_SCALE.md)
	for b in dinos518 dinol518; do for c in thyroid ovary capsule; do $(PY) scripts/run_thyroid_traps.py --backbone $$b --cohort $$c --generic --tag scale --arms erm mask balanced mte mte_balanced mte_protect mte_aug; done; done
finetune:
	for c in thyroid capsule ovary; do $(PY) scripts/run_finetune_spec.py --cohort $$c; done
	$(PY) scripts/run_finetune_spec.py --cohort thyroid --arch vit_small_patch16_224.augreg_in21k_ft_in1k --lr 3e-5
	$(PY) scripts/run_finetune_spec.py --cohort thyroid --arms mte_post --traps trapA   # fine-tune, then erase
sensitivity:
	for c in thyroid ovary; do \
	  $(PY) scripts/run_thyroid_traps.py --cohort $$c --generic --tag sens_px50 --min_px 50 --arms erm mask balanced mte mte_protect mte_balanced; \
	  $(PY) scripts/run_thyroid_traps.py --cohort $$c --generic --tag sens_strictloc --rA 0.7 --rB 0.05 --arms erm mask balanced mte mte_protect mte_balanced; done
	$(PY) scripts/run_thyroid_traps.py --cohort capsule --generic --tag sens_strictB --max_cover_B 0.05 --arms erm mask balanced mte mte_protect mte_balanced
	$(PY) scripts/data/embedding_groups.py   # stricter leakage groups (docs/PREREGISTRATION_EMBEDDING_GROUPS.md)
	$(PY) scripts/run_thyroid_traps.py --cohort thyroid --generic --groups_csv $${WTSS_DATA}/us/tncd/groups_emb.csv --tag emb_groups --arms erm mask balanced mte mte_balanced mte_protect mte_aug
	for c in ovary capsule; do $(PY) scripts/run_thyroid_traps.py --cohort $$c --generic --groups_csv $${WTSS_DATA}/$$c/groups_emb.csv --tag emb_groups --arms erm mask balanced mte mte_balanced mte_protect mte_aug; done
lama:
	for c in thyroid ovary capsule isic; do $(PY) scripts/data/precompute_lama.py --cohort $$c; done
	for c in thyroid ovary; do $(PY) scripts/run_thyroid_traps.py --cohort $$c --lama --tag lama --arms erm mask; done
	$(PY) scripts/run_spec_e13.py --backbone dino518 --lama --tag spec_lama --arms erm mask
baselines:   # SPLICE + adaptive selectors (need the *_universal runs with --save_val)
	$(PY) scripts/run_thyroid_traps.py --backbone dino518 --generic --tag splice --arms erm mask balanced splice mask_splice mte mte_protect mte_balanced
	$(PY) scripts/analysis/adaptive_select.py results/thyroid/dino518_universal results/capsule/dino518_universal results/ovary/dino518_universal results/spec_e13/dino518_spec_universal
analysis:
	$(PY) scripts/make_summary.py
	$(PY) scripts/analysis/primary_claims.py
	$(PY) scripts/make_cross_cohort_table.py
	$(PY) scripts/make_main_table.py
	$(PY) scripts/analysis/theory_sim.py
	$(PY) scripts/analysis/theory_predict.py      # theory fitted on clean+corr vs held-out reversed results
	$(PY) scripts/analysis/clinical_metrics.py    # AUPRC, Brier, ECE, sens/spec on the natural test sets
	$(PY) scripts/analysis/leakage_emb_compare.py # primary contrasts under embedding-based leakage groups
	$(PY) scripts/analysis/mechanism_figure.py    # counterfactual reliance vs overlap (controlled sweeps)
	$(PY) scripts/analysis/umte_ablation.py       # U-MtE erasure-rank ablation (cached features)
	$(PY) scripts/analysis/scale_compare.py       # DINOv2 ViT-S/B/L (needs the scale runs below)
	$(PY) scripts/make_figures.py
	$(PY) scripts/analysis/adhoc_bootstraps.py
	$(PY) scripts/verify/audit_paper_numbers.py   # every CI in the paper must trace to a result file

# ---------------------------------------------------------------- chest radiography (WP1)
cxr:
	$(PY) scripts/run_synthetic.py --cohort nih_ptx --backbone dino518 --artifact tube --tag main --arms erm mask balanced dfr leace --proposed
	$(PY) scripts/run_synthetic.py --cohort nih_ptx --backbone raddino518 --artifact tube --tag main --arms erm mask balanced dfr leace --proposed
	$(PY) scripts/run_traps.py --cohort nih_drain --backbone raddino518
	$(PY) scripts/run_traps.py --cohort nih_drain --backbone dino518

paper:
	$(PY) scripts/make_paper_tables.py
	cd paper && (tectonic -X compile main.tex && tectonic -X compile supplement.tex || echo "install tectonic or latexmk to build the PDF")
	cd paper_neurips && (tectonic -X compile main.tex || true)   # NeurIPS 2025 format (same style as SPLINCE)

report_full:   # complete standalone research record (report/full/full_report.pdf)
	$(PY) scripts/make_full_report.py
	cd report/full && (tectonic -X compile full_report.tex || echo "install tectonic to build the PDF")

test:
	$(PY) -m pytest -q tests

# ---------------------------------------------------------------- second review (docs/PREREGISTRATION_REVIEW2.md)
review2:
	bash scripts/queue_review2_us.sh
	bash scripts/queue_review2_isic.sh
	$(PY) scripts/analysis/operating_points.py
	$(PY) scripts/analysis/review2_summary.py
	$(PY) scripts/analysis/literature_table.py
	$(PY) scripts/analysis/clinical_metrics.py --suffix repro
	$(PY) scripts/verify/registry_table.py
	$(PY) scripts/make_review2_tables.py

# ---------------------------------------------------------------- third review (docs/PREREGISTRATION_REVIEW3.md)
review3:
	bash scripts/queue_review3.sh
	for c in thyroid ovary capsule; do $(PY) scripts/run_thyroid_traps.py --cohort $$c --tag matched05 --match --match_caliper 0.05 --arms erm mask; done
	$(PY) scripts/run_spec_e13.py --tag matched05 --match --match_caliper 0.05 --arms erm mask
	$(PY) scripts/analysis/crossed_ci.py
	$(PY) scripts/analysis/operating_points.py --crossed
	$(PY) scripts/make_review2_tables.py
	$(PY) scripts/make_review3_tables.py
