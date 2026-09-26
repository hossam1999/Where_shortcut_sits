# Where the Shortcut Sits

Location-dependent failure of region-of-interest (ROI) artifact mitigation in medical image
classification, what supervision each fix actually needs, and **Insert-to-Erase (I2E)**, a
localisation-free mitigation that works where the artifact lies *inside* the ROI — in
dermoscopy (hair, rulers) and chest radiography (chest drains).

This single repository replaces the four pilot folders of `isic_pcam_code_results/`
(`isic_overlap_pilot_self_contained`, `…patch_v2`, `…patch_v3`; `pcam_spurious_prototypes` belongs to an
earlier, abandoned thesis direction and is not part of this work). Every pilot number in the thesis proposal
that had code was re-derived from raw images with this repo (see *Verification*).

## Layout

```
src/wtss/
  stats.py            hierarchical paired bootstrap (pilot estimator, exact) + vectorised equivalent
  synthetic.py        controlled-overlap artifacts (ruler, tube), placements, trap environments
  ops.py              ROI masking (dilation), Telea inpainting
  backbones.py        DINOv2 ViT-B/14 (224/518), DermLIP/PanDerm (224), RAD-DINO (518)
  features.py         per-view feature cache, memmapped image caches
  heads.py            ERM, group-balanced, DFR, GroupDRO, LEACE (paired/unpaired), consistency, prevalence calibration
  methods/insertion.py  Insert-to-Erase (subspace erasure from insertion pairs) + ablations
  evaluation.py       clean-val threshold, worst-group accuracy, same-head |Δp|
  data/               ISIC 2018 pilot, ISIC 2019 traps, NIH ChestX-ray14 (synthetic tube, real drains)
  experiments/        synthetic driver, real-trap driver, end-to-end fine-tuning
scripts/              data preparation, experiment entry points, verification
frozen/               frozen pilot objects: leakage-safe manifest, placements, pre-committed Phase-2 criteria
docs/                 pre-registrations, thesis-claims audit, data licences
results/              tables written by the scripts (small CSV/MD/JSON; predictions are gitignored)
```

## Reproduce

```bash
pip install -r requirements.txt && pip install -e .
export WTSS_DATA=/path/with/100GB          # all downloads, caches and features live here
make data        # ≈75 GB download + preparation (U-Net, pHash groups, lung masks, drain detector)
make verify      # pilot CIs from archived predictions + rerun-vs-archive comparison
make synthetic   # thesis Results 1–2 (+ backbones, occlusion, dilation, appearance stress, leakage)
make isic2019    # thesis Result 3: real hair, Trap A (in-ROI) / Trap B (out-of-ROI)
make cxr         # chest radiography: synthetic tube + real chest-drain trap
make test
```

Hardware used: 1× RTX 3090 (24 GB), 8 CPU cores, 31 GB RAM. Backbones are frozen, so the whole
study is feature extraction plus linear heads; the fine-tuning robustness check is the only training.

## Protocol (unchanged from the pilot)
Leakage-safe grouped splits; training environment with P(A|Y=1)=0.9, P(A|Y=0)=0.1; correlated and
reversed (10/90) test environments plus an artifact-free clean test; C, λ and the decision threshold are
chosen on clean validation only; hierarchical paired bootstrap over training seeds / CV folds; success
criteria pre-registered (`docs/`) and reported whether positive or negative.

## Replication of the pilot (author spec: docs/REPLICATION_SPEC.md)
`python scripts/make_summary.py` writes `results/SUMMARY.md`: every expected number beside the obtained one with
MATCH / SIGN+CI / MISMATCH (spec tolerance rule). Deviations: `CHANGES.md`; claim-by-claim audit:
`docs/THESIS_CLAIMS_AUDIT.md`.

| Block | Entry point | Notes |
| --- | --- | --- |
| E1–E9 synthetic | `scripts/run_synthetic.py` | frozen manifest + placements |
| E10 real-hair linear | `scripts/analysis/e10_hair.py` | Mendeley masks, Bissoto labels |
| E12 contrast trap | `scripts/analysis/e12_contrast_trap.py` | |
| E13 / E15 real hair | `scripts/run_spec_e13.py --backbone {dino518,dermlip224}` | spec U-Net + cohort: `scripts/data/train_unet_spec.py`, `prepare_spec_cohort.py` |
| E7 / E14 | `scripts/analysis/e7_e14.py` | |
| leakage (§3.1) | `scripts/run_leakage_experiment.py` | |
| CXR multi-disease real devices | `scripts/run_cxr_traps.py --backbone {raddino518,medsiglip448,dino518} [--device_matched]` | CLiP × NIH linkage: `scripts/data/link_clip_nih.py`, `prepare_clip.py` |

Credentials: gated models (DermLIP since 2026-09-21, MedSigLIP) and Kaggle (RANZCR-CLiP) read `HF_TOKEN` /
`KAGGLE_API_TOKEN` from the environment (e.g. a `chmod 600` file sourced by the queue scripts); never commit them.

## Verification
- `scripts/verify/recompute_archived_cis.py` — all archived bridge CIs recomputed exactly (max diff 1e-16).
- `scripts/verify/compare_synthetic_to_archive.py` — rerun from raw images: 252 per-seed AUROCs within
  ≤ 0.0005; all 21 Δ-vs-ERM CIs and significance decisions identical.
- `docs/THESIS_CLAIMS_AUDIT.md` — every quantitative claim of the proposal, its status and evidence.

## Data
ISIC 2018/2019 (CC-BY-NC), HAM10000 lesion masks (Tschandl et al. 2020), ISIC 2019 artifact masks
(Wegley et al. 2026, doi:10.71674/man1-qa33), NIH ChestX-ray14, NEATX drain labels
(Jiménez-Sánchez et al., zenodo 14944064, CC BY-NC-SA). See `docs/DATA.md`.
