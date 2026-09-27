# Where the Shortcut Sits

**Masking cannot remove what is inside the region of interest.** This repository contains everything behind the
paper (`paper/main.pdf`, supplement `paper/supplement.pdf`): data preparation, pre-registered experiments,
statistics, figures and a backbone-agnostic tool.

Main findings (details: `docs/FINDINGS_OVERVIEW.md`; all numbers are 95 % hierarchical-bootstrap CIs):
- ROI masking helps much less for artifacts **inside** the ROI than outside, in four diseases / three modalities
  (dermoscopic hair, thyroid and ovarian sonographer calipers, capsule-endoscopy debris): crossover +0.15 to +0.37
  reversed-test AUROC, all Holm-significant; it becomes **harmful** when masking also removes disease context
  (dermoscopy, controlled sweeps, capsule fine-tuning, and on the unaltered thyroid test set: −0.10 on
  shortcut-conflicting cases).
- A linear-Gaussian model predicts these signs in closed form (`docs/THEORY.md`).
- **U-MtE** (mask-then-erase with generic synthetic overlays; no artifact example, mask or label) removes in-ROI
  shortcuts carried by distinct overlays; erasure beats augmentation; a disease-protected variant prevents failure when
  the artifact resembles the pathology; label-based reweighting is needed when the shortcut is carried by correlates.

## Install
```bash
git clone https://github.com/hossam1999/Where_shortcut_sits && cd Where_shortcut_sits
pip install -r requirements.txt && pip install -e .
export WTSS_DATA=/path/with/350GB     # downloads, caches, features
export HF_TOKEN=...                   # gated models: DermLIP (redlessone/DermLIP_PanDerm-base-w-PubMed-256), MedSigLIP
export KAGGLE_API_TOKEN=...           # Kaggle datasets; accept the RANZCR-CLiP competition rules on kaggle.com first
```
Keep tokens in a `chmod 600` file that you `source`; never commit them. Optional: `tectonic` to build the PDFs.

## Data (all public; licences in `docs/DATA.md`)
| Cohort | Source | Download |
|---|---|---|
| ISIC 2018 / 2019, HAM10000 masks | ISIC archive (CC BY-NC), Harvard Dataverse | `scripts/data/download_data.sh` |
| ISIC 2019 artifact masks | Wegley et al. 2026, doi:10.71674/man1-qa33 | `download_data.sh` |
| NIH ChestX-ray14 + NEATX drains | NIH (HF mirror), zenodo 14944064 | `download_data.sh` |
| RANZCR-CLiP devices | Kaggle competition | `scripts/data/download_extra.sh` |
| TN3K + TNCD labels (thyroid) | TRFE-Net / ACL repos (Google Drive archive, verified) | `download_extra.sh` |
| MMOTU OTU_2d (ovary) | Kaggle, CC BY 4.0 | `download_extra.sh` |
| SEE-AI (capsule) + expert contamination masks | Kaggle CC BY 4.0; figshare 27645021 | `download_extra.sh` |

## Reproduce everything
```bash
make data data_extra     # downloads + preparation (lesion U-Nets, pHash groups, lung masks, detectors, caches)
make verify              # pilot CIs recomputed from archived predictions
make synthetic isic2019  # dermoscopy: controlled rulers, real hair (author spec E13/E15)
make thyroid ovary capsule cxr drain
make natural finetune sensitivity lama baselines
make analysis            # SUMMARY, Holm-corrected primary claims, cross-cohort tables, theory check, figures
make paper               # LaTeX tables + paper/main.pdf, paper/supplement.pdf
make test
```
Runtime on 1× RTX 3090 (24 GB), 8 CPU cores, 31 GB RAM: about 2 days end to end, dominated by feature extraction
and fine-tuning. Run heavy targets **one at a time** (two concurrent DataLoader-heavy jobs fit in 31 GB, three do not).
Every method is deterministic given seed and features: the per-dataset `*_universal` run contains all arms; the
analysis scripts fall back to it when a historical per-ablation folder is absent (identical predictions, verified).

## Results you will get
- `results/SUMMARY.md` — pilot replication (73/88 numbers matched; 12 same sign and significance; 3 differ).
- `results/PRIMARY_CLAIMS.md` — Holm-corrected primary family (12/16 supported).
- `results/CROSS_COHORT.md`, `paper/tables/main_inroi.tex` — every arm × cohort × backbone.
- `docs/PREREGISTRATION_*.md` — each experiment's registration **and** its results section (supported / not).
- `paper/figures/` — dose–response, forest plot, theory, examples.

## Use the methods on your own data (any backbone, any artifact)
**U-MtE** (images + ROI masks only):
```bash
python -m wtss.umte fit   --backbone dino518 --images "train/*.png" --rois train_rois/ --out umte.pt
python -m wtss.umte embed --model umte.pt --images "test/*.png" --rois test_rois/ --out test_feats.npz
```
Python: `UMtE(backbone).fit(imgs, rois[, y=..., artifact_free=...])` → `.transform(imgs, rois)` → any head
(`UMtE.fit_head(Z, y, artifact=a)` for the group-balanced variant). Backbones: `dino518`, `dermlip224`, `raddino518`,
`medsiglip448`, `convnext384`, or any `wtss.backbones.Backend`. Reproduces the experimental features exactly.

**SLAS** (few-shot patch-token artifact localisation; annotate ~5–50 images of any artifact):
```bash
python -m wtss.slas fit   --backbone dinov2 --images "ann/*.jpg" --masks ann_masks/ --out probe.pt
python -m wtss.slas embed --probe probe.pt --images "test/*.jpg" --rois test_rois/ --out feats.npz
```

## Layout
```
src/wtss/
  stats.py            hierarchical paired bootstrap (+ p-values), crossover test
  synthetic.py        controlled artifacts (ruler, tube, caliper, debris), generic overlay library, placements
  heads.py            ERM, balanced, DFR, GroupDRO, JTT, LEACE, SPLICE, prevalence calibration, pseudo-groups
  methods/insertion.py  insertion-pair subspace erasure (I2E / U-MtE), disease protection
  umte.py, slas.py    public tools
  backbones.py        DINOv2, DermLIP, RAD-DINO, MedSigLIP, ConvNeXt
  data/               ISIC 2018/2019, thyroid & ovary (caliper detector us_markers.py), capsule, CXR
  experiments/        synthetic sweeps, real/spec traps, fine-tuning
scripts/              data preparation, run_* entry points, analysis/, verify/, make_* (tables, figures)
docs/                 pre-registrations + results, statistical plan, theory, audits, sensitivity, related work
paper/                main.tex, supplement.tex, sections/, tables/ (generated), figures/ (generated)
```
The original pilot folders (`isic_pcam_code_results/`) are superseded; the pilot's numbers are reproduced by
`make verify` and `scripts/make_summary.py` (author specification: `docs/REPLICATION_SPEC.md`).
