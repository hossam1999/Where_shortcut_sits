# New-dataset search (2026-10-03)

Goal: public datasets **not already used or rejected** (see `stages/EXTERNAL_VALIDATION_SEARCH.md`, Supplement S3)
that fit the trap design: a confirmed binary diagnosis, an ROI (mask or box), and a real artifact that sometimes lies
inside and sometimes outside that ROI, with enough images per cell (gate: ≥ 50 positive and ≥ 50 negative images in
each of Trap A, Trap B and the artifact-free group) and patient-grouped splits.

Already seen and excluded from this search: ISIC 2016–2020, HAM10000, Wegley masks, Kabir hair, DermArtifactDB,
IMA++, TN3K/TNCD, ThyUS2Path, TN-SCUI 2020, DDTI, Stanford AIMI thyroid cine-clips, MMOTU, SEE-AI, CECleanliness,
Kvasir-Capsule masks, BUSI, BUS-BRA, BUSClean, NIH-CXR14, NEATX, RANZCR-CLiP, CANDID-PTX, TCGA/GrandQC, BKAI-IGH.

**How facts were checked.** From this environment only GitHub is reachable; nature.com, PMC, figshare, Mendeley,
Zenodo and Kaggle are blocked by the egress proxy. Facts below come from search-engine abstracts of the data
descriptors and from the datasets' GitHub pages (including the curated atlas
github.com/TIanCat/open-medical-ultrasound-datasets, verified 2026-09). Anything marked **check** must be confirmed
on first download, before any label is read for analysis.

## Candidates, best fit first

| # | Dataset | Modality / disease | Size | Diagnosis | ROI | Artifact | Access, licence | Fit and blockers |
|---|---|---|---|---|---|---|---|---|
| 1 | **TN5000** (Zhang et al., Sci Data 2025; figshare 10.6084/m9.figshare.28455641) | thyroid US, benign vs malignant | 5,000 images (1,428 benign, 3,572 malignant) | FNA biopsy and pathology | nodule bounding boxes (VOC XML) | sonographer calipers (**check**: the descriptor says "only pure ultrasound images are included", which may mean overlays were excluded) | open, CC BY 4.0 | External validation of the thyroid cohort at a new hospital; boxes give an ROI (box mask, or MedSAM from the box), which ThyUS2Path lacked. **Check**: calipers present; patient identifiers (one image per patient view is stated); the frozen caliper detector on this scanner interface. |
| 2 | **DERM12345** (Yilmaz et al., Sci Data 2024; Harvard Dataverse 10.7910/DVN/DAXZ7P; ISIC collection 399) | dermoscopy, 40 subclasses incl. melanoma | 12,345 images, 1,627 patients, Turkey 2008–2021 | histopathology, follow-up or consensus | none (existing lesion U-Net) | hair, rulers, markings | open, CC BY | Second external validation of the hair traps on a new country and acquisition; frozen hair segmenter and U-Net apply unchanged. **Check**: melanoma count per cell against the gate; patient IDs in metadata. |
| 3 | **MILK10k** (MILK study team, J Invest Dermatol 2025; ISIC Archive) | dermoscopy + clinical close-up pairs, 48 diagnoses | 5,240 lesions, 10,480 images, 5 centres (AT, TR, US, MK, AU) | 95.7% histopathology | none (existing lesion U-Net) | hair; MONET concept probabilities for hair are released per image | open, CC BY-NC | As 2, with an independent hair label (MONET) as a sensitivity check. **Check**: drop lesions already in ISIC (an ISIC ID is given); patient IDs. |
| 4 | **EMBED** (Jeong et al., Radiol AI 2023) + artefact labels of Schueppert, Glocker, Roschewitz (ISBI 2025; github.com/biomedia-mira/mammo-artifacts) | mammography, screening/diagnostic outcome | open release 676,008 images, 23,264 patients; 22,012 images manually labelled for artefacts, detector predictions for all of EMBED | per-finding pathology (FNA, core, excision) | ROI boxes per finding (`ROI_coords`) | triangular skin markers (placed on palpable lumps, i.e. on or near the lesion) and circular markers (moles, scars: elsewhere); implants, devices, spot compression | **data-use agreement** (data.hitilab.com); artefact labels on GitHub | Strongest new modality: the marker type itself encodes location relative to the lesion, and palpability is causally linked to cancer. Blocker: the agreement must be requested by the author. Marker labels are image-level; marker location needs a simple detector (radio-opaque, high contrast). |
| 5 | **SMC-LUD** (Sci Data 2026; figshare 10.6084/m9.figshare.31112716) | liver US, HCC vs haemangioma | 5,385 images, 1,021 patients | HCC histopathology; haemangioma radiological | none stated | calipers (**check**) | open; CC BY 4.0 with non-commercial restriction | A fifth disease and organ. Blockers: no ROI annotations (a lesion segmenter would be needed); haemangioma not biopsy-proven. |
| 6 | **Galar** (Sci Data 2025; figshare+ 25304616; github.com/EKFZ-AI-Endoscopy/GalarCapsuleML) | capsule endoscopy, 29 frame labels (bubbles, dirt, ulcer, polyp, blood, angiectasia, …) | 80 videos, 3.5 M frames, ~580 GB | expert frame labels, five annotators | none (no lesion boxes) | bubbles and dirt, labelled per frame | open, CC BY 4.0 | Patient-grouped external test for the capsule cohort (video = patient). Blockers: no lesion ROI, so location cannot be measured without a lesion detector; size. |
| 7 | **REAL-Colon** (Sci Data 2024; figshare+ 22202866) | colonoscopy, polyp histology | 60 videos, 2.7 M frames, 132 resected polyps | histopathology per polyp | bounding boxes (COCO) | instruments, overlays (**check**) | open, CC BY | Histology-confirmed polyps with boxes. Blockers: the earlier colonoscopy attempt failed because no image was free of specular highlights; an artifact with a free group (e.g. instruments) would be needed; few polyps. |
| 8 | **PIBAdb** (Comput Methods Programs Biomed 2026) | colonoscopy, polyp histology, NBI and white light | 1,176 polyps, 31,946 polyp images, 14,124 non-polyp images | histopathology | bounding boxes | instruments, cleanliness levels | on request, non-profit, CC licence | Larger than REAL-Colon; same blocker as 7, plus access by request. |
| 9 | **BUSI_WHU** (927 images, 816 patients, masks; Mendeley), **BUS_UC** (811 image–mask pairs; CC BY 4.0), **BUS-UCLM** (Sci Data 2025; 683 images, 38 patients, biopsy; arrows, crosses and boxes overlap some masses) | breast US | see left | biopsy (BUS-UCLM); not stated for the others | masks | calipers, arrows, crosses | open | Same blocker that parked BUSI/BUS-BRA: almost no overlay-free images. Cheap to re-check with `scripts/data/breast_probe.py` (frozen thresholds). BUS-UCLM has too few patients for patient-grouped traps. |

## Support resources (not cohorts)
- **CheXmask** (Sci Data 2024; PhysioNet): lung masks for 657,566 radiographs of ChestX-ray8, CheXpert, MIMIC-CXR,
  PadChest and VinDr-CXR. Would give lung ROIs for a second chest-drain cohort if drain labels exist for that source.
- **Hand-annotated pen-mark masks** for TCGA slides (Patil et al., arXiv 2410.03289, preprint; github.com/abhijeetptl5/wsisegqc;
  also github.com/DIAGNijmegen/pathology-artifact-detection). Could address the reason the pathology cohort was
  rejected (GrandQC pen masks unreliable).
- Not useful: BUS-CoT (aggregate of existing breast sets incl. BUSI), Histology Tissue Fold Dataset (no diagnosis labels),
  KFGNet thyroid videos (244-video classification subset), PICCOLO (40 patients, by request).

## Recommendation
1. **TN5000 first**: open, CC BY, biopsy-confirmed, a new hospital, boxes for the ROI; it would turn the thyroid result
   into one with an external cohort. One download answers the two checks (calipers present? patient IDs?).
2. **DERM12345 or MILK10k** as a second external dermoscopy test, with the frozen hair pipeline and the existing gate.
3. **EMBED** only if the author applies for access: it is the best new modality but cannot start without the agreement.

Any of these would be a new confirmatory analysis and needs its own registration committed before labels are read.
