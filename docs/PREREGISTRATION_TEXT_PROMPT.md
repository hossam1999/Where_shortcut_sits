# Pre-registration — text-prompted artifact erasure and zero-shot reliance (vision–language backbones)

Committed before any model is fitted or any zero-shot score is computed for this analysis.

## Motivation
MedSigLIP and DermLIP embed images and text in one space. A text description of the artifact ("a thyroid ultrasound
image with measurement calipers") therefore gives an artifact direction with **no image, mask or label of the
artifact** — a modern annotation-free alternative (cf. biased-prompt debiasing, Chuang et al. 2023) to U-MtE's
synthetic overlays.

## Method (frozen; `wtss.text_directions`)
- For each cohort: 4 neutral templates × 4 artifact phrases; direction set = {emb(template + phrase) − emb(template)};
  SVD, rank k = smallest with 90 % energy (max 8). Prompts are fixed in `PROMPTS` (not tuned on any result).
- Erasure on the cached joint-space image features: x ↦ x − U Uᵀ (x − μ), μ = mean of training features.
- Arms: text_erase (unmasked), mask_text_erase, mask_text_erase_balanced, mask_text_erase_protect (disease
  protection as in U9). Compared with mask, U-MtE (mte), U-MtE_protect, U-MtE_balanced on the same deterministic
  environments.
- Backbones/cohorts: MedSigLIP — thyroid, capsule, ovary; DermLIP — ISIC 2019 hair (spec traps).

## Zero-shot reliance (no training)
Zero-shot score = cos(x, t_pos) − cos(x, t_neg) with the class prompts in `PROMPTS[cohort]["classes"]`, on the
unmasked and masked views of the trap test environments. Question: does a foundation model *without any task
training* already rely on in-ROI artifacts (correlated vs reversed AUROC gap), and does masking change that?

## Claims (reversed AUROC, Trap A, 95 % CI)
- **V1** mask_text_erase − mask > 0 (text-prompted erasure removes in-ROI shortcut).
- **V2** mask_text_erase − U-MtE: reported with CI, no directional prediction.
- **V3** mask_text_erase_balanced − balanced > 0.
- **Z1** zero-shot: report correlated − reversed AUROC gap (Trap A) per cohort (descriptive).
Reported whichever way they go.
