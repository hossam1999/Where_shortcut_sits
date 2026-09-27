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

## Results (Trap A reversed AUROC [95 % CI]; results/*/medsiglip448_text, results/spec_e13/dermlip224_spec_text)
| cohort (backbone) | V1 mask_text_erase − mask | U-MtE − mask | V2 mask_text_erase − U-MtE | V3 text_bal − balanced |
|---|---|---|---|---|
| thyroid (MedSigLIP) | +0.005 [+0.003, +0.008] | +0.204 [+0.174, +0.237] | −0.199 [−0.230, −0.170] | +0.056 [+0.038, +0.073] |
| ovary (MedSigLIP) | +0.008 [−0.001, +0.021] | **+0.215 [+0.182, +0.246]** | −0.207 [−0.235, −0.178] | −0.010 [−0.037, +0.020] |
| capsule (MedSigLIP) | +0.001 [−0.000, +0.003] | −0.236 [−0.252, −0.221] | +0.238 [+0.222, +0.253] | +0.066 [+0.038, +0.093] |
| ISIC hair (DermLIP) | +0.004 [−0.002, +0.011] | −0.076 [−0.094, −0.060] | +0.080 [+0.064, +0.097] | −0.112 [−0.132, −0.092] |
- **V1 essentially not supported**: text-prompted erasure is a near no-op (≤ +0.008; significant only on thyroid).
  The text-space artifact direction does not coincide with the image-space direction along which the artifact moves
  the representation (modality gap), so erasing it leaves the shortcut intact.
- V2: U-MtE is far better where the artifact is a distinct overlay (thyroid, ovary); text erasure "wins" on capsule and
  hair only because it does nothing while U-MtE (unprotected) removes disease signal there.
- V3: supported 2/4. New replication: U-MtE − mask on ovary with MedSigLIP = +0.215.

## Zero-shot reliance (Z1; results/zero_shot/)
No task training. Correlated − reversed AUROC gap (unmasked, Trap A): thyroid +0.066, capsule −0.398, ovary −0.126,
hair +0.081 — the foundation models' zero-shot scores are already swayed by artifacts (capsule debris by 0.40 AUROC).
Masking effect on zero-shot reversed AUROC, in-ROI vs out-of-ROI (post hoc crossover test):
thyroid +0.013 vs +0.116 (crossover +0.103 [+0.079, +0.127]); capsule +0.007 vs +0.056 (+0.049 [+0.024, +0.076]);
ovary −0.235 vs −0.054 (+0.181 [+0.135, +0.228]); hair (DermLIP) −0.078 vs −0.027 (+0.051 [+0.031, +0.071]).
**The location law holds even without any training** (4/4 crossovers > 0). Caveat: MedSigLIP zero-shot thyroid
classification is at chance (clean AUROC 0.45), so that row reflects artifact effects on an uninformative score.
