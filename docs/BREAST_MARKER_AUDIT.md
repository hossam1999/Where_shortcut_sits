# Breast ultrasound (BUSI + BUS-BRA) — marker-mask feasibility audit (2026-09-27): PARKED

Data: BUSI (Kaggle aryashah2k, 647 benign/malignant with tumour masks) + BUS-BRA (Kaggle orvile, 1,875 images,
1,064 patients, biopsy-proven pathology, tumour masks). Both show sonographer calipers, often inside the tumour
(BUSI dotted measurement lines cross the lesion — the in-ROI case).

Detector: `wtss.data.us_markers.marker_mask` (thyroid-frozen '+'/dotted-line rules) + an optional thick 'x'
template for GE-style asterisk calipers (`shapes=("plus", "x")`; thyroid default unchanged).
- v1 ('+' only): BUS-BRA "marker-free" images contained visible 'x' calipers in 7/8 audited images → unusable A0.
- v2 two-detector design (A0 = nothing found by a sensitive detector, thr 0.50/30; A1 = strict detector
  0.62/45): cells A0 386/190, Trap A 171/104, Trap B 470/184 (benign/malignant), but a fresh 10+10 audit found
  ~3/10 A0 images with faint grey '+' calipers missed, and text false positives among A1.
Decision: labels too noisy (A0 purity ≈ 70 %) for a confirmatory trap; not pre-registered, no model fitted.
Revisit with manual caliper annotation (≈ 50 images, SLAS few-shot probe) if needed.

## Attempt 2 (2026-09-27): annotation-free caliper probe — PARKED (definitive)
A DINOv2 patch-token caliper probe (wtss.slas) trained on thyroid + ovary detector masks reaches patch AUROC 0.983 on
held-out thyroid/ovary images (scripts/data/breast_probe.py). Applied to BUSI + BUS-BRA with thresholds fixed before
any breast label was inspected (A0: no patch >= 0.30; A1: >= 2 patches >= 0.70), only 19 of 2,522 images qualify as
caliper/overlay-free (BUS-BRA 6, BUSI 13): nearly every breast image carries on-screen markings, so no artifact-free
comparison group exists and the trap design is infeasible. Breast ultrasound is not used.
