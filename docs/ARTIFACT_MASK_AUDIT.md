# Artifact-mask audit (no clinician available; visual audit by the analysis agent)

Judging whether an on-screen caliper/overlay is present is a non-clinical visual task. Panels of 40 random images
per cell (seed 101; `scripts/audit/make_audit_panels.py`, ids saved next to the panels) were inspected at 200 px;
very faint marks may be missed at that size (limitation). An earlier 20-image audit is in THYROID_MARKER_AUDIT.md.

| cohort | cell | criterion | result |
|---|---|---|---|
| Thyroid (TN3K) | detected (A1) | detection lies on a real caliper / measurement mark | 38/40 (95 %); 2 border false positives; 1 dotted zoom box (overlay) |
| Thyroid | marker-free (A0) | no visible caliper | 38–39/40 (95–97 %); depth-scale ticks present in all images (not label-related) |
| Ovary (MMOTU) | detected (A1) | an on-screen overlay is present | 40/40; caliper specifically 31/40 (others: dotted zoom arcs, focus markers, thumbnail inset) |
| Ovary | marker-free (A0) | no visible caliper | 37/40 (92 %) |
| Ovary | marker-free (A0) | no overlay of any kind (arrows, boxes) | ~29/40 (72 %); text labels common |
| Capsule (SEE-AI) | probe debris masks | vs expert masks (540 held-out frames) | pixel accuracy 0.92, IoU 0.62 |
| ISIC 2019 | Wegley hair/ruler masks | published semi-automatic masks | not re-audited |

Implications: label noise on A dilutes the constructed correlation (biasing effects toward zero), it does not create
the location effect. The sensitivity analysis (docs/SENSITIVITY_ARTIFACT_LABELS.md) re-runs the thyroid and ovary
claims with large-marker-only and strict-location definitions. For the ovary cohort, "marker-free" means
caliper-free, not overlay-free — stated as a limitation in the paper.
