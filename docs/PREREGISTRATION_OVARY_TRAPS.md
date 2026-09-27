# Pre-registration — ovarian tumour ultrasound caliper traps (MMOTU OTU_2d)

Committed before any model is fitted on this cohort. Protocol = thyroid traps (docs/PREREGISTRATION_THYROID_TRAPS.md).
- Data: MMOTU OTU_2d (Kaggle orvile/mmotu-ovarian-ultrasound-images-dataset, CC BY 4.0), 1,469 images with tumour
  masks and 8 classes. Task: benign cystic (chocolate cyst, serous cystadenoma, simple cyst; y=0) vs lesions with
  solid components (teratoma, theca cell tumour, mucinous cystadenoma, high-grade serous carcinoma; y=1); normal
  ovary excluded → 1,202 images.
- Markers: the frozen thyroid detector, unchanged. Audit (8 per cell, scratch panel): Trap A detections are dotted
  measurement lines crossing the tumour (high precision); ~2/8 "marker-free" images show small marks (dilutes, does
  not bias); Trap B detections are mostly other on-screen overlays (dotted zoom-box arcs, focus markers).
- A0 = no detected marker pixel; A1 = >= 15 px; Trap A r >= 0.5, Trap B r < 0.1.
- Cells (A0Y0/A0Y1/A1Y0/A1Y1): Trap A 170/178/341/276, Trap B 170/178/91/103; reversed positives 196 (gates pass).
- Groups: pHash <= 8 (1,059 groups, largest 13); no patient ids (limitation).
- Backbone DINOv2@518 (primary). Arms: as thyroid (template + generic library, incl. mte_protect, mte_aug, jtt,
  mask_dfr); then the controlled synthetic-caliper sweep on the marker-free images (user rule: synthetic after real).
- Claims: O1 mask − ERM > 0 (Trap B); O2 mask − ERM < 0 (Trap A); O3 crossover > 0; O4 U-MtE − mask > 0 (Trap A);
  O5 U-MtE − mte_aug > 0 (Trap A); O6 U-MtE_balanced best by min(rev, corr) among label-using arms (Trap A).
Reported whichever way they go.

## Controlled synthetic-caliper sweep (registered with the real traps, before fitting)
Marker-free MMOTU images (343 after common support; test split 64 images — underpowered, wide CIs expected);
draw_caliper, 65x19 box, r ∈ {0,0.25,0.5,0.75,1}; seeds 42/123/456; DINOv2@518; claims as TS1–TS4.
