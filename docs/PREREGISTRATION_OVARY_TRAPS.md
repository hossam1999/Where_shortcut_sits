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

## Results — DINOv2@518, template arms (results/ovary/dino518_main)
- **O1 supported**: mask − ERM (Trap B) +0.30 (0.725 vs 0.423).
- **O2 not supported**: mask − ERM (Trap A) +0.13 (0.545 vs 0.414): masking helps less in-ROI, no harm.
- **O3 supported**: crossover +0.170 [+0.116, +0.224].
- Template MtE − mask (Trap A) +0.070 [+0.030, +0.106]; MtE_balanced reversed 0.729 / corr 0.790 (min 0.729) —
  best robust arm (balanced 0.707, DFR 0.689).

## Results — DINOv2@518, generic library (results/ovary/dino518_universal)
Trap A reversed / corr: mask 0.545 / 0.828; U-MtE 0.564 / 0.855; U-MtE_protect 0.588 / 0.832; mte_aug 0.529;
JTT 0.428; balanced 0.707 / 0.732; mask_balanced 0.687; U-MtE_balanced 0.736 / 0.811; mask+DFR 0.752 / 0.757.
- **O4 not supported**: U-MtE − mask +0.019 [−0.007, +0.043]; protected U-MtE − mask **+0.043 [+0.025, +0.061]**.
- **O5 supported**: U-MtE − mte_aug +0.035 [+0.010, +0.059]. U-MtE − JTT +0.137 [+0.098, +0.179].
- **O6 not supported**: by min(rev, corr) mask+DFR (0.752) > U-MtE_balanced (0.736) > balanced (0.707).

## Results — controlled synthetic calipers, DINOv2@518 (results/synthetic/ovary/dino518_caliper_corr_main; 343 images)
- TS3 not supported: mask − ERM at r=0 = +0.042 [−0.034, +0.119].
- **TS2 supported**: mask − ERM at r=1 = **−0.197 [−0.292, −0.100]** (masking harms when the caliper is inside).
- **TS1 supported**: location interaction +0.239 [+0.175, +0.306].
- **TS4 supported**: U-MtE − mask at r=1 = +0.141 [+0.091, +0.195]; protected +0.115 [+0.067, +0.165];
  U-MtE_balanced − mask +0.253 [+0.187, +0.325]; U-MtE_balanced − balanced −0.031 [−0.116, +0.054].
