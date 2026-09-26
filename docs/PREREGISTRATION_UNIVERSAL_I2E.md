# Pre-registration — artifact-agnostic erasure (Universal Insert-to-Erase, U-I2E; Universal Mask-then-Erase, U-MtE)

Committed before any U-I2E / U-MtE result exists. The insertion library is `wtss.synthetic.draw_generic_artifact`
(version "generic_artifact_v1"): one procedural family — thin curvilinear strokes, thick bands, solid/translucent
patches, blobs, dashed lines, small glyphs; random colour (half greyscale), opacity 0.35–1, width, 1–3 elements,
half centred inside the ROI. It contains **no ruler with ticks and no real hair**, and is **identical for every
artifact, dataset and modality** (no per-artifact tuning). Eraser: paired-difference subspace, energy 0.90,
max rank 64 (unchanged from I2E).

- **U-I2E**: erase the generic-insertion subspace from the original-image features; head on erased features.
- **U-MtE**: ROI-mask the image, insert generic structure *then* mask, erase that subspace in the masked view;
  head trained and tested on masked images (no artifact localisation, no artifact labels).
- `_balanced` variants add image-level artifact labels (group-balanced head).

## Claims (same protocols and bootstraps as the corresponding experiments)
- **U1** synthetic ruler (fixed and variable appearance), 100 % overlap: U-I2E − ERM reversed AUROC > 0 (CI).
- **U2** synthetic ruler, 100 % overlap: U-MtE − mask > 0 (removes masking's in-ROI harm) and U-MtE − ERM > 0.
- **U3** real hair Trap A (author protocol, DINOv2 @518 and DermLIP @224): U-MtE − mask > 0 (CI) and
  U-MtE − ERM ≥ 0; Trap B: U-MtE − ERM > 0.
- **U4** no gaming: clean AUROC loss ≤ 0.02 vs ERM and correlated ≥ reversed − 0.02 wherever U1–U3 are claimed.
- **U5** (labels available) U-I2E_balanced / U-MtE_balanced ≥ balanced on reversed AUROC with clean AUROC ≥ balanced's.
Outcomes are reported either way.

## Results (appended after the runs)
Synthetic ruler, DINOv2 @518, 100 % overlap (reversed AUROC Δ vs ERM): U-I2E +0.173 [+0.133, +0.218] (U1 ✅;
clean 0.822 vs ERM 0.792; corr 0.903 ≥ rev 0.688); U-MtE −0.007 [−0.053, +0.040] (U2 ❌); U-I2E_balanced +0.291
[+0.258, +0.325] vs balanced +0.281, clean 0.824 vs 0.806 (U5 ✅); U-MtE_balanced +0.243.
Real hair, author protocol, DINOv2 @518: U-I2E +0.046 [+0.025, +0.067] (A), +0.032 [+0.020, +0.044] (B);
U-MtE − mask +0.052 [+0.044, +0.062] (A), +0.025 [+0.009, +0.040] (B); U-MtE − ERM +0.015 [−0.004, +0.033] (A),
+0.132 [+0.105, +0.160] (B) (U3 ✅); U-I2E_balanced +0.238 (A) vs balanced +0.225 with clean 0.791 vs 0.775 (U5 ✅ A),
+0.207 vs +0.209 (B); U-MtE_balanced +0.254 (B, best). No arm inverts correlated/reversed except none (U4 ✅).
