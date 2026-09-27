# Pre-registration — modern inpainting baseline (LaMa)

Committed before any model is fitted on LaMa-inpainted images.
- Method: LaMa (big-lama; Suvorov et al., WACV 2022) inpaints the artifact pixels (artifact mask dilated 3 px) of
  every image with a non-empty mask (thyroid/ovary: detected calipers; capsule: probe debris masks; ISIC: Wegley
  hair/ruler masks) — an ORACLE w.r.t. the artifact masks, like the classical Telea arm. Cache:
  scripts/data/precompute_lama.py.
- Arms: lama (ERM head on LaMa-inpainted images), mask_lama (inpaint, then ROI-mask), compared with Telea inpaint,
  mask, U-MtE_protect and U-MtE_balanced (existing runs; same deterministic environments → paired).
- Claims (reversed AUROC, Trap A): **L1** lama − mask; **L2** U-MtE_protect − mask_lama; reported with CIs, no
  directional prediction (LaMa uses oracle masks; U-MtE uses none). Reported either way.
