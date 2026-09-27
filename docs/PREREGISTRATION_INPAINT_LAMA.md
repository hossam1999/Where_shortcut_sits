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

## Results — DINOv2@518, Trap A reversed AUROC (results/{thyroid,ovary}/dino518_lama; results/lama_comparison.json)
| | thyroid | ovary |
|---|---|---|
| LaMa − ERM | +0.136 [+0.130, +0.142] | +0.158 [+0.141, +0.176] |
| LaMa − mask | +0.025 [+0.009, +0.040] | +0.027 [−0.017, +0.069] |
| mask∘LaMa − mask | +0.226 [+0.213, +0.238] | +0.123 [+0.089, +0.155] |
| **L2** U-MtE_protect − mask∘LaMa | −0.009 [−0.045, +0.023] | −0.080 [−0.108, −0.053] |
| U-MtE_bal − mask∘LaMa | +0.162 [+0.152, +0.173] | +0.067 [+0.037, +0.095] |
Modern inpainting of the (detected) caliper pixels followed by masking is a strong baseline: it needs artifact
masks, which U-MtE does not. U-MtE_protect matches it on thyroid (n.s. difference) and is below it on ovary;
with image-level labels, U-MtE_bal beats it on both. Reversed / corr (clean) of mask∘LaMa: thyroid 0.626 / 0.825
(0.735), ovary 0.668 / 0.812 (0.741).
