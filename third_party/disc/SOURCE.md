# P8 — DISC: Wu, Yuksekgonul, Zhang, Zou, "Discover and Cure: Concept-aware Mitigation of Spurious Correlation", ICML 2023

Set up 2026-11-18 on machine 2 (author: "set up P8 with DISC"). Step 1 only (reproduce their ERM and DISC on the ISIC trap
sets) until an amendment registers the comparison.

- Paper: PMLR v202 (wu23w); arXiv 2305.00650. ISIC: Bissoto et al. (2020) trap sets, "each training split amplifies the
  correlations with 7 spurious attributes … hairs and gel bubbles could co-exist and cover the skin lesion region" (p. 6);
  metric: average AUROC "across the train-test splits, as standard in Bissoto et al. (2020)"; "All the experiments are repeated
  three times". Table 1 (ISIC, average AUROC): ERM 36.4 ± 0.7, ERM+aug 38.9 ± 1.5, UW 39.2, IRM 45.5 ± 3.6, IB-IRM 38.6,
  V-REx 24.5, CORAL 37.9, Fish 42.0, GroupDRO 36.4 ± 0.9, JTT 33.8, DM-ADA 35.8, LISA 38.0, **DISC 55.1 ± 2.3**. Methods that
  need domain labels use hair presence (Table 2d).
- Code: https://github.com/Wuyxin/DISC @ `d9865a8` (2025-11-17). Local `/workspace/third_party/disc`. ISIC commands:
  `scripts/isic.sh` (ERM: ResNet-50, lr 1e-3, batch 16, wd 1e-5, 100 epochs, `--seed $GROUP` = trap-set id; DISC: lr 5e-4,
  `--erm_path` the ERM best model, concept categories color-texture, 3 clusters). Upstream env: Python 3.9, torch 1.11.0.
  Ours: `/workspace/envs/disc`, Python 3.10, torch 2.11.0+cu128 (Blackwell).
- Data: their ISIC archive (Google Drive `1Os34EapIAJM34DrwZMw2rRRJij3HAUDV`, 11.2 GB) → `$WTSS_DATA/disc/isic`
  (`ISIC2018_Task1-2_Training_Input`, `trap-sets`). Checked against Bissoto f93c865: splits 2–5 byte-identical; split 1 same
  image ids, labels and artifact flags (only `.jpg` vs `.png` in `image`). Same data as P4.
- Concept bank: their generated Stable-Diffusion concept images (Drive `1EX0MP39xvUaYprCIReuTQWVZPj7xsops` texture,
  `1h-9uS7kDjDZSbbZc90da3lzMjPpwLj_g` other categories) → `$WTSS_DATA/disc/synthetic_concepts`.
- Related, same test (not P8): CBDebug (ICLR 2026) — see `docs/REPLICATION_STATUS.md` 2026-11-18 ~23:30.
