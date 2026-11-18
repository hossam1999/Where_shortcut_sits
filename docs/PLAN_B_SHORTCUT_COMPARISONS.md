# Plan B — securing one good paper to beat (and fallbacks)

Drafted 2026-11-18 on machine 2 for the author. Nothing here is registered; any part that is used gets a dated amendment
before its first result.

## 1. Why single-paper targets keep failing (four search rounds, `docs/REPLICATION_STATUS.md`)

| requirement | failure rate in the search | example |
|---|---|---|
| R7 a published number on a shortcut-broken test | most common | Friedrich NMI 2023 (accuracy flat, explanation score only), FastDiME (detection), SimBA (best method ≈ 0 disparity) |
| R4/R5 code and the test data | common | CVPR 2023 ConfDerm, SilverLining, Bevan ICML 2022 (private test), BackMix (private test) |
| R8 the artifact varies within each class | occasional, but fatal for mask_cmc | SpurBreast, MaskMedPaint |
| reproduction itself | P2, P5 failed; P4 failed its gate (rerun pending) | notebook checkpoints and repositories that postdate the paper |

So the scarce resource is not methods but **published numbers on shared shortcut tests**. Plan B therefore stops searching
paper by paper and instead (a) gathers many published numbers on a few shared tests, and (b) runs many published methods
from their official code on our tests.

## 2. Plan B for the target paper (author, 2026-11-18: the thesis needs one good paper, reproduced and beaten)

**Ranking of targets by fit** (venue, localized artifact that can sit inside the ROI, artifact label usable by our method,
code, data, published shortcut-broken number):
1. **P6 EPVT (MICCAI 2023) + PLDG (IEEE TMI 2025)** — passes every requirement: ISIC 2019 artifact domains (dark corner, hair,
   gel bubble, ruler — hair and bubbles lie over the lesion), artifact labels, code, data rebuilt (counts match the paper),
   trap-set numbers (bias 1: ERM 65.3, EPVT 74.6) and OOD Table 1 (EPVT 84.11). Citation search found no later main-venue
   paper on the same benchmark, so EPVT/PLDG is the state of the art there. **Make P6 the critical path:** bias-1
   reproduction → A30 → our arms a / b / d vs EPVT (and arm c vs PLDG) → bias 0.9 and the OOD sets.
2. **P3 ISNet (Nature Communications 2024; machine 1)** — strongest venue; its inside-the-lungs variant is the thesis question;
   stays descriptive unless its gate passes.
3. **P4 Saab (MLHC 2022)** — A31 rerun pending; if it fails, the authors are asked for the Table 1 code / checkpoints.

**Unlock near-misses through the authors** (cheap, parallel, no GPU; drafts in `docs/`, the author sends them):
- **Yan et al., CVPR 2023** ("Towards Trustable Skin Cancer Diagnosis via Rewriting Model's Decision"; same Monash group as
  EPVT/PLDG, whose code is public): ask for the ConfDerm splits and code. ConfDerm = ISIC 2019/2020 subsets, one confounder
  per class (dark corners, rulers, borders, air pockets, hair), unconfounded test; published per-class accuracies (e.g. dark
  corners: melanoma 23.48 % → 84.95 %). If shared, it is a top-venue target on the same images as P6.
- **Bevan & Atapour-Abarghouei, ICML 2022** (surgical markings and rulers; ruler test AUC 0.831 → 0.958 with CLGR; code public):
  ask whether the Heidelberg marked/ruler test set (Winkler et al.) can be shared under a data agreement.
- **Saab et al.**: the code version and checkpoints behind Table 1 (also resolves P4).
- **Pahde et al.**: the exact Table 1 configuration (draft already in `docs/EMAIL_DRAFT_PAHDE_RRCLARC.md`).

**Relax one requirement only where it keeps the comparison fair:** if a paper's test set can be rebuilt exactly from released
lists and our ERM reproduces its baseline, compare against its printed number even without its code (the papers' own
practice). Do not relax R8 (mask_cmc needs within-class artifact variation) or R6 for the primary target.

## 3. Fallback options if no target survives (most recommended first)

### B1 — Method zoo from official code on the location benchmark (phase 5 extended)  ★ recommended
**Idea.** Every comparison is "their method, their official code and recipe, on our shortcut test", identical data, folds and
selection rule for everyone. We do not need their published number (R7 dropped), only venue + code. One codebase can carry many
papers.
**Methods already available locally (official code, cloned and running on machine 2):**
- ClArC family (Pahde et al.): RR-ClArC (AAAI 2024 / ML 2025), A-ClArC, P-ClArC (same repository; R-ClArC CVPRW 2024 optional);
- ISNet (Bassi et al., Nat. Commun. 2024); Segmentation-ERM (Saab et al., MLHC 2022);
- Masked / ActDiff / GradMask / RRR (Viviano et al., ICLR 2021 code);
- EPVT (MICCAI 2023) and PLDG (TMI 2025), plus the 11 DomainBed baselines inside the EPVT repository with published skin
  numbers (ERM, GroupDRO, CORAL, MMD, DANN, IRM, SagNet, MLDG, CAD, DoPrompt, SelfReg);
- GroupDRO + NoiseCrop (Bissoto et al., ECCV-W 2022; code cloned).
**Methods to add from official code (no medical numbers needed):** DFR (ICLR 2023), JTT (ICML 2021), LfF (NeurIPS 2020), LEACE
(NeurIPS 2023), MaskTune (NeurIPS 2022) — e.g. via SubpopBench (Yang et al., ICML 2023, one framework with ~20 algorithms; to
be verified before use).
**Test.** Our registered design: Trap A (artifact inside the ROI) and Trap B (outside), correlated / reversed / clean tests,
3 folds, reversed AUROC primary; cohorts with real in-ROI artifacts: thyroid calipers (TN3K), ISIC hair, capsule debris, NIH
chest tubes (all built for the thesis).
**Claim it supports.** "Location-based fixes (masking, relevance outside the ROI, segmentation supervision, NoiseCrop) work in
Trap B and fail in Trap A; concept-level and group-robust fixes are the real competitors in Trap A; our constraint vs each."
This is the thesis claim itself, with many papers as competitors.
**Cost.** Phase-5 timing test first; roughly 1–3 GPU-h per method × cohort × trap × fold for small backbones; prioritise
thyroid + ISIC hair, Trap A, the 8–10 strongest methods (≈ 50–100 GPU-h).
**Risk.** Reviewers may say "not their test"; answer: identical official code and recipe, selection on validation only, and
the papers' own tests where they reproduced.

### B2 — Leaderboards on shared shortcut tests ("many papers, one test")
**Idea.** Use test sets on which several published papers already report numbers, reproduce one baseline (ERM) as a
calibration, and place our method against all published numbers on the same images.
| shared test | papers with numbers on it | our status |
|---|---|---|
| Bissoto ISIC 2019 trap sets (bias 0–1, 10 splits) | Bissoto ECCV-W 2022 (ERM 0.58, GroupDRO 0.68, +NoiseCrop 0.74 at bias 1), EPVT MICCAI 2023 and PLDG TMI 2025 (figures), Bissoto MICCAI-W 2023 (test-time selection) | data rebuilt (P6), ERM/EPVT reproduction queued |
| Bissoto ISIC 2018 trap sets (772-image test) | Bissoto CVPRW 2020, Saab MLHC 2022 (Binary 0.359, Seg 0.727) | P4: Binary reproduced, Seg rerun pending |
| EPVT OOD skin sets (Derm7pt-D/C, PH2, PAD-UFES-20) | 12 methods in EPVT Table 1 + PLDG Table I | needs Derm7pt (free account), PH2, PAD (public) |
| NIH ChestX-ray14 pneumothorax with drain labels (NEATX) | Jiménez-Sánchez ISBI 2023 (AUC with / without drains), FastDiME ECCV 2024 (correlation sets) | thesis has NIH + NEATX |
**Claim.** "On the shared test, our method is above every published number" — with the honest caveat that published numbers
come from different backbones and protocols; our own ERM reproduction anchors the comparison.
**Cost.** Low: one ERM calibration per test plus our heads (CPU) and arm d.

### B3 — Controlled location sweep with public generators
**Idea.** Use a published, coded generator so the artifact position is the only variable, and run that paper's own baselines:
- **SimBA** (MICCAI 2023 / JAMIA 2024; code and data public): their near / far bias scenarios as published, plus a new
  "inside the disease region" scenario made with their generator; their baselines (reweighing, unlearning, group models) and ours.
- Synthetic artifact insertion into real images (our thesis pipeline; OASIS-style synthetic skin artifacts, CVPRW 2026).
**Claim.** Causal evidence for "where the shortcut sits": the same method helps far/outside and fails near/inside.
**Cost.** SimBA 3D CNN, 4.3k volumes per scenario, 5 seeds (Keras/TF; a PyTorch port of their CNN would be recorded).

### B4 — Salvage failed replications ("their code, our test")
For every paper whose reproduction fails (P2, P5, possibly P4), keep their method as a competitor in B1 and report the
failed reproduction openly (already registered for P2, A18). No "beaten" claim against their printed number; "beaten" only
against our run of their code under identical conditions.

## 4. When to switch, and in what order
- **Trigger:** after the A31 P4 rerun and the P6 step-1 check. If P4 fails again and P6 (EPVT) does not reproduce, start B1
  with the methods already running here (ClArC family, ISNet, Seg-ERM, Masked/ActDiff, GroupDRO+NoiseCrop, EPVT) on thyroid
  and ISIC hair, Trap A first; add B2 on the shared ISIC trap sets in parallel (CPU heads); B3 if time remains.
- **Pre-registration:** one amendment per option before its first result: methods, official commits, recipes, cohorts, folds,
  metric, Holm family, success rule (the phase-3/phase-5 rule: paired CI above 0 on reversed AUROC, clean drop ≤ 0.01).
- **Reporting tiers (kept separate in every table):** (1) beaten on their own test after reproduction; (2) beaten under
  identical conditions on our test with their official code (B1); (3) above published numbers on a shared test (B2, with the
  protocol caveat); (4) controlled location evidence (B3).
