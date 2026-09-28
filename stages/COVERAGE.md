# Coverage checklist: every piece of work → the stage report that contains it

Sources checked (Stages 1–6): `report/full/full_report.tex` (the complete record on `main`, its Stages 0–15), every document in
`docs/`, `CHANGES.md`, `results/`, and the additions of the branch `stages-and-final-runs` (crossed bootstrap A1,
regression adjustment A2, ISIC 2020 A4, audit package, licences). Rule followed by the reports: a stage only refers
back to earlier stages, never forward.

| Work (where it lives in the repository) | Stage | Section |
|---|---|---|
| Clinical problem, question, design rationale | 1 | 2–3 |
| Pilot E1–E15, specification (`docs/REPLICATION_SPEC.md`), deviations (`CHANGES.md`) | 1 | 6 |
| Replication ledger (`results/rerun_2026-09-28/SUMMARY.md`), the five mismatches | 1 | 7.1, 7.6 |
| Controlled ruler sweep E1–E3, arms E8–E9 (inpaint, consistency, balanced, DFR, LEACE, GroupDRO) | 1 (numbers), 2 (analysis) | 1: 7.2; 2: 5.1 |
| E4–E7 occlusion, dilation, counterfactual, lesion size | 1 (replication), 2 (mechanism) | 2: 5.2 |
| E10 hair decodability and paired LEACE; DullRazor detector | 1 | 7.3 |
| E12 overlap-contrast design and why it fails | 1 | 7.4 |
| E13/E15 real hair traps (DINOv2, DermLIP), cell counts, source strata, E14 | 1, 3 | 1: 7.3; 3: 5.2, 5.6 |
| E11 atlas / vignetting trap (not re-implemented; reason given) | 1 | 7.5 |
| Thesis-claims audit (27 % leakage claim, U-Net, mask-source swap, archive re-run) | 1 | 7.5 |
| Leakage: grouped vs random splits | 1 | 5.1 |
| Crossed seed × image bootstrap (the correction) | 1 (method + pilot check); 2–5 (per-stage check) | "Sensitivity to the interval estimator" in each |
| Pre-registration practice and its caveat | 1 | 5.3 |
| Other-modality controlled sweeps (thyroid, ovary, capsule, chest tube; DINOv2, MedSigLIP, RAD-DINO), all baseline arms | 2 | 5.3 |
| Capsule sweep split reproducibility (32 of 71 frames) | 2 | 6 |
| Real traps: thyroid (TN3K/TNCD), ovary (MMOTU), capsule (SEE-AI); detectors, probe, audits | 3 | 3–5.1 |
| Real-trap crossovers with DINOv2, MedSigLIP, ConvNeXt; every baseline arm; environment AUROCs | 3 | 5.2, 5.4 |
| Four location tests (P1) with Holm | 3 (Holm over 4), 6 (full family of 16) | 3: 5.3; 6: 5.3 |
| Hair correlates (metadata, pixel share), author-independent reconstruction, metadata-matched traps | 3 | 5.5 |
| DermLIP exception | 3 (observation), 5 (explanation, theory) | 3: 5.6; 5: 5.4 |
| Chest radiography: device traps, device-matched follow-up, chest drains | 3 | 5.7 |
| Rejected datasets (breast ×2, CANDID-PTX, cardiomegaly, pathology pen, ISIC ink, EAD, NeoPolyp, DINOv3) | 3 | 5.8 |
| Covariate balance, matched traps (0.2 and 0.05 SD), regression adjustment A2 | 4 | 5.1–5.3 |
| Real-artifact transplant (T1–T4) and neutral paste (N1–N3) | 4 | 5.4 |
| Artifact-label sensitivity (large markers, strict location, capsule strict-B) | 4 | 5.5 |
| Embedding-based leakage groups (P1); remedy tests P2–P4 | 4 (P1), 6 (P2–P4) | 4: 5.5; 6: 5.9 |
| Rebuild on a second machine (R0) | 4 | 5.5 |
| Natural test sets, clinical metrics, operating points, n = 78 subgroup | 5 | 5.1–5.2 |
| Replication breadth: encoder scale S/B/L (ERM gap), fine-tuned ResNet-50 / ViT-S, zero-shot VLMs, ISIC 2020 external validation (A4) | 5 | 5.3 |
| Absolute performance vs literature | 5 | 5.3 |
| Theory: closed form, simulation, held-out prediction test T1–T4, verdict | 5 | 4, 5.5 |
| I2E → MtE → U-MtE, generic overlay library, protection, balanced variant | 6 | 3 |
| Remedies in all real traps, encoders; robustness min(rev, corr); primary family of 16 | 6 | 5.1–5.3 |
| Remedies in controlled sweeps (incl. protected sweeps, U-I2E) | 6 | 5.4 |
| Erase vs augment (U7), protection (U9), secondary contrasts | 6 | 5.5 |
| Rank ablation; encoder size for U-MtE | 6 | 5.6–5.7 |
| JTT, SPLINCE, LaMa, text-prompted erasure, SLAS, pseudo-groups (U8), erasure + DFR (U10), selectors (U13/U14), drains | 6 | 5.8 |
| Fine-tuned remedies: mte_ft, umte_ft, cons_ft, umte_cons_ft, fine-tune-then-erase, ovary power runs | 6 | 5.10 |
| Remedies on unaltered data (N1, N2, N4) | 6 | 5.11 |
| Decision guide; `wtss.umte` / `wtss.slas` tools | 6 | 3, 5.12 |
| Clinician audit package, licences | 6 | 6 |
| Manuscript (paper/), NeurIPS version, number audit, CLAIM 2024, related work, statements, limitations | 6 | 7–8 |
| Round 4 (`docs/PREREGISTRATION_ROUND4.md`, `scripts/round4/`, `results/round4/`): masking implementations R8 | 7 | 5.1 |
| Round 4: dose-response over the real overlap R9 (bin counts committed before fitting) | 7 | 5.2 |
| Round 4: ISIC 2019 → ISIC 2020 natural test, near-duplicate removal, operating points R10 | 7 | 5.3 |
| Round 4: prospective theory test R11 | 7 | 5.5 |
| Round 5 (`docs/PREREGISTRATION_ROUND5.md`, `results/round5/`): calibrated duplicate rule R12, cleaner hair groups R13 | 7 | 5.4 |
| Round 6 (`docs/PREREGISTRATION_ROUND6.md`, `scripts/round6/`, `results/round6/`): sources, licences, coverage A0 | 8 | 5.1 |
| Round 6: label agreement A1, traps on uncontradicted labels A2, differential error A3 (dermoscopy, thyroid, ovary, capsule) | 8 | 5.2–5.5 |
| Round 6: exploratory consensus view of the caliper labellers (E1), registered subgroup labels | 8 | 5.3 |
| Round 6: blinded rating tool and MedGemma audit answers (A4, A4b; rating not yet done) | 8 | 2, 6 |
| Round 6: fine-tuned thyroid model, recipe B1, FT0–FT4, operating points, comparison with the frozen probe | 8 | 5.6 |
| Round 7 (`docs/PREREGISTRATION_ROUND7.md`, `scripts/round7/`, `results/round7/`): thresholds matched on the test set, TM1–TM2 | 8 | 5.7 |
| Round 6 exploratory E1 (consensus traps) and E2 (confirmed subgroup) | 8 | 5.3 |

Analyses whose per-image predictions were not saved (chest-radiograph devices and drains, LaMa, text prompts, SLAS,
U8, U10, selectors, archived hair sensitivity designs) appear as point estimates labelled "not re-estimated"; their
values are written to `results/stage_derived/` by the stage generators so that the number audit can trace them.
