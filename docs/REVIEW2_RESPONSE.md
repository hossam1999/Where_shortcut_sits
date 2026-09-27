# Second external review (2026-09-27) — verification and responses

A colleague read `report/full/full_report.pdf` and ranked what stands between the work and acceptance at a Q1
journal. Each claim was first checked against the repository's own result files; the response column says what was
done. New experiments are pre-registered in `docs/PREREGISTRATION_REVIEW2.md` (pushed before any model was fitted).

## Verification of the review's factual claims
| # | Review claim | Verdict | Evidence in the repository |
|---|---|---|---|
| 1 | Trap A and Trap B are different image populations; in-ROI hair may go with larger lesions / body sites; hair is correlate-carried (oracle hair removal +0.015) | **Correct** | Traps share the artifact-free group but select different artifact-bearing images, matched only on source (`isic2019_spec.matched_pool`). `results/analysis/trap_metadata_*.csv`: e.g. BCN age 58.9 (Trap A) vs 51.8 (Trap B); head/neck 33 % vs 22 %. No lesion-size, nodule-size or artifact-amount matching; no real-artifact transplant existed. |
| 2 | In-ROI masking still helps on real thyroid/ovary/capsule (+0.111, +0.132, +0.096); natural harm is mixed (BCN −0.016, MSK −0.056, HAM +0.040); the universal claim is the gap | **Correct** | `results/{thyroid,ovary,capsule}/dino518_main`, `results/natural/*/natural_boot.json`; report Stage 4 already says "the universal claim is the gap". The abstract said masking "fails" in-ROI. |
| 2b | Thyroid sensitivity 0.734 → 0.347 needs CIs, several operating points, all seeds | **Correct** | `paper/tables/natural_clinical.tex` gives mean (SD) over seeds at one validation-fixed threshold only; AUROC is unchanged (0.736 vs 0.731), so the drop may be a threshold-transfer effect. |
| 3 | Remedies do not transfer to unaltered data (capsule U-MtE −0.005, protected −0.022; BCN +0.004; balancing capsule −0.086) | **Correct** | `report/full/gen/tab_natural.tex` (all numbers exact). On thyroid hard pairs U-MtE_bal − ERM ≈ −0.014. |
| 4 | Thyroid, ovary, capsule have no patient IDs | **Correct** | `docs/DATA.md`, `paper/sections/discussion.tex`. |
| 5 | Absolute performance is not compared with the literature; MedSigLIP zero-shot thyroid is at chance | **Correct** | No literature comparison in paper or report; zero-shot thyroid clean AUROC 0.45. ISIC hair also has no fine-tuned model. |
| 6 | Artifact labels are detector/probe based (thyroid 95 % precision; capsule IoU 0.62) without an independent expert audit | **Correct, and stronger** | `docs/ARTIFACT_MASK_AUDIT.md`: the visual audit was done "by the analysis agent", no clinician. |
| 7 | The 16-test family was defined after the individual registrations | **Correct** | `docs/STATISTICAL_PLAN.md` states it. Git order checked (`results/review2/registry.csv`): every registration precedes its outcome results (SLAS registration and its localisation result share one commit); but commit times are client-set and the work spans ~26 h, so the order is not externally verifiable. |
| F1 | r = 0.997 should not be headlined; the naive predictor 2·clean − corr reaches r = 0.989 | **Correct** | `results/theory/prediction_summary.json`: T2 ref (b) r = 0.989; T1 MAE 0.039 vs 0.108; T3 signs 42/45 vs 36/45. |
| F2 | The DermLIP hair crossover (−0.002) is omitted from the replication list | **Correct for the report** | Report executive summary lists replications without it; the paper mentions DermLIP but does not explain it. The theory fitted to DermLIP's clean/correlated AUROC gives −0.003 (`report/full/gen/tab_theory_x.tex`). |
| F3 | CLAIM checklist missing | **Correct** | No CLAIM checklist in the repository. |

## Responses (filled in as results arrive)
- **U-MtE demoted.** One paper, one story: location law → mechanism → theory → harm on unaltered data. Remedies
  become a compact decision guide ("what carries the shortcut decides the fix"); U-MtE stays as a tool and in the
  supplement. (Decision of the author, 2026-09-27.)
- Point 1: R1 (covariate balance, matched crossover) and R2 (real-artifact transplant) — `PREREGISTRATION_REVIEW2.md`.
- Point 2: title/abstract state the gap as the universal claim and harm as conditional; R3 operating points.
- Point 3: see U-MtE demotion.
- Point 4: external cohort with patient identifiers — search and feasibility below.
- Point 5: R4 (fine-tuned hair model; literature table).
- Point 6: expert audit kit (`scripts/audit/make_expert_audit_kit.py`) — needs two human raters.
- Point 7: registry table with commit hashes (`results/review2/registry.csv`); new registrations pushed before running.
- Framing: theory leads with held-out reversed AUROC MAE and sweep signs; DermLIP explained; CLAIM checklist added.
