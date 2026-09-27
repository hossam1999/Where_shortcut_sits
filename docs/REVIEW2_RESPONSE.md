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

## Responses (numbers: results/review2/*.csv; tables paper/tables/review2_*.tex)
All new analyses were pre-registered in `docs/PREREGISTRATION_REVIEW2.md` and pushed to GitHub before any model was
fitted (commit 29cd0da; amendment 1 in 50ac10a). All data were re-downloaded and every derived object rebuilt on new
hardware first.

**R0 — rebuild.** Every primary crossover reproduces within 0.01 (hair +0.147 [+0.119, +0.177], thyroid +0.239
[+0.212, +0.264], ovary +0.170 [+0.116, +0.223], capsule +0.377 [+0.339, +0.412]; archived +0.145, +0.230, +0.170,
+0.368). Makefile gaps found and fixed (native hair stats, U-Net stages).

**Point 1 — confounding of the real traps (the reviewer was right that the populations differ).**
- Covariate balance (R1a): in-ROI artifacts go with much larger lesions — hair SMD 2.13 (lesion area), capsule 2.57
  (lesion-box area), thyroid 0.40 (nodule area) — and more marker pixels (thyroid 0.53).
- *Real-artifact transplant (R2)*: the same real caliper / hair strand / debris pasted inside (overlap ≥ 0.95) or
  outside (overlap 0) the ROI of the same artifact-free images. Location interaction: hair +0.253 [+0.231, +0.275],
  thyroid +0.277 [+0.264, +0.290], ovary +0.074 [+0.055, +0.092], capsule +0.497 [+0.455, +0.539]; **4/4
  Holm-significant**. Outside the ROI masking removes the shortcut completely (reversed = correlated = clean AUROC);
  inside it survives. By the pre-registered interpretation rule, the location effect cannot be explained by
  population differences between Trap A and Trap B images in any cohort.
- *Covariate-matched traps (R1b)*: hair +0.154 [+0.125, +0.184], thyroid +0.238 [+0.207, +0.269], ovary +0.163
  [+0.087, +0.237] (unchanged from unmatched; max |SMD| after matching ≤ 0.11); capsule infeasible by the pre-registered
  rule (19 matched pairs in one label; exploratory +0.230 [+0.125, +0.337]).

**Point 2 — the harm claim.** Title kept (it states the gap); abstract rewritten: the universal claim is the gap,
harm is conditional. New real-artifact evidence of harm: with transplanted real hair in the lesion masking lowers
reversed AUROC by −0.075 [−0.095, −0.055], with real debris on the lesion by −0.175 [−0.210, −0.141] (masking improves
clean AUROC there: harm from the artifact becoming a larger share of what remains — theory text refined accordingly).
Thyroid sensitivity made airtight (R3): on the patient-disjoint official split masking lowers sensitivity at **5/5**
validation-fixed operating points (e.g. 0.699 → 0.424, −0.275 [−0.399, −0.182] at maximal balanced accuracy); at
validation specificity 0.80 sensitivity for malignant nodules carrying an in-ROI caliper falls by −0.372
[−0.436, −0.291]. The pre-registered decision rule allows the sentence "masking lowers sensitivity". (The rebuilt run
gives 0.699 → 0.424 where the archived run gave 0.734 → 0.347: threshold-based metrics move with small numeric
differences; the paper reports the rebuilt values with CIs.)

**Point 3 — remedies on unaltered data (correct).** U-MtE is demoted: the paper is one story (law → mechanism →
theory → consequences on unaltered data), and the remedies are a compact decision guide ("what carries the shortcut
decides the fix") that states explicitly that no remedy transfers uniformly to unaltered data. U-MtE, its ablations
and all remedy results moved to the supplement (S8).

**Point 4 — external validation with patient IDs.** Not done. The official TNCD test split used for the unaltered
thyroid analysis *is* patient-disjoint (Gong et al., MICCAI 2022; now stated). Searched: TN-SCUI 2020 (one image per
patient, masks) forbids use outside the challenge; ThyUS2Path (842 patients, pathology, CC BY 4.0) has real calipers
but no nodule masks, and the frozen caliper detector responds to its scanner interface — feasible only with a
transferred nodule segmentation and a re-validated detector (future work, documented in the supplement S3).

**Point 5 — absolute performance.** Thyroid (only cohort with a published benchmark on the same split): frozen
DINOv2 + linear head 0.735 AUROC (rebuilt; archived 0.736) vs 0.773–0.784 for fine-tuned CNNs with cross-entropy and
0.799 with curriculum learning (Gong et al. 2022) — stated as a limitation. Fine-tuned crossovers: capsule +0.584
[+0.469, +0.691], thyroid ViT-S +0.168 [+0.074, +0.265]; thyroid ResNet-50 and ovary were not significant with 5
clusters. New: a fine-tuned ResNet-50 on the hair traps reproduces the crossover (+0.143 [+0.062, +0.219]); the ovary
re-run with 15 clusters (amendment 1) does not (+0.058 [−0.037, +0.157]) — the fine-tuned network barely learns the ovary
task (clean AUROC 0.596 / 0.500), so this null is uninformative. The paper now says "fine-tuned networks in three of
four cohorts" (results/review2/R4_finetune.csv). MedSigLIP zero-shot numbers are used only to show the effect is in the representation.

**Point 6 — expert validation.** Not done (needs clinicians). The existing audits were by the analysis team, now
stated as such. A blinded two-rater audit kit (240 items, 60 per cohort) and a scoring script (Cohen's κ, precision
and recall of the automatic labels vs rater consensus) are ready: `scripts/audit/make_expert_audit_kit.py`,
`scripts/audit/score_expert_audit.py`. A clinical co-author is recommended.

**Point 7 — public pre-registration.** Registry of every registration's commit vs its first outcome commit
(`results/review2/registry.csv`, supplement S1): all registrations precede their outcome results (SLAS registration
and its localisation result share a commit; the primary family was grouped post hoc — both stated). Commit times are
client-set; the second-round registrations were pushed to GitHub before running. Depositing the repository snapshot
on OSF/Zenodo (DOI) is a step for the author.

**Framing.** Theory leads with held-out reversed AUROC (MAE 0.039 vs 0.108 for the symmetry heuristic) and sweep signs
(42/45 vs 36/45); r = 0.997 is explicitly not used as evidence (heuristic 0.989). DermLIP's missing hair gap is
explained with data (masking costs DermLIP clean AUROC −0.060/−0.081 at both locations; oracle hair removal only
+0.044/+0.045) and anticipated by the theory (−0.003). CLAIM 2024 checklist in the supplement (S12), with the items the
study does not meet marked (18 inter-rater variability, 21 sample size, 33 external data partial, 44 funding).
