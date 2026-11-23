# Final submission check: BMC Medical Imaging (Research article)

Manuscript: `manuscript_bmc/manuscript.tex` → `manuscript_bmc/manuscript.pdf` (20 pages, double-spaced, line-numbered).
Branch: `claude/festive-bohr-95evo5`. Editorial pass: 2026-11-22.

## 1. Summary of changes

| Area | Change |
|---|---|
| Authors | Three authors in the requested order: Hossam Elkady (corresponding, es-Hossam.Elkady2022@alexu.edu.eg), Nagia Ghanem (nagia.ghanem@alexu.edu.eg) and Samia Hafez (s.hafez@alexu.edu.eg). Shared affiliation: Department of Computer and Systems Engineering, Faculty of Engineering, Alexandria University, Alexandria, Egypt. No titles, ORCID iDs or contributions were added. A red confirmation note covers the official name spelling and the affiliation. |
| Language | Wording made consistent throughout: ROI, trap, crossover, shortcut-conflicting, "reproduced / preliminary / not reproduced / descriptive". Hedged claims kept hedged. "Beat" replaced by "outperformed" where a registered rule applies. |
| Central finding | Stated in the Results and in the Discussion: masking removes out-of-ROI shortcuts but leaves in-ROI shortcuts largely intact. Masking is not always harmful: in-ROI reversed AUROC still rose in thyroid, ovary and capsule (0.29→0.40, 0.41→0.55, 0.58→0.67) and fell for hair (0.51→0.46). |
| Thyroid threshold analysis | The validation-fixed result (0.744→0.223) and the oracle test-set specificity match (0.544→0.390, Holm p = 0.019) are reported separately. The oracle analysis is labelled "not a deployable procedure" in Methods, Results and Table 2. Overall sensitivity at matched specificity (0.557 vs 0.540) is now reported, so the subgroup deficit is not mistaken for a global one. |
| Constraint | Now stated in the Abstract, Results and Discussion: the constraint did not dominate masking (40/42 components met). Its two failures are named: ISIC 2020 −0.021 [−0.027, −0.016], and inconclusive non-inferiority on the thyroid split. |
| Neutral-paste control | The Results now say the transplant does not separate the artifact from the paste edge in ovary and capsule. There, the location evidence rests on the real and matched traps. This is also limitation 5. |
| Patient-level clustering | Limitation 4 now says that TN3K has no patient IDs, so the intervals for the unaltered thyroid analysis treat images as independent. The grouped bootstrap covered the trap crossovers only. |
| Ovarian detector | The Methods give the audit result (31/40 calipers; other hits were overlays) and redefine "artifact-free" as caliper-free. The final limitation repeats this. |
| EPVT / tolerance | Methods: the initial registration used 0.03, widened to 0.05 for all papers after the first reproduction checks and before EPVT's head-to-head comparison. Results: the bias-1 gap is 0.0497, so EPVT passes 0.05 by 0.0003. That margin is smaller than the ≈0.005 figure-reading uncertainty, and the gap would fail 0.03. Bias 0.9 (gap 0.017) passes both tolerances. Table 4 and limitation 7 repeat this. Added (2026-11-23): under 0.03 the EPVT comparison would be descriptive, with the same differences in favour of the constraint, so the conclusion does not depend on the tolerance. |
| Audit categories | Reproduced, preliminary, not reproduced and descriptive are defined in Methods and used consistently in the Abstract, Results and Table 4. |
| Viviano | The original-method reproduction (1 seed, preliminary, no verdict) is now separate from the three-seed location experiment (masked model 0.615 inside vs 0.896 outside; constraint 0.856–0.880 per seed; gain over their standard model not significant). |
| Claims of superiority | EPVT: outperformed under the registered rule, with the reproduction caveat, and 0.02–0.04 AUROC lower on clean images. RR-ClArC 2024: not outperformed. Saab et al.: tie. DISC, RR-ClArC 2025 and ISNet: descriptive only. |
| Declarations | Repository URL inserted. Ethics paragraph lists every dataset and licence. Every unverifiable item is marked `[AUTHOR CONFIRMATION REQUIRED: …]`. |
| Layout | Table 1 scaled to the text width. Table 2 threshold labels shortened. No overfull boxes remain. |

## 2. Experimental claim verification (every number in the manuscript against the repository)

| Claim | Manuscript | Source | Status |
|---|---|---|---|
| Location crossovers (real, matched, transplant, transplant−neutral) | Table 1 | `paper/tables/review2_law.tex` | ✓ |
| Neutral-paste crossovers +0.204 / +0.145 / +0.052 / +0.459 | Results 2 | `paper/tables/review2_law.tex` | ✓ |
| 16/16 alternative masking implementations, Holm p ≤ 0.010 | Results 1 | `paper/sections/results_*` (review-2 sensitivity) | ✓ |
| Grouped bootstrap: width ratio 0.97–1.16, capsule 1.50 (129 blocks), Holm p = 0.0008 | Results 1 | `results/grouped_bootstrap/SUMMARY.md` | ✓ |
| In-ROI masking 0.29→0.40, 0.41→0.55, 0.58→0.67, hair 0.51→0.46 | Results 1 | `paper/tables/main_inroi.tex` | ✓ |
| Prevalences 37.6%/24.1% and 31%/20% | Results 3 | `paper/sections/results_natural.tex` | ✓ |
| Thyroid overall −0.004 [−0.048, +0.041]; conflicting −0.095 / −0.102 | Results 3 | `paper/tables/main_natural.tex` | ✓ |
| 0.744→0.223, −0.521 [−0.642, −0.382]; frozen 0.549→0.177, −0.372 | Table 2 | `paper/tables/main_natural.tex` | ✓ |
| Oracle 0.544→0.390, −0.154 [−0.302, −0.023], Holm 0.019; frozen 0.505→0.331, −0.174; overall 0.557 vs 0.540; spec 0.806/0.804 | Table 2, Results 3 | `results/round7/SUMMARY.md` | ✓ |
| ISIC 2020 counts: 32,997 / 7,657 / 25,340 / 2,044 / 483 | Methods | `results/round4/natural_isic2020/counts.json` | ✓ |
| ISIC 2020 overall +0.016; conflicting −0.026 | Results 3 | `paper/tables/main_natural.tex` | ✓ |
| Held-out ISIC 2019 sources: 2 negative, 1 positive (BCN −0.025, HAM +0.050, MSK −0.065) | Results 3 | `paper/tables/main_natural.tex` | ✓ |
| Capsule masking +0.077 overall | Results 3 | `paper/tables/main_natural.tex` | ✓ |
| Constraint +0.118 to +0.330 (0.400→0.730); 40/42; −0.002; −0.021 | Results 4 | `paper/sections/guide.tex` | ✓ |
| Detector audits: thyroid 38/40; ovary 40/40 overlay, 31/40 caliper | Methods | `docs/ARTIFACT_MASK_AUDIT.md` | ✓ |
| Thyroid traps 1,150 / 326 / 1,801; fine-tuned AUROC 0.769 vs 0.773 | Methods | `paper/sections/data.tex`, round 6/7 summaries | ✓ |
| Phase 5 (Table 3) | Table 3 | `docs/REPLICATION_RESULTS.md` (phase 5) | ✓ |
| EPVT 0.696 vs 0.746 (gap 0.0497), 0.775 vs 0.792; +0.086, +0.031 | Results 5 | `docs/REPLICATION_RESULTS.md` (P6) | ✓ |
| Viviano 0.615 / 0.896; constraint 0.856–0.880, means 0.867/0.876; CI to −0.006 | Results 5 | `docs/REPLICATION_RESULTS.md` (P1 location, 3 seeds) | ✓ |
| DISC 0.736 vs 0.489; RR-ClArC 2025 0.785 vs 0.524; Saab +0.023 [−0.029, +0.076] | Results 5 | `docs/REPLICATION_RESULTS.md` (P8, P2, P4) | ✓ |
| ISNet 0.84/0.83/0.83 → 0.998/0.167/0.000 | Results 5 | `docs/REPLICATION_RESULTS.md` (P3) | ✓ |
| RR-ClArC 2024: within 0.03 on ResNet-18/EfficientNet-B0, within 0.05 on VGG-16; not outperformed | Results 5 | `docs/REPLICATION_RESULTS.md` (P7) | ✓ |

No experiments were run in this pass, and none were needed: every claim could be traced to an existing result file.

## 3. BMC formatting compliance

**Caveat:** the official instructions at bmcmedimaging.biomedcentral.com could not be fetched from this environment (host blocked by the network policy), and web searches did not return the current page text. The checks below follow BMC's standard Research article requirements as generally published. **Re-check them against the live "Submission guidelines → Preparing your manuscript → Research article" page before submission.**

| Requirement (standard BMC Research article) | Status |
|---|---|
| Title page: title, all authors, affiliations, corresponding author's email | ✓ |
| Structured abstract (Background, Methods, Results, Conclusions), ≤350 words | ✓ 349 words |
| 3–10 keywords | ✓ 9 |
| Sections: Background, Methods, Results, Discussion, Conclusions | ✓ |
| List of abbreviations | ✓ |
| Declarations: ethics and consent, consent for publication, availability of data and materials, competing interests, funding, authors' contributions, acknowledgements | ✓ headings present; four need author input (section 4) |
| Clinical trial number ("Not applicable") | ✓ (included per recent BMC templates) |
| Vancouver-style numbered references | ✓ `vancouver.bst`, 0 unresolved |
| Line numbers and double spacing | ✓ |
| Figures uploaded as separate files at submission | ⚠ embedded for review. Upload `paper/figures/design.pdf` and `examples.pdf` separately (BMC accepts PDF/EPS/TIFF/PNG) |
| Tables within the text, numbered and captioned | ✓ 4 tables |
| Supplementary files | None cited. The repository serves as the data and materials source. |
| Word limit | BMC Medical Imaging sets no strict limit for Research articles. The main text is ≈3,300 words excluding tables. |
| LaTeX submission | BMC accepts LaTeX via its template or a PDF. The `article` class is acceptable for an initial submission. Converting to the Springer Nature LaTeX template may be requested at revision. |

## 4. Remaining author confirmations (red `AUTHOR CONFIRMATION REQUIRED` in the PDF)

1. **Names and affiliation.** Confirm the official spelling of each name and the affiliation of all three authors. Neither could be verified online: the alexu.edu.eg staff pages are blocked here, and search results did not list Dr Ghanem or Dr Hafez.
2. **Ethics.** Obtain a determination from the Alexandria University ethics committee that a secondary analysis of public, de-identified data needs no further approval, with a reference number and date. Otherwise, state the committee's actual position. No exemption has been claimed.
3. **Competing interests.** Each author confirms.
4. **Funding.** Give the sources, or state that there are none.
5. **Authors' contributions.** CRediT roles for H.E., N.G. and S.H., agreed by all three authors.
6. **Co-author approval.** "All authors read and approved the final manuscript." Do not submit until both co-authors have actually approved.
7. **Acknowledgements.**
8. **Repository URL.** The authors plan a new public repository holding only code and pre-registrations (export: `python scripts/release/make_code_release.py <dir>`). Replace the URL in *Availability of data and materials* once it exists; the pre-registrations, including amendment A38, must be in it because the manuscript's "registered before results" statements rest on them.

## 5. Remaining scientific weaknesses (for reviewers)

1. **EPVT reproduction is borderline.** It passes only the widened 0.05 tolerance, by 0.0003, which is below the figure-reading error. The widening came after the first reproduction checks. The EPVT comparison is real but fragile, and reviewers may treat it as descriptive.
2. **The audit is thin.** Only two of eight papers reproduced fully. Most head-to-heads are descriptive, and the only clean confirmatory win on in-ROI artifacts is the thyroid phase 5 comparison (three folds; the ISNet comparison's CI starts at +0.019).
3. **No patient-level clustering for TN3K.** The intervals for the unaltered thyroid analysis may be too narrow. The 78-nodule subgroup was defined post hoc for the frozen model, though it was registered for the fine-tuned model.
4. **The neutral-paste control explains much of the transplant effect.** In ovary and capsule the transplant does not separate the artifact from the paste edge.
5. **The 90/10 trap association is stronger than in clinical data.** Unaltered-data evidence covers three cohorts, and the ISIC 2020 conflicting-pair effect is marginal (CI upper bound −0.000).
6. **Artifact labels come from automatic detectors.** They were audited by a non-clinician, and the ovarian detector captures non-caliper overlays (9/40 audited).
7. **The constraint has limits.** It needs artifact labels, costs 0.02–0.04 clean AUROC, and fails on ISIC 2020 and on class-specific marks.

## 6. Compilation results

- Build: `pdflatex → bibtex → pdflatex → pdflatex` in `manuscript_bmc/`.
- LaTeX errors: **0**. Undefined citations or references: **0**. Overfull boxes: **0** (after the table fixes).
- Figures: both found (`../paper/figures/design.pdf`, `examples.pdf`).
- Pages: 20.
- BibTeX warnings: only "empty pages", "missing publisher" and "empty author and editor" on `@misc` entries (arXiv preprints, Kaggle dataset). Authors still print in the bibliography, so these are cosmetic.
- The rendered title page and the Table 1–4 pages were checked visually.
- Build files (`.aux`, `.bbl`, `.blg`, `.log`, `.out`) are ignored by `.gitignore` and not committed.

## 7. Scores

**Scientific writing: 8/10.**
- The argument is clear and consistent: a location law, then clinical consequences, then a remedy with its failures, then an external audit.
- Claims are hedged to match the evidence, and the oracle and descriptive analyses are labelled.
- Every number traces to a repository file.
- Deductions:
  - The text is dense. Results 5 packs four kinds of comparison into one subsection.
  - Some Methods sentences carry several qualifiers.
  - The audit could be moved to a supplement for readability.

**Submission readiness: 6/10.** This score is split into two kinds of problem.

*Scientific (moderate; acceptable for submission with the stated limitations):*
- the borderline EPVT reproduction;
- no patient-level clustering in TN3K;
- the neutral-paste confound in two cohorts;
- detector-derived labels.

None of these blocks submission, but expect reviewer questions on EPVT and on clustering.

*Administrative (blocking; submission is not possible until they are resolved):*
- the ethics determination;
- confirmation of funding, competing interests, contributions and acknowledgements;
- both co-authors' approval of the final manuscript;
- verification of the author names and affiliation;
- a check against the live BMC guidelines;
- separate figure files at upload.

With these resolved, readiness would rise to about 8/10.
