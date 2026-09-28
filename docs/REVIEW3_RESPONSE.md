# Third review round (2026-09-27) — verification and responses

Seven comments on the second-review results, checked against the result files first. New analyses are registered in
`docs/PREREGISTRATION_REVIEW3.md` (pushed before computing).

## Verification
| # | Comment | Verdict | Evidence |
|---|---|---|---|
| 1 | The transplant needs a paste-edge control | **Valid** | No control for pasting seams exists; R5 registered. |
| 2 | Transplanted hair is pixel-carried; natural hair is correlate-carried | **Correct** | Oracle hair-pixel removal gains +0.015 on the real hair trap; the transplant constructs a pixel-carried shortcut by design. |
| 3 | Capsule's real-trap crossover is not supported by matching | **Correct** | Matching left 19 polyp-like pairs (< 30, infeasible); lesion-box area SMD 2.57. Capsule's location evidence rests on the transplant. |
| 4 | Report the matched analysis fully (balance < 0.1, before/after, images kept) | **Partly met before** | Pooled max \|SMD\| after matching 0.108 / 0.045 / 0.079 (hair / thyroid / ovary), but within label up to 0.164 / 0.150 / 0.140; images kept 1,829 / 316 / 179 per trap of 5,438 / 1,150 / 617 in-ROI. R7 (stricter calliper) registered. |
| 5 | The ovary fine-tuned model is uninformative, not a failed replication | **Correct** | ERM clean AUROC 0.596 (Trap A) and 0.500 (Trap B) with 15 clusters. |
| 6 | Report n, CI and registration status of the −0.372 subgroup | **Valid; and the CI was too narrow** | n = 78 malignant nodules with an in-ROI caliper (official test split, 614 images; the same 78 for every seed). Subgroup defined post hoc; S3 registered before computing. The CI came from a bootstrap that resamples the shared test images independently per seed — too narrow (R6). |
| 7 | Label new analyses by registration status; keep them out of the 16-test family | **Valid** | They are separate Holm families already but the paper did not say so; status now stated. |

## Additional issue found while checking (not in the review)
The hierarchical bootstrap resamples test images independently within each seed cluster even when clusters share the
same test images; this understates the test-set sampling variance. R6 re-estimates the headline CIs with a crossed
(seed × image) bootstrap.

## Responses (results/review3/*.csv; tables paper/tables/review3_*.tex and the regenerated review2_* tables)
1. **Paste-edge control (R5).** Neutral tissue pasted through the same masks, positions and blending gives location
   interactions of hair +0.204 [+0.164, +0.244], thyroid +0.145 [+0.130, +0.161], ovary +0.052 [+0.021, +0.084],
   capsule +0.459 [+0.383, +0.535] — 52–92 % of the artifact transplant's. The artifact adds significantly beyond the
   paste in hair (+0.049 [+0.026, +0.072]) and thyroid (+0.132 [+0.116, +0.148]), not in ovary or capsule. By the
   pre-registered rule the transplant signal is **not** attributable to the artifact alone in any cohort: seams / the
   pasted patch are part of the signal. The paper now says what the transplant does show — on fixed images, *where* a
   label-correlated pattern sits decides what masking removes (the image-population explanation is excluded) — and what
   it does not: anything about a particular artifact's appearance, or the in-ROI harm of real artifacts in context.
2. **Hair.** Stated explicitly: the transplant shows the location effect for pixel-carried hair; natural hair's
   shortcut is mostly carried by correlates (oracle removal +0.015), so +0.253 is not a claim about natural hair.
3. **Capsule.** Its real-trap crossover is labelled descriptive (table footnote and text); its location evidence rests on
   the transplant (which, per point 1, shows location dependence of a pasted pattern).
4. **Matched analysis, full.** Supplement S11 table: images kept (hair 1,829 of 5,438 in-ROI / 8,329 out; thyroid 316
   of 1,150 / 326; ovary 179 of 617 / 194), pooled max |SMD| before → after (2.13 → 0.11, 0.53 → 0.04, 0.39 → 0.08),
   within label up to 0.16 / 0.15 / 0.14. Stricter calliper (R7, 0.05 SD) does not improve within-label balance (up to
   0.18 / 0.19 / 0.18) and shrinks the crossover by 16 / 9 / 11 % (+0.124, +0.218, +0.152), all CIs excluding zero.
   Stated as a shrinking-but-surviving effect with residual imbalance.
5. **Ovary fine-tuned model.** Now "not evaluable" (ERM clean AUROC 0.60 / 0.50): the fine-tuned replication holds in
   all three evaluable cohorts (hair ResNet-50 +0.143 [+0.056, +0.231], capsule, thyroid ViT-S).
6. **Subgroup.** n = 78 malignant nodules with an in-ROI caliper (of 236 malignant; 614-image official test split).
   Crossed CI: −0.372 [−0.499, −0.236] (was [−0.436, −0.291]). Defined post hoc; the operating-point test was registered
   before computing. All 15 thyroid operating-point contrasts still exclude zero with the crossed bootstrap.
7. **Registration status.** Main-text paragraph and supplement table (S1): transplant, matching, operating points and the
   paste-edge control were designed after earlier results and registered before their own; each is its own Holm family,
   outside the unchanged 16-test primary family.

**R6 crossed bootstrap.** Headline CIs are 1.1–2.2× wider (natural analyses ≈ 2×). Every headline claim still excludes
zero. Now including zero: balancing's transplant interaction for ovary and capsule, MSK all-pairs mask − ERM, and the
ovary out-of-ROI transplant mask − ERM is borderline (+0.064 [+0.001, +0.126]). Controlled sweeps and supplement
mitigation analyses keep per-seed CIs (predictions not regenerated) — stated as a limitation.

**Also answered:** which labels are automatic — supplement S4 provenance table (e.g. capsule debris masks: probe for
5,459 of 5,481 frames, expert for 22; non-HAM ISIC lesion masks: U-Net for 15,316 images; all caliper masks: rule-based
detector; all audits by the analysis team including an AI agent).
