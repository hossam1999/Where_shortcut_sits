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

## Responses
(filled in when the registered analyses are computed)
