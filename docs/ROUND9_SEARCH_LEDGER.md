# Round 9 — exploratory search ledger (a SEARCH on development data, not a test)

## MORNING SUMMARY (2026-09-29, 01:15; plain language)
**Round 8 (registered, final): no remedy dominates masking.** The fixed sequence stopped at its first candidate,
mask_cmc: it met 40 of the 42 components. It was better than masking wherever masking fails (every Trap A, every hard-pair
cell, including ISIC 2020) and kept masking's behaviour outside the ROI. It lost on one cell, **ISIC 2019 → 2020 all pairs
(−0.021 [−0.027, −0.016])**, and thyroid all pairs was inconclusive (−0.002 [−0.011, +0.007]). mask_bal met every D2
component but lost on thyroid, MSK and ISIC 2020 all pairs. By the registered rule the paper says: *wins where masking fails but costs
elsewhere* (mask_cmc, mask_bal). The sweeps claim D5 was supported for mask_cmc and mask_bal; the fine-tuned paste
remedy (D6) was not. Every cohort's pasted-vs-real probe exceeded 0.75 (0.90–0.99), and the paste plans could not balance the
traps (too few training-set donors): the paste results should not be read as evidence about real artifacts.
(`results/round8/SUMMARY.md`, commit 0d66e42.)

**Round-9 search (development data only; 12 candidates, the limit; about 50 minutes of CPU, no GPU).** No candidate met
all 29 development proxy components. Three eligible candidates met 28 or 27 with **no loss**: V-REx on masked features
(28/29), IRMv1 (28/29) and conditional adversarial debiasing (27/29). Product of experts and group logit adjustment
also reached 28/29 but **flip the shortcut** in thyroid Trap A (D4), so they are not eligible (rule 4). The honest caveat:
round 8's mask_bal also scored 28/29 on this proxy and still lost on ISIC 2020, and the proxy cannot see ISIC 2020.
The one pattern worth testing: **conditional adversarial debiasing is the only arm whose all-pairs contrast with masking
is ≥ −0.001 in every development cohort (positive on all three ISIC sources)**, i.e. the cell where round 8 failed; it
pays with smaller Trap A gains (+0.09 to +0.17 instead of +0.18 to +0.38) and more seed variance.

**Needs your approval**: (1) whether round 9 should be run at all, given that the proxy cannot distinguish the
search's best from round 8's mask_bal; (2) if yes, `docs/PREREGISTRATION_ROUND9_DRAFT.md` (3 candidates, fixed
sequence mask_condadv → mask_irm → mask_vrex, reserved seeds 9101–9505, ovary held out). No round-9 confirmation was run.

## 0. Start (2026-09-29 00:15) and firewall
- Round 8 is final (commit 0d66e42, `results/round8/SUMMARY.md`): **no primary candidate dominates masking**. Decision
  by the registered rule: *wins where masking fails but costs elsewhere*: mask_cmc (40 of 42 components; loss: ISIC
  2019 → 2020 all pairs −0.021 [−0.027, −0.016]; thyroid all pairs inconclusive) and mask_bal (losses: all pairs in
  thyroid, MSK and ISIC 2020). Round 8 is not changed again.
- **Reserved round-9 confirmation seeds: 9101, 9202, 9303, 9404, 9505.** No data is generated with them during the search.
- **Allowed**: development seeds 42, 123, 456, 789, 2026, with their trap test environments (reversed, correlated,
  clean) for thyroid, capsule and ISIC hair; the validation folds of the natural training sets (thyroid, capsule,
  ISIC BCN / HAM / MSK hold-out designs). **Forbidden**: seeds 8101–8505 and 9101–9505; every natural test set
  (thyroid official test split, capsule test split, the BCN / HAM / MSK hold-out sources); ISIC 2020; ovary.
  `scripts/round9_search/firewall.py` raises on any forbidden request; unit-tested (`tests/test_round9_search.py`).
  The search never builds a natural test split: thyroid uses its train/validation part only, ISIC the two training
  sources, capsule drops each seed's test groups before anything else.
- **Blind spot, stated in advance**: mask_cmc's one loss is on ISIC 2020, which the search may not touch. The closest
  allowed proxy is the ISIC 2019 validation folds (D1'). A candidate can look perfect here and still fail ISIC 2020.
- **Development proxy criterion** (ranking only): D1' all pairs on the evaluation half of each natural validation fold,
  NI 0.01 (thyroid, capsule, BCN, HAM, MSK); D2' Trap A min(rev, corr) SUP (thyroid, capsule, ISIC) and hard pairs on the
  evaluation half, SUP (thyroid, BCN, MSK); D3' Trap B reversed, Trap B min(rev, corr), clean (Trap A and Trap B
  models), NI 0.03 per cell (thyroid, capsule, ISIC); D4 no flipping. 29 components per candidate. Each natural
  validation fold is split by image group into a selection half (C and the round-8 rule: worst-group objective,
  same-artifact guard −0.005, masking as fallback) and an evaluation half. Crossed seed × image bootstrap, 2,000
  replicates (search only). "Worst slack" = the smallest (lower bound − threshold) over the tested components.
- **Eligibility rules of the author** (a candidate that breaks one is not eligible, whatever it scores): at test time
  only the image, its ROI mask and automatic detectors run on that image; training uses training images and labels,
  the repository's automatic artifact labels and masks, validation for selection; one recipe for every cohort;
  must keep masking's out-of-ROI behaviour and must not flip the shortcut (D3', D4); inference cost stated.
- **Budget**: at most 12 search candidates; the 1-GPU-hour cap was lifted by the author; stop at 08:15 or when two
  candidates meet every proxy component. Reference arms (masking, round-8 heads) are run for context and are not
  counted.

## 1. Candidates (sources verified; predictions written before any run)
All are heads on frozen DINOv2-B/14 @ 518 features of the masked image (mask_*) or the full image (full_*). Every
candidate uses the automatic image-level artifact labels **only in training**; at test time it scores the (masked)
image with one linear head or one small MLP + linear head — the same inference as masking (1 encoder pass + ROI mask),
no detector at test time. None is tuned on a test set.

| # | candidate | idea and source (verified) | label | needs (train / test) | theory's prediction (before running) |
|---|---|---|---|---|---|
| 1 | mask_ba | backdoor / regression adjustment: the artifact indicator A enters the head as an unpenalised covariate and is dropped at test (constant for AUROC). Pearl's backdoor adjustment; in vision CONTA, Zhang et al. 2020 (arXiv 2009.12547, "backdoor adjustment" p. 2); "eliminate proxy effects while maintaining predictive accuracy", Pope & Sydnor 2011 (doi 10.1257/pol.3.3.206; abstract verified, method details not read in full text) | adaptation (published in statistics / econometrics; not found in medical-imaging shortcut work) | A labels, ROI / image + ROI | the head learns disease from comparisons within artifact status (the A column absorbs the association), like CMC and balancing but without reweighting: Trap A large gain; Trap B ≈ masking; same-artifact pairs ≈ masking, so D1' close to masking; hard pairs gain. Graded artifact strength beyond binary A stays. |
| 2 | full_ba | the same on unmasked features | adaptation | A labels / image | keeps context (dermoscopy gains); out-of-ROI association also absorbed by A, so Trap B could hold; thyroid Trap A lower than masked variants (masking denoises) |
| 3 | mask_cmc_ba | round-8 mask_cmc projection + backdoor covariate (idea f) | adaptation | A labels, ROI / image + ROI | removes both the first-moment direction and the A-explained part; ≈ best of 1 and mask_cmc |
| 4 | full_cmc_ba | the same on unmasked features | adaptation | A labels / image | as 2 with the CMC constraint |
| 5 | mask_poe | product of experts with a bias-only model on A: fixed offset log-odds(Y \| A) in training, dropped at test. Clark et al. 2019 (arXiv 1909.03683), He et al. 2019 (arXiv 1908.10763), Karimi Mahabadi et al. 2020 (arXiv 1909.06321), all NLP | adaptation (published outside medical imaging) | A labels, ROI / image + ROI | fixed marginal offset may over-correct where image features already explain part of the association: Trap A gain, D4 flipping risk, D1' cost like balancing |
| 6 | mask_la | group-conditional logit adjustment: offset τ·log-odds(Y \| A), τ ∈ {0.5, 1.5, 2} by the rule. Menon et al. 2021 (arXiv 2007.07314; class-prior version) | adaptation | as 5 | τ picks the strength; between masking and PoE |
| 7 | mask_moments | penalty on the within-class difference of the score's mean and variance between A = 1 and A = 0 (score-level CORAL, Sun & Saenko 2016, arXiv 1607.01719; second-moment extension of the conditional MMD of Veitch et al. 2021) | adaptation | as 5 | a soft CMC with a variance term: ≈ mask_cmc |
| 8 | mask_condadv (g) | conditional adversarial debiasing: MLP projection; adversary predicts A from the projection **given Y**, through gradient reversal λ ∈ {0.1, 1, 10}. Ganin et al. 2016 (arXiv 1505.07818), Zhang et al. 2018 (arXiv 1801.07593, adversary given the true label for equality of odds), Kim et al. 2019 (arXiv 1812.10352) | published (fairness / vision), adaptation here | as 5 | removes A-information within class (nonlinear CMC), but the projection can also lose disease signal and adversarial training is unstable: D1' loss risk, seed variance (reported) |
| 9 | mask_cnc (h1) | Correct-n-Contrast with the artifact × label groups given (Zhang et al. 2022, arXiv 2203.01517): positives = same label, other artifact status | published, adaptation (groups given) | as 5 | pulls same-class images together across artifact status: CMC-like; projection may cost S |
| 10 | mask_cfc (h2) | counterfactual contrastive learning (Roschewitz et al. 2024, arXiv 2403.09605, published in medical imaging with generated counterfactuals): positives = the image and its renders with the artifact inserted inside the ROI (generic overlays for every image; real pasted instances for artifact-free images) | published (medical imaging), adaptation (our renders as counterfactuals) | artifact masks of donors + ROI / image + ROI | invariance to the rendered directions: like U-MtE (calipers yes, hair no); real pastes are distinguishable from real artifacts (round-8 probe 0.90–0.99), so the gain may not transfer |
| 11 | mask_vrex | V-REx, groups = artifact × label as environments (Krueger et al. 2021, arXiv 2003.00688) | published, applied here | as 5 | balancing-like: Trap A gain, D1' cost on same-artifact pairs |
| 12 | mask_irm | IRMv1 (Arjovsky et al. 2019, arXiv 1907.02893) | published, applied here | as 5 | linear IRM "more often fails" (Rosenfeld et al. 2021, arXiv 2010.05761): ≈ masking |

Not run: **d. background randomisation** (Xiao et al. 2021, arXiv 2006.09994, Mixed-Rand). Masking already removes the
background at test, and randomising it in training would also remove the lesion context that dermoscopy needs (theory:
S_m < S); it needs new renders, and the 12-candidate limit was used for g and h added by the author.

## 2. Results (appended after every candidate; `results/round9_search/summary.csv`, `components_<candidate>.csv`)

### 1. mask_ba
- development proxy: **22/29 components met**, worst slack -0.068 (D3' Trap B min(rev,corr) thyroid -0.060); Trap A reversed seed SD 0.034; settings chosen: {'ba': 126, 'mask': 49}
- losses: D3' Trap B reversed thyroid -0.060 [-0.085]; D3' Trap B min(rev,corr) thyroid -0.060 [-0.098]; D3' clean (Trap B models) thyroid -0.066 [-0.097]; D4 no flipping thyroid trapA -0.175; D3' clean (Trap B models) capsule -0.017 [-0.030]; D3' clean (Trap B models) isic -0.051 [-0.085]
- inconclusive: D2' val hard pairs thyroid +0.001 [+0.000]
- status: does not meet every proxy component

### 2. full_ba
- development proxy: **22/29 components met**, worst slack -0.017 (D2' val hard pairs isic_BCN -0.003); Trap A reversed seed SD 0.055; settings chosen: {'mask': 118, 'ba': 57}
- losses: D2' val hard pairs isic_BCN -0.003 [-0.017]
- inconclusive: D1' val all pairs isic_MSK +0.000 [-0.018]; D2' val hard pairs thyroid +0.000 [+0.000]; D2' val hard pairs isic_MSK +0.016 [-0.006]; D3' clean (Trap A models) thyroid -0.013 [-0.031]; D2' Trap A min(rev,corr) capsule +0.001 [-0.003]; D3' clean (Trap B models) isic +0.002 [-0.030]
- status: does not meet every proxy component

### 3. mask_cmc_ba
- development proxy: **23/29 components met**, worst slack -0.036 (D3' clean (Trap B models) thyroid -0.034); Trap A reversed seed SD 0.021; settings chosen: {'per_class': 107, 'pooled': 40, 'mask': 28}
- losses: D3' Trap B reversed thyroid -0.021 [-0.049]; D3' Trap B min(rev,corr) thyroid -0.021 [-0.054]; D3' clean (Trap B models) thyroid -0.035 [-0.066]; D3' clean (Trap B models) isic -0.017 [-0.039]
- inconclusive: D1' val all pairs thyroid -0.003 [-0.014]; D1' val all pairs isic_MSK -0.005 [-0.011]
- status: does not meet every proxy component

### 4. full_cmc_ba
- development proxy: **23/29 components met**, worst slack -0.013 (D1' val all pairs isic_MSK -0.006); Trap A reversed seed SD 0.012; settings chosen: {'mask': 125, 'pooled': 26, 'per_class': 24}
- losses: D1' val all pairs isic_MSK -0.006 [-0.023]
- inconclusive: D1' val all pairs isic_BCN -0.004 [-0.017]; D2' val hard pairs thyroid +0.000 [+0.000]; D2' val hard pairs isic_BCN +0.012 [-0.008]; D2' Trap A min(rev,corr) thyroid +0.015 [+0.000]; D2' Trap A min(rev,corr) capsule +0.003 [-0.001]
- status: does not meet every proxy component

### 5. mask_poe
- development proxy: **28/29 components met**, worst slack +0.002 (D1' val all pairs thyroid -0.001); Trap A reversed seed SD 0.035; settings chosen: {'poe': 133, 'mask': 42}
- losses: D4 no flipping thyroid trapA -0.052
- inconclusive: none
- status: does not meet every proxy component

### 6. mask_la
- development proxy: **28/29 components met**, worst slack +0.000 (D1' val all pairs isic_MSK -0.005); Trap A reversed seed SD 0.048; settings chosen: {'tau0.5': 92, 'tau1.5': 48, 'mask': 22, 'tau2': 13}
- losses: D4 no flipping thyroid trapA -0.041
- inconclusive: none
- status: does not meet every proxy component

### 7. mask_moments
- development proxy: **26/29 components met**, worst slack -0.002 (D1' val all pairs isic_MSK -0.005); Trap A reversed seed SD 0.023; settings chosen: {'gamma100': 76, 'gamma10': 46, 'mask': 28, 'gamma1': 25}
- losses: D1' val all pairs isic_MSK -0.005 [-0.012]
- inconclusive: D2' val hard pairs thyroid +0.013 [+0.000]; D2' val hard pairs isic_BCN +0.006 [+0.000]
- status: does not meet every proxy component

### 8. mask_condadv
- development proxy: **27/29 components met**, worst slack -0.000 (D2' val hard pairs isic_MSK +0.013); Trap A reversed seed SD 0.026; settings chosen: {'lam1': 60, 'mask': 56, 'lam0.1': 42, 'lam10': 17}
- losses: none
- inconclusive: D2' val hard pairs thyroid +0.020 [+0.000]; D2' val hard pairs isic_MSK +0.013 [-0.000]
- status: does not meet every proxy component

### 9. mask_cnc
- development proxy: **20/29 components met**, worst slack -0.010 (D1' val all pairs capsule -0.007); Trap A reversed seed SD 0.037; settings chosen: {'mask': 95, 'w0.5': 34, 'w2': 29, 'w1': 17}
- losses: D1' val all pairs capsule -0.007 [-0.020]; D1' val all pairs isic_MSK -0.009 [-0.019]; D3' clean (Trap B models) capsule -0.015 [-0.034]
- inconclusive: D1' val all pairs isic_BCN -0.001 [-0.014]; D1' val all pairs isic_HAM -0.004 [-0.015]; D2' val hard pairs thyroid +0.000 [+0.000]; D2' val hard pairs isic_MSK +0.012 [-0.002]; D3' Trap B reversed capsule -0.014 [-0.039]; D3' Trap B min(rev,corr) capsule -0.014 [-0.039]
- status: does not meet every proxy component

### 10. mask_cfc
- development proxy: **19/29 components met**, worst slack -0.007 (D3' Trap B reversed capsule -0.017); Trap A reversed seed SD 0.015; settings chosen: {'mask': 75, 'w2': 45, 'w0.5': 33, 'w1': 22}
- losses: D2' val hard pairs isic_MSK -0.000 [-0.005]; D3' Trap B reversed capsule -0.017 [-0.037]; D3' Trap B min(rev,corr) capsule -0.017 [-0.037]; D3' clean (Trap B models) capsule -0.017 [-0.035]
- inconclusive: D1' val all pairs isic_HAM -0.001 [-0.011]; D2' val hard pairs thyroid +0.047 [-0.000]; D2' val hard pairs isic_BCN +0.000 [+0.000]; D3' Trap B reversed thyroid -0.010 [-0.032]; D3' Trap B min(rev,corr) thyroid -0.010 [-0.030]; D3' clean (Trap B models) thyroid -0.009 [-0.031]
- status: does not meet every proxy component

### 11. mask_vrex
- development proxy: **28/29 components met**, worst slack -0.001 (D1' val all pairs isic_HAM -0.005); Trap A reversed seed SD 0.018; settings chosen: {'beta100': 122, 'beta10': 20, 'beta1': 18, 'mask': 15}
- losses: none
- inconclusive: D1' val all pairs isic_HAM -0.005 [-0.011]
- status: does not meet every proxy component

### 12. mask_irm
- development proxy: **28/29 components met**, worst slack -0.000 (D2' val hard pairs isic_BCN +0.006); Trap A reversed seed SD 0.028; settings chosen: {'lam100': 71, 'mask': 59, 'lam1': 25, 'lam10': 20}
- losses: none
- inconclusive: D2' val hard pairs isic_BCN +0.006 [-0.000]
- status: does not meet every proxy component


## 3. Synthesis (01:15)
**Candidates tried: 12** (the limit), plus 5 reference arms (erm, mask_balanced, and round 8's mask_cmc, full_cmc,
mask_bal on development data), which are not counted. Wall-clock about 50 minutes, CPU only (no GPU used). No crash.
No candidate met every proxy component, so the "two candidates" stopping rule did not trigger; the search ended at the
12-candidate limit.

| # | candidate | met / 29 | losses | eligible | kept for the draft? | why |
|---|---|---|---|---|---|---|
| ref | mask_bal (round 8) | 28 | 0 | yes | — | reference: 28/29 here, yet lost on ISIC 2020 in round-8 confirmation |
| ref | mask_cmc (round 8) | 27 | 1 | yes | — | reference |
| 1 | mask_ba | 22 | 6 | no (D4 flip, thyroid Trap A) | dropped | the unpenalised A term over-absorbs: flips thyroid Trap A, loses Trap B / clean |
| 2 | full_ba | 22 | 1 | yes | dropped | loses BCN hard pairs; falls back to masking in 67 % of training sets |
| 3 | mask_cmc_ba | 23 | 4 | yes | dropped | Trap B and clean losses in thyroid |
| 4 | full_cmc_ba | 23 | 1 | yes | dropped | MSK all-pairs loss |
| 5 | mask_poe | 28 | 1 (D4) | **no** (flips thyroid Trap A, −0.052) | dropped | rule 4 |
| 6 | mask_la | 28 | 1 (D4) | **no** (flips thyroid Trap A, −0.041) | dropped | rule 4 |
| 7 | mask_moments | 26 | 1 | yes | dropped | MSK all-pairs loss |
| 8 | mask_condadv | 27 | 0 | yes | **kept (1st)** | only arm with all-pairs contrast ≥ −0.001 everywhere (positive on BCN, HAM, MSK) — the round-8 failure cell's proxy; smaller Trap A gains |
| 9 | mask_cnc | 20 | 3 | yes | dropped | all-pairs losses (capsule, MSK), capsule Trap B |
| 10 | mask_cfc | 19 | 4 | yes | dropped | capsule Trap B / clean losses; needs donor masks |
| 11 | mask_vrex | 28 | 0 | yes | **kept (3rd)** | behaves like round-8 mask_bal (same small all-pairs costs; HAM −0.005 inconclusive) |
| 12 | mask_irm | 28 | 0 | yes | **kept (2nd)** | all-pairs ≈ 0 everywhere; smaller Trap A gains (capsule +0.020); contrary to the prediction (≈ masking) it helps in thyroid and ISIC Trap A |

Predictions versus outcomes: backdoor adjustment — wrong (predicted ≈ CMC); the unpenalised artifact covariate
over-corrects (D4 flip in thyroid Trap A). PoE/logit adjustment: the predicted flipping risk occurred.
Moments ≈ CMC: yes. Conditional adversary: the predicted instability is moderate (seed SD 0.026 vs 0.017 for
mask_bal); the predicted D1' risk did not occur. CnC and counterfactual contrastive: the predicted disease-signal cost
occurred (D1' / D3' losses). V-REx ≈ balancing: yes. IRM ≈ masking: wrong (it moves off masking in 66 % of training
sets and gains in Trap A).

**Requirements and inference cost (rule 5, 6)** of the kept candidates: training needs the image-level automatic artifact
labels already in the repository (caliper detector, hair masks, debris probe) and the ROI masks; selection needs the
validation split with the same labels; test time needs only the image and its ROI mask — one frozen-encoder pass,
masking, then a linear head (V-REx, IRM) or a 768→256→128 MLP plus a linear head (conditional adversary): the same
encoder cost as masking, a negligible head cost. No detector runs at test time. CPU training in seconds to minutes.
