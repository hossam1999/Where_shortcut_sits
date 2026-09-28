# Pre-registration — round 6: artifact labels without a clinician (Part A) and a benchmark-level fine-tuned thyroid model (Part B)

Committed on 2026-09-28 before any analysis below. Implementation notes: `scripts/round6/README.md`. Outputs:
`results/round6/`. Changes after this commit are recorded as dated amendments at the end, with the reason. Existing
results are not re-estimated; everything here is added evidence and is reported whichever way it goes.

## Part A — validating the real-artifact labels without a clinician

**Why.** The real traps (Stage 3) use automatic artifact labels: hair masks (published, Wegley et al.), a caliper
detector (thyroid, ovary; our rule-based detector), a contamination probe (capsule; trained on expert masks), and lesion
masks (dataset experts; U-Net for non-HAM ISIC images). No clinician will audit them. Two facts make a clinician-free
validation sufficient for the claims made. (i) Artifact presence and position are visual facts, not diagnoses.
(ii) Under non-differential misclassification of trap membership the observed crossover is attenuated,
C_obs ≈ (1 − e_A − e_B) · C_true, where e_A (e_B) is the share of Trap-A (Trap-B) artifact images whose artifact is
really on the other side of the ROI boundary; random label errors therefore cannot create a positive crossover. The
checks below measure the error rates, test whether they differ by diagnosis, and re-run the traps on labels that no
independent source contradicts.

### A0 Sources, licences and gates
| source | used for | licence (as found on 2026-09-28) | rule |
|---|---|---|---|
| DermArtifactDB (Zenodo 18324962) | ISIC 2019 hair present/absent, expert-reviewed | CC BY 4.0 (labels) | match by ISIC ID (suffix `_downsampled` removed) |
| IMA++ (Zenodo 14201693) | ISIC lesion masks, 1–5 annotators per image | CC BY-NC-ND 4.0 | measurement only: its masks and anything derived at pixel level are never committed |
| Multicentre clear/contaminated capsule masks (Mendeley vmxhn95j8z, v3) | SEE-AI contamination masks, expert | CC BY 4.0 | match to our SEE-AI frames by pHash (Hamming ≤ 2, as in `prepare_capsule.py`) |
| BUSClean (github hawaii-ai/bus-cleaning) | independent caliper detector, out of the box (no adaptation, no tuning on our data) | to be read from the repository | used only if its licence permits research use; otherwise skipped and recorded |
| MedGemma 1.5 4B-it (Hugging Face, gated) | independent vision–language labeller, run locally only | model terms of use | fixed prompts, greedy decoding; no image leaves the machine |
The licence text, version and download date of every source are written to `results/round6/sources.json` before use.
A source is analysed only if it covers ≥ 200 images of the cohort; otherwise it is reported as not feasible.

### A1 Agreement of our labels with each independent source (descriptive, with 95 % bootstrap CIs)
- Hair: Cohen's κ of our hair-free / hair status (hair-free = ≤ 30 native hair pixels) with DermArtifactDB hair.
- Lesion masks: Dice of our ROI with the IMA++ majority mask; overlap r recomputed with the IMA++ mask; agreement of the
  trap class (Trap A: r ≥ 0.5; Trap B: r < 0.1; neither) among artifact-bearing images.
- Capsule: IoU of our contamination mask with the expert mask; contamination share and r recomputed; agreement of the
  cell (artifact-free < 0.03; Trap A ≥ 0.10 and r ≥ 0.5; Trap B ≥ 0.10 and r < 0.1).
- Calipers: κ for presence (ours: ≥ 15 marker pixels vs 0) with BUSClean and with MedGemma; for location, agreement of
  our class (r ≥ 0.5 vs r < 0.1) with MedGemma's answer to "is any caliper mark inside the outlined nodule?" (nodule
  outline drawn in green), and with BUSClean where it returns caliper positions.
- Every agreement is also reported separately for Y = 0 and Y = 1 (differential error, A3).
- A model labeller (MedGemma; BUSClean) counts as an informative source only if its κ is ≥ 0.40 against the images on
  which the two other caliper labels agree; otherwise it is reported as uninformative and not used in A2.

### A2 Traps on cleaned labels (primary of Part A)
For each cohort, an image is **contradicted** if an informative independent source available for it places it in a
different cell (artifact present vs absent, or the opposite side of the ROI boundary). Contradicted images are removed;
images without an independent source are kept. The four DINOv2 real traps (Stage 3 protocol: matching, seeds, folds,
ERM and mask arms, crossed bootstrap) are re-run on the cleaned cohorts, with the count gate of Stage 7 (≥ 25 per matched
cell, ≥ 40 reversed-test positives for seed 42).
- **HA2**: crossover > 0 in each cohort that passes the gate (Holm over the cohorts).
- Reported with it: the share of images verified, contradicted and unverified per cell; the original crossover.

### A3 Differential error and bias analysis
- Error rates of our cells against each informative source (and against the human rating, A4, once available) by
  trap and diagnosis. Differential error is flagged if the error rates of Y = 1 and Y = 0 differ by > 0.10 with a 95 % CI
  excluding 0; a flagged cohort's conclusion then rests on A2.
- The attenuation formula is a first-order approximation (it treats the mask gain of a trap as a mixture of the gains
  of its correctly and incorrectly placed images) and is stated as such; A2 is its empirical check.
- For each cohort: the measured error sum e_A + e_B and the corrected crossover C_obs / (1 − e_A − e_B) with the
  interval scaled the same way, reported only for cohorts without flagged differential error.

### A4 Blinded non-clinician rating of the audit sample (can run later; does not block A1–A3)
- The existing 240-image package (`audit/`; key fixed and its SHA-256 committed before any review) is rated in a local
  web tool served on the GPU machine (localhost only, viewed through an SSH tunnel), in random order, without cell
  information, using the fields of `audit/review_sheet.csv`. The rater is the author, a non-clinician; this is stated
  in the paper. Pass 2 is done ≥ 7 days after pass 1 in a new random order (intra-rater κ). A second non-clinician rater
  is added if available (inter-rater κ).
- MedGemma answers the same presence and location questions for the 240 images.
- Reported: intra- and inter-rater κ; κ of the rater with MedGemma and with our labels; error rates of our cells by
  diagnosis (feeds A3). The key is opened only after pass 2 is saved.

## Part B — a benchmark-level fine-tuned model on the patient-disjoint thyroid split

**Why.** The frozen linear probe reaches a test AUROC of about 0.74 on the official patient-disjoint TNCD split, below
published fine-tuned models (0.773–0.799, Gong et al. 2022). A reviewer can argue that the location law and the
sensitivity loss are properties of weak models.

### B1 Recipe, chosen on validation only
Natural thyroid split of Stage 5 (train on TN3K trainval with its natural caliper–label association, test on the
official test split of 614 images; group-safe 80/20 train/validation split per seed). Grid for the ERM arm, seed 42,
selected by validation AUROC only: architecture ∈ {`convnext_tiny.fb_in22k_ft_in1k`, `resnet50`,
`vit_base_patch14_dinov2.lvd142m`} × learning rate ∈ {1e-4, 3e-5} × epochs ∈ {8, 20}; 224 px, the augmentation, optimiser
and schedule of `wtss.experiments.finetune`. No test image is scored during selection. The chosen recipe is written to
`results/round6/ft_recipe.json` and committed before any final model is trained.

### B2 Final runs with the chosen recipe
- Natural split: 5 seeds × arms {ERM, mask, balanced, mask + balanced}; validation predictions saved for operating
  points (thresholds from validation only, OP1–OP5 as in Stage 5).
- Thyroid traps (Stage 3 protocol): arms {ERM, mask}; env seeds 42 and 123 × 5 folds = 10 bootstrap clusters.

### B3 Hypotheses (crossed bootstrap; Holm over FT1–FT4)
- **FT0** (descriptive): mean test AUROC of ERM; called benchmark-level if ≥ 0.773 (the lowest published cross-entropy
  baseline on the same split). FT1–FT4 are tested and reported whether or not FT0 is met.
- **FT1**: natural test set, hard-pair AUROC mask − ERM < 0 (hard pairs as in Stage 5: malignant nodules with an in-ROI
  caliper vs benign nodules without; the direction of the training association).
- **FT2**: masking lowers sensitivity (crossed CI < 0) at ≥ 3 of OP2–OP5.
- **FT3**: at OP2, sensitivity among malignant nodules with an in-ROI caliper, mask − ERM < 0. This subgroup was post hoc
  in Stage 5; for this model it is registered before any result.
- **FT4**: thyroid trap crossover (mask − ERM)_B − (mask − ERM)_A > 0.

## Amendment 1 (2026-09-28, before any analysis of this round)
Reason: the coverage of our images by each source is unknown, and some sources overlap the training data of our own
labellers. Changes:
1. **Exclude labeller training data from every agreement statistic**: ISIC 2018 Task 1 training images (training set of
   the lesion U-Net) from the IMA++ comparison; the expert-masked frames of figshare 27645021 (training set of the
   capsule probe) from the capsule comparison. Their counts are reported.
2. **Coverage report first**: A0 writes, per cohort and source, the number of matched images after exclusion, by cell
   and by Y. Sources with ≥ 200 matched images are analysed as registered; 50–199 are reported descriptively only;
   fewer than 50 are reported as not feasible.
3. **Additional independent hair masks**: the Mendeley hair masks of Kabir et al. (doi:10.17632/j5ywpd2p27.2,
   CC BY 4.0; already on the machine) are compared with our hair masks where they overlap (Dice; r recomputed with
   our lesion mask; trap-class agreement), under the same coverage rule.
4. **Larger human rating for calipers (A4b, optional)**: because no public caliper annotation exists, an additional
   stratified sample of 150 thyroid and 150 ovary images (50 per cell, 25 per diagnosis; fixed seed; blind key hashed
   and committed before rating) is added to the rating tool for presence and location of calipers only. If the author
   does not rate it, this is reported.

## Amendment 2 (2026-09-28, before any analysis of this round; after reviewing the implementation)
Reason: a code review before the first run found places where the implementation did not yet do what A0–A4 register,
and places where the registration left a rule implicit. No data of this round had been analysed. Changes:
1. **MedGemma on the caliper cohorts** (A1, as registered): every thyroid and ovary image outside the gap (1–14
   detected marker pixels) is asked the presence prompt on the plain image and the location prompt on the image with
   only the green ROI contour. Our automatic marker mask is never shown to a model labeller (this also applies to the
   240-image audit, where the location question uses a contour-only image, not the review overlay).
2. **BUSClean validity guard**: if BUSClean is positive on more than half of our caliper-free images, it detects
   on-screen annotation rather than calipers; it is then reported as invalid for calipers and not used.
3. **Only one model labeller available**: the A1 rule (κ ≥ 0.40 against the images on which the two other labels
   agree) needs two other labels. If only one model labeller is usable for a cohort, its informativeness is judged
   against the author's blinded rating (A4/A4b) with the same threshold, and A2 for that cohort is re-run after the
   rating (`scripts/round6/after_rating.sh`).
4. **Like-for-like overlap for ISIC** (A1): r is recomputed at 518 px with our hair mask and each lesion mask (ours,
   IMA++), and hair masks (ours, Kabir) are compared inside our lesion mask. A source's location class is used for a
   contradiction only where our own 518-px class agrees with our registered cell; other images are
   resolution-ambiguous and count as unverified. IMA++ Dice is also reported by lesion-mask source (HAM manual vs U-Net).
5. **Capsule frames whose own label is an expert mask** (22 frames of Stage 3) are excluded from the capsule
   comparison, because it would compare expert with expert.
6. **A3**: the differential-error flag is evaluated within each trap (Trap A, Trap B), as "by trap and diagnosis"
   registers; e_A, e_B and the corrected crossover are reported per source; the correction applies to the original
   Stage 3 crossover (the one the measured errors refer to), not to the cleaned crossover of A2.
7. **Reference crossover for ISIC**: the crossed-bootstrap estimate of the regenerated run (Stage 3), as for the
   other cohorts.

## Amendment 3 (2026-09-28, before any rating)
Reason: for a non-clinician the region is not identifiable on raw capsule frames (it is the union of expert lesion
boxes) and uncertain on ultrasound, so a location answer given on the raw image alone would measure the rater's guess
of the region rather than the artifact's position relative to the region our cells use. Change: in step 1 of the
rating tool, the raw image is shown next to the same image with only the green outline of the dataset's expert region
(lesion mask, nodule or tumour mask, lesion boxes; never our automatic artifact mask), and the page states the cohort's
artifact and region. Step 2 (mask questions) still shows the review overlay. Contour images are generated locally into
the git-ignored `audit_local/contours/`. No rating had been saved.

## Results (added after the run; the text above is unchanged)
Coverage committed at `4c3263d`, recipe at `b13bcdb`, A4b key hash at `4b32a1f`, all before the steps that use them;
results at `26ce34e`; write-up in Stage 8. The author's rating (A4, A4b) has not been done.

- **A0.** Analysed: DermArtifactDB (20,519 ISIC images), IMA++ (241 after excluding 2,503 ISIC 2018 training images),
  Kabir hair masks (485), BUSClean (thyroid 3,493; ovary 1,202), MedGemma (thyroid 3,492; ovary 1,198). Capsule masks:
  29 frames after exclusions → not feasible.
- **A1, dermoscopy.** Hair presence κ = 0.076 [0.070, 0.082]: our artifact-free cell is clean (97 % hair-free in
  DermArtifactDB), but our masks exceed 30 hair pixels on 93 % of the images DermArtifactDB calls hair-free. Position:
  IMA++ Dice 0.851 [0.830, 0.870], trap class agreement 0.841 [0.791, 0.887], opposite side 0/15 (Trap A) and 4/158
  (Trap B); Kabir hair-mask Dice 0.618 [0.600, 0.637], opposite side 0/31 and 1/233. κ with Kabir is not informative
  (our masks mark hair on all 485 images; presence agreement 97.3 %).
- **A1, thyroid.** BUSClean passes the validity guard (5.2 % positive on caliper-free images); both labellers are
  informative (κ 0.505 and 0.654 on the images where the other two labels agree). Presence κ with our detector: BUSClean
  0.483 [0.456, 0.509], MedGemma 0.663 [0.638, 0.686].
- **A1, ovary.** BUSClean invalid (positive on 90.5 % of caliper-free images). MedGemma κ with our detector 0.194
  [0.148, 0.247]; its informativeness is judged against the rating (Amendment 2), pending.
- **A2 (HA2).** Dermoscopy supported: 0.199 [0.164, 0.233] on uncontradicted labels vs 0.147 [0.111, 0.185] (Stage 3).
  Thyroid: count gate not met (the registered rule removed 61 % of Trap A and 87 % of Trap B; cleaned Trap B has 29
  benign and 14 malignant caliper images). Capsule and ovary: no image removed; the re-run reproduces Stage 3 and is not
  evidence about the labels.
- **A3.** Differential error flagged for dermoscopy (DermArtifactDB, both traps: +0.133 [+0.100, +0.168],
  +0.166 [+0.134, +0.198]) and thyroid (BUSClean, Trap A: −0.163 [−0.229, −0.099]); by the registered rule their
  conclusions rest on A2 and no attenuation correction is reported.
- **Reporting correction after the run (no estimate changed).** `bias_analysis.py` had reported the correction per
  source even when another source of the same cohort flagged differential error; A3 withholds it for the whole cohort,
  and the tables were re-generated from the saved per-image files. e_A and e_B against DermArtifactDB (presence only) are
  now undefined instead of 0.
- **Part B.** Recipe ConvNeXt-T (IN-22k), lr 1e-4, 8 epochs (validation AUROC 0.865). FT0 0.769: below the benchmark
  of 0.773. FT1 −0.102 [−0.178, −0.026], FT2 4 of 4 operating points, FT3 −0.521 [−0.642, −0.382], FT4 +0.187
  [+0.111, +0.268]; all supported after Holm.
- **Exploratory, not registered (decided after the results above).** Consensus view (an image is contradicted only
  when BUSClean and MedGemma agree with each other against our cell): 2 % of caliper-free, 2 % of Trap A and 14 % of
  Trap B thyroid images; of the 78 FT3 subgroup nodules none is contradicted by both and 36 are confirmed by both.
  `scripts/round6/exploratory.sh` re-runs the thyroid traps under this view (E1) and FT3 in the confirmed nodules (E2);
  their outputs go to `results/round6/exploratory/`.
