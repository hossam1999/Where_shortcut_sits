# Draft email to Yan, Ge et al. — EPVT trap-set reproduction (not sent; the author sends it)

**To:** Siyuan Yan <siyuan.yan@monash.edu> (first author), Zongyuan Ge <zongyuan.ge@monash.edu>
*Addresses as printed on p. 1 of the PLDG paper (IEEE TMI 44(1), 2025); the EPVT MICCAI 2023 paper prints none and neither
paper marks a corresponding author. Monash addresses may have changed since.*
**Subject:** Question about the ISIC trap-set results (bias 1) of EPVT (MICCAI 2023) / PLDG (TMI 2025)

Dear Dr. Yan, dear Prof. Ge,

I am an MSc student working on shortcut learning in medical imaging, specifically on how the position of an artifact
relative to the lesion decides whether region-based fixes work. EPVT and PLDG are central comparisons in my thesis, and I
am reproducing the trap-set debiasing result (EPVT Fig. 3a / PLDG Fig. 8) with your public code
(github.com/SiyuanYan1/EPVT @ 8d46593) and Bissoto et al.'s trap sets. Thank you for releasing it.

At bias 1 (Bissoto's split 1; your grouping script; your README hyper-parameters; ViT-B/16 ImageNet; batch 26 × 5 domains;
100 epochs in your code's epoch unit, int(min domain size / 26) steps; early stopping patience 22 on the validation
ROC-AUC, as in the paper; the best-validation checkpoint scored once on `test_bias_1_1`), our ERM reproduces your figure
but EPVT does not:

| | ours | your figure (digitised) |
|---|---|---|
| ERM, 5 seeds | 0.668 ± 0.017 | 0.653 |
| EPVT, released code, seed 0 | 0.655 | 0.746 |
| EPVT, mixup as in Eq. 3 (see 1. below), 5 seeds | 0.696 ± 0.006 | 0.746 |

Before contacting you we audited every element of the pipeline against your code and paper (a 37-item table, attached:
data and domain grouping, preprocessing, augmentation, model and prompts, registry defaults, optimisers, epoch unit,
selection and scoring). We found no difference in our port, but a few points where we could not tell what produced the
figure:

1. **Domain mixup labels.** In `DoPrompt_group_decompose.forward_prompt` (`domainbed/algorithms.py:2106–2109`), x_i, x_j,
   y_i and y_j each come from a separate call to `random_pairs_of_minibatches`, i.e. from four independent domain
   permutations, so in about 99 % of steps the mixed images are trained against labels of images from other domains. Eq. 3
   of the paper uses the labels corresponding to x_i and x_j. Which version produced Fig. 3a? (With one pairing per step,
   EPVT improves from 0.655 to 0.696 in our runs, but stays below 0.746.)
2. **Trap-set scoring.** `test_epvt.py` scores only the four OOD sets; we scored the in-domain trap test as in
   `test_erm.py` (MelanomaDataset, Resize 224, softmax ROC-AUC). Is this how Fig. 3a was computed, and which Bissoto split
   (1–10) and how many epochs / which patience (paper 22, code 200) were used for the trap sets?
3. **Training set size.** The paper reports 12,360 training images; the released `train_bias_0_1.csv` has 12,343 and
   `train_bias_1_1.csv` 12,359 rows. Was another file used?
4. If available, could you share the trap-set result logs or checkpoints behind Fig. 3a?

We will report your answer, cite your papers and code, and are happy to share our results and the audit with you before
submission.

Thank you very much for your time.

Kind regards,
[Name]
[MSc programme, university, supervisor]

*Attachment for the author to add: `docs/P6_EPVT_AUDIT.md` (audit table) and the A34 per-seed results
(`results/replication/epvt/A34_gate.json`).*
