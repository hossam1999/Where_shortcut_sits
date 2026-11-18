# Draft email to Yan et al. (not sent; the author sends it)

**To:** corresponding author of "Towards Trustable Skin Cancer Diagnosis via Rewriting Model's Decision" (CVPR 2023) —
address from the paper's first page (Zongyuan Ge's group, Monash University); **Cc:** Siyuan Yan (first author)
**Subject:** Request: ConfDerm splits and code, "Towards Trustable Skin Cancer Diagnosis via Rewriting Model's Decision" (CVPR 2023)

Dear Dr. Ge and Dr. Yan,

I am an MSc student working on shortcut learning in medical imaging, specifically on how the position of an artifact
relative to the lesion (inside vs outside the region of interest) decides whether region-based fixes work. Your group's
work is central to this: I am currently reproducing EPVT (MICCAI 2023) and PLDG (IEEE TMI 2025) from your public
repositories, and the released code has made that possible — thank you.

Your CVPR 2023 paper would be an ideal second benchmark for my thesis, because ConfDerm assigns one confounder per class
(dark corners, rulers, borders, air pockets, hair) with an unconfounded test set, and several of these confounders lie over
the lesion. I could not find a public release of ConfDerm or of the code (the project page siyuanyan1.github.io/projects/
trust_derm returns 404, and I did not find a repository). Would you be willing to share:

1. the ConfDerm split lists (image IDs of the training, validation and test sets of each of the five subsets) and the
   artifact annotations used to build them;
2. the code for the baseline and for the rewriting method (SpRAy clustering, CAV concept space, first-order-logic
   intervention), or, if that is not possible, the training settings of the baseline CNN and the exact evaluation protocol
   behind the per-class accuracies (e.g. melanoma with dark corners: 23.48 % → 84.95 %);
3. if available, the trained baseline checkpoints.

I would use them only for research, report your numbers alongside mine, cite the paper, dataset and code, and share my
results with you before submission if you wish. I am happy to sign a data-use agreement if one is needed.

Thank you very much for your time.

Kind regards,
[Name]
[MSc programme, university, supervisor]
