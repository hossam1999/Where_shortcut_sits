# Draft email to Bevan & Atapour-Abarghouei (not sent; the author sends it)

**To:** Peter Bevan and Amir Atapour-Abarghouei (Durham University) — addresses from the paper's first page
**Subject:** Request: Heidelberg test set and trained weights, "Skin Deep Unlearning" (ICML 2022)

Dear Mr. Bevan and Dr. Atapour-Abarghouei,

I am an MSc student working on shortcut learning in medical imaging, specifically on how the position of an artifact
relative to the lesion (inside vs outside the region of interest) decides whether region-based fixes work. Your ICML 2022
paper "Skin Deep Unlearning: Artefact and Instrument Debiasing in the Context of Melanoma Classification" is an ideal
comparison for my thesis: surgical markings and rulers are exactly the kind of artefacts I study, and your public code
(github.com/pbevan1/Skin-Deep-Unlearning) makes the training reproducible — thank you for releasing it.

As your README notes, the surgical-marking / ruler test set from Heidelberg University (Winkler et al., JAMA Dermatology
2019; 130 lesions, each with and without markings) is not publicly available, and it is the test behind your Table 1
(e.g. ruler bias: baseline AUC 0.831, CLGR 0.958). I would like to ask:

1. whether the Heidelberg test set could be made available to me for research under a data-use agreement, or whom I should
   contact to request it (e.g. Prof. Haenssle's or Dr. Winkler's group);
2. whether you could share the trained weights for the Table 1 models (baseline, LNTL, TABE and CLGR, surgical markings and
   rulers), which your README offers on request;
3. the exact skewed training sets used for Table 1 (the image lists after the d_m = 20 / d_r = 18 duplication), if they were
   saved, together with the surgical-marking and ruler labels for ISIC 2020.

I would use everything only for research, report your numbers alongside mine, cite your paper and code, and share my
results with you before submission if you wish.

Thank you very much for your time.

Kind regards,
[Name]
[MSc programme, university, supervisor]
