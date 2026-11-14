# Deep-research prompt: find a paper we can replicate and compare against

Copy everything below the line into the deep-research model.

---

You are helping an MSc student find **one published paper to replicate and then beat**. Search thoroughly, read full texts (not just abstracts), and verify every claim against the paper itself.

## The student's thesis, in short
Medical image classifiers often learn **shortcuts**: an image artifact that is correlated with the diagnosis but is not evidence of disease. Examples: a ruler or ink next to melanomas, sonographer **calipers** on thyroid or breast nodules, a **chest drain** in pneumothorax X-rays, hair on skin lesions, and debris in capsule endoscopy.

A common fix is **ROI masking**: segment the lesion or organ and let the model see only that region. The thesis shows that this works **only when the artifact lies outside the ROI**. When the artifact lies **inside** the ROI (a caliper on the nodule, a ruler or hair over the lesion, a drain over the lung), masking cannot remove it. The model then relies on it as much as before or more, and performance on cases where the artifact points to the wrong diagnosis drops.

- **Evaluation:** models are trained with the artifact on 90% of positive and 10% of negative images. They are tested on a correlated set (same rates), a **reversed** set (rates swapped, 10/90) and a **clean** set. The main metric is reversed-test **AUROC**.
- **Datasets:** ISIC 2018/2019/2020 (dermoscopy), TN3K/TNCD (thyroid ultrasound), MMOTU (ovarian ultrasound), SEE-AI/Kvasir-Capsule (capsule endoscopy) and NIH ChestX-ray14.
- **The thesis also has a fix:** a class-conditional feature constraint applied after masking, plus a label-free variant. Both repair the in-ROI case.

## What we need
A peer-reviewed paper whose **numbers we can reproduce and then compare against**. The plan is:
1. Rebuild their setup and reproduce their reported number.
2. Show that their method, or plain ROI masking in their setup, fails when the shortcut lies inside the ROI.
3. Add our fix and beat their number on the same test.

The paper must be about **the same problem**: an ROI restriction (or ROI-based method) meeting a spurious feature, ideally one that can lie inside the ROI.

## Must have
1. **Medical image classification** with a spurious feature correlated with the label. The feature can be real (calipers, rulers, ink, hair, drains, tubes, text, markers, electrodes, bubbles, debris, scanner overlays) or synthetic (pasted marks or patches).
2. **Some form of ROI restriction** among the methods or baselines. Any of these count:
   - segmentation masking or cropping to the lesion or organ;
   - background removal or normalisation;
   - inpainting outside the ROI;
   - relevance or attention penalties outside a mask (e.g. ISNet, "right for the right reasons", GradMask, attention supervision);
   - foreground-only training;
   - test-time masking.
3. **A reported number on a test where the shortcut is broken.** Examples: a reversed or "trap" test set, an artifact-free or shortcut-conflicting subgroup, or a worst-group or out-of-distribution set. It must be something we can rebuild.
4. **Public data**, and preferably public code or splits.
5. **Peer-reviewed**, 2019–2026. Prefer journals and main conferences (MICCAI, MIDL, IPMI, ISBI, CVPR/ICCV/ECCV, NeurIPS/ICML/ICLR, Medical Image Analysis, IEEE TMI, Nature family, Radiology AI, ML4H/CHIL proceedings). Flag clearly if something is a workshop paper or preprint.

## Strong pluses (rank these higher)
- The spurious feature **can lie inside the ROI**. Best of all, the authors **mention this** anywhere: limitations, discussion, future work, an appendix or a figure caption. For example: "masking cannot remove artifacts that overlap the lesion", "if the spurious feature overlaps the region of interest…", "markers on the lesion remain after cropping", or "the drain lies over the lung field".
- The metric is AUROC, or one we can convert to (balanced accuracy, worst-group accuracy, F1).
- They use a dataset we already have: ISIC 2018/2019/2020, HAM10000, NIH ChestX-ray14, TN3K/TNCD/DDTI, MMOTU, Kvasir-Capsule. Breast ultrasound with calipers is also welcome: BUSI, BrEaST, UDIAT, BUS-BRA.
- It is feasible on one GPU within about 2 weeks.
- It is well cited, or from a strong group or venue.

## Where to look (not exhaustive)
- **Dermoscopy:** artifact debiasing (rulers, ink, hair, dark corners, colour patches) with lesion segmentation or cropping; trap sets; "skin deep unlearning"; targeted data augmentation.
- **Breast and thyroid ultrasound:** papers that note **calipers or annotation marks** on BUSI or thyroid images as a shortcut and then crop to the ROI, inpaint, or remove marks. Papers curating BUSI often mention this.
- **Chest X-ray:** COVID or pneumothorax with **lung segmentation** where a shortcut survives inside the lungs (drains, tubes, source or intensity bias); ISNet-like relevance penalties; GradMask; lung-masking studies.
- **Methods with ROI or foreground priors:** saliency or attention supervision with masks; counterfactual inpainting outside the ROI (e.g. MaskMedPaint); "foreground bias" or "spurious features within the object".
- **Natural-image analogues, only as a fallback:** in-object spurious features (Salient ImageNet core vs spurious features, ImageNet-9 backgrounds, foreground-only training) where a mask keeps the spurious cue.

## Already known to us
For these, only report **new** facts such as exact numbers, splits, code links or an in-ROI quote. Do not re-propose them without saying what is new.
- Bissoto, Valle, Avila 2020, CVPRW, "Debiasing skin lesion datasets and models? Not so fast" (trap sets plus background normalisation). Also Bissoto 2022 ECCVW (artifact-based DG) and Bissoto 2023 MICCAIW (test-time selection).
- Bassi et al. 2024, Nature Communications (ISNet).
- Jin, Gerych, Ghassemi 2024, MaskMedPaint (ML4H).
- Sourget et al., Mask of Truth (J Imaging Inform Med 2026).
- Maron et al. 2021, JMIR.
- Teixeira et al. 2021, Sensors.
- Nauta et al. 2022, Diagnostics.
- Stanley et al. 2024, JAMIA (SimBA).
- Boland et al. 2024, MIDL.
- Corbetta et al. 2026, MICCAI.
- Puli et al. 2022, ICLR (NuRD).
- Lin et al., segmentation shortcuts (calipers in fetal ultrasound).
- Weng et al., thyroid ultrasound across scanners.
- Jiménez-Sánchez et al. 2023, ISBI (chest drains).
- Li et al. 2023, CVPR (Whac-A-Mole).
- Winkler et al. (surgical markings; scale bars).
- Rueckel et al. 2021.
- Aniraj et al. 2023, ICCVW (masking strategies).

## Output
1. **A ranked shortlist of the 5 best candidates.** For each one, give:
   - **Reference:** authors, title, venue, year, DOI.
   - **Peer review:** peer-reviewed? Main track or workshop?
   - **Data and artifact:** dataset(s) with sizes; the spurious feature, and whether it can lie inside the ROI (and how often, if reported).
   - **ROI restriction:** exactly how it is done.
   - **Test design:** train and test correlation rates, and the reversed, conflicting or worst-group set.
   - **Metric and headline numbers:** with the table or figure number, e.g. "Table 3: ROI-masked ResNet-50, AUROC 0.71 on the artifact-free test".
   - **Exact quote(s)** where the authors touch on in-ROI or overlapping shortcuts, with section or page. Write "none found" if there are none.
   - **Code, splits and labels:** availability with links (GitHub, dataset pages).
   - **Compute:** estimated cost to reproduce.
   - **Replication plan:** which number we reproduce, how we show the in-ROI failure in their setup, and which number our fix must beat.
   - **Risks:** for example missing splits, unclear preprocessing, or weak venue.
2. **One recommendation**, with the reason.
3. **Search log:** the queries used, and anything relevant that you rejected and why (e.g. no broken-shortcut test, private data, preprint only).

Rules:
- Read full texts.
- Mark anything you could not verify as UNVERIFIED.
- Never invent numbers, quotes, DOIs or links.
- If a paper only partly fits, say exactly which criteria it misses.
