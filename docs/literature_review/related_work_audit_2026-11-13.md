# Where the Shortcut Sits
## Full-text audit of the MSc presentation’s related work

**Audit date:** 13 November 2026

**Reading the tables.** CONFIRMED means the claim is supported by the full text cited in the row. WRONG means the claim needs the correction given in that row. NOT FOUND means the stated detail was absent, unquantified, or unavailable for verification. A dash means the slide supplied no claim for that point. Publication details can also come from the publisher’s bibliographic record. Numbers calculated from released data are explicitly marked. The same-title full manuscripts were used for C and R, alongside their publication records. P’s requested journal version could not be opened.

**Test designs.** For a binary cue, rates are conditional on the label unless specified otherwise. A reversed test changes the direction of the association seen in training. A test with equal cue prevalence across labels is balanced, and a test with the cue removed is artifact-free. Counterfactual image edits, subgroup evaluation, cue-positive test sets and ordinary external validation are identified by their actual construction. For spatial experiments, the tables identify what moved and which region defines its location.

**Closest precedents.** Bassi et al. already combine matched, reversed and artifact-free tests with background-bias mitigation, including an ROI-only pipeline. Puli et al. use the 90/10 → 10/90 chest X-ray source reversal, also test independence, and include a crop baseline. Bissoto et al. reverse seven dermoscopy artifact associations and evaluate masking. Stanley et al. move a synthetic bias between two anatomical regions outside the disease region. These precedents leave a narrower comparison for the thesis: crossing the same artifact’s placement inside versus outside the ROI with ROI masking and evaluating each condition on matched, reversed and clean tests.

---

## A. Brown et al. — ShorT

[Full journal text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10354021/) · [Supplement](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-39902-7/MediaObjects/41467_2023_39902_MOESM1_ESM.pdf)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Alex Brown, Nenad Tomasev, Jan Freyberg, Yuan Liu, Alan Karthikesalingam and Jessica Schrouff. *Detecting shortcut learning for fair medical AI using shortcut testing*. Nature Communications 14, 4314 (2023). DOI 10.1038/s41467-023-39902-7. Peer-reviewed journal. | Citation and article header | CONFIRMED |
| Method | Train a shared model with disease and demographic outputs. Scale the demographic gradient before it enters the shared layers, then measure how well a frozen representation supports an independent demographic probe and how the disease model’s fairness changes. | Methods, “Shortcut testing” and “Modulating the representation”; Fig. 1 | CONFIRMED |
| Data | NIH ChestX-ray14, with effusion, atelectasis and “abnormal” tasks. Original training set 65,394. Biased/balanced training sizes are 55,634/61,029 for effusion, 57,800/59,349 for atelectasis and 55,128/62,294 for abnormal. The dermatology application uses 12,027 training cases and 1,925 independent test patients. The supplement also tests race encoding for CheXpert cardiomegaly and simulates a colored square on MNIST. | Supplementary Methods, data and patient cohort tables; Supplementary Fig. 1 | CONFIRMED |
| Models | BiT ResNet-101×3 with a shared trunk and separate heads for the clinical target and demographic attribute. Frozen linear age probe. | Methods, “Model architecture” and representation measurement | — |
| Metrics | Disease AUROC. Age recoverability is age-prediction mean absolute error (MAE, years), so lower MAE means more recoverable age. The separation gap is the mean absolute slope of two logistic fits of the thresholded prediction against age, fitted separately in positive and negative cases. ShorT reports the Spearman correlation between encoding and separation measures. | Methods, “Metrics”; Figs. 2–4 | — |
| Results | Effusion age MAE versus separation, Spearman ρ = −0.668 on the biased setup (p = 8.11×10⁻¹⁷), versus −0.224 on the original and +0.116 on the balanced setup. Negative gradient scaling roughly halves the biased-model separation gap in the illustrated comparison. | Figs. 3c and 4a–c | CONFIRMED |
| Test design | Age is continuous. In original/biased/balanced effusion training, positive-case prevalence is 10.3%/10.1%/10.5%. Mean ages in positive versus negative cases are 51.3/48.2, 55.8/44.6 and 50.1/50.0 years respectively. NIH’s original test has 17,723 examples; Fig. 3 lists 15,148 and 16,093 for the subsampled biased and balanced evaluations. The paper also cross-evaluates dataset versions. For CheXpert cardiomegaly, the resplit is 85/5/10; overall disease prevalence is 19.8% in Black and 11.5% in White patients. No sign-reversed clinical test is prescribed. In supplemental MNIST, the authors sweep red-square prevalence and switch its color in paired counterfactual images. | Supplementary Methods, effusion and cardiomegaly cohorts; Figs. 3–4; Supplementary Figs. 1, 7–8 | — |
| Location | Age is a demographic attribute. In supplemental MNIST a colored square is placed randomly, with no foreground/background strata. The clinical experiments have no spatial placement intervention. | Methods; Supplementary Note 1 | — |
| Masking | The clinical classifier uses the image. The intervention is gradient scaling, with no anatomical ROI restriction. | Fig. 1; Methods, model and image preprocessing | — |

## B. Yang et al. — Fairness under external validation

[Full publisher PDF](https://www.nature.com/articles/s41591-024-03113-4.pdf) · [Journal article](https://www.nature.com/articles/s41591-024-03113-4)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Yuzhe Yang, Haoran Zhang, Judy W. Gichoya, Dina Katabi and Marzyeh Ghassemi. *The limits of fair medical imaging AI in real-world generalization*. Nature Medicine 30, 2838–2848 (2024). DOI 10.1038/s41591-024-03113-4. Peer-reviewed journal. | Article header and citation | CONFIRMED |
| Method | Train and tune six algorithms, measure demographic recoverability from frozen features, and compare disease discrimination and group gaps on internal and external datasets. Methods include ERM, ReSample, GroupDRO, DANN, CDANN and MA (exponential moving average). The MIMIC-CXR grid alone has 3,456 models. | Fig. 1; Methods, “Algorithms” and experimental setup | CONFIRMED |
| Data | Counts as analyzed: MIMIC-CXR 357,167; CheXpert 222,792; NIH 112,120; SIIM 11,582; PadChest 144,478; VinDr 6,354; ISIC 32,259; ODIR 6,800. The main CXR grid trains on MIMIC-CXR. Additional experiments train on CheXpert, ISIC and ODIR. The main external CXR evaluation uses the other five CXR sources. | Fig. 1; Methods, “Datasets”; Extended Data Fig. 7 | CONFIRMED |
| Models | DenseNet-121 clinical classifiers. Logistic probes use frozen image representations to predict demographic groups. | Methods, model training and demographic encoding | — |
| Metrics | Underdiagnosis is the false-positive rate (FPR) for “No Finding,” since a false positive predicts healthy for a patient with findings. For specific findings it is the false-negative rate (FNR). Fairness gaps are differences between selected demographic groups. FPR is FP/(FP+TN), and FNR is FN/(FN+TP). The paper also uses clinical AUROC, macro-AUROC for demographics, and the expected calibration error (bin-size-weighted discrepancy between observed and predicted rates). | Methods, “Evaluation metrics”; Figs. 1–2 | CONFIRMED |
| Results | In Fig. 2g, demographic encoding and fairness gap correlate at R = 0.82 for age/No Finding and R = 0.71 for race/effusion. In a CheXpert-trained example, the sex gap for No Finding is −0.1 percentage points internally and 3.2 points on MIMIC-CXR. Fairness transfer varies by task, demographic and dataset. | Fig. 2g; Fig. 4; Extended Data Fig. 7 | CONFIRMED, as empirical associations and examples |
| Test design | Standard dataset splits are 70% training, 15% validation and 15% testing. A separate analysis rebalances demographic cells and makes label prevalence uniform across cells. External cohorts have naturally varying prevalence. There is no prescribed reversal of a binary artifact–label link. | Methods, “Datasets” and evaluation setup; Fig. 4 | — |
| Location | Demographic recoverability and source transfer are studied. No cue location is experimentally assigned. | Methods, demographic encoding and cross-dataset evaluation | — |
| Masking | Standard image preprocessing and training augmentation include cropping. There is no anatomical ROI-only or ROI-removed comparison. | Methods, image transformations | — |

## C. Jiménez-Sánchez et al. — Chest drains and pneumothorax

[Full same-title manuscript](https://arxiv.org/pdf/2211.04279) · [IEEE proceedings record](https://ieeexplore.ieee.org/document/10230572)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Amelia Jiménez-Sánchez, Dovile Juodelyte, Bethany Chamberlain and Veronika Cheplygina. *Detecting Shortcuts in Medical Images – A Case Study in Chest X-Rays*. IEEE ISBI 2023, pp. 1–5. DOI 10.1109/ISBI53787.2023.10230572. Peer-reviewed main conference. | Manuscript title page; IEEE proceedings record | CONFIRMED |
| Method | Train pneumothorax classifiers on increasing CheXpert sample sizes. Evaluate them on two sources using controlled proportions of visible drains among pneumothorax cases, and inspect the learned representations. | §§3.1–3.2; Fig. 3 | CONFIRMED |
| Data | Source corpora: CheXpert 224,314 images, 65,240 patients; NIH-CXR14 112,120 images, 30,805 patients. CheXpert alone supplies training and validation. The training/validation subsets contain 4,000, 8,000, 16,000 or 24,000 images, resampled three times. The authors annotated 300 CheXpert pneumothorax-with-drain images for their test design. | §§3.1–3.2; Table 1 | CONFIRMED, with training source specified |
| Models | DenseNet-121 with probabilistic class-activation-map pooling and a final fully connected layer. | §3.2 | — |
| Metrics | Pneumothorax AUROC, the area under the sensitivity versus false-positive-rate curve, averaged across runs with standard deviation. t-SNE is used to display the embeddings. | §3.2; Table 1; Figs. 2–3 | — |
| Results | With 24,000 development images, CheXpert baseline / drain-positive / drain-negative tests score 81.0 ± 0.6%, 84.7 ± 0.8%, 76.7 ± 0.7%. NIH scores 74.4 ± 2.4%, 81.1 ± 3.5%, 65.5 ± 2.6%. The 0.94 versus 0.77 comparison in the background is from an earlier study. | Table 1, 24k columns; §4 | CONFIRMED |
| Test design | Development subsets are label-balanced and split 80/20 for training/validation. Training drain prevalence is not specified. In each 600-image test, the 300 negatives are all drain-free. Among the 300 positives, 150/300, 300/300 or 0/300 have drains in the three tests. This gives a drain-free test and two positively associated tests, with no reversal. | §3.2; Table 1 | — |
| Location | Real drains are annotated as present or absent. Their positions are not experimentally moved or stratified by ROI. | §§3.1–3.2 | — |
| Masking | The classifier uses the chest radiograph. No anatomical input mask or outside-ROI relevance penalty is evaluated. | §3.2 | — |
| NEATX attribution | The approximately 3,500 NIH and 1,000 PadChest annotations belong to a later paper with a different author list. See C2. | C §3.1; C2 §§3.1–3.2 | WRONG |

### C2. Cheplygina et al. — NEATX

[Full manuscript](https://arxiv.org/pdf/2309.02244) · [Published MIUA chapter](https://doi.org/10.1007/978-3-031-98688-8_10)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Veronika Cheplygina, Cathrine Damgaard, Trine Naja Eriksen, Dovile Juodelyte and Amelia Jiménez-Sánchez. *Augmenting Chest X-ray Datasets with Non-Expert Annotations*. Medical Image Understanding and Analysis (MIUA), LNCS 15916, pp. 133–144 (2025). DOI 10.1007/978-3-031-98688-8_10. Peer-reviewed conference proceedings. | Title page; published chapter record | CONFIRMED as the separate NEATX reference |
| Method | Non-experts label tubes and drains on CXRs. The authors compare their agreement with experts and test how detectors trained with the new annotations generalize to expert labels. | §§3–4 | — |
| Data | NIH-CXR14 has 3,709 initially annotated images and 3,543 accepted annotations. PadChest has 1,011 annotated images for chest drains, tracheostomy, nasogastric and endotracheal tubes. Source datasets have 112,120 and 160,868 images, respectively. | §§3.1–3.2 | CONFIRMED for NEATX |
| Models | ResNet-50, InceptionV3, DenseNet-121 and a modified DenseNet. | §3.3; Table 1 | — |
| Metrics | AUROC for tube detection and Cohen’s κ for annotator agreement, which adjusts for chance agreement. | §3.3; Tables 1–2 | — |
| Results | Modified DenseNet drain-detection AUROC is 92.1 ± 1.3%, 92.6 ± 0.2% and 89.6 ± 0.6% on the three test subsets reported in Table 1 (NonExp and the two expert subsets). | Table 1 | — |
| Test design | Patient-separated detection splits and non-expert/expert test subsets. Tube presence is the detection target. No engineered tube–disease reversal. | §§3.3 and 4; Table 1 | — |
| Location | Tubes are identified by type or presence. No controlled near/far or inside/outside movement. | §3.2 | — |
| Masking | The detection benchmark uses radiographs. No anatomical ROI masking experiment. | §§3.3 and 4 | — |

## D. Pahde et al. — Reveal to Revise

[Full manuscript and appendix](https://arxiv.org/pdf/2303.12641) · [MICCAI chapter](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_56)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Frederik Pahde, Maximilian Dreyer, Wojciech Samek and Sebastian Lapuschkin. *Reveal to Revise: An Explainable AI Life Cycle for Iterative Bias Correction of Deep Models*. MICCAI 2023, LNCS 14221, pp. 596–606. DOI 10.1007/978-3-031-43895-0_56. Peer-reviewed main conference. | Title page; proceedings record | CONFIRMED |
| Method | Use SpRAy with explanation maps to find shortcut patterns, then localize concepts and iteratively correct reliance using RRR, CDEP and ClArC variants. | §§3–4; Fig. 1 | CONFIRMED |
| Data | ISIC 2019, 25,331 images and eight diagnosis classes, with band-aids, rulers, skin markers and an inserted text artifact. Pediatric Bone Age, 12,611 images grouped into five age bins, includes a natural “L” marker and a synthetic artifact. | §4.1; Appendix A.2 | CONFIRMED |
| Models | VGG-16, ResNet-18 and EfficientNet-B0. CDEP was not applied to EfficientNet-B0 in the reported comparison. | §4.2; Table 1 | CONFIRMED |
| Metrics | Classification accuracy and F1, the harmonic mean of precision and recall. A further metric is the fraction of explanation relevance in the artifact region. The F1 averaging rule is not identified in the paper. | §§4.2–4.3; Tables 1–2 | — |
| Results | With synthetic text in the poisoned ISIC test, VGG-16 F1 rises 7.7→39.2 with RRR and ResNet-18 39.0→70.4. In the iterative ISIC correction, ruler-region relevance falls 24.2%→8.5%. On poisoned Bone Age images, VGG-16 F1 rises 16.2→55.1 with a-ClArC. | Tables 1–2; supplementary Table 6 | — |
| Test design | The datasets use 80/10/10 splits. Synthetic text is inserted in 10% of melanomas or 50% of images in the 0–46-month age bin, with 0% in other classes. In the poisoned test it appears in every image, so presence is constant across labels. The original held-out images are also evaluated. The paper does not build a reversed-prevalence test. | §§4.1–4.2; Appendix A.2, Table 3 | — |
| Location | Text is inserted with randomized location, scale and rotation. The experiment does not assign or compare placements inside versus outside an anatomical ROI. | Supplement, synthetic artifact construction | — |
| Masking | Correction methods use annotated artifact pixels or concept maps to reduce relevance in shortcut regions. There is no anatomical ROI-only classifier. | §3.2, RRR/CDEP; §4.2 | — |

## E. Lin, Weng et al. — Shortcuts in medical image segmentation

[Full MICCAI proceedings paper](https://papers.miccai.org/miccai-2024/paper/0423_paper.pdf) · [ISIC 2017 dataset protocol](https://arxiv.org/abs/1710.05006)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Manxi Lin, Nina Weng, Kamil Mikolaj, Zahra Bashir, Morten B.S. Svendsen, Martin G. Tolsgaard, Anders N. Christensen and Aasa Feragen. *Shortcut Learning in Medical Image Segmentation*. MICCAI 2024, LNCS 15008, pp. 623–633. DOI 10.1007/978-3-031-72111-3_59. Peer-reviewed main conference. | Proceedings title page and citation | CONFIRMED |
| Method | Inpaint calipers and text to test whether ultrasound segmentation depends on them, then train a model to reduce that dependence. For skin lesions, alter crop location to test how zero padding and centering affect performance across the image. | §§3–4; Figs. 2–4 | CONFIRMED |
| Data | 3,775 ultrasound stills from third-trimester fetal examinations, 13 structures across head, abdomen, femur and maternal cervix views; 110 videos, including 10 cervix videos. The skin experiment uses the official train/test ISIC 2017 sets (2,000/600). The challenge also released 150 validation images. | §§3.1 and 4.1; ISIC challenge protocol | CONFIRMED |
| Models | DTU-Net for ultrasound; standard U-Net for skin. | §§3.1 and 4.1 | — |
| Metrics | Dice is twice the overlap divided by the sum of the two mask areas. Table 1 compares predictions with ground truth. Video Dice in Table 2 compares frame predictions with the final-frame prediction, a temporal-consistency measure. | §3.1; Tables 1–2 | — |
| Results | Original→inpainted mean Dice on the still test: head 76.97→70.85%, abdomen 82.06→78.85%, femur 93.82→91.84%, cervix 76.29→71.81%. Mitigation improves video consistency, e.g. head 89.07→98.39% and cervix 82.42→97.00%. Skin Dice is plotted by ring in Fig. 4; a global numeric comparison is not tabulated. | Tables 1–2; Fig. 4 | CONFIRMED for the Dice comparisons stated here |
| Test design | Ultrasound stills use an 80/10/10 split. The same held-out images are evaluated with original and inpainted annotations. The annotation prevalence by structure in train/test is not reported. No cue–label reversal is specified. | §3.1; Table 1 | — |
| Location | The skin study varies where the lesion lies relative to the crop boundary and evaluates ring-specific effects. Ultrasound calipers and text are removed in place. No same-artifact relocation across an anatomical ROI boundary. | §§3–4; Fig. 4 | — |
| Masking | Uses annotation-region inpainting and crop interventions. The network predicts segmentation masks and does not use an anatomical input mask to restrict a classifier. | §§3.1 and 4.1 | — |

## F. Cassidy et al. — ISIC duplication and benchmarks

[Full publisher PDF via institutional repository](https://e-space.mmu.ac.uk/631025/7/1-s2.0-S1361841521003509-main.pdf)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Bill Cassidy, Connah Kendrick, Andrzej Brodzicki, Joanna Jaworek-Korjakowska and Moi Hoon Yap. *Analysis of the ISIC image datasets: Usage, benchmarks and recommendations*. Medical Image Analysis 75, 102305 (2022). DOI 10.1016/j.media.2021.102305. Peer-reviewed journal. | Publisher PDF first page | CONFIRMED |
| Method | Audit exact and downsampled duplicates within and between ISIC releases. Keep official tests intact, remove matching training copies, and benchmark lesion classification on curated images. | §§3 and 4.1–4.3 | CONFIRMED |
| Data | ISIC training/test counts by release: 2016, 900/379; 2017, 2,000/600; 2018, 10,015/1,512; 2019, 25,331/8,238; 2020, 33,126/10,982. Deduplication leaves 57,062 training images; the balanced binary benchmark uses 9,810 images. | Tables 2 and 8; §4.2 | CONFIRMED |
| Models | Nineteen benchmark CNNs: DenseNet-121/169/201; EfficientNet-B0–B4; InceptionResNetV2 and InceptionV3; ResNet-50/50V2/101/101V2/152/152V2; VGG-16/19; Xception. | §4.3; benchmark tables | — |
| Metrics | Duplicate matching uses exact-file comparison and identifiers for known resized copies, with similarity metrics discussed separately. The ISIC 2020 binary benchmark uses AUROC. The 2017 benchmark also reports precision, recall, F1 and accuracy. Multiclass leaderboard performance uses normalized accuracy (mean per-class recall). | §§4.1–4.3; Tables 10–13 | — |
| Results | Exact duplicates counted separately are 12,039 in the pooled training files and 1,592 in pooled test files; pooling both gives 13,976 exact duplicates. Within-release train/test matches: three in 2016, two in 2017 and zero in each later release. The audit identifies 2,263 downsampled matches, removes 1,927 downsampled files, and removes 14,310 training files in total. For the ISIC 2020 binary benchmark VGG-19 scores 0.80 AUROC, versus 0.77 for VGG-16 and DenseNet-121. | Tables 5–8 and 10; §4.1 | CONFIRMED |
| Test design | Duplicate and split hygiene is the intervention. The balanced binary benchmark has 7,848 train and 1,962 validation images, with external ISIC challenge tests. The paper also excludes all images bearing a given artifact and rebalances the remaining development set. Cue prevalence is 0% in those curated arms for the excluded artifact, with no reversed test. | §§4.1–4.4; Tables 14–15 | — |
| Location | Artifact examples are illustrated, with no assigned position intervention. | Fig. 5; §3 | — |
| Masking | Standard image resizing/center-cropping is used in the benchmark. No lesion mask comparison. | §4.3 | — |

## G. Sourget et al. — Mask of Truth

[Full publisher article](https://link.springer.com/article/10.1007/s10278-025-01531-5) · [PMC copy](https://pmc.ncbi.nlm.nih.gov/articles/PMC12920878/)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Théo Sourget, Michelle Hestbek-Møller, Amelia Jiménez-Sánchez, Jack Junchi Xu and Veronika Cheplygina. *Mask of Truth: Model Sensitivity to Unexpected Regions of Medical Images*. Journal of Imaging Informatics in Medicine 39(1), 767–784. DOI 10.1007/s10278-025-01531-5. Peer-reviewed journal. Online May 2025; assigned to the February 2026 issue. | Publisher article history and recommended citation | WRONG if 2025 is used as the volume/issue year |
| Method | Cross five training inputs with five test inputs: full images, exact ROI only, exact ROI removed, ROI bounding box only and ROI bounding box removed. Compare masked predictions, explanations and embedding similarities, and have a radiology resident read masked examples. | Methods, “Masking,” “Experimental Setup” and “Expert Evaluation”; Fig. 2 | CONFIRMED |
| Data | PadChest 93,203 filtered frontal CXRs, with quality-filtered CheXmask lung masks. Chákṣu 1,345 fundus images with optic-disc masks. External tests use 22,004 NIH images and 18,000 AIROGS images. | Methods, “Datasets”; Table 1 | CONFIRMED |
| Models | DenseNet-121 is the image classifier, and logistic regression is used for metadata checks. CheXmask lung segmentations were generated by HybridGNet. | Methods, “Models” and “Datasets” | CONFIRMED with architecture specified |
| Metrics | AUROC, computed per finding for CXR and for glaucoma in fundus images; SHAP maps; t-SNE and cosine similarity on learned embeddings; descriptive clinician identification of findings. | Methods, evaluation and explainability; Tables 3–4 | CONFIRMED |
| Results | Full-image-trained PadChest effusion AUROC is 0.95 full input, 0.86 lungs removed and 0.47 lungs only. For full-image-trained glaucoma, the corresponding scores are 0.68, 0.59 and 0.66. Training and testing on optic disc only yields 0.78. With full training, external AUROC is 0.72 for NIH effusion and 0.65 for AIROGS glaucoma. The resident identifies 2 of 35 findings in the ROI-removed material, out of an evaluation of 75 images across masking conditions. | Fig. 4a–b; Table 2; “Study with a Domain Expert” | CONFIRMED for high residual CXR performance; scope the claim by task |
| Test design | PadChest split 80/20. Chákṣu uses its predefined split. Image support changes systematically across the train/test matrix, while external sets are evaluated with full images. No controlled artifact prevalence or reversed cue–label test. | Methods, “Datasets” and experimental design; Table 2 | — |
| Location | Tests whether useful signal is inside or outside the lungs/disc and also dilates the ROI. No specific artifact is repositioned. The authors discuss information carried by mask geometry. | Results, masking experiments; Discussion | — |
| Masking | Yes. Precise and bounding-box ROI-only and ROI-removed inputs are applied during training and/or testing. | Methods, “Masking”; Figs. 2 and 4 | CONFIRMED |

## H. Bassi et al. — ISNet

[Full journal text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10767127/) · [Supplement](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-44371-z/MediaObjects/41467_2023_44371_MOESM1_ESM.pdf)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Pedro R.A.S. Bassi, Sergio S.J. Dertkigil and Andrea Cavalli. *Improving deep neural network generalization and robustness to background bias via layer-wise relevance propagation optimization*. Nature Communications 15, 291 (2024). DOI 10.1038/s41467-023-44371-z. Peer-reviewed journal. | Article header | CONFIRMED |
| Method | Jointly optimize class prediction and an LRP loss that suppresses background relevance outside training masks. At inference ISNet uses the unsegmented image. A separate U-Net plus classifier actually segments the input before classification. | Fig. 1; Methods, “ISNet” and segmentation pipeline | CONFIRMED |
| Data | COVID CXR development uses 4,644 images per class before its split; Brixia and CheXpert supply development data. The 3,126-image external set uses BIMCV COVID-19 (1,515), NIH pneumonia (1,295) and Montgomery/Shenzhen normal (406). TB development starts with 6,728 TB Portals TB CXRs and 6,728 CheXpert normals; internal test is 1,284 and external test is 753 (383 TB, 370 normal). Dogs uses 240/60/201 train/validation/test images and CelebA-HQ uses 24,183/2,993/2,824. | Methods, datasets; Tables 4–5; supplement | — |
| Models | DenseNet-121 for CXR and CelebA-HQ. Stanford Dogs uses VGG-19. The comparison includes a U-Net plus DenseNet-121 pipeline. There are eight alternatives in the overall study and seven alternatives in the TB table, which contains eight models including ISNet. | Results overview; Tables 1 and 5; supplement, architectures | WRONG for ResNet-50 on Dogs; comparison with U-Net confirmed |
| Metrics | Macro-F1, the unweighted mean of classwise harmonic means of precision and recall; AUROC and other classification metrics. The ± values in the two cited tables are 95% confidence-interval half-widths. | Methods, statistical evaluation; Tables 1 and 5 | CONFIRMED |
| Results | External TB macro-F1: ISNet 0.738 ± 0.044, U-Net pipeline 0.576 ± 0.050, plain DenseNet 0.566 ± 0.050. AUROC is 0.809 ± 0.031, 0.667 ± 0.039 and 0.576 ± 0.040, respectively. In synthetic-bias COVID, ISNet macro-F1 is 0.775 across all three tests; plain DenseNet gives 0.775 matched, 0.434 clean and 0.195 reversed. The U-Net pipeline gives 0.618, 0.619 and 0.618 on the same tests. | Table 5, external TB; Table 1, synthetic-bias COVID | CONFIRMED |
| Test design | In the synthetic COVID experiment all train/validation images receive their class-specific upper-left mark before augmentation: square normal, circle pneumonia, triangle COVID. Training augmentation can remove some assigned marks. At test the rule is matched on all images, removed on all images, or cyclically reassigned (normal gets circle, pneumonia triangle, COVID square). Thus matched = 100% original cue, reversed = 0% original cue, clean = 0% marks. Dogs similarly use 100% class-specific marks in train, matched, cyclically reassigned (0% originally matched) or unmarked at test. In CelebA-HQ, each shape signals its attribute (100% presence conditional on attribute-positive, 0% on negative) in train/matched test; those rates swap to 0%/100% in reversed testing, and both are 0% in the unmarked test. TB is evaluated externally without these marks. | Results, artificial background bias; Table 1; supplement, artificial biases | — |
| Location | The synthetic CXR shape is in the upper-left background, outside the lungs. Dogs/CelebA-HQ also use background marks. No comparison of the same mark placed inside and outside the target ROI. | Methods and supplement, artificial background biases | — |
| Masking | Yes. ISNet uses anatomical masks for an outside-ROI relevance penalty during training. The U-Net baseline uses a segmented input at inference. | Fig. 1; Methods, “ISNet” and benchmark models | CONFIRMED |

## I. Li et al. — Whac-A-Mole

[Full CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_A_Whac-a-Mole_Dilemma_Shortcuts_Come_in_Multiples_Where_Mitigating_One_CVPR_2023_paper.pdf) · [Supplement](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Li_A_Whac-a-Mole_Dilemma_CVPR_2023_supplemental.pdf)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Zhiheng Li, Ivan Evtimov, Albert Gordo, Caner Hazirbas, Tal Hassner, Cristian Canton Ferrer, Chenliang Xu and Mark Ibrahim. *A Whac-a-Mole Dilemma: Shortcuts Come in Multiples Where Mitigating One Amplifies Others*. CVPR 2023, pp. 20071–20082. DOI 10.1109/CVPR52729.2023.01922. Peer-reviewed main conference. | CVPR paper and IEEE bibliographic record | CONFIRMED |
| Method | Build UrbanCars with two separately controllable cues and apply a watermark to ImageNet validation images. Compare robust training strategies and propose Last Layer Ensemble (LLE), which combines shortcut-specific classifier heads. | §§2–5; Fig. 5 | CONFIRMED |
| Data | UrbanCars has 8,000 training, 500 validation and 500 test composites. ImageNet-W is the watermarked ImageNet validation set, with 50,000 images across 1,000 classes. | §§2.1–2.2; supplement, UrbanCars construction | CONFIRMED |
| Models | ResNet-50 is the main controlled backbone. The ImageNet comparisons also include ViT, ConvNeXt, RegNet and other variants listed in the model tables. | §§3–4; Tables 1 and 6 | — |
| Metrics | In-distribution accuracy weights group accuracies by training proportions. BG Gap and CoObj Gap subtract ID accuracy from accuracy for examples where only the named cue is uncommon, in percentage points. The joint gap uses examples with both cues uncommon. ImageNet-W Gap subtracts unmodified ImageNet accuracy from watermarked accuracy, and Carton Gap is the class-specific gain for carton. | §2.1, evaluation protocol; Tables 1–3 | CONFIRMED |
| Results | UrbanCars ERM BG/CoObj/joint gaps are −15.3/−11.2/−69.2 points, versus −2.1/−2.7/−5.9 for LLE. EIIL reduces the BG gap to −4.2 but widens the CoObj gap to −24.7, an example of the trade-off. | Table 3; §3 | CONFIRMED |
| Test design | In UrbanCars training, background and co-object each match the car class with probability 0.95, independently given the car class. Both match with probability 0.9025. Validation and test each set both cue match rates to 0.50. The doubly uncommon group is analyzed separately. ImageNet-W overlays a watermark on the validation set. No whole-test 95→5 reversal. | §§2.1–2.2; supplement, UrbanCars construction | CONFIRMED for the proportions stated here |
| Location | The UrbanCars cues occupy different roles, background and separate co-object. ImageNet-W uses a watermark and augmentation can vary its position. No same-cue inside/outside-anatomical-ROI comparison. | Figs. 2–3; supplement, ImageNet-W | — |
| Masking | Uses masks to composite foreground/background and produce shortcut-specific augmentations. No medical ROI restriction. | §§2.1 and 4; supplement | — |

## J. Kirichenko et al. — Deep Feature Reweighting

[Full ICLR paper and appendices](https://arxiv.org/pdf/2204.02937) · [Original Waterbirds split protocol](https://arxiv.org/pdf/1911.08731)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Polina Kirichenko, Pavel Izmailov and Andrew Gordon Wilson. *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations*. ICLR 2023, main-track peer-reviewed paper. No conference volume/pages or proceedings DOI; arXiv 2204.02937, DOI 10.48550/arXiv.2204.02937 for the archive record. | Conference PDF title page | CONFIRMED |
| Method | Train a feature extractor by ERM, freeze it, then refit its final linear head by regularized logistic regression on a group-balanced subset. | §§5–6; Appendix C | CONFIRMED |
| Data | Waterbirds has 4,795 training and 1,199 validation images; the original protocol uses 5,794 test images. CelebA has 162,770 training and 19,867 validation images and uses the official test split. Other experiments include Dominoes, ColorMNIST, MultiNLI, CivilComments and ImageNet variants. | Figs. 6–7; Appendices B–D; original Waterbirds protocol, Appendix C.1 | CONFIRMED |
| Models | ImageNet-pretrained ResNet-50 for Waterbirds and CelebA, ResNet-20 for Dominoes, a small LeNet-like CNN for ColorMNIST and BERT for text. Additional ImageNet backbones are evaluated in §7. | §6; Appendices B–D | — |
| Metrics | Worst-group accuracy is the minimum of groupwise accuracies. Reported average accuracy uses group weighting appropriate to the benchmark; Waterbirds uses train-group proportions. | §6; Table 2; Appendix C | CONFIRMED |
| Results | Worst-group accuracy: Waterbirds DFR 92.9 ± 0.2%, Group DRO 91.4%, ERM 74.9%; CelebA DFR 88.3 ± 1.1%, Group DRO 88.9%, ERM 46.9%. | Table 2 | CONFIRMED |
| Test design | Standard Waterbirds training aligns habitat and bird type about 95% of the time within class. The standard test has balanced backgrounds within each class, about 50/50. Additional training variants are 50% and 100% alignment. “Inverse problem” swaps the prediction target and shortcut roles. A foreground-only test removes the background cue. Five-class ColorMNIST randomizes test color independently (20% chance of each color), while training agreement varies from 80% to 100% plus the independent baseline. | §§4, 6; Appendices B.1 and B.3; Figs. 4 and 6 | CONFIRMED for 95% training and balanced test |
| Location | Foreground and background are separated for diagnosis. The same cue is not moved across the bird boundary. | §4; Fig. 4 | — |
| Masking | Yes, in diagnostic evaluations that isolate the bird or background. The standard DFR classifier uses composite images. | §4; Appendix B.1 | — |

## K. Zong et al. — MEDFAIR

[Full ICLR paper and appendices](https://arxiv.org/pdf/2210.01725)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Yongshuo Zong, Yongxin Yang and Timothy Hospedales. *MEDFAIR: Benchmarking Fairness for Medical Imaging*. ICLR 2023, main-track peer-reviewed paper. No conference volume/pages or proceedings DOI; arXiv 2210.01725, DOI 10.48550/arXiv.2210.01725 for the archive record. | Conference PDF title page | CONFIRMED |
| Method | Compare ERM plus methods from five families (subgroup rebalancing, domain-independence, adversarial training, disentanglement and domain generalization), 11 algorithms total, with three checkpoint-selection rules: overall performance, minimax Pareto and distance-to-optimal. Compare performance and fairness across datasets, demographic groups and selection rules. | §§3.2–3.3 | CONFIRMED |
| Data | CheXpert 222,793; MIMIC-CXR 370,955; PAPILA 420; HAM10000 9,948; Fitzpatrick17k 16,012; OL3I 8,139; COVID-CT-MD 308; OCT 384; ADNI 1.5T 550; ADNI 3T 110. The four sensitive attributes are age, sex, race and skin type. | Table 1; §3.1 | CONFIRMED |
| Models | Two-dimensional and three-dimensional ResNet-18 classifiers for images and volumes, respectively. | Appendix B.1 | — |
| Metrics | Overall AUROC pools all examples. Worst-group AUROC is the minimum subgroup AUROC. AUROC gap is the maximum minus minimum subgroup AUROC. The appendix also reports loss, calibration and other thresholded measures. | §§2 and 4; Appendix B | CONFIRMED |
| Results | For HAM10000/sex, ERM overall/worst/gap is 85.20/83.12/3.31 points and SWAD is 91.44/91.14/0.44. No evaluated algorithm significantly improves fairness over ERM across the benchmark in the critical-difference comparison. | Table A9; Fig. 5 | CONFIRMED, at the benchmark level |
| Test design | Standard train/validation/test ratios are 80/10/10 for 2D and 70/10/20 for 3D. External tests use paired sources where available. There are subgroup-reweighting interventions in training. No deliberately reversed artifact–label test. | Appendix B.1; §4, external evaluation | — |
| Location | Sensitive demographic attributes are studied as metadata groups, with no controlled image placement. | §3.1; Table 1 | — |
| Masking | No anatomical ROI-only comparison or penalty. | Appendix B.1, architectures and transforms | — |

## L. Ktena et al. — Diffusion augmentation

[Full journal text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11031395/) · [Full 2023 preprint v1](https://arxiv.org/pdf/2304.09218v1)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Ira Ktena, Olivia Wiles, Isabela Albuquerque, Sylvestre-Alvise Rebuffi, Ryutaro Tanno, Abhijit Guha Roy, Shekoofeh Azizi, Danielle Belgrave, Pushmeet Kohli, Taylan Cemgil, Alan Karthikesalingam and Sven Gowal. *Generative models improve fairness of medical classifiers under distribution shifts*. Nature Medicine 30, 1166–1173 (2024). DOI 10.1038/s41591-024-02838-6. Peer-reviewed journal. | Journal article header | CONFIRMED |
| Method | Train conditional diffusion generators on labeled and unlabeled medical images, sample images for target class/attribute combinations, and train classifiers with mixtures of synthetic and real images. Evaluate accuracy and fairness under source and prevalence shifts. | Fig. 1; Methods, generative modeling and classifier training | CONFIRMED |
| Data | CAMELYON17: 455,954 labeled and about 1.8 million unlabeled patches; Fig. 2 captions list 33,560 internal and 85,054 external evaluation patches. CheXpert: 201,055 train; the Results give 13,332 ID test images and Fig. 3 captions give 23,261 ID evaluation images. ChestX-ray14 external test: 17,723. Dermatology: US 16,530 cases, Australian dataset 6,639 images and Colombian dataset 3,900 images. | Results, modality subsections; Figs. 2–4; Methods, “Datasets” | CONFIRMED for datasets and roles; ID CXR counts differ between passages |
| Models | Conditional denoising diffusion with U-Net denoisers and modality-dependent upsampling; classifier backbones are ResNet-152 for histopathology, BiT-ResNet-152 for CXR and BiT-ResNet for dermatology. | Methods, model architectures and modality hyperparameters | — |
| Metrics | Histopathology accuracy; CXR mean ROC-AUC over findings; dermatology high-risk sensitivity, i.e. the true-positive rate for the grouped high-risk conditions. Fairness gaps are differences between subgroup performances for the indicated metric. | Table 1; Methods, evaluation measures | — |
| Results, histopathology | The preprint reports +7.7 percentage points versus baseline. The final article reports +48.5% relative to the unaugmented baseline and +3.2% relative to color augmentation. Final Fig. 2b shows roughly 64% for the unaugmented baseline, 94% for color augmentation, and 95–96% for synthetic data plus color augmentation on the test hospital. Use the final paper’s stated headline and label the plotted values as approximate. | Preprint v1, histopathology Results; final Table 1 and Fig. 2 | WRONG if +7.7 is attributed to the final article |
| Results, chest X-ray | The final text retains +5.2% relative mean OOD ROC-AUC and a 44.6% relative reduction in the OOD sex gap. Final Table 1 gives 0.031 as the absolute reduction for that gap. | Final CXR Results; Fig. 3; Table 1 | CONFIRMED |
| Test design | Tests hospital/source, prevalence and demographic changes. The paper varies labeled data availability and mixes synthetic/real batches, including 50/50, 100/0 and 75/25 depending on modality. These proportions describe data mixtures. The paper does not engineer an artifact inversion or artifact-free image cohort. | Methods, evaluation and modality experiments; Table 1 | — |
| Location | No fixed image artifact is moved relative to an anatomical region. | Methods, diffusion conditioning and evaluation | — |
| Masking | No anatomical ROI-only versus ROI-removed experiment. | Fig. 1; Methods, classifier training | — |

## M. Christodoulou et al. — Segmentation confidence intervals

[Full MICCAI manuscript](https://discovery.ucl.ac.uk/id/eprint/10205944/1/2409.17763v2.pdf) · [Proceedings chapter](https://doi.org/10.1007/978-3-031-72117-5_12)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Evangelia Christodoulou, Annika Reinke, Rola Houhou, Piotr Kalinowski, Selen Erkan, Carole H. Sudre, Ninon Burgos, Sofiène Boutaj, Sophie Loizillon, Maëlys Solal, Nicola Rieke, Veronika Cheplygina, Michela Antonelli, Leon D. Mayer, Minu D. Tizabi, M. Jorge Cardoso, Amber Simpson, Paul F. Jäger, Annette Kopp-Schneider, Gaël Varoquaux, Olivier Colliot and Lena Maier-Hein. *Confidence Intervals Uncovered: Are We Ready for Real-World Medical Imaging AI?* MICCAI 2024, LNCS 15010, pp. 124–132. DOI 10.1007/978-3-031-72117-5_12. Peer-reviewed main conference. | Title page; Springer chapter, “pp 124–132” | WRONG for the last page (132) |
| Method | Survey statistical reporting in MICCAI 2023 segmentation papers. Fit a relationship between mean Dice and its standard deviation using challenge data, then estimate intervals for published results with sufficient reporting. | §§2.1–2.3; Figs. 1–2 | CONFIRMED |
| Data | 221 eligible segmentation papers among 730 MICCAI 2023 papers. Modeling data cover 19 models and ten Medical Segmentation Decathlon tasks, with 189 scores. Validation uses 56 prior MICCAI challenges covering 124 tasks and 213 methods. | §§2.1–2.2; Figs. 1–2 | CONFIRMED for the 221-paper survey |
| Models | The models under review are published segmentation systems. The fitted statistical model is a generalized linear model with a log link and a quadratic function of mean Dice; this study does not train an image classifier. | §2.2 | — |
| Metrics | Dice is a segmentation overlap measure. The inferred 95% confidence interval is based on the estimated between-case SD and sample size. Rank comparison checks whether the second method’s mean lies inside the first method’s interval. | §§2.2–2.3; Fig. 4 | — |
| Results, reporting | 54.8% report no variability measure, and one paper (0.5%) reports confidence intervals. | Fig. 3; §3, RQ1 | CONFIRMED |
| Results, rank comparison | Among 77 eligible comparisons, median inferred 95% CI width is 0.03 Dice, versus a median top-two difference of 0.01; 64.9% have the second mean inside the winner’s interval. The interval is for the winner’s mean, so this is not a paired significance test. | Fig. 4b; §3, RQ3; Supplementary Fig. 5 | CONFIRMED with the 77-comparison denominator |
| Test design | Statistical validation on historical challenge results. No engineered cue distribution. | §§2.2–2.3 | — |
| Location | The segmentation tasks span anatomical regions. No experimental cue placement. | §2.2 | — |
| Masking | No ROI masking intervention in the literature audit. | §§2–3 | — |

## N. Kapoor & Narayanan — Leakage taxonomy

[Full journal text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Sayash Kapoor and Arvind Narayanan. *Leakage and the reproducibility crisis in machine-learning-based science*. Patterns 4(9), 100804 (2023). DOI 10.1016/j.patter.2023.100804. Peer-reviewed journal. | Article header | CONFIRMED |
| Method | Review reports of leakage, organize eight failure modes, and propose model information sheets. Reanalyze civil-war onset prediction to assess published claims after leakage and evaluation errors are corrected. | Fig. 1; taxonomy and civil-war case sections | CONFIRMED |
| Data | 22 studies collectively identify leakage affecting at least 294 papers across 17 fields. The civil-war review starts with 124 papers, selects 12 for detailed evaluation, and reanalyzes four that claim advantages over logistic regression. | Figs. 1 and 3; civil-war case-study methods | CONFIRMED |
| Models | Civil-war comparisons include logistic regression and published ML estimators such as random forest and AdaBoost. No image-classifier architecture is benchmarked in this case study. | Fig. 3; Supplemental Experimental Procedures S3 | CONFIRMED for the logistic-regression comparison |
| Metrics | ROC AUC and uncertainty in comparative AUC estimates; civil-war onset is a rare event and the authors report uncertainty when comparing classifiers. | Fig. 3; case-study discussion | — |
| Results | The taxonomy has eight leakage types. The corrected analyses show no substantive AUC advantage over logistic regression in the four case-study papers that claimed one. In the one case with a remaining positive difference, the gain falls from 0.14 to 0.01 AUC. | Fig. 1; Fig. 3 and civil-war case-study results | CONFIRMED |
| Test design | Separation, prediction timing and dependence between observations are the focus. No deliberately reversed artifact distribution or image test set. | Taxonomy L1–L3; civil-war case methods | — |
| Location | No image-cue placement in the empirical reanalysis. | Civil-war case-study methods | — |
| Masking | No image ROI in the case-study data. | Civil-war case-study methods | — |

## O. Stanley et al. — SimBA, near versus far

[Full JAMIA text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11491635/) · [Released SimBA splits](https://doi.org/10.5683/SP3/0A1IA0)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Emma A.M. Stanley, Raissa Souza, Anthony J. Winder, Vedant Gulve, Kimberly Amador, Matthias Wilms and Nils D. Forkert. *Towards objective and systematic evaluation of bias in artificial intelligence for medical imaging*. JAMIA 31(11), 2613–2621 (2024). DOI 10.1093/jamia/ocae165. Peer-reviewed journal. | Article header | CONFIRMED |
| Method | Use SimBA to simulate matched cohorts of T1 brain MRIs with no bias, a morphological bias near the disease region, or a bias far away. Train separate matched pipelines and compare reweighing, unlearning and group-specific models. | Methods, “Dataset generation” and “Bias mitigation”; Fig. 1 | CONFIRMED |
| Data | 2,002 synthetic MRIs per scenario (1,000 disease and 1,002 non-disease), using the SRI24 atlas/LPBA40 labels and effects estimated with 50 IXI scans. Train/validation/test are 50/25/25. | Methods, data generation and modeling; Supplementary Table S2 | CONFIRMED |
| Models | Five-block 3D CNN with 32 to 512 filters, plus the mitigation variants and bias head for unlearning. | Methods, “Deep learning model” and “Bias mitigation” | — |
| Metrics | Accuracy; ΔTPR and ΔFPR are rates for the bias group minus rates for the non-bias group. Relative disparities subtract the matched no-bias reference disparity. SmoothGrad saliency is summarized by region as a share of total saliency. | Methods, “Model evaluation” and “Quantitative explainability”; Equations 1–2 | — |
| Results | Naïve accuracy for no/near/far bias is 84.81/87.68/86.77%. Relative ΔTPR/ΔFPR are 22.12/20.32 points for near bias and 17.35/11.88 for far bias. Reweighing gives 0.44/−0.69 and −0.80/−0.32 points, respectively, with the most consistent near-zero disparities. | Table 1; Fig. 2 and related Results | CONFIRMED |
| Test design | The intended prevalence is 70% in disease and 30% in non-disease. The released CSVs give 360/500 versus 145/505 in train (72.0%/28.7%), 180/252 versus 70/246 in validation (71.4%/28.5%), and 180/248 versus 68/251 in test (72.6%/27.1%). Each biased model uses the same scenario for train and test. The no-bias model is trained and tested on its own counterpart, with no simulated cue. No inverted test. | Fig. 1; Methods; counts calculated from released split CSVs | CONFIRMED for the 70/30 schedule |
| Location | Disease morphology is in the left insula. Bias is in the left putamen (near) or right postcentral gyrus (far). Both bias regions are distinct from the disease region. | Fig. 1; Methods, bias scenarios | CONFIRMED |
| Masking | Brain-region masks define simulation and saliency measurements. The classifier is trained on the volumes; none of the three mitigations restricts it to the disease ROI. | Methods, modeling, mitigation and saliency | — |

### O2. Stanley et al. — Bias encoding across CNN layers

[Full eBioMedicine text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11700256/)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Emma A.M. Stanley, Raissa Souza, Matthias Wilms and Nils D. Forkert. *Where, why, and how is bias learned in medical image analysis models? A study of bias encoding within convolutional networks using synthetic data*. eBioMedicine 111, 105501 (January 2025 issue; online December 2024). DOI 10.1016/j.ebiom.2024.105501. Peer-reviewed journal. | Article header and publication history | CONFIRMED |
| Method | Generate no-bias, morphology-only, and morphology-plus-intensity counterparts for the same synthetic subjects. Probe intermediate CNN representations with PCA and logistic regression for disease and bias attributes. | Fig. 2; Methods, “Bias encoding analysis” | — |
| Data | 4,303 T1-weighted simulated scans per scenario, 12,909 across three counterparts; 2,153 disease and 2,150 non-disease per scenario. Matched 50/25/25 train/validation/test. | Fig. 2; Table 1; Methods, “Datasets” | — |
| Models | Six-block 3D CNN including the pooling block, with filters increasing from 16 to 256, and logistic-regression probes of principal components. | Methods, classification and bias encoding; Fig. 3 | — |
| Metrics | Disease classification accuracy, bias-subgroup ΔTPR and ΔFPR, and probe accuracy for disease and bias encoding. | Methods, model performance and encoding | — |
| Results | Overall accuracy is 84.9/86.1/88.3% for control, morphology-only and morphology-plus-intensity. Subgroup TPR/FPR gaps are 16.7/19.0 points for the morphology-only comparison and 40.4/37.7 for the both-cues versus neither-cue comparison. The morphology attribute reaches 100% probe accuracy in the aggregate representation. | Tables 2–3; Fig. 4 and Supplementary Table S2 | — |
| Test design | Nominal cue rates are 70/30 by disease status. In the two-cue data, the morphology rate is 724/1,078 versus 355/1,078 in train and 361/530 versus 179/544 in test. The intensity rate is 716/1,078 versus 358/1,078 in train and 356/530 versus 179/544 in test. Each model is evaluated on a held-out subset of its own scenario. No reversed test. | Fig. 2; Table 1 (morphology counts summed over intensity status) | — |
| Location | The localized bias remains in the left putamen. The second cue is a global 20% intensity/contrast change. No near/far or inside/outside relocation in this follow-up. | Fig. 2; Methods, simulated bias effects | — |
| Masking | No anatomical input restriction. Spatial information is used to generate the known simulated effects. | Methods, simulation and CNN | — |

## P. Germani et al. — Mammography foundation-model embeddings

[Journal record](https://doi.org/10.1016/j.media.2026.104258) · [HAL record](https://inserm.hal.science/inserm-05727793) · [Full related 2025 manuscript](https://arxiv.org/pdf/2505.10579) · [Separate MICCAI chapter](https://doi.org/10.1007/978-3-032-05185-1_3)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Elodie Germani, Fatima Zeineddine, Charbel Mourad and Shadi Albarqouni. *When multi-institutional dataset aggregation masks shortcut-like behavior in Foundation Models for medical imaging*. Medical Image Analysis 114, 104258 (2026), DOI 10.1016/j.media.2026.104258. It is a journal publication. The publisher and repository did not deliver a readable full text for this version. | DOI and HAL bibliographic records | Publication metadata verified; full-text claims NOT FOUND |
| Method | The journal experiment could not be checked. A separate 2025 MICCAI paper, available in full as an arXiv manuscript under a different title and with an additional author, uses frozen embeddings and downstream probes across sites. | Preprint §3; journal full text inaccessible | NOT FOUND for the journal |
| Data | The preprint lists CBIS-DDSM 2,844; RSNA 9,594; INbreast 410; MIAS 322; CMMD 2,742; VinDR 3,709; CDD-CESM 1,003; KAU-BCMD 1,774; MMD 745; LBMD 3,090 selected scans. The journal’s counts could not be checked. | Related preprint Table 2 | NOT FOUND for the journal |
| Models | The preprint compares MammoCLIP, MedCLIP, GLoRIA, CLIP and DINOv2 with frozen features. The journal’s exact backbone/probe list could not be checked. | Related preprint §3 and Table 3 | NOT FOUND for the journal |
| Metrics | The preprint reports F1 for diagnosis and includes embedding visualization. The journal’s metric definitions and dataset-identification evaluation could not be checked. | Related preprint §3 and Table 3 | NOT FOUND for the journal |
| Results | The preprint reports MammoCLIP diagnosis F1 of 0.73 ± 0.11 internal, 0.32 ± 0.11 external and 0.65 ± 0.14 pooled. The journal claim that site identity is easier to recover than the clinical label and that pooling hides this could not be verified. | Related preprint Table 3; journal full text inaccessible | NOT FOUND |
| Test design | The preprint compares internal, external and pooled settings. The journal’s construction and proportions could not be checked. | Related preprint §3; journal full text inaccessible | NOT FOUND |
| Location | A spatial intervention could not be verified. | Journal full text inaccessible | NOT FOUND |
| Masking | An ROI mask experiment could not be verified. | Journal full text inaccessible | NOT FOUND |

## Q. Cooke et al. — RoentMod

[Full journal article](https://www.nature.com/articles/s41746-026-02497-6) · [AUC table](https://www.nature.com/articles/s41746-026-02497-6/tables/1)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Lauren H. Cooke, Matthias Jung, Jan M. Brendel, Nora M. Kerkovits, Borek Foldyna, Michael T. Lu and Vineet K. Raghu. *RoentMod: a synthetic chest X-ray modification model to identify and correct image interpretation model shortcuts*. npj Digital Medicine 9, 324 (2026). DOI 10.1038/s41746-026-02497-6. Peer-reviewed journal. | Article header | CONFIRMED |
| Method | Adapt the MIMIC-trained RoentGen diffusion model with image-to-image editing to add a finding to a real CXR. Compare model predictions before and after edits, then train an interpretation model with paired synthetic images. | Fig. 1; Methods, “RoentMod” and model training | CONFIRMED, with MIMIC and NIH roles distinguished |
| Data | Reader experiment: 100 normal MIMIC-CXR images edited for eight findings, giving 800 edits. Stress tests use MIMIC-CXR and NIH ChestX-ray14. The training experiment reports 20,000 NIH patients and 65,315 real radiographs plus a 136,000-scan synthetic training subset of 170,000 generated. The separate NIH cohort description reports 64,628 filtered scans. The other filtered cohorts are MIMIC-CXR (94,067), CheXpert (29,453) and PadChest (88,109). | Fig. 1; Results, reader and augmentation experiments; Methods | WRONG if reader validation is attributed to NIH alone |
| Models | RoentGen’s denoising U-Net/text conditioning and Stable Diffusion image-to-image encoder/decoder; two TorchXRayVision DenseNet-121 models, Ark+ and ElixrB for interpretation stress tests; a multitask DenseNet-121 for augmentation. | Methods, model development and model evaluation | — |
| Metrics | Reader realism and correct-addition rates; paired classifier probability changes reported as percentile shifts; feature-based pFID for generated images; ROC AUC for finding classification. | Methods, evaluations; Figs. 1–2; Table 1 | — |
| Results, realism | 745/800 edits (93.1%) rated realistic. Six findings were correctly added 89–99% of the time. Emphysema (76%) and lung nodules (70%) were excluded from later experiments. Off-target prediction changes are observed on paired images, e.g. for edema predicted on images edited to add a different condition. | Results, “Reader study” and shortcut experiments; Figs. 1–2 | CONFIRMED for the qualified range and shortcut evidence |
| Results, augmentation | Against the NIH TorchXRayVision comparator, NIH edema AUC 0.766→0.901 (+0.135), pneumonia 0.678→0.746 (+0.068), hernia 0.820→0.922 (+0.102). MIMIC hernia 0.660→0.881 (+0.221). Some scores decline, e.g. PadChest mass 0.841→0.833. | Table 1 | CONFIRMED for specific improvements |
| Test design | Matched original and edited images test changes caused by adding a specific finding. The source inputs for the reader experiment are normal. No cohort-wide artifact–label rates or engineered reversed population. | Fig. 1; Methods, stress-test construction | — |
| Location | Findings are generated in plausible locations. Their placement is not a randomized inside/outside-ROI factor. | Fig. 1; Methods, image editing | — |
| Masking | No anatomical input masking or outside-ROI penalty in the reported shortcut experiment. | Methods, interpretation model and image editing | — |

## R. Weng et al. — Thyroid ultrasound across scanners

[Full same-title manuscript](https://arxiv.org/pdf/2207.13765) · [Published article record](https://scholars.duke.edu/publication/1573675)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Jingxi Weng, Benjamin Wildman-Tobriner, Mateusz Buda, Jichen Yang, Lisa M. Ho, Brian C. Allen, Wendy L. Ehieli, Chad M. Miller, Jikai Zhang and Maciej A. Mazurowski. *Deep learning for classification of thyroid nodules on ultrasound: validation on an independent dataset*. Clinical Imaging 99, 60–66 (2023). DOI 10.1016/j.clinimag.2023.04.010. Peer-reviewed journal. Full-text checks use the authors’ same-title manuscript. | Manuscript title page; journal publication record | CONFIRMED |
| Method | Apply a previously trained two-stage thyroid-nodule classifier to a separate hospital cohort and compare performance with four radiologists and across scanners. Examine scanner images and compare model performance with versus without manual nodule contouring. | Methods, algorithm and cohort; Discussion; Figs. 5–6 | CONFIRMED |
| Data | The abstract reports 1,278 training nodules; the original development cohort had 1,631 nodules and an original 99-nodule test. Independent validation: 378 nodules from 320 patients, 756 images, 147 malignant and 231 benign. Main device groups contain 64 ATL HDI 5000, 218 Philips iU22 and 79 GE LOGIQ E9 nodules. | Abstract; Methods, datasets; Table 1 and Fig. 4 | CONFIRMED |
| Models | Faster R-CNN with ResNet-101 locates the calipers. A square ROI with 32-pixel margin around them is resized to 160×160 and passed to a custom six-convolution classifier. | Methods, “Deep learning algorithm” | — |
| Metrics | ROC AUC with 95% CI; bootstrap tests compare AUCs, and Cohen’s κ measures chance-adjusted inter-reader agreement. | Statistical analysis; Tables 2–3 | — |
| Results | Independent AUC 0.69 (95% CI 0.64–0.75), versus 0.87 on the original test. Across the three main devices, model AUC is 0.56, 0.72 and 0.75. GE versus ATL model AUC differs at p = 0.040. The authors report equivalent performance with and without manual contouring, with no separate caliper-ablation AUC printed. | Results; Tables 2–3; Discussion, Figs. 5–6 | CONFIRMED for scanner validation and the reported caliper check |
| Test design | Malignancy prevalence is 39% externally versus 9.9% in original training. The device mix changes from a predominantly Siemens development cohort. There is no assigned train/test caliper–label inversion. | Methods and Discussion, scanner effects | — |
| Location | Caliper styles differed visually by scanner. The authors say their contouring comparison did not explain the scanner performance differences. They did not relocate calipers or quantify their effect with a controlled edit. | Discussion; Figs. 5–6 | CONFIRMED for the qualified conclusion |
| Masking | Yes to localization and ROI cropping, and to the reported contouring comparison. No dedicated caliper-inpainting or controlled inside/outside comparison. Validation also excludes images with extra or deficient calipers. | Methods, classifier; Discussion | — |

## S. Arjovsky et al. — Invariant Risk Minimization

[Full preprint](https://arxiv.org/pdf/1907.02893) · [Released Colored MNIST code](https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Martin Arjovsky, Léon Bottou, Ishaan Gulrajani and David Lopez-Paz. *Invariant Risk Minimization*. arXiv:1907.02893, first posted 2019. DOI 10.48550/arXiv.1907.02893. Preprint for this reference; no conference volume/pages. | Archive title page and version history | CONFIRMED as a preprint |
| Method | Learn a representation supporting the same optimal classifier across environments. Evaluate a practical invariance penalty on synthetic and Colored MNIST tasks. | §§3 and 5 | — |
| Data | Colored MNIST is constructed from MNIST digits with noisy binary labels. The released code splits the source training data into 25,000 and 25,000 training images and 10,000 test images. | §5.2; code, split construction | — |
| Models | Shared multilayer perceptron. The released implementation has two 256-unit hidden layers. | §5.2; code, MLP and default arguments | — |
| Metrics | Binary accuracy, the fraction correctly classified, reported as mean and SD over ten runs. | Table 1 | — |
| Results | Train/test accuracy: ERM 87.4/17.1%; IRM 70.8/66.9%; grayscale oracle test 73.0%. | Table 1 | — |
| Test design | Color agrees with the assigned binary label 80% and 90% in the two training environments, and 10% at test. Equivalently, the three color flip probabilities are 20%, 10% and 90%. The underlying digit-derived label is flipped with probability 25%. The grayscale oracle removes color. | §5.2; Fig. 3; Table 1 | CONFIRMED, with exact rates |
| Location | The digit is globally colored. No location manipulation. | §5.2; Fig. 3 | — |
| Masking | No spatial ROI mask. Grayscale is a color-removal baseline. | §5.2; Table 1 | — |

## T. Bissoto, Valle & Avila — ISIC trap sets

[Full CVPR workshop paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w42/Bissoto_Debiasing_Skin_Lesion_Datasets_and_Models_Not_So_Fast_CVPRW_2020_paper.pdf) · [Released split and annotation files](https://github.com/alceubissoto/debiasing-skin/tree/master/trap-sets)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Alceu Bissoto, Eduardo Valle and Sandra Avila. *Debiasing Skin Lesion Datasets and Models? Not So Fast*. CVPR Workshops 2020, pp. 3192–3201. DOI 10.1109/CVPRW50498.2020.00378. Workshop proceedings. | CVF/IEEE records; paper title page | CONFIRMED |
| Method | Annotate seven visible cues, then select disjoint training and test subsets with opposing artifact–malignancy associations, accounting for co-occurring artifacts. Compare background normalization, occlusion and adversarial debiasing. | §§3.2–3.4 | CONFIRMED |
| Data | ISIC 2018 Tasks 1–2, 2,594 dermoscopy images; external Dermoscopic Atlas 872 and Clinical Atlas 839. In each of five released trap splits the train CSV has 1,822 images (365 malignant/1,457 benign), and the test CSV 772 (154/618). The 259-image validation CSV is a subset of the released train CSV. | §3.1; §§3.4 and 4.2; counts calculated from released CSVs | CONFIRMED for datasets; counts outside the paper marked |
| Models | Inception-v4, ResNet-18 and ResNet-152. The adversarial LNTL implementation predicts the seven shortcut attributes through auxiliary heads. | §§3.3 and 4; Table 3 | — |
| Metrics | Malignancy ROC AUC, with mean ± SD across repeated traps. Other experiments measure how well artifacts can be predicted. | §§4.1–4.2; Tables 2–3 | CONFIRMED |
| Results | ResNet-18 trap-test AUC is 44.7 ± 1.5% original, 62.4 ± 3.3% with normalized background and 51.4 ± 1.7% with LNTL. | Table 3 | — |
| Test design | Yes, each artifact’s association changes direction. Artifacts are dark corners, hair, gel border, gel bubbles, rulers, ink and patches. The paper describes split selection without publishing a single nominal schedule. The exact observed CSV rates are listed below. Background normalization removes exterior pixels but can leave overlying artifacts. | §§3.2–3.4 and 4.2; released CSVs | CONFIRMED for reversal |
| Location | Artifacts can overlap the lesion; the discussion gives hair, gel bubbles and rulers as examples. The paper compares exterior versus lesion visibility through masking. No same artifact is systematically relocated across the lesion boundary. | §§3.2 and 4.4; Fig. 6 | — |
| Masking | Yes. Normalize pixels outside the lesion segmentation mask to a training mean, and occlude lesion pixels with several bounding-box sizes. | §§3.2 and 4.4; Figs. 2 and 6 | CONFIRMED |

**Exact trap-set rates from the released CSVs.** Numerators are artifact-positive image counts. In each of the five splits these full train-CSV and test-CSV rates agree. The train denominators include the listed validation subset. These are realized proportions, not a stated design target.

| Artifact | Train malignant | Train benign | Test malignant | Test benign |
|---|---:|---:|---:|---:|
| Dark corners | 224/365 (61.37%) | 386/1,457 (26.49%) | 10/154 (6.49%) | 340/618 (55.02%) |
| Hair | 148/365 (40.55%) | 1,045/1,457 (71.72%) | 116/154 (75.32%) | 209/618 (33.82%) |
| Gel border | 89/365 (24.38%) | 193/1,457 (13.25%) | 0/154 (0%) | 397/618 (64.24%) |
| Gel bubble | 128/365 (35.07%) | 846/1,457 (58.06%) | 129/154 (83.77%) | 164/618 (26.54%) |
| Ruler | 296/365 (81.10%) | 446/1,457 (30.61%) | 13/154 (8.44%) | 529/618 (85.60%) |
| Ink | 63/365 (17.26%) | 57/1,457 (3.91%) | 0/154 (0%) | 322/618 (52.10%) |
| Patches | 2/365 (0.55%) | 111/1,457 (7.62%) | 0/154 (0%) | 74/618 (11.97%) |

**If the listed validation IDs are withheld**, the fitting pool contains 1,563 images per split and yields the following train-only counts. Each cell lists malignant then benign. All seven directions still flip relative to the test. The paper does not specify whether its training command excludes the validation IDs from the released train CSV.

| Artifact | Split 1 | Split 2 | Split 3 | Split 4 | Split 5 |
|---|---|---|---|---|---|
| Dark corners | 186/298, 338/1,265 | 195/314, 335/1,249 | 188/299, 338/1,264 | 200/315, 334/1,248 | 187/306, 335/1,257 |
| Hair | 125/298, 906/1,265 | 131/314, 892/1,249 | 122/299, 913/1,264 | 125/315, 892/1,248 | 125/306, 898/1,257 |
| Gel border | 74/298, 163/1,265 | 75/314, 159/1,249 | 77/299, 170/1,264 | 69/315, 163/1,248 | 71/306, 172/1,257 |
| Gel bubble | 102/298, 742/1,265 | 111/314, 742/1,249 | 104/299, 739/1,264 | 114/315, 720/1,248 | 106/306, 720/1,257 |
| Ruler | 240/298, 398/1,265 | 251/314, 377/1,249 | 247/299, 382/1,264 | 253/315, 389/1,248 | 245/306, 390/1,257 |
| Ink | 51/298, 50/1,265 | 55/314, 50/1,249 | 53/299, 50/1,264 | 47/315, 49/1,248 | 51/306, 50/1,257 |
| Patches | 2/298, 96/1,265 | 2/314, 90/1,249 | 2/299, 101/1,264 | 2/315, 96/1,248 | 2/306, 96/1,257 |

## U. Puli et al. — NuRD

[Full ICLR paper and appendices](https://cims.nyu.edu/~rajeshr/papers/2022_puli_nurd_iclr.pdf) · [PMC copy](https://pmc.ncbi.nlm.nih.gov/articles/PMC13697191/)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Aahlad Puli, Lily H. Zhang, Eric K. Oermann and Rajesh Ranganath. *Out-of-Distribution Generalization in the Presence of Nuisance-Induced Spurious Correlations*. ICLR 2022, main-track peer-reviewed paper. No proceedings volume/pages or DOI. | Conference PDF title page | CONFIRMED |
| Method | Use reweighting or conditional generation to break a nuisance–label dependence, then distill a representation that avoids nuisance-induced correlation. | §§3–4 | CONFIRMED |
| Data | CheXpert/MIMIC pneumonia mix: 12,446 train and 400 test, with balanced healthy/pneumonia labels. Waterbirds 3,510/400; Colored MNIST 4,851/4,945; Gaussian simulation 10,000/2,000 train/test. | §§5.1–5.4 | — |
| Models | Three-layer CNN for CXR, ResNet-18 for Waterbirds, four-layer CNN for Colored MNIST; generative variants use conditional image models described in the appendix. | Appendix B, implementation details | — |
| Metrics | Binary classification accuracy with mean ± standard error. The main result tables include held-out samples from the original training and nuisance-randomized distributions as well as the shifted test. | §5; Figs. 1–2; Tables 2–3 | — |
| Results | Reversed CXR: ERM 37 ± 1%, reweighting-NuRD 61 ± 1%, generative-NuRD 41 ± 2%. Reversed Waterbirds: ERM 66 ± 2%, reweighting 83 ± 2%. Reversed Colored MNIST: 10 ± 0% versus 75 ± 0% for ERM and reweighting. | Figs. 1–2; Table 3 | — |
| Test design | CXR train: 90% of pneumonia from MIMIC and 90% of healthy from CheXpert. Test: 90% of pneumonia from CheXpert and 90% of healthy from MIMIC. Waterbirds and Colored MNIST also flip 90% alignment to 10%. For CXR and Waterbirds the appendix sweeps test misalignment 0.50, 0.70 and 0.90 against train misalignment 0.10. The Gaussian simulation shifts a distribution parameter from +0.5 to −0.9. | §§5.1–5.4; Appendix B.1, Fig. 4 | CONFIRMED |
| Location | CXR source signatures can affect the whole image. The 4-pixel border of the 32×32 image serves as the measured nuisance. Waterbirds uses a border too. No controlled cue relocation. | §§5.3–5.4; Figs. 1–2 | — |
| Masking | Central-crop ERM is an explicit diagnostic baseline, 39 ± 0.5% versus full-image ERM 37 ± 1% on CXR. NuRD itself uses full images at inference and uses the border as nuisance input during training. No anatomical segmentation mask in the CXR pipeline. | §5.4; Appendix B.1, crop baseline; Fig. 2 | CONFIRMED for the crop baseline |

## V. Makar et al. — Causal shortcut removal

[Full AISTATS proceedings paper](https://proceedings.mlr.press/v151/makar22a/makar22a.pdf) · [Waterbirds sweep code](https://github.com/mymakar/causally_motivated_shortcut_removal/blob/master/waterbirds/main.py)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Maggie Makar, Ben Packer, Dan Moldovan, Davis Blalock, Yoni Halpern and Alexander D’Amour. *Causally Motivated Shortcut Removal Using Auxiliary Labels*. AISTATS 2022, PMLR 151, pp. 739–766. Peer-reviewed main conference. No DOI in the proceedings record. | PMLR title page and citation | CONFIRMED |
| Method | Reweight examples and penalize dependence between the learned representation and the auxiliary shortcut label, using weighted maximum mean discrepancy. Evaluate performance over changes in the shortcut association. | §§3–5 | CONFIRMED |
| Data | Waterbirds composites draw on CUB and Places; medical experiment uses CheXpert pneumonia labels and sex. The paper describes resampling for 20 Waterbirds and five CheXpert simulations; exact final image counts per split were not found. | §§5.1–5.2; Appendix F | NOT FOUND for exact experimental split counts |
| Models | ImageNet-pretrained ResNet-50 for Waterbirds and DenseNet-121 for CheXpert. | §§5.1–5.2 | — |
| Metrics | AUROC (ROC area), with mean and standard error; Brier score (mean squared probability error) and logistic loss in supplementary robustness comparisons. | Fig. 2; Table 1; Appendix F | — |
| Results | CheXpert shifted/unchanged AUROC: weighted MMD 0.75/0.85, unweighted L2 0.62/0.71, weighted L2 0.69/0.86. | Table 1 | — |
| Test design | **CheXpert:** train P(pneumonia given female) = P(no pneumonia given male) = 0.90, with disease prevalence 0.30. Test either uses the same conditionals or makes both 0.50. **Waterbirds:** train alignment is 0.50 or 0.90. The test sweep includes 0.10, 0.50 and 0.90, so the 0.90-trained model is tested at a reversed endpoint. | §§5.1–5.2; Fig. 2; released Waterbirds sweep code | CONFIRMED for Waterbirds reversal; WRONG if asserted for CheXpert |
| Location | No artifact relocation relative to an anatomical ROI. Waterbirds uses background composites, and CheXpert uses sex as a metadata-defined proxy. | §§5.1–5.2 | — |
| Masking | Bird segmentation is used to compose the benchmark. The medical experiment does not mask to lungs. | §5.1; Appendix F | — |

## W. Jabbour et al. — Exploiting and preventing CXR shortcuts

[Full MLHC proceedings paper](https://proceedings.mlr.press/v126/jabbour20a/jabbour20a.pdf)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Sarah Jabbour, David Fouhey, Ella Kazerooni, Michael W. Sjoding and Jenna Wiens. *Deep Learning Applied to Chest X-Rays: Exploiting and Preventing Shortcuts*. Machine Learning for Healthcare 2020, PMLR 126, pp. 750–782. Peer-reviewed main conference. No DOI in the proceedings record. | PMLR title page and citation | CONFIRMED |
| Method | Select training cases so that a demographic or treatment attribute perfectly tracks the disease, then test on data independent of that attribute. To mitigate, learn CXR features on unskewed pneumonia data and refit the target classifier. | §§4 and 5.1.2; Fig. 1 | CONFIRMED |
| Data | Institutional acute hypoxemic respiratory failure (AHRF) cohort: 1,296 patients. Also uses MIMIC-CXR and CheXpert. In the skew experiment, test size is 111 patients for AHRF CHF and 1,000 for each pneumonia dataset. Training sizes depend on shortcut and cohort, e.g. sex rows use 479/16,641/6,412 cases across the three cohorts. | §4.2; §5.1.2; Appendix Table 5 | — |
| Models | DenseNet-121 encoder and final linear classifier. Transfer experiments initialize from CXR models and compare adapting the last layer versus more layers. | §§4.1–4.2; Fig. 1 | — |
| Metrics | Disease AUROC and AUROC against the shortcut label, with 95% CIs from 1,000 bootstrap draws. | §4.3 | — |
| Results | With sex-skewed versus unskewed training, clean-test AUC falls from 0.81→0.66, 0.78→0.53 and 0.82→0.63 across the three cohorts. In the age-skewed CHF comparison, last-layer transfer improves AUC 0.66 (95% CI 0.54–0.77)→0.84 (0.73–0.92). | Fig. 4; Appendix Table 7 | — |
| Test design | A “1:1” training/validation link means shortcut and disease labels agree perfectly by selection. Tests are resampled so the disease is independent of the attribute. The synthetic −1 to +1 correlation sweep applies to source pretraining, not to oppositely skewed medical test sets. | §5.1.2; Fig. 4; synthetic transfer experiment | WRONG if presented as opposite-direction train and test skew |
| Location | Attributes are demographic, clinical or synthetic image-wide noise. Saliency is examined, with no controlled relocation relative to a disease ROI. | §§5.1–5.2; Figs. 2–3 | — |
| Masking | Standard image crops/augmentation and saliency. No anatomy-only input restriction. | §4.2; Figs. 2–3 | — |

## X. Additional medical-imaging precedents

This is a targeted full-text search of 2018–2026 publications. The tables distinguish deliberately inverted populations, deliberately incompatible one-class or paired edits, spatially varied cues, and studies using only independence at test. A main-track or journal result has the publication status shown in its row. Workshops are labeled explicitly.

### X1. Corbetta et al. — Cyclic artifact inversion in two medical modalities

[Full MICCAI paper](https://papers.miccai.org/miccai-2026/paper/5420_paper.pdf) · [Proceedings record](https://doi.org/10.1007/978-3-032-38062-3_6)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Valentina Corbetta, Antonio Portaluri, Muzhen He, Daniël Boeke, Regina Beets-Tan, Veronica Lachi, Elisabeth Wetzer, Robert Jenssen, Wilson Silva and Kristoffer Wickstrøm. *Beyond Clean Test Sets: Spurious Correlations in Medical Vision-Language Models and the Role of Concept Supervision*. MICCAI 2026, LNCS 16879, pp. 56–66. DOI 10.1007/978-3-032-38062-3_6. Peer-reviewed main conference. | Title page; proceedings record | CONFIRMED |
| Method | Inject simulated acquisition artifacts at controlled prevalence and evaluate five models with differing clinical concept supervision under artifact removal and inversion. | §§2–3; Table 1 | CONFIRMED |
| Data and models | FGADR Segmentation Set, 1,842 fundus images; EMBED subset, 100 mammography cases/400 images. RetCLIP ViT-B/16 and Mammo-CLIP EfficientNet-b2, with LoRA and five alternatives from image-only to concept bottleneck. | §3; Table 1 | CONFIRMED |
| Metrics and results | Balanced accuracy (mean class recall), averaged over five splits. Inverted fundus VLM baseline declines from 0.63 at p = 0 to 0.53 at p = 25%. Most models approach 0.20 at p = 100%. Inverted mammography approaches 0.33 at p = 100%. | §3, Metrics; Fig. 2 | CONFIRMED |
| Test design | Four fundus overlays for five classes and two mammography overlays for three classes. The lowest-severity class stays clean in training. Among eligible classes the fraction overlaid is p = 0, 25, 50, 75 or 100%. Clean testing removes all overlays. Inversion cyclically shifts the class associations and is evaluated over the same p sweep. A separate correlated test curve and a full test contingency table are not reported. | §2, “Synthetic Confounder Design” and “Evaluation Protocols”; Fig. 2 | CONFIRMED |
| Location and masking | Clinically plausible overlay locations and lesion/concept annotations. No systematic same-overlay move across an anatomical ROI or ROI-masked classifier input. | §§2–3; Fig. 1 | CONFIRMED for the stated scope |

### X2. Drenkow et al. — G-AUDIT

[Full journal article](https://www.nature.com/articles/s41746-026-02807-y)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Nathan Drenkow, Mitchell Pavlak, Keith Harrigian, Ayah Zirikly, Adarsh Subbaswamy, Mohammad Mehdi Farhangi, Nicholas Petrick and Mathias Unberath. *Detecting dataset bias in medical AI using a generalized and modality agnostic auditing approach*. npj Digital Medicine 9, 654 (2026). DOI 10.1038/s41746-026-02807-y. Peer-reviewed journal. | Article title page | CONFIRMED |
| Method | Score candidate shortcuts by their statistical relation to the label and their recoverability from the input. Calibrate potential performance loss by inserting an artificial attribute and evaluating a maximally adverse version. | Methods, “Utility and detectability” and “Bounding Performance Risk” | CONFIRMED |
| Data and models | ISIC 2019 skin lesion data, 25,331 images. Swin-T task classifier and ResNet-18 surrogate, with surrogate architecture sensitivity analysis. Separate experiments cover notes and tabular ICU data. | Methods, “Skin Lesion Classification”; Table 1 | CONFIRMED |
| Metrics and results | Utility is adjusted normalized mutual information between attribute and label. Detectability is mutual information between the attribute and its surrogate prediction. ISIC AUC falls by approximately 0.2 in the illustrated worst-case calibration at observed utility levels. | Fig. 2b; Methods | CONFIRMED |
| Test design | Define the synthetic attribute as A = Y initially and randomly flip N of the attribute labels to sweep utility (alignment before rendering is 1 − N/n). On the worst-case test, A = 1 − Y for every case. No single fixed training percentage or per-class rendered watermark prevalence is reported. | Methods, “Bounding Performance Risk”; Fig. 2b | CONFIRMED for the reversed synthetic attribute |
| Location and masking | The image example uses a fixed-position visual cue. There is no ROI-relative relocation or anatomy-only input. | Methods, “Bounding Performance Risk” | CONFIRMED for the stated scope |

### X3. Unnikrishnan et al. — SilverLining

[Full WACV paper](https://openaccess.thecvf.com/content/WACV2026/papers/Unnikrishnan_SilverLining_Data-First_Mitigation_of_Spatial_and_Spectral_Shortcuts_Without_Introducing_WACV_2026_paper.pdf) · [Proceedings DOI](https://doi.org/10.1109/WACV61042.2026.00132) · [Released PneumoniaMNIST loader](https://github.com/theidentity/SilverLining_WACV2026/blob/main/dataloaders/pnu_dataset.py)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Balagopal Unnikrishnan, Michael Brudno and Chris McIntosh. *SilverLining: Data-First Mitigation of Spatial and Spectral Shortcuts Without Introducing New Confounders*. WACV 2026, pp. 1294–1303. DOI 10.1109/WACV61042.2026.00132. Peer-reviewed main conference. | CVF title page; proceedings citation | CONFIRMED |
| Method | Detect spatially shortcut-bearing patches and mitigate by consistent patch masking. Use spectral preprocessing for shortcuts such as intensity variations. | §3; Figs. 1–2 | CONFIRMED |
| Data and models | Controlled PneumoniaMNIST (the supplement reports 10,000 images for runtime benchmarking), MIMIC-CXR→CheXpert, and HyperKvasir colonoscopy; Animals is CIFAR-10-derived. ResNet-18 patch features and DenseNet-121 for the clinical CXR arm. | §§3–4; Table 1 | CONFIRMED |
| Metrics and results | Counter-shortcut AUROC on PneumoniaMNIST, mean ± SD across three runs. Spatial experiment is 0.02 ± 0.03 for ERM and 0.93 ± 0.00 after spatial mitigation. With both cues, ERM 0.02 versus joint mitigation 0.94. | Table 1; §4.2 | CONFIRMED |
| Test design | PneumoniaMNIST test has opposite position/intensity shortcuts. The paper states the reversal without giving percentages. The released PneumoniaMNIST loader assigns the class-congruent cue to 95% of training images and leaves 5% clean; at test it assigns a cue to 50%, with the mapping matched on the regular test and flipped on the counter-shortcut test. The other 50% stay clean. In each cue-bearing arm, the mapping is 100% aligned or 100% flipped conditional on the cue being applied. In HyperKvasir the natural green marker occurs in 40.57% of non-polyp and 0.11% of polyp images, with counterfactual additions to previously unmarked images. | §§4.1–4.3; Table 1; released loader, get_dataloaders and apply_bias | CONFIRMED for reversal; rates from code |
| Location and masking | Yes for left-versus-right placement of markers and patch-level shortcut removal. Position is defined in the image rather than relative to an anatomical ROI. | §3.1; Fig. 2; §4.1 | CONFIRMED |

### X4. Yang et al. — KnoBo

[Full NeurIPS proceedings paper](https://papers.neurips.cc/paper_files/paper/2024/file/a4e683f0ce6b91e7fbdae9d32642d88f-Paper-Conference.pdf) · [Full PMC copy](https://pmc.ncbi.nlm.nih.gov/articles/PMC12064272/)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Yue Yang, Mona Gandhi, Yufei Wang, Yifan Wu, Michael S. Yao, Chris Callison-Burch, James C. Gee and Mark Yatskar. *A Textbook Remedy for Domain Shifts: Knowledge Priors for Medical Image Analysis*. NeurIPS 2024, Advances in Neural Information Processing Systems 37, pp. 90683–90713. DOI 10.52202/079017-2879 in the proceedings distributor record. Peer-reviewed main conference. | Title page; proceedings metadata | CONFIRMED |
| Method | Generate clinically interpretable concepts from retrieval-augmented medical sources, ground them in images and use a concept bottleneck with priors. Test against baseline architectures on reversed confounds. | §§4–6 | CONFIRMED |
| Data and models | NIH-CXR, CheXpert and ISIC supply 10 confounded tasks. Example train/validation/test sizes are NIH–CheXpert 5,000/1,000/1,000, ISIC-hospital 2,400/280/280 and NIH projection 3,000/300/300. Ten additional tasks use random splits. ViT-L/14 CLIP features, DenseNet-121 and concept-bottleneck baselines. | §5; Appendix A, Tables 7–8 | CONFIRMED |
| Metrics and results | Accuracy on matched validation and inverted test, their mean and their absolute gap. On NIH–CheXpert, ViT-L/14 scores 99.9% matched and 0.1% inverted, versus 91.6% and 52.3% for KnoBo. On ISIC-hospital the inverted result is 10.0% versus 77.5%. | Table 1; §5.2 | CONFIRMED |
| Test design | Resample the two target classes from opposing groups and exchange the group assignments at test. Explicitly exclusive sources imply 100%→0% match for NIH–CheXpert and ISIC-hospital. Other inverted covariates include sex, age, race, projection, skin color and lesion body site. Validation has the training assignments. Random-split benchmarks are separate. | §1; Fig. 1; §5.1; Appendix A | CONFIRMED |
| Location and masking | Studies patient position during radiography and lesion body site as naturally occurring group cues. It does not move a visual artifact relative to a lesion mask or use ROI-only inputs. | §5.1; Appendix A; §4 | CONFIRMED for the stated scope |

### X5. Müller et al. — 95/5 inversion on CheXpert

[Full paper](https://arxiv.org/pdf/2407.18792) · [Workshop proceedings](https://link.springer.com/chapter/10.1007/978-3-031-73290-4_6)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Sarah Müller, Louisa Fay, Lisa M. Koch, Sergios Gatidis, Thomas Küstner and Philipp Berens. *Benchmarking Dependence Measures to Prevent Shortcut Learning in Medical Imaging*. MLMI workshop at MICCAI 2024, LNCS 15242, pp. 53–62. DOI 10.1007/978-3-031-73290-4_6. Published online October 2024 and cited by Springer as 2025. Workshop proceedings. | Title page; Springer citation | CONFIRMED |
| Method | Compare dependence penalties that decouple representations for disease and a correlated attribute, using the same train, matched validation, inverted and balanced evaluation construction. | §§2.2–2.4 | CONFIRMED |
| Data and models | Morpho-MNIST and CheXpert pleural effusion/sex. CheXpert provides 100,014 development-pool and 2,183 held-out-pool images before biased subsampling. Correlated training cohort 21,892. ResNet-50 for CXR. | §§2.1–2.4; Table 2 | CONFIRMED |
| Metrics and results | Primary-task AUROC for matched, inverted and balanced splits. CXR baseline 98.5/45.8/80.5%, compared with MINE 93.8/87.7/92.4%. | Table 5 | CONFIRMED |
| Test design | Training has 10,400 images in each aligned cell and 546 in each opposite cell, i.e. approximately 95/5. Matched validation has 95/5 alignment, inverted test 5/95 and balanced test 50/50. The medical cue is sex, not an inserted overlay. | §2.3; Table 2 | CONFIRMED |
| Location and masking | Fixed image preprocessing crop. No controlled cue placement or anatomical ROI mask. | §§2.1 and 2.4 | CONFIRMED for the stated scope |

### X6. Fay et al. — MIMM-X

[Full paper](https://arxiv.org/pdf/2511.22990) · [Workshop proceedings](https://doi.org/10.1007/978-3-032-05870-6_10) · [Released sampling code](https://github.com/lab-midas/MIMM-X/blob/main/Dataset/loaders.py) · [Pleural-effusion configuration](https://github.com/lab-midas/MIMM-X/blob/main/config/config_pleural.yml)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Louisa Fay, Hajer Reguigui, Bin Yang, Sergios Gatidis and Thomas Küstner. *MIMM-X: Disentangling Spurious Correlations for Medical Image Analysis*. FAIMI workshop at MICCAI 2025, LNCS 15976, pp. 94–103. DOI 10.1007/978-3-032-05870-6_10. Workshop proceedings. | Title page; Springer citation | CONFIRMED |
| Method | Minimize mutual information between feature subspaces for the target and multiple nuisance attributes, then cross-predict labels to check their separation. | §§2–3 | CONFIRMED |
| Data and models | NAKO/UKB brain MRI, 2,896/244/538/1,760 train/matched validation/inverted/balanced. CheXpert experiments use 4,126/408/2,692/4,432 and 9,850/1,114/4,064/2,204 respectively. Four-layer MRI encoder and DenseNet-121 for CXR. | §3, Experiments 1–3; Fig. 2 | CONFIRMED |
| Metrics and results | Target accuracy and cross-prediction accuracy from the opposite feature subspace. Inverted brain MRI accuracy 82.8% versus 27.2% baseline. Inverted CXR accuracy for MIMM-X is 70.4% and 70.2% in the two experiments. | Tables 1–3; §4 | CONFIRMED |
| Test design | Experiments 1–2 target 90% cue alignment in training per class, matched validation, inverse test and a balanced test. The MRI diagram gives exact sex-aligned counts per class 1,304/1,448 training, 110/122 validation, 26/269 inverted and 440/880 balanced. Site is also inverted. Summing over sex gives the same site-alignment counts as for sex in the MRI diagram. In CheXpert experiment 3 the contingency cells for the imposed sex cue are 0.45/0.05 and 0.05/0.45 (90/10 within class). Age and co-disease there retain natural associations. The released pleural-effusion configuration samples the imposed sex cue at nominal 90/10 matched, 10/90 inverted and 50/50 balanced. Exact realized CXR cell counts are not printed. | Fig. 2a–c; §3; released configuration and sampling code | CONFIRMED |
| Location and masking | Metadata/source cues, no cue relocation. Central axial MRI slice selection; no anatomy-only ROI mask. | §3, experiment descriptions | CONFIRMED for the stated scope |

### X7. Boland et al. — Fixed and random synthetic positions

[Full MIDL proceedings paper](https://raw.githubusercontent.com/mlresearch/v250/main/assets/boland24a/boland24a.pdf) · [Proceedings record](https://proceedings.mlr.press/v250/boland24a.html)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Christopher Boland, Keith A. Goatman, Sotirios A. Tsaftaris and Sonia Dahdouh. *There Are No Shortcuts to Anywhere Worth Going: Identifying Shortcuts in Deep Learning Models for Medical Image Analysis*. MIDL 2024 full-paper track, PMLR 250, pp. 131–150. No DOI in the proceedings record. | PMLR title page and record | CONFIRMED |
| Method | Compare the prediction depths of models trained with and without synthetic shortcuts, and localize shortcut-sensitive layers using differences between depth distributions. | §§2–4; Fig. 1 | CONFIRMED |
| Data and models | Waterbirds, CheXpert pneumothorax subset and ISIC 2017 benign/malignant. Class-balanced subsampling; exact medical split counts not given. ResNet-34, DenseNet-121, InceptionNet and VGG-16. | §3.3 | CONFIRMED |
| Metrics and results | Prediction depth is the earliest layer after which k-nearest-neighbor predictions stay consistent. Layerwise KL divergence compares distributions. Fixed square on Waterbirds shifts the depth distribution across backbones (e.g. VGG-16 p = 6.497×10⁻⁵⁴ in Table 1). In the CheXpert DenseNet-121 experiment, the simplest square reduces AUC by 10–15% across learning rates, versus 3–7% for the complex cue. | §§2, 4.1 and 4.3; Tables 1–2 | CONFIRMED |
| Test design | Shortcut model training uses a perfectly label-correlated cue. Test cues are balanced across labels; the paper evaluates shortcut-bearing and shortcut-free examples within that design. No inverted population. | §§3.1 and 4.1 | CONFIRMED |
| Location and masking | Yes, compare a red square at fixed versus random image positions, and jitter a curved drain-like cue. The positions are not defined by an anatomy or lesion mask. No ROI restriction. | §3.2; Figs. 2 and 5 | CONFIRMED |

### X8. Nauta et al. — Color-patch insertion and removal

[Full journal article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8774502/)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Meike Nauta, Ricky Walsh, Adam Dubowski and Christin Seifert. *Uncovering and Correcting Shortcut Learning in Machine Learning Models for Skin Cancer Diagnosis*. Diagnostics 12(1), 40 (2022 issue, online 2021). DOI 10.3390/diagnostics12010040. Peer-reviewed journal. | Article header | CONFIRMED |
| Method | Inpaint existing color calibration patches in benign cases, paste patches into malignant cases, compare predictions, and retrain after inpainting training patches. | §§3.1–3.5 | CONFIRMED |
| Data and models | ISIC Archive: 2,506 malignant and 19,298 benign; random 80/20. VGG-16 classifier and generative multi-column CNN (GMCNN) for inpainting. | §§3.1–3.2 | CONFIRMED |
| Metrics and results | Decision flip rate and mean absolute/negative probability changes, plus sensitivity and specificity. Malignant patch insertion changes decisions in 69.5% of analyzed cases for the original model versus 33.5% after retraining. Benign patch removal gives 21.1% versus 0.6%. | Tables 1 and 3; §§4.2–4.4 | CONFIRMED |
| Test design | 8,973/19,298 benign cases (46.5%) and 0/2,506 malignant cases contain the artifact in the source corpus. The held-out analyses remove patches from originally patched benign cases or add patches to malignant cases. In the inserted malignant subset every image has the patch. The effects are reported on these paired subsets separately, without a jointly sampled reversed test population. | §§3.1–3.4; Tables 1 and 3 | CONFIRMED for paired interventions |
| Location and masking | Inpainting or insertion changes the patch region. No assignment of the same patch inside versus outside the lesion, and no classifier restricted to the lesion mask. | §§3.2–3.3; Fig. 3 | CONFIRMED for the stated scope |

### X9. Pérez-García, Bond-Taylor et al. — RadEdit

[Full ECCV proceedings paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01923.pdf) · [Springer record](https://doi.org/10.1007/978-3-031-73254-6_21)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Fernando Pérez-García, Sam Bond-Taylor, Pedro P. Sanchez, Boris van Breugel, Daniel C. Castro, Harshita Sharma, Valentina Salvatelli, Maria T.A. Wetscherek, Hannah Richardson, Matthew P. Lungren, Aditya Nori, Javier Alvarez-Valle, Ozan Oktay and Maximilian Ilse. *RadEdit: Stress-Testing Biomedical Vision Models via Diffusion Image Editing*. ECCV 2024 main conference, LNCS 15070, pp. 358–376, Springer citation year 2025. DOI 10.1007/978-3-031-73254-6_21. | Title page; Springer citation | CONFIRMED |
| Method | Use diffusion and complementary edit/keep masks to change disease evidence while preserving image shortcuts. Compare classifiers trained under differing degrees of confounding. | §§3–4; Figs. 2–3 | CONFIRMED |
| Data and models | BIMCV and MIMIC-CXR for the acquisition stress test, including 2,774 edited negatives. CANDID-PTX for the drain test, with 16 real rare negatives and 628 edited in the text (629 in Table 2). ResNet-50 classifier and a joint segmentation/classification U-Net for the stronger PTX model. | §§4.4–4.5; Tables 1–2 | CONFIRMED with the published count discrepancy |
| Metrics and results | Accuracy on biased versus adversarially edited sets, mean ± SD. Weak COVID-19 classifier 99.1→5.5%. Weak pneumothorax classifier 93.3→17.9% when tested on edited drain-bearing negatives. | Tables 1–2 | CONFIRMED |
| Test design | In the confounded COVID-19 training set, positives come from BIMCV+ and negatives from MIMIC-CXR. The edited test contains only negatives, all retaining the BIMCV+ acquisition signature. In CANDID-PTX about 1% of source images have a drain without PTX; the edited test contains only PTX-negative images with the drain preserved. These are one-class conflicts with training associations. | §§4.4–4.5; Tables 1–2 | CONFIRMED for partial reversal |
| Location and masking | Masks edit disease while preserving the shortcut, e.g. the drain. A strong model has a segmentation auxiliary task. No controlled same-artifact move across the ROI. | §§3 and 4.5; Fig. 3 | CONFIRMED |

### X10. Weng, Pegios et al. — FastDiME, a related counterfactual control

[Full ECCV proceedings paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11674.pdf) · [Springer record](https://doi.org/10.1007/978-3-031-73016-0_20)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Nina Weng, Paraskevas Pegios, Eike Petersen, Aasa Feragen and Siavash Bigdeli. *Fast Diffusion-Based Counterfactuals for Shortcut Removal and Generation*. ECCV 2024 main conference, LNCS 15144, pp. 338–357, Springer citation year 2025. DOI 10.1007/978-3-031-73016-0_20. | Title page; Springer citation | CONFIRMED |
| Method | Generate image pairs that differ in a shortcut, using a mask to constrain diffusion edits. Compare predictions across matched, balanced and individually flipped examples. | §§3–4; Fig. 4 | CONFIRMED |
| Data and models | CheXpert cardiomegaly/pacemaker; NIH pneumothorax/drain; ISIC 2018 malignancy/ruler; also CelebA. ResNet-18 for main shortcut-detection tests, DDPM generator and a U-Net encoder for shortcut guidance. | §§4.1 and 4.3 | CONFIRMED |
| Metrics and results | Target AUC and mean absolute deviation in scores after editing. Under 100% training alignment, CheXpert AUC is 0.98 matched, 0.58 balanced and 0.63 counterfactual balanced; ISIC is 0.98, 0.70 and 0.79. | Table 4; §4.3 | CONFIRMED |
| Test design | The design uses k = 100%, 75% or 50% shortcut prevalence in the positive class and 100−k% in the negative class (Fig. 4), for training and the matched tests. The balanced test has equal cells. Each shortcut is flipped on that balanced test, so the intended aggregate remains balanced. The medical experiments do not evaluate a population-level inverse of biased training. | §3.2, steps 1–2; Fig. 4 | CONFIRMED for individual flips; no population reversal |
| Location and masking | Uses localized edit masks to modify a cue in place. No same-cue ROI-crossing or anatomy-only classifier input. | §3.1; Figs. 2–3 | CONFIRMED |

### X11. DeGrave, Janizek & Lee — Paired marker swap and projection shift

[Full publisher PDF](https://www.nature.com/articles/s42256-021-00338-7.pdf)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Alex J. DeGrave, Joseph D. Janizek and Su-In Lee. *AI for radiographic COVID-19 detection selects shortcuts over signal*. Nature Machine Intelligence 3, 610–619 (2021). DOI 10.1038/s42256-021-00338-7. Peer-reviewed journal. | Article title page | CONFIRMED |
| Method | Train source-confounded models, cross-test on external cohorts and edit selected images to test cues identified with saliency and CycleGAN. | Figs. 1–3; Methods, “Experimental validation of feature attributions” | CONFIRMED |
| Data and models | Dataset I: ChestX-ray14/GitHub-COVID, 112,528 source images. Dataset II: PadChest/BIMCV-COVID-19+, 97,866. Also evaluate same-system BIMCV negatives. DenseNet-121 main model with secondary architectures. | Fig. 1b; Methods, “Datasets and preprocessing” | CONFIRMED |
| Metrics and results | ROC AUC, mean ± SD, and change in output log odds on paired edits. Dataset I model 0.992→0.76 internally to externally, dataset II model 0.995→0.70. The two swapped-marker examples change by −5.63 and +13.85 log odds. | Figs. 1c and 3a | CONFIRMED |
| Test design | Each combined cohort intentionally confounds case status with source; external sources differ. Projection association changes naturally across cohorts. Figure 3 swaps markers between one positive and one negative case and compares with random-patch controls. No cohort-wide prescribed inverse marker schedule. | Figs. 1b, 3a and 4; Results, “Shortcuts have a variable effect on generalization” | CONFIRMED for paired swap and natural shift |
| Location and masking | Move shoulder regions to upper corners in an example and exchange laterality marks between paired images. No comparison of the same marker inside and outside a defined anatomical ROI, and no ROI-only classifier. | Fig. 3; Methods, validation of feature attributions | CONFIRMED |

### X12. Winkler et al. — Surgical ink and cropping

[Full JAMA Dermatology article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6694463/)

| Point | What the paper says | Quote or location | Our claim |
|---|---|---|---|
| Reference | Julia K. Winkler, Christine Fink, Ferdinand Toberer, Alexander Enk, Teresa Deinlein, Rainer Hofmann-Wellenhof, Luc Thomas, Aimilios Lallas, Andreas Blum, Wilhelm Stolz and Holger A. Haenssle. *Association Between Surgical Skin Markings in Dermoscopic Images and Diagnostic Performance of a Deep Learning Convolutional Neural Network for Melanoma Recognition*. JAMA Dermatology 155(10), 1135–1141 (2019). DOI 10.1001/jamadermatol.2019.1735. Peer-reviewed journal. | Journal article header | CONFIRMED |
| Method | Use a fixed pretrained classifier on matched unmarked, marked and cropped image sets. Mark benign lesions in vivo and superimpose analogous ink on melanoma images. | Methods, image acquisition and modifications; Fig. 1 | CONFIRMED |
| Data and models | Heidelberg test series has 130 lesions per set, 107 benign nevi and 23 melanomas. Inception-v4 Moleanalyzer-Pro was previously trained on over 120,000 dermoscopic images. | Methods and Results, lesion characteristics | CONFIRMED |
| Metrics and results | ROC AUC, sensitivity and specificity at the stated 0.5 threshold. Specificity on unmarked/marked/cropped sets is 84.1%/45.8%/97.2%; AUROC is 0.969/0.922/0.993. Mean predicted malignancy for benign cases is 0.16/0.54/0.03. | Figs. 2 and 5; Results | CONFIRMED |
| Test design | Cue present in 0% of both classes in the unmarked condition and 100% of both classes in the marked condition. Training artifact–label proportions are not given. No correlated training manipulation or reversed test. | Methods, image acquisition and modifications | CONFIRMED for paired cue presence/absence |
| Location and masking | Ink is placed adjacent to nevi, with analogous synthetic markings for melanomas. Cropping reduces background and focuses on the lesion. This tests adjacent-marker effects and cropping without placing the same marker inside the lesion. | Methods and Discussion; Fig. 1 | CONFIRMED |

**Related controls.** Sun, Koch and Baumgartner (MICCAI 2023 main conference) varied the fraction of positive CXRs receiving a tag, colored line, or occlusion (0%, 20%, 50%, 80%, 100%, with none in negatives). The different cue types occupy different image areas. The matched test and the clean evaluation do not systematically move a single cue across a disease ROI. Rieger et al. (ICML 2020 main conference) penalize ISIC patch reliance and test on images with or without the patch. Their reversal is in Colored MNIST. These studies support the interpretation and mitigation discussion. [Sun et al., full paper](https://arxiv.org/pdf/2307.12344) · [Rieger et al., full paper](https://proceedings.mlr.press/v119/rieger20a/rieger20a.pdf)

## Summary table for S–X

“Yes” denotes an engineered inverted test. “Partly” denotes inversion in only a nonmedical arm of a mixed study, a one-class stress test, a paired edit, or a naturally changing association. The masking column specifies whether masks change the classifier input or serve another purpose.

| Paper | Built a reversed test? | Varied location? | Used masking? | Medical images? |
|---|---|---|---|---|
| S · Arjovsky, IRM | **Yes.** Color agreement train 80% and 90%, test 10%. | No | No spatial mask; grayscale oracle | No |
| T · Bissoto, trap sets | **Yes.** All seven associations flip. Example ruler, malignant/benign 81.10/30.61%→8.44/85.60%. | No controlled relocation; observed overlap | Yes, exterior normalization and lesion occlusion | Yes, dermoscopy |
| U · Puli, NuRD | **Yes.** Original CXR source agreement 90% train/matched held-out →10% inverted; intermediate 30% and independent 50% also tested. Waterbirds and Colored MNIST 90%→10%. | No; CXR source signatures can be global | Yes, center-crop baseline; NuRD uses full input | Yes, CXR, and nonmedical |
| V · Makar | **Partly.** Waterbirds 90%→10% in the sweep. CheXpert 90%→90% or 50% for disease conditional on sex. | No | Bird compositing only | Yes, CXR, and nonmedical |
| W · Jabbour | **No.** Perfect training association → independence on test. | No | No anatomical ROI mask | Yes, CXR |
| X1 · Corbetta | **Yes.** Cyclic reassignment at overlay prevalence 0/25/50/75/100%; clean test removes all overlays. | No ROI-relative movement | Concept delineations; no input restriction | Yes, fundus and mammography |
| X2 · Drenkow | **Yes.** Variable synthetic alignment → A = 1−Y for all test images. | No; fixed position example | No anatomical ROI mask | Yes, dermoscopy, plus nonimaging |
| X3 · SilverLining | **Yes.** PneumoniaMNIST code assigns 95% congruent cues in train and 50% opposite cues/50% clean in the counter-shortcut test. | Yes, left/right markers; no anatomical crossing | Yes, masks artifact patches | Yes, CXR and colonoscopy, and nonmedical |
| X4 · Yang, KnoBo | **Yes.** Opposite group assignments; source/hospital-exclusive variants 100%→0% match. | Natural projection and lesion body-site groups only | No anatomy-only input | Yes, CXR and skin |
| X5 · Müller, MLMI workshop | **Yes.** 95/5 matched → 5/95 inverted; 50/50 balanced. | No | No anatomical ROI mask | Yes, CXR, and nonmedical |
| X6 · Fay, FAIMI workshop | **Yes.** Nominal 90/10 → 10/90 in the MRI construction, plus 50/50 balanced and matched validation. CheXpert also inverted; 90/10→10/90 for the imposed sex cue in released configuration. | No | Central slice selection only | Yes, MRI and CXR |
| X7 · Boland, MIDL | **No.** Perfect train association → balanced test. | Yes, fixed/random square and cue jitter; no ROI strata | No anatomy-only input | Yes, CXR and dermoscopy, and nonmedical |
| X8 · Nauta | **Partly.** Benign-only source cue, 46.5% vs 0%; insertion into malignant test cases and removal from patched benign cases as paired analyses. | No controlled ROI-relative movement | Artifact masks for inpainting | Yes, dermoscopy |
| X9 · RadEdit | **Partly.** Perfect positive source link → edited negatives only, all carrying positive-source signature; separate drain-positive negative-only test. | No | Edit/keep masks and an auxiliary disease segmentation task | Yes, CXR |
| X10 · FastDiME | **No.** Match 100/75/50% train–test rates. Flip individual cues on a balanced test, keeping aggregate balance by design. | No | Local masks for edits | Yes, CXR and dermoscopy, and nonmedical |
| X11 · DeGrave | **Partly.** Paired marker swap plus natural projection association shift; no prescribed inverse cohort. | Yes, shoulder movement in an example; no ROI-crossing of a marker | No classifier ROI mask | Yes, CXR |
| X12 · Winkler | **No.** Marked 100% and unmarked 0% in both classes; training prevalence unavailable. | Adjacent ink, with background reduced by cropping | Yes, lesion-focused crop | Yes, dermoscopy |

Two A–R precedents are especially important for interpreting this table. **H** combines a fully matched test, a fully cyclic reversed test and an artifact-free test, with the marker outside the lungs and an actual segmentation pipeline among the comparators. **O** explicitly varies near/far morphology at the same nominal 70/30 bias prevalence, with both locations outside the disease region. [ISNet, Table 1 and supplement](https://pmc.ncbi.nlm.nih.gov/articles/PMC10767127/) · [SimBA, Figs. 1–2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11491635/)

## Corrections for the presentation

1. **Novelty and takeaway.** Use the narrower contribution of experimentally crossing artifact location relative to the ROI with masking and the three test environments. Cite H, T, U and X1 for prior reversal designs and O and X7 for controlled spatial variation. A result statement suited to the talk is “In our experiments, ROI masking removed reliance on artifacts placed wholly outside the ROI. Artifacts retained inside the mask could still act as shortcuts.” The additional papers establish precedence for several parts of the experiment. The audit found no full-text example of the complete same-artifact inside/outside × mask × three-test design.
2. **A, ShorT.** Name both variables in −0.668: frozen age-probe MAE and the separation gap for effusion in the biased setup. Lower MAE means more recoverable age. Say “roughly halved the separation gap” for the comparison illustrated in Fig. 3c.
3. **B, Yang.** Attribute the 3,456-model grid to MIMIC-CXR. Specify that fairness can degrade or reappear externally and that transfer varies with task and group. Use FPR for No Finding and FNR for specific findings, with the group comparison stated.
4. **C, chest drains.** State that training is on CheXpert and the drain-controlled tests use CheXpert and NIH. Use AUROC 84.7% with drains and 76.7% without for CheXpert, and 81.1% with and 65.5% without for NIH, at the largest size. All negative test cases are drain-free. Cite Cheplygina et al. (MIUA 2025) separately for NEATX: 3,543 accepted NIH drain annotations and 1,011 PadChest images with four tube types.
5. **D, Reveal to Revise.** Name F1, accuracy and artifact relevance. The poisoned test adds the cue to every image. Its result shows sensitivity to cue addition and the effect of correction.
6. **E, segmentation.** Name Lin first. Report still-image Dice against ground truth separately from video prediction consistency. The skin experiment moves the lesion relative to crop boundaries and probes a centering/zero-padding shortcut.
7. **F, ISIC leakage.** Report 0.80 as VGG-19’s AUROC on ISIC 2020. Distinguish 13,976 pooled exact duplicates, 2,263 downsampled matches identified, 1,927 downsampled files removed, and 14,310 total training files removed. For same-release train/test overlap use 3 in 2016 and 2 in 2017. The combined counts use different pools.
8. **G, Mask of Truth.** Cite 2026 for volume 39(1), with online publication in 2025 if relevant. Specify DenseNet-121 and the train/test input pair. The 0.95 full / 0.86 lungs removed / 0.47 lungs only values are for a full-image-trained effusion model. Quote glaucoma separately.
9. **H, ISNet.** Replace ResNet-50 with VGG-19 for Stanford Dogs. Use eight alternatives overall and seven alternatives in the external TB comparison. State that ISNet applies its LRP penalty during training and the U-Net baseline uses actual segmented input. Include the 100% matched, 0% original-match cyclic and artifact-free COVID test settings.
10. **I and J, computer-vision benchmarks.** UrbanCars uses two independent 95%-aligned training cues and balances each at 50% in validation/test. DFR’s Waterbirds test also balances backgrounds. An uncommon group score and an “inverse problem” that changes the target are different evaluations from an inverted test population.
11. **K, MEDFAIR.** List the ten datasets from Table 1. Say no significant fairness improvement over ERM across the benchmark. Individual method/dataset wins are compatible with that finding.
12. **L, diffusion augmentation.** Attribute the +7.7 percentage-point histopathology number to the preprint. For the final paper use +48.5% relative versus no augmentation and +3.2% relative versus color augmentation. Keep +5.2% relative mean OOD CXR ROC-AUC and qualify the 44.6% relative reduction as the OOD sex gap.
13. **M, confidence intervals.** Correct the citation to pp. 124–132. Report 54.8% with no variability measure and 1 of 221 with CIs. State the denominator of 77 eligible comparisons for the 0.03 vs 0.01 median and the 64.9% within-interval result.
14. **N, leakage.** Say “at least 294 papers in 17 fields.” Limit the civil-war conclusion to the four corrected cases. One complex model retained a 0.01 AUC advantage after correction.
15. **O and O2, SimBA.** Near putamen and far postcentral gyrus are both outside the left-insula disease region. The biased train/test cases use the same nominal 70/30 schedule. The released splits show the realized train and test rates (about 72/29 and 73/27). Reweighing produced the most consistent near-zero relative gaps in those experiments. In the 2025 follow-up the putamen cue stays fixed and the added cue is global intensity.
16. **P, Germani.** The requested journal full text was inaccessible. Hold its quantitative and methodological claims as NOT FOUND for this audit. The separate 2025 MICCAI paper has a different title and author list. Its open manuscript provides related numbers that can be cited under the 2025 reference.
17. **Q, RoentMod.** The 800 reader-study edits are from 100 MIMIC images. Six findings reach 89–99% correct addition, with emphysema and nodules lower and excluded. Cite Table 1 in absolute AUC units, e.g. +0.135 for NIH edema, +0.068 for NIH pneumonia and +0.102 for NIH hernia, against the NIH TorchXRayVision comparator. Improvements vary by finding and cohort.
18. **R, thyroid ultrasound.** Say the detector uses calipers to locate the crop. The authors examined visible scanner-dependent caliper styles and compared performance with and without manual contouring. They report equivalent performance in that check and conclude caliper differences did not explain their scanner gap. The full manuscript gives no dedicated ablation AUC.
19. **S–U, direct reversal citations.** Colored MNIST in IRM uses 80% and 90% training alignment and 10% test alignment and is a preprint. Bissoto’s workshop trap sets truly invert all seven artifact associations at different rates. NuRD gives the closest exact binary schedule, 90/10→10/90 in medical images, using source as the cue.
20. **V–W, near misses.** Makar’s reversed sweep is in Waterbirds. Its CheXpert arm compares the matched 90% conditionals with 50%. Jabbour’s medical test removes the train association through resampling.
