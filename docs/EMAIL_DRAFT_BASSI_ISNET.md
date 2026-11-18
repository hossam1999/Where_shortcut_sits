# Draft email to Bassi et al. (not sent; the author sends it)

**Sent 2026-11-18 by the author (no Cc). Answer, when it arrives, is recorded in docs/REPLICATION_STATUS.md.**

**To:** pedro.salvadorbassi2@unibo.it, andrea.cavalli@unibo.it (addresses printed in the paper)
**Subject:** Question about the released COVIDIsNet.pt model (ISNet, Nature Communications 2024)

Dear Dr. Bassi and Prof. Cavalli,

I am an MSc student working on shortcut mitigation in medical imaging, and I am replicating your ISNet COVID-19 results
("Improving deep neural network generalization and robustness to background bias via layer-wise relevance propagation
optimization", Nat. Commun. 15:291, 2024). Thank you for releasing the code and the pre-trained models.

To check that our rebuilt o.o.d. test matches yours, we scored the released `COVIDIsNet.pt` (from the Google Drive folder
in the ISNet README). We load it strictly (all 1,144 keys match a fresh `IsDense121Lightning(classes=3)` built from your
code), in eval mode, and take the class logits from the forward pass with `heat=False` (the standard torchvision
DenseNet-121 with the same weights gives the same predictions). The input follows Supplementary Note 6.1: grayscale,
histogram equalization, values rescaled to [0, 1], resized to 224 × 224, repeated in 3 channels; class order normal 0,
pneumonia 1, COVID-19 2 (Supplementary Note 6.2). The input scale agrees with the first BatchNorm layer's running
statistics.

Even on images from your training sources (BrixIA COVID-19 and CheXpert normal / pneumonia), the model predicts
"pneumonia" for 95% of the BrixIA COVID-19 images and for 73% of the CheXpert normal images, and it never predicts
COVID-19. We have probably missed a preprocessing detail, and would be grateful for your help with four questions:

1. Is `COVIDIsNet.pt` the COVID-19 ISNet evaluated in Table 2 (trained without synthetic bias)?
2. Which CheXpert release did you use (the original high-resolution images or CheXpert-v1.0-small)?
3. Could you share the exact histogram-equalization code (library and function, and whether it is applied at the
   original bit depth, before or after resizing)?
4. Does the model expect any other preprocessing (for example cropping, lung masking, a different value range or
   normalization)?

We will report your answer, and cite your paper and code.

Thank you very much for your time.

Kind regards,
[Name]
[MSc programme, university]
