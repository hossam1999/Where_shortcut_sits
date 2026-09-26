# Thyroid ultrasound (TN3K/TNCD) marker detector — visual audit

Detector: `wtss.data.us_markers.marker_mask` (dotted measurement lines = >= 7 evenly spaced isolated bright dots,
spacing CV <= 0.20; '+' calipers by template matching on the top-hat image; top 8 % / bottom 6 px excluded).
Tuned on one random sample (seed 3, 16 images) and a first audit sheet (seed 2026, images 0-19); **frozen**
before the final audit on images 20-39 of the seed-2026 sample, which had not been viewed.

Final audit (20 unseen images, image-level presence judged by eye from the raw image):
- true positives 6 (+2 probable: 33, 37), true negatives 10, false negatives 1 (thin solid-line Doppler box, #22),
  false positives 1 (single tiny dot, #28) → precision ≈ 0.86–0.89, recall ≈ 0.86–0.89.
- Known weakness: thin *solid* box outlines (Doppler/ROI boxes) are not detected; dotted boxes are.
Pixel masks under-cover faint dotted lines (the detector marks dots, not the full segment), which biases the
overlap r mildly toward the caliper ends.
