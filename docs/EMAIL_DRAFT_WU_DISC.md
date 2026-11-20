# Draft email to Wu, Zou et al. — DISC ISIC trap-set results and the shared image cache (not sent; the author sends it)

**Sent 2026-11-20 by the author to shirwu@cs.stanford.edu and jamesz@stanford.edu (signed "Hossam Elkady"); no bounce at send time.**

**To:** Shirley Wu <shirwu@cs.stanford.edu>, James Zou <jamesz@stanford.edu>
*Addresses as printed on p. 1 of the ICML 2023 paper ("Correspondence to: Shirley Wu …, James Zou …"). Stanford addresses may
have changed since.*
**Subject:** Question about the ISIC trap-set results of DISC (ICML 2023) and the image cache in the released code

Dear Ms. Wu, dear Prof. Zou,

I am an MSc student working on shortcut learning in medical imaging, specifically on how the position of an artifact
relative to the lesion decides whether region-based fixes work. DISC is a central comparison in my thesis, and I am
reproducing its ISIC result (Table 1: ERM 36.4, DISC 55.1, average AUROC over Bissoto et al.'s trap sets 1–5) with your
public code (github.com/Wuyxin/DISC @ d9865a8) and your released data and concept bank. Thank you for releasing them.

While doing so we noticed one detail of the released code that we would like to check with you, politely and in case it
is useful:

1. **One image cache for all trap sets.** `ISICDataset` (`disc/dataset/isic_dataset.py`) saves the transformed images once to
   `<root_dir>/features/feature_mat.npy` and reloads that file in later runs; the file name does not contain the trap-set
   id, and `scripts/isic.sh` uses the same `--root_dir` for every GROUP. Running the trap sets one after another with one
   data folder (as we first did), every trap set after the first therefore loads the first trap set's images in the first
   trap set's row order, while names, labels and the hair attribute come from its own CSV. We checked this row by row: on
   trap sets 2–5 only 3–21 % of the cached rows show the image their CSV row describes (a hue-invariant pixel comparison
   against a fresh resize of each named image; trap set 1 itself matches exactly).
2. **Results with one data folder per trap set** (each with its own cache, all verified row by row; your commands and
   hyper-parameters otherwise unchanged; test ROC-AUC at the best-validation epoch, as `run_expt.py`):

   | | trap set 1 | 2 | 3 | 4 | 5 | mean | Table 1 |
   |---|---|---|---|---|---|---|---|
   | ERM | 0.531 | 0.506 | 0.524 | 0.502 | 0.474 | 0.507 | 0.364 |
   | DISC | 0.451 | 0.493 | 0.514 | 0.525 | 0.464 | 0.489 | 0.551 |

   With the shared cache (our first, invalid runs), ERM averaged 0.368 over trap sets 1–3, close to Table 1's 0.364.
3. A smaller point: the rows after the training set are listed as [20 % part, 80 % part] but labelled [test, val]
   (`isic_dataset.py`, `split_array`), so the 154-image validation set is the last 154 images of the 80 % part rather than
   the 20 % part. This does not affect our comparison (we used exactly your rows), but we mention it in case it was
   unintended.

Could you tell us whether the published ISIC runs were made with one data folder for all trap sets (so with the shared
cache), or with a fresh cache per trap set? If any logs or checkpoints of the Table 1 runs are still available, they would
help us report your result accurately.

We will report your answer, cite your paper and code, and are happy to share our scripts (including the row-by-row cache
check) and results with you before submission.

Thank you very much for your time.

Kind regards,
[Name]
[MSc programme, university, supervisor]

*Attachments for the author to add: `results/replication/disc/gate.md` and the cache check
(`scripts/replication/p8_cache_check.py` with its output in `docs/REPLICATION_STATUS.md`, 2026-11-19 ~15:05).*
