# After the machine is released: what is lost and how to rebuild it

Written 2026-09-29 before releasing the GPU machine that produced every result. What remains after release: the git
repository (code, registrations, every committed result table and summary) and the author's download of
`/root/release_archive/` (`wtss_outputs.tar[.partNN]`: every per-image prediction file, model weights that produce labels
or reported predictions, run logs; `wtss_private_keys.tar`: the two blinding keys and the uncommitted rating answers —
private, never into git or Zenodo). Checksums: `SHA256SUMS` in the archive folder and `results/ARCHIVE_MANIFEST.csv`.

## Lost with the machine

| what | size on this machine | why it is not archived |
|---|---|---|
| raw datasets (`$WTSS_DATA`: ISIC 2018/2019/2020, HAM10000 masks, artifact masks, TN3K/TNCD, MMOTU, SEE-AI, capsule masks, NIH ChestX-ray14, NEATX, RANZCR-CLiP, round-6 external sets, ThyUS2Path) | ≈ 146 GB (see `docs/DATA_MANIFEST.md`) | third-party licences; re-download with the commands below; `docs/DATA_MANIFEST.md` holds a listing SHA256 per dataset to check a re-download |
| derived image caches (`*/cache_518*`, `isic2019/prepared`, `cxr/prepared`, `cache/images`) | ≈ 90 GB | pixel-level derivatives (resized images, lung masks, hair masks, LaMa-inpainted images) must not be redistributed; rebuilt by the prepare scripts |
| feature caches (`$WTSS_CACHE/features`) | 5.5 GB | derived from the images; rebuilt automatically (every run script extracts a missing view, and `scripts/round4/common.assemble_views` reuses cached rows) |
| audit images (`audit/*/` image folders, `audit_local/`) and the round-6 rating images (`audit_local/contours`, `results/round6/_local/`) | small | images of licence-restricted datasets and the blinded key; regenerated deterministically (below) |
| model weights downloaded from hubs (DINOv2, RAD-DINO, MedSigLIP, DermLIP, ConvNeXt, MedGemma, LaMa, torchxrayvision) | ≈ 20 GB | public or gated; revisions and SHA256 in `docs/ENVIRONMENT.md` |

## Rebuild on a new machine

```bash
git clone git@github.com:hossam1999/Where_shortcut_sits.git && cd Where_shortcut_sits
git checkout claude/festive-bohr-95evo5
python -m venv /venv/main && source /venv/main/bin/activate && pip install -r requirements.txt   # versions: docs/ENVIRONMENT.md
export WTSS_DATA=/root/data PYTHONPATH=$PWD/src WTSS_BOOTSTRAP=crossed
# credentials the author must provide on the new machine (never commit them):
#   Hugging Face token with the MedSigLIP and MedGemma terms accepted; Kaggle token whose account accepted the
#   RANZCR-CLiP competition rules
make data          # ISIC, HAM10000 masks, artifact masks, NIH ChestX-ray14, NEATX; lesion U-Net; ISIC caches;
                   # chest-radiograph caches and lung masks; drain detector and drain cohort
make data_extra    # TN3K/TNCD, MMOTU, SEE-AI, capsule masks, RANZCR-CLiP (+ link to NIH), hair statistics, spec U-Net
make lama          # LaMa-inpainted caches (only for the LaMa baseline)
```
Check each re-download against `docs/DATA_MANIFEST.md` (file count and listing SHA256; a changed upstream release shows
up as a mismatch). Predictions need not be recomputed: restore them from the archive into `results/` (same paths).

**Feature caches** are rebuilt by the run that needs them, e.g. for the thyroid and ovary DINOv2@518 views used by
rounds 4–9 and by the rating analysis:
```bash
python scripts/run_thyroid_traps.py --cohort thyroid --backbone dino518 --generic --tag universal --save_val --arms erm mask balanced dfr mte mte_balanced mte_protect mte_protect_balanced jtt mask_jtt umte_jtt mask_balanced mask_dfr mte_aug
python scripts/run_thyroid_traps.py --cohort ovary   --backbone dino518 --generic --tag universal --save_val --arms erm mask balanced dfr mte mte_balanced mte_protect mte_protect_balanced jtt mask_jtt umte_jtt mask_balanced mask_dfr mte_aug
```
(`WTSS_RESULTS=/tmp/scratch_results` keeps these rebuild runs away from the committed results; only the feature files
under `$WTSS_CACHE/features/{thyroid,ovary}/dinov2_b14_518/` are needed.) The Makefile targets `thyroid`, `ovary`,
`capsule`, `drain`, `natural`, `cxr` rebuild the remaining views the same way.

## The blinded rating (author)
1. **Regenerate the audit package** (images, overlays, blind key) from the rebuilt caches and check the key against the
   committed hashes before anyone looks at it:
   ```bash
   python audit/make_audit_sample.py                    # writes audit/*/ images and audit_local/KEY_open_after_review.csv
   sha256sum audit_local/KEY_open_after_review.csv      # must equal audit/KEY_SHA256.txt
   python scripts/round6/a4b_sample.py                  # A4b caliper sample; its key must equal results/round6/a4b_KEY_SHA256.txt
   ```
   A mismatch means the regenerated sample differs from the registered one: stop and do not rate. (Alternatively
   restore the two keys from `wtss_private_keys.tar`, verify them the same way, and regenerate only the images.)
2. **Restore the rating answers already given** (pass 1) from `wtss_private_keys.tar` into `results/round6/rating/`.
3. **Rate** with the local tool (binds 127.0.0.1 only; open it through an SSH tunnel):
   ```bash
   python scripts/round6/rating_app.py --pass 1 --check        # every image present
   python scripts/round6/rating_app.py --pass 2                # main package, second pass
   python scripts/round6/rating_app.py --pass 2 --sheet a4b    # A4b caliper sample, second pass
   ```
   **Pass 2 must start at least 7 days after pass 1** (intra-rater reliability; `docs/PREREGISTRATION_ROUND6.md`).
4. **After rating**, rebuild the thyroid and ovary DINOv2@518 features first (commands above), then
   ```bash
   bash scripts/round6/after_rating.sh
   ```
   which scores the rating and re-runs the label-agreement, cleaned-trap, bias and summary steps it can change.

## What cannot be rebuilt exactly
- Hub models can be re-released under the same name: check the revisions and SHA256 in `docs/ENVIRONMENT.md`.
- LaMa inpainting and fp16 feature extraction are deterministic up to GPU numerics; small differences in regenerated
  features are expected (the rebuilt drain cohort of 2026-09-29 matched the archived size exactly, see
  `results/rerun_2026-09-28/SUMMARY_archived_reruns.md`).
