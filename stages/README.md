# Stage reports

Eight standalone progress reports, meant to be handed over one at a time. Each report builds only on the stages
before it and never refers to a later one; `COVERAGE.md` maps every piece of work in the repository to the stage and
section that contains it. Each folder holds the LaTeX source, the built PDF, generated
tables and figures, and the script that regenerates them from result files:

| Folder | Theme |
|---|---|
| `stage1_problem_pilot_protocol/` | the clinical problem, the pilot (E1–E15) and its independent replication, deviations, the audit of the pilot's claims, leakage, the protocol and the crossed bootstrap |
| `stage2_location_law_mechanism/` | controlled overlap sweeps, the location law and its mechanism (occlusion, dilation, counterfactual, lesion size), sweeps in four more modalities with every baseline arm |
| `stage3_real_artifacts/` | real artifacts across modalities and encoders: hair, calipers, debris; every baseline arm; detector validity; what carries the hair shortcut; DermLIP; chest-radiograph devices and drains; rejected datasets; the four location tests |
| `stage4_causal_checks/` | balance (SMD), matched traps and the regression-adjusted crossover (A2), real-artifact transplant, neutral paste, artifact-label sensitivity, embedding leakage groups, rebuild on a second machine |
| `stage5_practice_theory/` | natural test sets, clinical metrics and operating points, the n = 78 thyroid subgroup, scale, fine-tuning, zero-shot models, ISIC 2020 external validation, literature comparison, the DermLIP exception, the theory |
| `stage6_remedies_audit_paper/` | U-MtE and every remedy and baseline (frozen and fine-tuned), the 16-test family, the decision guide, the clinician audit package, the manuscript, CLAIM, statements, limitations |
| `stage7_robustness_external/` | round 4: masking implementations (black, blur, box crop, crop of the masked image), dose-response over the real overlap, ISIC 2019 → ISIC 2020 natural test with operating points, prospective theory test |
| `stage8_labels_strong_model/` | round 6: artifact labels checked without a clinician (DermArtifactDB, IMA++, Kabir hair masks, capsule masks, BUSClean, MedGemma), traps on uncontradicted labels, differential error; a fine-tuned thyroid model near the benchmark with the registered subgroup |

Layout of each stage: `stageN.tex`, `stageN.pdf`, `figures/`, `tables/` (generated; `numbers.tex` holds every number
quoted in the prose as a macro), `make_stageN.py`. Shared files are in `common/` (`preamble.tex`, `notation.tex`,
`references.bib`, `stagelib.py`).

`combined/full_report.pdf` joins the eight stages into one document (one section per stage, an overview and an overall
conclusion); `combined/make_combined.py` assembles it from the stage sources.

Build everything and run the number audit:

```bash
bash stages/build_all.sh
```

Rules the reports follow: every number is read from a result file by `make_stageN.py` (nothing typed by hand;
summaries computed by the generators are written to `results/stage_derived/`); every
interval comes from the regenerated runs in `results/rerun_2026-09-28/` (Stage 7: `results/round4/`, `results/round5/`; Stage 8: `results/round6/`) with the corrected (crossed seed × image)
bootstrap; registration commits are read from git (`stagelib.reg`), and commit times come from this machine, so they
are not independent proof of order. `scripts/verify/audit_numbers_final.py --docs stages` fails the build if a number
cannot be traced. Decisions taken during the unattended run are in `DECISIONS_LOG.md`.
