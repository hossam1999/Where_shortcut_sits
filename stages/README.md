# Stage reports

Six standalone reports, one per supervisor meeting. Each folder holds the LaTeX source, the built PDF, generated
tables and figures, and the script that regenerates them from result files:

| Folder | Theme |
|---|---|
| `stage1_problem_pilot_protocol/` | the clinical problem, the pilot (E1–E15), its independent replication, the protocol and the bootstrap correction |
| `stage2_location_law_mechanism/` | controlled overlap sweeps, the location law and its mechanism (occlusion, dilation, counterfactual, lesion size), other modalities |
| `stage3_real_artifacts/` | real artifacts across modalities: hair, calipers, debris, devices; the failed E12 design; detector validity; chest-radiograph boundary; the 16-test Holm family |
| `stage4_causal_checks/` | balance (SMD), matched traps and the regression-adjusted crossover (A2), real-artifact transplant, neutral paste, the bootstrap correction and every claim it changed |
| `stage5_practice_theory/` | natural test sets, clinical metrics, the n = 78 thyroid subgroup, replication breadth, the DermLIP exception, the theory judged by held-out error |
| `stage6_remedies_audit_paper/` | remedies (decision guide), the image-label audit, external validation, limitations, the paper, CLAIM, submission statements |

Layout of each stage: `stageN.tex`, `stageN.pdf`, `figures/`, `tables/` (generated; `numbers.tex` holds every number
quoted in the prose as a macro), `make_stageN.py`. Shared files are in `common/` (`preamble.tex`, `notation.tex`,
`references.bib`, `stagelib.py`).

Build everything and run the number audit:

```bash
bash stages/build_all.sh
```

Rules the reports follow: every number is read from a result file by `make_stageN.py` (nothing typed by hand); every
interval comes from the regenerated runs in `results/rerun_2026-09-28/` with the corrected (crossed seed × image)
bootstrap; registration commits are read from git (`stagelib.reg`), and commit times come from this machine, so they
are not independent proof of order. `scripts/verify/audit_numbers_final.py --docs stages` fails the build if a number
cannot be traced. Decisions taken during the unattended run are in `DECISIONS_LOG.md`.
