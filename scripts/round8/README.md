# Round 8 — runbook

Plan, hypotheses and decision rules: `docs/PREREGISTRATION_ROUND8.md` (frozen; changes only as dated amendments at its
end). Design and scoreboard: `docs/ROUND8_DESIGN.md`. Ground rules as rounds 4–7: code here, results in
`results/round8/`, feature caches only under `$WTSS_CACHE/features/round8/`, never commit images, features,
predictions or external masks; every script has `--smoke` and is resumable.

```bash
export WTSS_DATA=/root/data
bash scripts/round8/run_all.sh smoke   # every step on small subsets; validation images stand in for every test set
bash scripts/round8/run_all.sh         # development -> freeze (committed + pushed) -> confirmation -> analysis
bash scripts/round8/phase1.sh          # Phase 1: scoreboard of existing results (no new model)
```

| script | what |
|---|---|
| `common.py` | crossed bootstrap with shared replicates, pair-type AUROCs, the selection rule (`select_setting`), seeds |
| `candidates.py` | every frozen-feature arm (mask_bal(λ), CMC, locrand plans, AFR, GroupDRO, CFS, ensembles) |
| `locrand.py` | N3 placements (K = 4 donor versions per artifact-free image), pasted masked features (GPU), probe check |
| `heads_run.py` | traps / natural sets / ISIC 2020: `--stage dev` (validation only) or `--stage confirm` |
| `freeze.py` | `results/round8/frozen_choice.json` from the development choices (held-out ovary settings) |
| `sweeps.py` | D5: cross-fitted controlled sweeps at r = 0 and r = 1 |
| `ft.py` | D6: locrand during fine-tuning of the thyroid ConvNeXt-T |
| `analyse.py` | D1–D6, IUT, Holm, replication, every arm × cell, `results/round8/SUMMARY.md` |
| `scoreboard.py`, `pair_decomposition.py`, `criterion_check.py`, `design_tables.py` | Phase 1 |

## Absolute AUROC for the paper's remedies table (after release)
The paper's remedies table reports each arm minus masking from `confirm/descriptive_all_cells.csv`. The absolute AUROCs
need the git-ignored confirmation predictions, which are in `wtss_outputs.tar`:

    tar -xf wtss_outputs.tar --wildcards 'results/round8/confirm/trap/*_dino518/predictions.csv.gz' \
                                         'results/round8/confirm/natural/*_dino518/predictions.csv.gz'
    PYTHONPATH=src python scripts/round8/absolute_scores.py      # CPU only
    git add results/round8/confirm/absolute_auroc.csv            # the predictions stay git-ignored

The script checks that every arm minus masking reproduces the registered differences (it stops otherwise).
`scripts/make_main_summary_tables.py` then switches the table to absolute AUROCs with a masking column.
