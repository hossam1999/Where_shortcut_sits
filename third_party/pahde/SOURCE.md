# Pahde et al., Ensuring medical AI safety (Machine Learning 114:206, 2025)

- Repository: https://github.com/frederikpahde/medical-ai-safety (linked from the paper's code-availability section, not a mirror)
- Commit: `4990898b0f0b88ba42458ebb9caa89d1f99c393f` (cloned 2026-11-16)
- Upstream environment file: `requirements.txt (no versions pinned)`
- Local clone: `/workspace/third_party/pahde` (linked here as `code/`; git-ignored)
- Environment used here: not built yet. The pinned versions cannot run on the Blackwell GPU (cc 12.0 needs CUDA ≥ 12.8);
  see `docs/REPLICATION_STATUS.md`. Any change to their code is stored next to this file as a `.patch`.

## Checkpointing and resume (checked 2026-11-17)
- `model_training/train_model.py::store_model` saves `epoch`, `model_state_dict` and `optimizer_state_dict` (Adam state)
  every `store_every_n_epochs` (100 in their configs) and at the end. The MultiStepLR schedule is not saved; on resume
  `start_training.py` replays it by stepping `start_epoch` times, which is equivalent.
- **Their resume trains `num_epochs` more epochs** (`range(start_epoch + 1, start_epoch + num_epochs + 1)`), so a resumed run
  must set `num_epochs = 300 − start_epoch` to keep 300 in total. If a resume ever happens, this and whether the optimizer
  state was restored are recorded next to the P2 result.
- The first P2 Vanilla run has no checkpoint before epoch 100, which falls short of scheduling rule 5; the interval cannot be
  changed in a running process. Nothing risky runs beside it until `checkpoint_resnet50d_100.pth` exists. For P2 runs not
  yet started (RR-ClArC fine-tunes), a shorter interval is an engineering note, not an amendment.
- Pause safety: their DataLoaders set no timeout, W&B logging is disabled (`WANDB_MODE=disabled`), and there is no watchdog.
  P2 was paused with SIGSTOP from 08:37 while ISNet ran one epoch alone, and is resumed with SIGCONT afterwards.
