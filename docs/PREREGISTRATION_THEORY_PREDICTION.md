# Pre-registration — does the linear-Gaussian theory *predict* real results? (theory → empirical test)

Committed before any prediction is computed. Motivation: a reviewer noted that the theory (docs/THEORY.md) was only
checked against simulations drawn from its own assumptions. This test uses it as a **zero-free-parameter predictor
of held-out environments** on the real experiments already run.

## Model (docs/THEORY.md, generalised to any test composition)
Training P(A=1|Y=1)=p1=0.9, P(A=1|Y=0)=p0=0.1 (the design value of every trap and sweep). With disease SNR S, artifact
SNR A, v = [p1(1−p1)+p0(1−p0)]/2 and an LDA-optimal head (w_s = √S, w_a = √A (p1−p0)/(1+vA)), a test environment with
P(A=1|Y=y) = q_y has
  Δ = S + (p1−p0)(q1−q0) A/(1+vA),  σ_y² = S + w_a² (1 + A q_y(1−q_y)),  AUROC = Φ(Δ / √(σ1² + σ0²)).
(Reduces to AUROC_rev = Φ((S−Ã)/√(2(S+Ã))) for q = 0.1/0.9.)

## Procedure (`scripts/analysis/theory_predict.py`)
For every cell (result directory × trap or overlap × arm):
1. Observed AUROC per environment = mean over training seeds/folds; q_y per environment read from the saved test
   predictions (`artifact_present` by label).
2. Fit (S ≥ 0, A ≥ 0) so that the model reproduces the **clean** and **correlated** AUROC (least squares in probit
   space; two equations, two unknowns).
3. **Predict the reversed AUROC**, which is never used in the fit. No parameter is shared across cells or tuned.

Cells (fixed now; one canonical run per cohort × backbone):
- Real location traps (Trap A in-ROI, Trap B out-of-ROI): ISIC hair (spec_e13/dino518_spec, dermlip224_spec); thyroid
  (dino518_main, medsiglip448_main, convnext384_universal); capsule (dino518_main, medsiglip448_main,
  convnext384_universal); ovary (dino518_main, medsiglip448_text); chest-radiograph traps (cxr_traps/raddino518 ×
  Atelectasis, Consolidation, Effusion, Infiltration).
- Chest drains (in-ROI only): cxr_drain/raddino518_universal.
- Controlled overlap sweeps (5 overlaps each): synthetic/isic2018 {dino518, dermlip224, dino224}_ruler_fixed_corr_main;
  nih_ptx {raddino518, dino518}_tube_corr_main; capsule {dino518, medsiglip448}_debris_corr_main; thyroid
  {dino518, medsiglip448}_caliper_corr_main; ovary dino518_caliper_corr_main.
- Secondary (outside the model's assumptions — end-to-end networks, not linear heads on frozen features):
  finetune/{thyroid, capsule, ovary}/resnet50, finetune/thyroid/ViT-S.

Arms: **primary** = ERM and mask (the two arms that define the location law). **Secondary** = every other arm whose
head is plain ERM on the 0.9/0.1 training set (erasure/inpainting/projection arms). Arms that reweight or change the
training distribution (balanced, DFR, JTT, GroupDRO, prevcal, augmentation, selectors) are excluded because the
model's p1/p0 no longer describe their training set.

## Endpoints
- **T1** reversed AUROC, predicted vs observed, over all primary cells: mean absolute error (MAE), Pearson r,
  share within ±0.05. Compared with two reference predictors that use the same inputs: (a) rev = clean (“no shortcut
  harm”), (b) rev = 2·clean − corr (linear symmetry).
- **T2** location crossover, predicted vs observed, per real trap cohort × backbone:
  [rev(mask) − rev(ERM)]_TrapB − [rev(mask) − rev(ERM)]_TrapA. Sign agreement, Pearson r, MAE.
- **T3** controlled sweeps: predicted vs observed mask − ERM reversed-AUROC gain at every overlap. Sign agreement,
  Pearson r, MAE.
- **T4** (secondary) in-ROI mask effect: does sign(S_mask − S_ERM) (fitted from clean/correlated only) predict the
  sign of the observed Trap-A mask gain? Accuracy.

## Decision rule (fixed now)
We call the theory **quantitatively predictive** if T2 signs agree in all cells and Pearson r ≥ 0.7 for both T1
and T2, and the theory's T1 MAE is smaller than reference (a). Otherwise the paper describes it as a **qualitative
account** ("reproduces signs and orderings") and does not use the word "predicts" for magnitudes. Reported whichever
way it goes.

## Results (results/theory/prediction_summary.json, prediction_cells.csv, prediction_crossover.csv; Fig. theory_predict)
Deviation: the DINOv2 chest-tube sweep (synthetic/nih_ptx/dino518_tube_corr_main) has no saved predictions and was
skipped; everything else as listed.

| endpoint | theory | ref. (a) rev = clean | ref. (b) rev = 2·clean − corr |
|---|---|---|---|
| T1 reversed AUROC, ERM & mask (148 cells) | MAE **0.039**, r = 0.917, 76 % within ±0.05 | MAE 0.234, r = 0.592 | MAE 0.108, r = 0.763 |
| T1 all ERM-head arms (570 cells) | MAE 0.031, r = 0.945 | MAE 0.163 | MAE 0.069 |
| T2 crossover (14 cohort × backbone) | **r = 0.997, MAE 0.014, signs 13/14** | r = 0.930, MAE 0.108, 12/14 | r = 0.989, MAE 0.036, 13/14 |
| T3 sweeps, mask gain (45 overlap points) | r = 0.965, MAE 0.043, signs 42/45 | r = 0.670, 31/45 | r = 0.869, 36/45 |
| T4 in-ROI sign from S_mask − S_ERM | 11/14 | | |
| End-to-end fine-tuning (outside the model) | crossover signs 4/4; rev AUROC MAE 0.018 (16 cells) | | |

- **Verdict by the pre-registered rule: "qualitative account"**, because one of 14 crossover signs disagrees:
  chest-radiograph Consolidation, observed −0.002 vs predicted +0.010 (both ≈ 0). All other criteria are met with
  wide margins (T1 r = 0.92 ≥ 0.7; T2 r = 0.997 ≥ 0.7; T1 MAE 0.039 < 0.234).
- Fitted only to clean and correlated performance, the closed form's out-of-sample estimates agree with the held-out
  reversed-test AUROC to within 0.04 on average and with the location crossover to within 0.014, better than both
  reference predictors on every endpoint; they also agree for end-to-end fine-tuned networks (4/4 crossover signs),
  which violate its assumptions. Per the decision rule, the paper does not say the theory "predicts" magnitudes; it
  reports these agreements and calls the theory a qualitative account.
- The clear failure is the chest-drain cohort (predicted 0.70–0.76, observed 0.19–0.24): the clean and correlated
  environments do not reveal a shortcut carried by correlates of the drain (the treated lung), which is consistent
  with the correlate-carried account (THEORY.md §6) and marks the model's boundary.
