# Pre-registration (DRAFT, not approved): improving mask_cmc where it fails

Status: **draft for the author's approval** (2026-11-18). Nothing here is scored on any test set. This track does not
replace or delay anything registered (ISNet chain, P4, phase 5 keep priority).

## Where mask_cmc fails (the motivation, from results already seen)
1. **Trained encoders that absorb a strong in-ROI shortcut.** P3 90/10 inside, arm d (EfficientNet-B0 trained on
   lung-masked images with class-specific marks): deceiving-test macro-F1 0.010 (registered, pooled) and 0.048 (post hoc,
   per_class); standard (no marks) 0.40. Removing 1–3 mean-difference directions does not remove a shortcut the encoder has
   spread over many directions. P1 RSNA (view shortcut): arm d 0.498 vs ActDiff 0.708.
2. **U-MtE inside the ROI.** P1 location test, tube inside: U-MtE reversed AUROC 0.598 vs 0.86–0.88 for mask_cmc arms.
3. **Clean-accuracy cost.** P2 arm b (frozen DINOv2 on lesion-masked images): clean 0.685 vs Vanilla 0.839.

## Literature (closest work; nothing below is claimed as new)
- **LEACE** — Belrose et al., *LEACE: Perfect linear concept erasure in closed form*, NeurIPS 2023 (arXiv 2306.03819):
  the least-squares closed-form eraser that makes a concept linearly unrecoverable with minimal change (whitened metric).
- **INLP** — Ravfogel et al., *Null It Out*, ACL 2020: iterative nullspace projection of probe directions.
- **RLACE** — Ravfogel et al., *Linear Adversarial Concept Erasure*, ICML 2022: linear minimax erasure.
- **LEOPARD** — Saillenfest & Lemberger, ECAI 2025 (arXiv 2507.12341): erasure that makes *class-conditional* distributions
  of the projected representations indistinguishable (closest to "class-conditional erasure").
- **CIRCE** — Pogodin et al., *Efficient Conditionally Invariant Representation Learning*, ICLR 2023 (arXiv 2212.08645):
  training-time penalty for φ(X) ⊥ Z | Y (closest to candidate b).
- **Medical shortcut penalties** — Müller, Fay et al., MLMI 2024 (arXiv 2407.18792): benchmark of dependence penalties
  (MI, distance correlation, adversarial) for shortcut learning in medical imaging; MIMM-X (arXiv 2511.22990).
- **Label-free spurious directions** — DIAL (ICLR 2026, sparse-autoencoder spurious directions, projected out); joint
  subspace estimation (arXiv 2310.11991). Closest to candidate c, which differs in using synthetic in-ROI overlays.
- Our own mask_cmc projects out the within-class artifact mean differences in the Euclidean metric — a first-moment,
  class-conditional special case of the above. What would be new is the *evaluation* (where the shortcut sits) and the
  specific class-conditional forms below, not linear or conditional erasure as such.

## Candidates (at most 3)

### (a) Class-conditional LEACE with an iterative stop (CC-LEACE)
- **Definition.** Features X (n × d), label y ∈ {1..K}, artifact label A (one or more binary flags; training images only).
  For each class y with ≥ 2 images in both artifact groups, compute the within-class covariance Σ_y and cross-covariance
  Σ_{XA|y}; pool them over classes, weighted by class size: Σ_w = Σ_y π_y Σ_y, C_w = Σ_y π_y Σ_{XA|y}. Apply the LEACE
  eraser with (Σ_w, C_w): x ↦ x − W⁺ P W (x − μ_y-free mean), W = (Σ_w + εI)^{-1/2}, P the orthogonal projector onto the
  column space of W C_w (Belrose et al., Theorem 3.1, with the class-conditional covariances in place of the marginal ones).
  Then iterate (INLP-style, at most T = 8 rounds): fit a within-class logistic probe for A on the erased training features;
  if its validation AUROC (within class, averaged) exceeds 0.55, project out its weight vector in the whitened metric and
  repeat.
- **Head.** Class-balanced logistic regression (binary or multinomial) on the erased features, as mask_cmc.
- **Selection (validation only).** ε ∈ {1e-4, 1e-3, 1e-2} × trace(Σ_w)/d, T chosen by the stopping rule above, C in the
  A21 grid {0.001, 0.003, 0.01, 0.03, 0.1, 1, 10}; all by the paper's/cohort's registered validation metric.
- **Why it might help.** In the whitened metric the eraser removes the artifact signal that is spread over many
  correlated directions (failure 1), which a Euclidean mean-difference projection leaves.

### (b) Class-conditional mean-and-covariance penalty during training (design only; GPU needs approval)
- **Definition.** Encoder f trained with L = CE(h(f(x)), y) + λ Σ_y [ ‖μ(f | A=1, y) − μ(f | A=0, y)‖² +
  ‖Σ(f | A=1, y) − Σ(f | A=0, y)‖_F² ] on each minibatch (groups with < 2 images in a batch skipped), features
  L2-normalised; a class-balanced batch sampler that guarantees both artifact groups per class. After training, CC-LEACE
  (candidate a) on the final features.
- **Selection.** λ ∈ {0.1, 1, 10} and early stopping on the registered validation metric; everything else as arm d.
- **Closest work.** CIRCE (conditional independence penalty), conditional MMD / CIDDG, Müller et al. 2024; ours differs
  only in being a first- and second-moment class-conditional penalty matched to the post-hoc eraser.

### (c) U-MtE inside the ROI: multi-family in-ROI overlays with held-out-family selection
- **Definition.** As U-MtE (label-free; no artifact labels or examples), but the difference subspace is estimated from
  F = 4 overlay families drawn **inside the ROI** (bars, text strings, rings, dot clusters; random position, size and
  contrast), each giving paired features (x, x + overlay_f). The subspace dimension k is chosen by leave-one-family-out:
  fit on 3 families, measure the fraction of the held-out family's feature shift energy removed, pick the smallest k
  reaching 90% averaged over the 4 hold-outs.
- **Selection.** k as above (label-free); head C on the registered validation metric.
- **Why it might help.** U-MtE's single generic overlay family may not span the real in-ROI artifact's shift (failure 2);
  requiring the subspace to generalise to an unseen family is a label-free proxy for generalising to the real artifact.

## Development protocol (until the author approves test scoring)
- **Data allowed:** validation splits only (P1–P4 validation; phase-5 thyroid `val_clean` is NOT used — phase 5 is the
  confirmatory test and stays unseen), and the thesis's development data (ISIC 2018 pilot / synthetic-ruler cohort,
  synthetic traps). CPU and cached features only (no GPU without approval; candidate b is designed, not run).
- **Never** score a candidate on any P1–P4 test set or on phase 5 during development.
- **Counterfactual validation views are allowed** when built from validation images only (e.g. P3 validation images with
  the deceiving marks drawn, P1 location validation with the tube moved), because a biased validation alone cannot show
  whether a shortcut is removed. They are development tools, never reported as test results.
- Each candidate is compared with mask_cmc on the same validation data; reported: validation metric, within-class artifact
  probe AUROC after erasure (should approach 0.5), and the clean-vs-correlated validation gap where available.
- Timebox: 2–3 days. Weekly plain-English report (validation numbers only), with a recommendation of which candidate(s) to
  register.
- **Confirmatory test:** phase 5 (thyroid, not yet seen), added by a dated amendment to `PREREGISTRATION_PHASE5.md` after
  approval. Any score on a P1–P4 test is labelled **post hoc exploratory**.
