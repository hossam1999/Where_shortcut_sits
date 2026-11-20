# Reproduction gates at 0.03 and 0.05 (A38; unrounded values; no result changes)

The tolerance was widened from 0.03 to 0.05 by the author on 2026-11-20, **after the first gate results**, for every paper alike (P2–P8; P1 keeps its ±1 SD rule). Values are read unrounded from the committed result files.

| paper | method | metric | ours | published | gap | passes at 0.03 | passes at 0.05 | source |
|---|---|---|---|---|---|---|---|---|
| P2 Pahde 2025 | RR-ClArC (A18) | biased-test accuracy | 0.523885 | 0.760 | 0.236115 | no | no | `pahde/p2_reproduction_a18.csv` |
| P3 Bassi 2024 | released COVIDIsNet.pt, primary test (A9) | deceiving macro-F1 | 0.223012 | 0.773 | 0.549988 | no | no | `isnet/covidisnet_check.csv` |
| P3 Bassi 2024 | released COVIDIsNet.pt, size-matched test (A9) | deceiving macro-F1 | 0.259983 | 0.773 | 0.513017 | no | no | `isnet/covidisnet_check.csv` |
| P4 Saab 2022 | Binary-ERM (3 trap sets) | robust AUROC | 0.363234 | 0.359 | 0.004234 | yes | yes | `saab/p4_gate_seg_pil.csv` |
| P4 Saab 2022 | Seg-ERM (3 trap sets) | robust AUROC | 0.646773 | 0.727 | 0.080227 | no | no | `saab/p4_gate_seg_pil.csv` — A31 rerun; also misses the notebook value 0.772 |
| P5 R2R (step 1) | released Vanilla, ch test | accuracy | 0.838137 | 0.895 | 0.056863 | no | no | `r2r/step1.csv` |
| P5 R2R (step 1) | released Vanilla, ch test | macro-F1 | 0.772320 | 0.821 | 0.048680 | no | yes | `r2r/step1.csv` |
| P5 R2R (step 1) | released Vanilla, clean test | accuracy | 0.833004 | 0.895 | 0.061996 | no | no | `r2r/step1.csv` |
| P5 R2R (step 1) | released Vanilla, clean test | macro-F1 | 0.769040 | 0.821 | 0.051960 | no | no | `r2r/step1.csv` |
| P5 R2R (step 1) | released Vanilla, attacked test | accuracy | 0.552310 | 0.356 | 0.196310 | no | no | `r2r/step1.csv` |
| P5 R2R (step 1) | released Vanilla, attacked test | macro-F1 | 0.622479 | 0.390 | 0.232479 | no | no | `r2r/step1.csv` |
| P6 Yan 2023 (EPVT) | ERM, bias 1, 5 seeds | AUROC | 0.667727 | 0.653 | 0.014727 | yes | yes | `epvt/A34_gate.json` — target digitised (±≈0.005) |
| P6 Yan 2023 (EPVT) | EPVT (A36), bias 1, 5 seeds | AUROC | 0.696288 | 0.746 | 0.049712 | no | yes | `epvt/A34_gate.json` — target digitised (±≈0.005) |
| P7 Dreyer 2024 (RR-ClArC) | RR-ClArC vgg16 | biased-test accuracy | 0.531386 | 0.570 | 0.038614 | no | yes | `rrclarc/p7/vgg16_summary.csv` |
| P7 Dreyer 2024 (RR-ClArC) | RR-ClArC resnet18 | biased-test accuracy | 0.713383 | 0.712 | 0.001383 | yes | yes | `rrclarc/p7/resnet18_summary.csv` |
| P7 Dreyer 2024 (RR-ClArC) | RR-ClArC efficientnet_b0 | biased-test accuracy | 0.728385 | 0.756 | 0.027615 | yes | yes | `rrclarc/p7/efficientnet_b0_summary.csv` |
| P8 Wu 2023 (DISC) | ERM, 5 trap sets | AUROC | 0.507410 | 0.364 | 0.143410 | no | no | `disc/gate.json` |
| P8 Wu 2023 (DISC) | DISC, 5 trap sets | AUROC | 0.489418 | 0.551 | 0.061582 | no | no | `disc/gate.json` |

**Gate verdicts (registered combination rule per paper):**

| paper | rule | at 0.03 | at 0.05 |
|---|---|---|---|
| P2 Pahde 2025 | RR-ClArC | not reproduced | not reproduced |
| P3 Bassi 2024 | A9 check (either variant) | not reproduced | not reproduced |
| P4 Saab 2022 | both | not reproduced | not reproduced |
| P5 R2R (step 1) | every number | not reproduced | not reproduced |
| P6 Yan 2023 (EPVT) | both | not reproduced | reproduced |
| P7 Dreyer 2024 (RR-ClArC) | per architecture | vgg16 no, resnet18 yes, efficientnet_b0 yes | vgg16 yes, resnet18 yes, efficientnet_b0 yes |
| P8 Wu 2023 (DISC) | both | not reproduced | not reproduced |

Descriptive published-number row (P8 table): **CBDebug (Enouen & Galhotra, ICLR 2026) 0.580 ± 0.116** on the same Bissoto trap sets — *not reproduced: no ISIC code released*.

P6 note: the EPVT gap (0.049712) is inside 0.05 by less than the digitisation uncertainty of the published figure (±≈0.005); the pass is stated with that caveat.
