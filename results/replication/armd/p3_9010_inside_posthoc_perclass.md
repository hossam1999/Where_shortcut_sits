# POST HOC, exploratory — arm d on P3 90/10 inside with mode = per_class (author, 2026-11-18)

**Not a registered result.** Same EfficientNet-B0 features as the registered arm d; mode fixed to per_class (span of the 3 per-class mark directions); C by the same validation rule (macro-F1 on P3's validation split with its marks, A21 grid): C = 1 (edge: no; validation macro-F1 0.851). Test scored once. The registered row (pooled, C = 0.03) is shown for comparison.

| test | variant | registered (pooled, C = 0.03) | POST HOC per_class |
|---|---|---|---|
| biased | primary | 0.997 | 0.906 |
| biased | size-matched | 0.996 | 0.892 |
| standard | primary | 0.410 | 0.404 |
| standard | size-matched | 0.404 | 0.398 |
| deceiving | primary | 0.010 | 0.048 |
| deceiving | size-matched | 0.008 | 0.037 |

Validation macro-F1 by C (per_class): 0.001: 0.772, 0.003: 0.805, 0.01: 0.830, 0.03: 0.843, 0.1: 0.851, 1: 0.851, 10: 0.838
