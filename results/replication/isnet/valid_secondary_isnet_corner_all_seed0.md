# SECONDARY (A24): isnet_corner_all_seed0 on P3's cohort-validation images, in-distribution

**Secondary:** these images were used for epoch/model selection (P3's validation split). Image-only bootstrap 95% CIs (2,000 replicates); one seed. The rebuilt external test stays the registered primary test.

Design: marks at position **corner**, presence **all** (biased = own-class shape per the presence rule; standard = no mark; deceiving = P3's swapped shapes on every image).

| condition | n | macro-F1 [95% CI] |
|---|---|---|
| biased | 3483 | 0.841 [0.829, 0.853] |
| standard | 3483 | 0.834 [0.822, 0.846] |
| deceiving | 3483 | 0.831 [0.819, 0.843] |
