# SECONDARY (A24): isnet_inside_all_seed0 on P3's cohort-validation images, in-distribution

**Secondary:** these images were used for epoch/model selection (P3's validation split). Image-only bootstrap 95% CIs (2,000 replicates); one seed. The rebuilt external test stays the registered primary test.

Design: marks at position **inside**, presence **all** (biased = own-class shape per the presence rule; standard = no mark; deceiving = P3's swapped shapes on every image).

| condition | n | macro-F1 [95% CI] |
|---|---|---|
| biased | 3483 | 0.998 [0.997, 0.999] |
| standard | 3483 | 0.167 [0.161, 0.173] |
| deceiving | 3483 | 0.000 [0.000, 0.000] |
