# SECONDARY (A24): isnet_inside_9010_seed0 on P3's cohort-validation images, in-distribution

**Secondary:** these images were used for epoch/model selection (P3's validation split). Image-only bootstrap 95% CIs (2,000 replicates); one seed. The rebuilt external test stays the registered primary test.

Design: marks at position **inside**, presence **9010** (biased = own-class shape per the presence rule; standard = no mark; deceiving = P3's swapped shapes on every image).

| condition | n | macro-F1 [95% CI] |
|---|---|---|
| biased | 3483 | 0.974 [0.969, 0.979] |
| standard | 3483 | 0.745 [0.730, 0.759] |
| deceiving | 3483 | 0.033 [0.028, 0.039] |
