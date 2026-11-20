# P8 gate (DISC, ICML 2023; A32, A35, A37) — machine 2

Their code and commands; one data root per trap set (A37, caches verified row by row); per set the test ROC-AUC (618 images, DISC's split) at the epoch with the highest validation ROC-AUC (154 images).

| method | trap set 1 | 2 | 3 | 4 | 5 | mean ± SD | paper | Δ | within 0.03 |
|---|---|---|---|---|---|---|---|---|---|
| ERM | 0.531 (ep 18) | 0.506 (ep 14) | 0.524 (ep 70) | 0.502 (ep 92) | 0.474 (ep 71) | 0.507 ± 0.022 | 0.364 | +0.143 | no |
| DISC | 0.451 (ep 22) | 0.493 (ep 36) | 0.514 (ep 27) | 0.525 (ep 13) | 0.464 (ep 72) | 0.489 ± 0.032 | 0.551 | -0.062 | no |

**Gate: FAILED.** ERM mean misses 0.364 by > 0.03: gate FAILED; the published numbers may carry the shared-cache bug of DISC's released code (A37 hypothesis).
