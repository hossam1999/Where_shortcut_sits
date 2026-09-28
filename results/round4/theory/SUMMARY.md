# R11 — prospective theory test (fit on clean + correlated, predict reversed; no shared parameter)

- **T1'** reversed AUROC, 84 new cells: theory MAE 0.011, r = 0.996, 100 % within ±0.05; reference (a) rev = clean MAE 0.177; reference (b) MAE 0.036.
- **T2'** R8 crossovers: signs 16/16, r = 0.992, MAE 0.011.
- **T3'** R9 slopes: signs 4/4, MAE 0.018; bin gains: signs 18/18, MAE 0.014.

**Verdict by the pre-registered rule: quantitatively predictive.**

Crossovers (observed vs predicted):

| cohort | view | obs | pred |
|---|---|---|---|
| Capsule debris | crop_box | 0.317 | 0.307 |
| Capsule debris | crop_mask | 0.358 | 0.343 |
| Capsule debris | mask | 0.377 | 0.357 |
| Capsule debris | mask_black | 0.367 | 0.347 |
| Capsule debris | mask_blur | 0.278 | 0.249 |
| ISIC hair | crop_box | 0.091 | 0.089 |
| ISIC hair | crop_mask | 0.154 | 0.154 |
| ISIC hair | mask | 0.147 | 0.150 |
| ISIC hair | mask_black | 0.170 | 0.182 |
| ISIC hair | mask_blur | 0.154 | 0.164 |
| Ovary calipers | crop_box | 0.145 | 0.158 |
| Ovary calipers | crop_mask | 0.160 | 0.163 |
| Ovary calipers | mask | 0.170 | 0.160 |
| Ovary calipers | mask_black | 0.153 | 0.151 |
| Ovary calipers | mask_blur | 0.112 | 0.094 |
| Thyroid calipers | crop_box | 0.246 | 0.223 |
| Thyroid calipers | crop_mask | 0.250 | 0.232 |
| Thyroid calipers | mask | 0.239 | 0.228 |
| Thyroid calipers | mask_black | 0.223 | 0.220 |
| Thyroid calipers | mask_blur | 0.218 | 0.214 |

Slopes (observed vs predicted):

| cohort | n_bins | obs_slope | pred_slope |
|---|---|---|---|
| Capsule debris | 5 | -0.489 | -0.454 |
| ISIC hair | 5 | -0.235 | -0.238 |
| Ovary calipers | 3 | -0.205 | -0.186 |
| Thyroid calipers | 5 | -0.250 | -0.263 |
