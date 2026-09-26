# Synthetic rerun from raw images vs archived pilot

Per-seed AUROC agreement (rerun − archived):

| method                   | env       |   n |   mean_abs_diff |   max_abs_diff |
|:-------------------------|:----------|----:|----------------:|---------------:|
| balanced                 | clean     |   9 |          0.0001 |         0.0003 |
| balanced                 | test_corr |   9 |          0.0001 |         0.0003 |
| balanced                 | test_rev  |   9 |          0.0001 |         0.0005 |
| dfr                      | clean     |   9 |          0.0001 |         0.0003 |
| dfr                      | test_corr |   9 |          0.0001 |         0.0002 |
| dfr                      | test_rev  |   9 |          0.0001 |         0.0002 |
| erm                      | clean     |   9 |          0.0001 |         0.0002 |
| erm                      | test_corr |   9 |          0.0001 |         0.0001 |
| erm                      | test_rev  |   9 |          0.0002 |         0.0003 |
| groupdro                 | clean     |   9 |          0.0001 |         0.0003 |
| groupdro                 | test_corr |   9 |          0.0001 |         0.0003 |
| groupdro                 | test_rev  |   9 |          0.0001 |         0.0002 |
| inpaint                  | clean     |   9 |          0.0001 |         0.0002 |
| inpaint                  | test_corr |   9 |          0      |         0.0001 |
| inpaint                  | test_rev  |   9 |          0.0001 |         0.0002 |
| inpaint_consistency      | clean     |   9 |          0.0002 |         0.0005 |
| inpaint_consistency      | test_corr |   9 |          0.0009 |         0.0024 |
| inpaint_consistency      | test_rev  |   9 |          0.0021 |         0.0082 |
| inpaint_consistency_lam0 | clean     |  12 |          0.0001 |         0.0003 |
| inpaint_consistency_lam0 | test_corr |  12 |          0.0001 |         0.0002 |
| inpaint_consistency_lam0 | test_rev  |  12 |          0.0002 |         0.0004 |
| leace                    | clean     |   9 |          0.0001 |         0.0001 |
| leace                    | test_corr |   9 |          0.0001 |         0.0002 |
| leace                    | test_rev  |   9 |          0.0001 |         0.0002 |
| mask                     | clean     |   9 |          0.0001 |         0.0002 |
| mask                     | test_corr |   9 |          0      |         0.0001 |
| mask                     | test_rev  |   9 |          0.0001 |         0.0002 |

Reversed-test Δ vs ERM (hierarchical bootstrap):

| method              |   overlap |   seed_delta_mean_arch |   ci95_lo_arch |   ci95_hi_arch |   seed_delta_mean_new |   ci95_lo_new |   ci95_hi_new |   delta_diff | same_sign_and_ci_decision   |
|:--------------------|----------:|-----------------------:|---------------:|---------------:|----------------------:|--------------:|--------------:|-------------:|:----------------------------|
| mask                |       0   |                  0.184 |          0.146 |          0.224 |                 0.184 |         0.146 |         0.224 |        0     | True                        |
| inpaint             |       0   |                  0.141 |          0.12  |          0.162 |                 0.141 |         0.12  |         0.162 |        0     | True                        |
| inpaint_consistency |       0   |                  0.038 |          0.011 |          0.065 |                 0.042 |         0.015 |         0.069 |        0.004 | True                        |
| mask                |       0.5 |                 -0.097 |         -0.138 |         -0.056 |                -0.097 |        -0.138 |        -0.056 |       -0     | True                        |
| inpaint             |       0.5 |                  0.211 |          0.187 |          0.234 |                 0.211 |         0.187 |         0.234 |       -0     | True                        |
| inpaint_consistency |       0.5 |                  0.046 |          0.017 |          0.076 |                 0.043 |         0.014 |         0.072 |       -0.003 | True                        |
| mask                |       1   |                 -0.128 |         -0.17  |         -0.086 |                -0.129 |        -0.17  |        -0.087 |       -0     | True                        |
| inpaint             |       1   |                  0.244 |          0.218 |          0.269 |                 0.243 |         0.218 |         0.269 |       -0     | True                        |
| inpaint_consistency |       1   |                  0.043 |          0.012 |          0.073 |                 0.042 |         0.012 |         0.073 |       -0     | True                        |
| balanced            |       0   |                  0.138 |          0.109 |          0.168 |                 0.137 |         0.109 |         0.168 |       -0     | True                        |
| groupdro            |       0   |                  0.081 |          0.056 |          0.105 |                 0.08  |         0.056 |         0.105 |       -0     | True                        |
| leace               |       0   |                  0.167 |          0.139 |          0.197 |                 0.167 |         0.139 |         0.197 |        0     | True                        |
| dfr                 |       0   |                  0.124 |          0.091 |          0.161 |                 0.124 |         0.091 |         0.16  |       -0     | True                        |
| balanced            |       0.5 |                  0.219 |          0.188 |          0.252 |                 0.219 |         0.188 |         0.252 |        0     | True                        |
| groupdro            |       0.5 |                  0.152 |          0.124 |          0.179 |                 0.152 |         0.124 |         0.18  |        0     | True                        |
| leace               |       0.5 |                  0.237 |          0.212 |          0.263 |                 0.237 |         0.212 |         0.263 |       -0     | True                        |
| dfr                 |       0.5 |                  0.221 |          0.184 |          0.261 |                 0.221 |         0.184 |         0.261 |       -0     | True                        |
| balanced            |       1   |                  0.281 |          0.249 |          0.315 |                 0.281 |         0.248 |         0.315 |       -0     | True                        |
| groupdro            |       1   |                  0.197 |          0.168 |          0.226 |                 0.197 |         0.168 |         0.226 |       -0     | True                        |
| leace               |       1   |                  0.294 |          0.265 |          0.325 |                 0.294 |         0.265 |         0.325 |       -0     | True                        |
| dfr                 |       1   |                  0.279 |          0.24  |          0.319 |                 0.279 |         0.24  |         0.32  |       -0     | True                        |
