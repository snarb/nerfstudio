# Static leader campaign monitor

Generated from campaign manifests and `metrics_compact.csv`; training-batch metrics are not quality evidence.

| Campaign | Seed | Status | Latest step | Legacy ARM points | Total points incl. fixed warm-up |
|---|---:|---|---:|---:|---:|
| leader_stableocc_S0_seed42 | 42 | complete | 106316 | 259,835,319,470 | 264,130,286,766 |
| leader_stableocc_S1_seed42 | 42 | complete | 106316 | 257,053,358,820 | 261,348,326,116 |

## Common full-eval checkpoints

| Step | Current / archive ARM points | All-run PSNR range | All-run SSIM range | All-run LPIPS range | Repeated-seed range P/S/L | Same-seed tolerance | Archive PSNR / SSIM / LPIPS |
|---:|---:|---:|---:|---:|---|---|---|
| 15188 | 18.035–18.211 / 17.565 B | 0.033900 | 0.001696 | 0.003957 | seed42 n=2: 0.033900/0.001696/0.003957 | FAIL | 28.5960 / 0.651726 / 0.371653 |
| 30376 | 52.211–52.633 / 50.671 B | 0.035000 | 0.000033 | 0.003597 | seed42 n=2: 0.035000/0.000033/0.003597 | FAIL | 29.2098 / 0.676160 / 0.305969 |
| 45564 | 90.218–91.056 / 87.566 B | 0.032100 | 0.001386 | 0.002619 | seed42 n=2: 0.032100/0.001386/0.002619 | FAIL | 29.3952 / 0.673553 / 0.279821 |
| 60752 | 130.340–131.589 / 126.588 B | 0.038800 | 0.000502 | 0.001427 | seed42 n=2: 0.038800/0.000502/0.001427 | FAIL | 29.5279 / 0.677030 / 0.262007 |
| 75940 | 171.770–173.479 / 166.919 B | 0.040000 | 0.001424 | 0.001169 | seed42 n=2: 0.040000/0.001424/0.001169 | FAIL | 29.6217 / 0.675272 / 0.252857 |
| 91128 | 214.088–216.287 / 208.140 B | 0.015000 | 0.003384 | 0.000710 | seed42 n=2: 0.015000/0.003384/0.000710 | FAIL | 29.6920 / 0.672744 / 0.240396 |
| 106316 | 257.053–259.835 / 250.035 B | 0.049100 | 0.000246 | 0.000932 | seed42 n=2: 0.049100/0.000246/0.000932 | FAIL | 29.6180 / 0.668451 / 0.231120 |

## Accepted scheduled candidates

| Campaign | Checkpoint | PSNR | SSIM | LPIPS | Significant artifacts | Serious ROI | Numeric | Automatic | Detail reference |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| leader_stableocc_S1_seed42 | step-000091128.ckpt | 29.840143 | 0.669203 | 0.219455 | 0 | 0 | pass | pass | FAIL |

## Per-campaign trajectory

| Campaign | Step | ARM points | PSNR (delta) | SSIM (delta) | LPIPS (delta) | Numeric gate |
|---|---:|---:|---:|---:|---:|---|
| leader_stableocc_S0_seed42 | 15188 | 18.211 B | 28.781900 (+0.185900) | 0.647596 (-0.004130) | 0.357641 (-0.014012) | FAIL |
| leader_stableocc_S0_seed42 | 30376 | 52.633 B | 29.419600 (+0.209800) | 0.660633 (-0.015527) | 0.291537 (-0.014432) | FAIL |
| leader_stableocc_S0_seed42 | 45564 | 91.056 B | 29.724800 (+0.329600) | 0.668729 (-0.004824) | 0.260964 (-0.018857) | FAIL |
| leader_stableocc_S0_seed42 | 60752 | 131.589 B | 29.818300 (+0.290400) | 0.669892 (-0.007138) | 0.243945 (-0.018062) | FAIL |
| leader_stableocc_S0_seed42 | 75940 | 173.479 B | 29.885500 (+0.263800) | 0.673041 (-0.002231) | 0.233830 (-0.019027) | FAIL |
| leader_stableocc_S0_seed42 | 91128 | 216.287 B | 29.855100 (+0.163100) | 0.672587 (-0.000157) | 0.220189 (-0.020207) | pass |
| leader_stableocc_S0_seed42 | 106316 | 259.835 B | 29.866600 (+0.248634) | 0.667140 (-0.001311) | 0.212544 (-0.018576) | FAIL |
| leader_stableocc_S1_seed42 | 15188 | 18.035 B | 28.748000 (+0.152000) | 0.645900 (-0.005826) | 0.361598 (-0.010055) | FAIL |
| leader_stableocc_S1_seed42 | 30376 | 52.211 B | 29.454600 (+0.244800) | 0.660666 (-0.015494) | 0.295134 (-0.010835) | FAIL |
| leader_stableocc_S1_seed42 | 45564 | 90.218 B | 29.692700 (+0.297500) | 0.667343 (-0.006210) | 0.263583 (-0.016238) | FAIL |
| leader_stableocc_S1_seed42 | 60752 | 130.340 B | 29.779500 (+0.251600) | 0.669390 (-0.007640) | 0.245372 (-0.016635) | FAIL |
| leader_stableocc_S1_seed42 | 75940 | 171.770 B | 29.845500 (+0.223800) | 0.671617 (-0.003655) | 0.232661 (-0.020196) | FAIL |
| leader_stableocc_S1_seed42 | 91128 | 214.088 B | 29.840100 (+0.148100) | 0.669203 (-0.003541) | 0.219479 (-0.020917) | pass |
| leader_stableocc_S1_seed42 | 106316 | 257.053 B | 29.817500 (+0.199534) | 0.667386 (-0.001065) | 0.213476 (-0.017644) | FAIL |
