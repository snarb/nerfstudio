# Static leader campaign monitor

Generated from campaign manifests and `metrics_compact.csv`; training-batch metrics are not quality evidence.

| Campaign | Seed | Status | Latest step | Legacy ARM points | Total points incl. fixed warm-up |
|---|---:|---|---:|---:|---:|
| leader_stableocc_gradfp32_F0_seed42 | 42 | complete | 106316 | 257,995,138,600 | 262,290,105,896 |
| leader_stableocc_gradfp32_F1_seed42 | 42 | complete | 106316 | 258,735,338,020 | 263,030,305,316 |

## Common full-eval checkpoints

| Step | Current / archive ARM points | All-run PSNR range | All-run SSIM range | All-run LPIPS range | Repeated-seed range P/S/L | Same-seed tolerance | Archive PSNR / SSIM / LPIPS |
|---:|---:|---:|---:|---:|---|---|---|
| 15188 | 18.206–18.252 / 17.565 B | 0.007000 | 0.003522 | 0.001158 | seed42 n=2: 0.007000/0.003522/0.001158 | pass | 28.5960 / 0.651726 / 0.371653 |
| 30376 | 52.526–52.635 / 50.671 B | 0.030600 | 0.010149 | 0.001023 | seed42 n=2: 0.030600/0.010149/0.001023 | FAIL | 29.2098 / 0.676160 / 0.305969 |
| 45564 | 90.827–90.861 / 87.566 B | 0.046500 | 0.008851 | 0.001651 | seed42 n=2: 0.046500/0.008851/0.001651 | pass | 29.3952 / 0.673553 / 0.279821 |
| 60752 | 131.089–131.216 / 126.588 B | 0.037100 | 0.007283 | 0.001027 | seed42 n=2: 0.037100/0.007283/0.001027 | pass | 29.5279 / 0.677030 / 0.262007 |
| 75940 | 172.578–172.931 / 166.919 B | 0.030800 | 0.011486 | 0.002044 | seed42 n=2: 0.030800/0.011486/0.002044 | FAIL | 29.6217 / 0.675272 / 0.252857 |
| 91128 | 214.945–215.493 / 208.140 B | 0.057600 | 0.005810 | 0.001409 | seed42 n=2: 0.057600/0.005810/0.001409 | pass | 29.6920 / 0.672744 / 0.240396 |
| 106316 | 257.995–258.735 / 250.035 B | 0.079700 | 0.009092 | 0.000174 | seed42 n=2: 0.079700/0.009092/0.000174 | FAIL | 29.6180 / 0.668451 / 0.231120 |

## Accepted scheduled candidates

| Campaign | Checkpoint | PSNR | SSIM | LPIPS | Significant artifacts | Serious ROI | Numeric | Automatic | Detail reference |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| leader_stableocc_gradfp32_F0_seed42 | step-000091128.ckpt | 29.888805 | 0.675644 | 0.220226 | 0 | 0 | pass | pass | FAIL |
| leader_stableocc_gradfp32_F1_seed42 | step-000091128.ckpt | 29.831200 | 0.681454 | 0.218816 | 0 | 0 | pass | pass | FAIL |

## Per-campaign trajectory

| Campaign | Step | ARM points | PSNR (delta) | SSIM (delta) | LPIPS (delta) | Numeric gate |
|---|---:|---:|---:|---:|---:|---|
| leader_stableocc_gradfp32_F0_seed42 | 15188 | 18.252 B | 28.759700 (+0.163700) | 0.650683 (-0.001043) | 0.360250 (-0.011403) | FAIL |
| leader_stableocc_gradfp32_F0_seed42 | 30376 | 52.635 B | 29.402600 (+0.192800) | 0.665068 (-0.011092) | 0.294471 (-0.011498) | FAIL |
| leader_stableocc_gradfp32_F0_seed42 | 45564 | 90.861 B | 29.729100 (+0.333900) | 0.670683 (-0.002870) | 0.262234 (-0.017587) | FAIL |
| leader_stableocc_gradfp32_F0_seed42 | 60752 | 131.089 B | 29.827500 (+0.299600) | 0.671679 (-0.005351) | 0.245760 (-0.016247) | FAIL |
| leader_stableocc_gradfp32_F0_seed42 | 75940 | 172.578 B | 29.901200 (+0.279500) | 0.673394 (-0.001878) | 0.234071 (-0.018786) | FAIL |
| leader_stableocc_gradfp32_F0_seed42 | 91128 | 214.945 B | 29.888800 (+0.196800) | 0.675644 (+0.002900) | 0.220255 (-0.020141) | pass |
| leader_stableocc_gradfp32_F0_seed42 | 106316 | 257.995 B | 29.858300 (+0.240334) | 0.670751 (+0.002300) | 0.211213 (-0.019907) | pass |
| leader_stableocc_gradfp32_F1_seed42 | 15188 | 18.206 B | 28.766700 (+0.170700) | 0.647161 (-0.004565) | 0.359092 (-0.012561) | FAIL |
| leader_stableocc_gradfp32_F1_seed42 | 30376 | 52.526 B | 29.433200 (+0.223400) | 0.675217 (-0.000943) | 0.295494 (-0.010475) | FAIL |
| leader_stableocc_gradfp32_F1_seed42 | 45564 | 90.827 B | 29.682600 (+0.287400) | 0.679534 (+0.005981) | 0.260583 (-0.019238) | FAIL |
| leader_stableocc_gradfp32_F1_seed42 | 60752 | 131.216 B | 29.790400 (+0.262500) | 0.678962 (+0.001932) | 0.244733 (-0.017274) | FAIL |
| leader_stableocc_gradfp32_F1_seed42 | 75940 | 172.931 B | 29.870400 (+0.248700) | 0.684880 (+0.009608) | 0.232027 (-0.020830) | FAIL |
| leader_stableocc_gradfp32_F1_seed42 | 91128 | 215.493 B | 29.831200 (+0.139200) | 0.681454 (+0.008710) | 0.218846 (-0.021550) | pass |
| leader_stableocc_gradfp32_F1_seed42 | 106316 | 258.735 B | 29.778600 (+0.160634) | 0.679843 (+0.011392) | 0.211387 (-0.019733) | pass |
