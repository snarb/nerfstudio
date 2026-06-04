# Scene Scale 1.5 Reproducible Runs

Dataset:
`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns`

Scope: `scene_scale=1.5` clean reproducibility runs only. The unreproduced baseline run is intentionally excluded from these aggregate numbers.

| Run | Timestamp | Selected checkpoint | Eval loss | PSNR | SSIM | LPIPS |
|---|---|---|---:|---:|---:|---:|
| A | `repro_scene150_clean_A_20260528_081832` | `step-000015188.ckpt` | 0.00438475 | 23.715738 | 0.637653 | 0.499827 |
| B | `repro_scene150_clean_B_20260528_083138` | `step-000045564.ckpt` | 0.00410401 | 23.816729 | 0.666949 | 0.470307 |
| C | `repro_scene150_clean_C_20260528_085748` | `step-000030376.ckpt` | 0.00432915 | 23.659456 | 0.647519 | 0.479323 |

## Aggregates

| Statistic | Eval loss | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Mean | 0.00427264 | 23.730641 | 0.650707 | 0.483152 |
| Max | 0.00438475 | 23.816729 | 0.666949 | 0.499827 |
| Min | 0.00410401 | 23.659456 | 0.637653 | 0.470307 |

Best-by-metric:

| Metric | Best run | Value |
|---|---|---:|
| Eval loss, lower is better | B | 0.00410401 |
| PSNR, higher is better | B | 23.816729 |
| SSIM, higher is better | B | 0.666949 |
| LPIPS, lower is better | B | 0.470307 |
