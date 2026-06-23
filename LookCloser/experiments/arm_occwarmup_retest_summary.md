Per-run metrics:

| Seed | Selected step | Eval loss | PSNR | SSIM | LPIPS | Artifact score | Serious artifact | ROI score | ROI serious score | ROI serious count | Stand connector | Train time | Eval time | Artifact time | Total time | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 8192 | 0.02973320 | 28.501562 | 0.652204 | 0.447453 | 0.280000 | 0.280000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1713.262s | 40.230s | 28.170s | 2056.347s | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h40_newocc_seed42/renders_artifact_selection_step-000008192` |
| 42 | 16384 | 0.02799450 | 29.410427 | 0.679574 | 0.403663 | 0.469000 | 0.469000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2254.054s | 41.392s | 24.366s | 2594.566s | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h41_newocc_seed42/renders_artifact_selection_step-000016384` |

Mean metrics:

| SSIM | LPIPS | PSNR | Eval loss | Artifact score | Serious artifact | ROI score | ROI serious score | ROI serious count | Stand connector | Train time | Eval time | Artifact time | Total time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.665889 | 0.425558 | 28.955995 | 0.02886385 | 0.374500 | 0.374500 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1983.658s | 40.811s | 26.268s | 2325.456s |

Best single result by metric:

| Metric | Best seed | Value | Render directory |
|---|---:|---:|---|
| Artifact score, lower better | 42 | 0.280000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h40_newocc_seed42/renders_artifact_selection_step-000008192` |
| Serious artifact score, lower better | 42 | 0.280000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h40_newocc_seed42/renders_artifact_selection_step-000008192` |
| ROI artifact score, lower better | 42 | 0.000000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h40_newocc_seed42/renders_artifact_selection_step-000008192` |
| ROI serious artifact score, lower better | 42 | 0.000000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h40_newocc_seed42/renders_artifact_selection_step-000008192` |
| ROI serious count, lower better | 42 | 0.000000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h40_newocc_seed42/renders_artifact_selection_step-000008192` |
| Stand connector ROI, lower better | 42 | 0.000000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h40_newocc_seed42/renders_artifact_selection_step-000008192` |
| SSIM, higher better | 42 | 0.679574 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h41_newocc_seed42/renders_artifact_selection_step-000016384` |
| LPIPS, lower better | 42 | 0.403663 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h41_newocc_seed42/renders_artifact_selection_step-000016384` |
| PSNR, higher better | 42 | 29.410427 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h41_newocc_seed42/renders_artifact_selection_step-000016384` |
| Eval loss, lower better | 42 | 0.027994 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h41_newocc_seed42/renders_artifact_selection_step-000016384` |
| Train time, lower better | 42 | 1713.262496s | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h40_newocc_seed42/renders_artifact_selection_step-000008192` |
| Eval time, lower better | 42 | 40.230113s | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h40_newocc_seed42/renders_artifact_selection_step-000008192` |
| Artifact detector time, lower better | 42 | 24.366194s | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h41_newocc_seed42/renders_artifact_selection_step-000016384` |
| Total time, lower better | 42 | 2056.346581s | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h40_newocc_seed42/renders_artifact_selection_step-000008192` |
