Per-run metrics:

| Seed | Selected step | Eval loss | PSNR | SSIM | LPIPS | Artifact score | Serious artifact | ROI score | ROI serious score | ROI serious count | Stand connector | Train time | Eval time | Artifact time | Total time | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 4096 | 0.08352780 | 19.613150 | 0.640016 | 0.530901 | 0.415000 | 0.415000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3606.318s | 46.047s | 33.311s | 4114.164s | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs025_newocc_seed42/renders_artifact_selection_step-000004096` |
| 42 | 4096 | 0.08315150 | 19.717342 | 0.640571 | 0.527097 | 0.279000 | 0.279000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2614.243s | 46.088s | 33.547s | 2979.758s | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs050_newocc_seed42/renders_artifact_selection_step-000004096` |

Mean metrics:

| SSIM | LPIPS | PSNR | Eval loss | Artifact score | Serious artifact | ROI score | ROI serious score | ROI serious count | Stand connector | Train time | Eval time | Artifact time | Total time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.640294 | 0.528999 | 19.665246 | 0.08333965 | 0.347000 | 0.347000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3110.281s | 46.068s | 33.429s | 3546.961s |

Best single result by metric:

| Metric | Best seed | Value | Render directory |
|---|---:|---:|---|
| Artifact score, lower better | 42 | 0.279000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs050_newocc_seed42/renders_artifact_selection_step-000004096` |
| Serious artifact score, lower better | 42 | 0.279000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs050_newocc_seed42/renders_artifact_selection_step-000004096` |
| ROI artifact score, lower better | 42 | 0.000000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs025_newocc_seed42/renders_artifact_selection_step-000004096` |
| ROI serious artifact score, lower better | 42 | 0.000000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs025_newocc_seed42/renders_artifact_selection_step-000004096` |
| ROI serious count, lower better | 42 | 0.000000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs025_newocc_seed42/renders_artifact_selection_step-000004096` |
| Stand connector ROI, lower better | 42 | 0.000000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs025_newocc_seed42/renders_artifact_selection_step-000004096` |
| SSIM, higher better | 42 | 0.640571 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs050_newocc_seed42/renders_artifact_selection_step-000004096` |
| LPIPS, lower better | 42 | 0.527097 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs050_newocc_seed42/renders_artifact_selection_step-000004096` |
| PSNR, higher better | 42 | 19.717342 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs050_newocc_seed42/renders_artifact_selection_step-000004096` |
| Eval loss, lower better | 42 | 0.083152 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs050_newocc_seed42/renders_artifact_selection_step-000004096` |
| Train time, lower better | 42 | 2614.243489s | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs050_newocc_seed42/renders_artifact_selection_step-000004096` |
| Eval time, lower better | 42 | 46.047418s | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs025_newocc_seed42/renders_artifact_selection_step-000004096` |
| Artifact detector time, lower better | 42 | 33.310798s | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs025_newocc_seed42/renders_artifact_selection_step-000004096` |
| Total time, lower better | 42 | 2979.757946s | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs050_newocc_seed42/renders_artifact_selection_step-000004096` |
