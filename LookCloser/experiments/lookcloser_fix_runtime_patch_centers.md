# LookCloser Frequency Grid Optimization

## What was tested

- Dataset: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns`
- Seeds per candidate: `42, 43, 44`
- Max iterations per run: `60752`
- Checkpoint protocol: final eval on the best in-training eval-loss checkpoint.
- Selection order: mean SSIM, mean LPIPS, mean PSNR, mean eval loss, mean training time.

## Results

- Current best carried config: `{"background_color": "black", "center_method": "focus", "enable_adaptive_ray_marching": false, "enable_fas": true, "enable_feature_reweighting": true, "enable_frequency_grid": true, "fallback_frequency_level": 0.0, "grid_resolution": 64, "grid_update_batch_size": 4096, "grid_update_interval": 512, "max_res": null, "max_res_base": 2048.0, "min_res": 16.0, "num_frequency_levels": 16, "orientation_method": "up", "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.15, "scene_scale": 2.0, "train_num_rays_per_batch": 1024}`
- Mean metrics: SSIM `0.555012`, LPIPS `0.431763`, PSNR `25.700442`, eval loss `0.03699280`, training time `2334.873s`
- Best SSIM candidate: `control/current=baseline` with max SSIM `0.556117`
- Best LPIPS candidate: `control/current=baseline` with min LPIPS `0.421140`
- Best PSNR candidate: `control/current=baseline` with max PSNR `25.815132`
- Best eval-loss candidate: `control/current=baseline` with min eval loss `0.03610380`
- Best single run render directory: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_patch_centers/lookcloser/control_current_baseline_seed43/renders_best_step-000030376`

| Stage | Param | Value | Rank | Carried | Mean SSIM | Max SSIM | Mean LPIPS | Min LPIPS | Mean PSNR | Max PSNR | Mean Eval Loss | Min Eval Loss | Mean Train s | Min Train s | Config |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| control | current | `baseline` | 1 | yes | 0.555012 | 0.556117 | 0.431763 | 0.421140 | 25.700442 | 25.815132 | 0.03699280 | 0.03610380 | 2334.873 | 1743.746 | `{"background_color": "black", "center_method": "focus", "enable_adaptive_ray_marching": false, "enable_fas": true, "enable_feature_reweighting": true, "enable_frequency_grid": true, "fallback_frequency_level": 0.0, "grid_resolution": 64, "grid_update_batch_size": 4096, "grid_update_interval": 512, "max_res": null, "max_res_base": 2048.0, "min_res": 16.0, "num_frequency_levels": 16, "orientation_method": "up", "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.15, "scene_scale": 2.0, "train_num_rays_per_batch": 1024}` |

## Per-run results

| Timestamp | Stage | Param | Value | Seed | Checkpoint | Eval Loss | PSNR | SSIM | LPIPS | Train s | Eval JSON | Renders |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| control_current_baseline_seed42 | control | current | `baseline` | 42 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_patch_centers/lookcloser/control_current_baseline_seed42/nerfstudio_models/step-000030376.ckpt` | 0.037135 | 25.701994 | 0.554071 | 0.424751 | 2645.454002 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_patch_centers/lookcloser/control_current_baseline_seed42/eval_best_step-000030376.json` | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_patch_centers/lookcloser/control_current_baseline_seed42/renders_best_step-000030376` |
| control_current_baseline_seed43 | control | current | `baseline` | 43 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_patch_centers/lookcloser/control_current_baseline_seed43/nerfstudio_models/step-000030376.ckpt` | 0.036104 | 25.815132 | 0.556117 | 0.421140 | 2615.419393 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_patch_centers/lookcloser/control_current_baseline_seed43/eval_best_step-000030376.json` | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_patch_centers/lookcloser/control_current_baseline_seed43/renders_best_step-000030376` |
| control_current_baseline_seed44 | control | current | `baseline` | 44 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_patch_centers/lookcloser/control_current_baseline_seed44/nerfstudio_models/step-000015188.ckpt` | 0.037739 | 25.584200 | 0.554847 | 0.449399 | 1743.746168 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_patch_centers/lookcloser/control_current_baseline_seed44/eval_best_step-000015188.json` | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_patch_centers/lookcloser/control_current_baseline_seed44/renders_best_step-000015188` |

## Insights

- Compared to the carried update-sweep baseline (`SSIM=0.555427`, `LPIPS=0.425247`, `PSNR=25.729128`, `eval loss=0.03694857`, `train=2635.440s`), patch-center runtime updates are worse on the primary mean metrics: SSIM `-0.000415`, LPIPS `+0.006516` (worse), PSNR `-0.028686`, and eval loss `+0.00004423` (worse).
- The code path was stable across all three seeds, but the mean metric regression says this implementation-doubt fix should be reverted for now.
- To revert only this experiment's code change, restore the `_update_frequency_grid` sampling block in `nerfstudio/pipelines/lookcloser_pipeline.py` from patch-cell sampling back to the prior arbitrary-pixel sampling plus pixel-to-frequency-map lookup. The relevant hunk begins at `# --- 1. Sample Random Patches ---` and ends at construction of `f2d_tensor`.
