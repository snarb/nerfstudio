# LookCloser Frequency Grid Optimization

## What was tested

- Dataset: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns`
- Seeds per candidate: `42, 43, 44`
- Max iterations per run: `60752`
- Checkpoint protocol: final eval on the best in-training eval-loss checkpoint.
- Selection order: mean SSIM, mean LPIPS, mean PSNR, mean eval loss, mean training time.

## Results

- Current best carried config: `{"background_color": "black", "center_method": "focus", "enable_adaptive_ray_marching": false, "enable_fas": true, "enable_feature_reweighting": true, "enable_frequency_grid": true, "fallback_frequency_level": 0.0, "grid_resolution": 128, "grid_update_batch_size": 2048, "grid_update_interval": 1024, "max_res": null, "max_res_base": 2048.0, "min_res": 16.0, "num_frequency_levels": 16, "orientation_method": "up", "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.15, "scene_scale": 2.0, "train_num_rays_per_batch": 1024}`
- Mean metrics: SSIM `0.554821`, LPIPS `0.425556`, PSNR `25.707525`, eval loss `0.03668333`, training time `2885.513s`
- Best SSIM candidate: `control/current=baseline` with max SSIM `0.555698`
- Best LPIPS candidate: `control/current=baseline` with min LPIPS `0.415132`
- Best PSNR candidate: `control/current=baseline` with max PSNR `25.803509`
- Best eval-loss candidate: `control/current=baseline` with min eval loss `0.03651520`
- Best single run render directory: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_frequency_grid/lookcloser/control_current_baseline_seed43/renders_best_step-000045564`

| Stage | Param | Value | Rank | Carried | Mean SSIM | Max SSIM | Mean LPIPS | Min LPIPS | Mean PSNR | Max PSNR | Mean Eval Loss | Min Eval Loss | Mean Train s | Min Train s | Config |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| control | current | `baseline` | 1 | yes | 0.554821 | 0.555698 | 0.425556 | 0.415132 | 25.707525 | 25.803509 | 0.03668333 | 0.03651520 | 2885.513 | 1743.550 | `{"background_color": "black", "center_method": "focus", "enable_adaptive_ray_marching": false, "enable_fas": true, "enable_feature_reweighting": true, "enable_frequency_grid": true, "fallback_frequency_level": 0.0, "grid_resolution": 128, "grid_update_batch_size": 2048, "grid_update_interval": 1024, "max_res": null, "max_res_base": 2048.0, "min_res": 16.0, "num_frequency_levels": 16, "orientation_method": "up", "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.15, "scene_scale": 2.0, "train_num_rays_per_batch": 1024}` |

## Per-run results

| Timestamp | Stage | Param | Value | Seed | Checkpoint | Eval Loss | PSNR | SSIM | LPIPS | Train s | Eval JSON | Renders |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| control_current_baseline_seed42 | control | current | `baseline` | 42 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_frequency_grid/lookcloser/control_current_baseline_seed42/nerfstudio_models/step-000045564.ckpt` | 0.036515 | 25.731192 | 0.554881 | 0.416669 | 3456.511000 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_frequency_grid/lookcloser/control_current_baseline_seed42/eval_best_step-000045564.json` | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_frequency_grid/lookcloser/control_current_baseline_seed42/renders_best_step-000045564` |
| control_current_baseline_seed43 | control | current | `baseline` | 43 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_frequency_grid/lookcloser/control_current_baseline_seed43/nerfstudio_models/step-000045564.ckpt` | 0.036728 | 25.803509 | 0.555698 | 0.415132 | 3456.477961 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_frequency_grid/lookcloser/control_current_baseline_seed43/eval_best_step-000045564.json` | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_frequency_grid/lookcloser/control_current_baseline_seed43/renders_best_step-000045564` |
| control_current_baseline_seed44 | control | current | `baseline` | 44 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_frequency_grid/lookcloser/control_current_baseline_seed44/nerfstudio_models/step-000015188.ckpt` | 0.036807 | 25.587873 | 0.553884 | 0.444867 | 1743.550332 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_frequency_grid/lookcloser/control_current_baseline_seed44/eval_best_step-000015188.json` | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_frequency_grid/lookcloser/control_current_baseline_seed44/renders_best_step-000015188` |

## Insights

- Visual inspection of seed 43 renders shows plausible global structure but heavy smoothing on cables, brick mortar, floor cracks, labels, and thin rigging. The current baseline is therefore usable for comparisons, but not yet acceptable for tiny-detail reconstruction.
- Baseline variance is non-trivial: seed 44 stopped at the first eval and had worse LPIPS, while seeds 42/43 improved through step 45564. Continue using 3-seed means for decisions.
- Next experiment: schedule hyperparameters (`max_res_base`, then `num_frequency_levels`) using the same preprocessing maps and fixed-ray-marching baseline.
