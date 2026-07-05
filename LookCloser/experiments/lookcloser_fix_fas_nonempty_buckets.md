# LookCloser Frequency Grid Optimization

## What was tested

- Dataset: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns`
- Seeds per candidate: `42, 43, 44`
- Max iterations per run: `60752`
- Checkpoint protocol: final eval on the best in-training eval-loss checkpoint.
- Selection order: mean SSIM, mean LPIPS, mean PSNR, mean eval loss, mean training time.

## Results

- Current best carried config: `{"background_color": "black", "center_method": "focus", "enable_adaptive_ray_marching": false, "enable_fas": true, "enable_feature_reweighting": true, "enable_frequency_grid": true, "fallback_frequency_level": 0.0, "grid_resolution": 64, "grid_update_batch_size": 4096, "grid_update_interval": 512, "max_res": null, "max_res_base": 2048.0, "min_res": 16.0, "num_frequency_levels": 16, "orientation_method": "up", "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.15, "scene_scale": 2.0, "train_num_rays_per_batch": 1024}`
- Mean metrics: SSIM `0.555089`, LPIPS `0.423117`, PSNR `25.745732`, eval loss `0.03662763`, training time `2625.339s`
- Best SSIM candidate: `control/current=baseline` with max SSIM `0.555835`
- Best LPIPS candidate: `control/current=baseline` with min LPIPS `0.421698`
- Best PSNR candidate: `control/current=baseline` with max PSNR `25.827042`
- Best eval-loss candidate: `control/current=baseline` with min eval loss `0.03620270`
- Best single run render directory: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_fas_nonempty/lookcloser/control_current_baseline_seed44/renders_best_step-000030376`

| Stage | Param | Value | Rank | Carried | Mean SSIM | Max SSIM | Mean LPIPS | Min LPIPS | Mean PSNR | Max PSNR | Mean Eval Loss | Min Eval Loss | Mean Train s | Min Train s | Config |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| control | current | `baseline` | 1 | yes | 0.555089 | 0.555835 | 0.423117 | 0.421698 | 25.745732 | 25.827042 | 0.03662763 | 0.03620270 | 2625.339 | 2615.219 | `{"background_color": "black", "center_method": "focus", "enable_adaptive_ray_marching": false, "enable_fas": true, "enable_feature_reweighting": true, "enable_frequency_grid": true, "fallback_frequency_level": 0.0, "grid_resolution": 64, "grid_update_batch_size": 4096, "grid_update_interval": 512, "max_res": null, "max_res_base": 2048.0, "min_res": 16.0, "num_frequency_levels": 16, "orientation_method": "up", "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.15, "scene_scale": 2.0, "train_num_rays_per_batch": 1024}` |

## Per-run results

| Timestamp | Stage | Param | Value | Seed | Checkpoint | Eval Loss | PSNR | SSIM | LPIPS | Train s | Eval JSON | Renders |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| control_current_baseline_seed42 | control | current | `baseline` | 42 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_fas_nonempty/lookcloser/control_current_baseline_seed42/nerfstudio_models/step-000030376.ckpt` | 0.036579 | 25.687208 | 0.553734 | 0.425020 | 2615.267437 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_fas_nonempty/lookcloser/control_current_baseline_seed42/eval_best_step-000030376.json` | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_fas_nonempty/lookcloser/control_current_baseline_seed42/renders_best_step-000030376` |
| control_current_baseline_seed43 | control | current | `baseline` | 43 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_fas_nonempty/lookcloser/control_current_baseline_seed43/nerfstudio_models/step-000030376.ckpt` | 0.036203 | 25.827042 | 0.555699 | 0.421698 | 2645.532411 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_fas_nonempty/lookcloser/control_current_baseline_seed43/eval_best_step-000030376.json` | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_fas_nonempty/lookcloser/control_current_baseline_seed43/renders_best_step-000030376` |
| control_current_baseline_seed44 | control | current | `baseline` | 44 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_fas_nonempty/lookcloser/control_current_baseline_seed44/nerfstudio_models/step-000030376.ckpt` | 0.037101 | 25.722946 | 0.555835 | 0.422635 | 2615.218580 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_fas_nonempty/lookcloser/control_current_baseline_seed44/eval_best_step-000030376.json` | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_fas_nonempty/lookcloser/control_current_baseline_seed44/renders_best_step-000030376` |

## Insights

- Compared with the carried baseline from `experiments/lookcloser_frequency_grid_update_sweep.md` (mean SSIM `0.555427`, LPIPS `0.425247`, PSNR `25.729128`, eval loss `0.03694857`, train time `2635.440s`), this fix produced mean SSIM `0.555089`, LPIPS `0.423117`, PSNR `25.745732`, eval loss `0.03662763`, train time `2625.339s`.
- Delta versus carried baseline: SSIM `-0.000338`, LPIPS `-0.002130` (better), PSNR `+0.016604`, eval loss `-0.00032094`, train time `-10.101s`.
- Recommendation: revert the sampler code change because mean SSIM did not improve, despite the LPIPS and PSNR gains. Revert only the non-empty bucket sampling change in `../nerfstudio/lookcloser_pixel_sampler.py`; keep this experiment report as the record.
