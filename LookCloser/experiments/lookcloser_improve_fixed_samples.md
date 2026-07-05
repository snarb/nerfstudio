# LookCloser Fixed Sample Count Improvement

## What was tested

Hypothesis: adaptive ray marching is not currently viable, but the fixed-step renderer may be under-sampling at `fixed_num_samples_per_ray=256`. Increasing only `fixed_num_samples_per_ray` to `384` may improve reconstruction/detail metrics while preserving the current frequency-grid logic.

Candidate config:
- `fixed_num_samples_per_ray=384`
- `grid_resolution=64`
- `max_res_base=2048`
- `num_frequency_levels=16`
- `grid_update_interval=512`
- `grid_update_batch_size=4096`
- fixed ray marching with Frequency Grid, FAS, and Feature Re-weighting enabled
- seeds `42`, `43`, `44`
- `train_num_rays_per_batch=1024`, matching the carried reference runs
- final eval selected the checkpoint with lowest training-time eval loss

Experiment artifacts:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_384`

## Results

Per-run metrics from final eval on the selected best-eval-loss checkpoint:

| Seed | Selected checkpoint | Eval loss | PSNR | SSIM | LPIPS | Train time | Renders |
|---:|---|---:|---:|---:|---:|---:|---|
| 42 | `step-000030376` | 0.03472760 | 26.654209 | 0.590933 | 0.395277 | 3186.324s | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_384/lookcloser/fixed_samples_384_seed42/renders_best_step-000030376` |
| 43 | `step-000015188` | 0.03442380 | 26.581749 | 0.581699 | 0.419325 | 2224.263s | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_384/lookcloser/fixed_samples_384_seed43/renders_best_step-000015188` |
| 44 | `step-000030376` | 0.03412760 | 26.612181 | 0.584819 | 0.392803 | 3396.668s | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_384/lookcloser/fixed_samples_384_seed44/renders_best_step-000030376` |

Mean metrics:

| Candidate | SSIM | LPIPS | PSNR | Eval loss | Train time |
|---|---:|---:|---:|---:|---:|
| Carried reference, 256 samples | 0.555427 | 0.425247 | 25.729128 | 0.03694857 | 2635.440s |
| Fixed 384 samples | 0.585817 | 0.402468 | 26.616046 | 0.03442633 | 2935.752s |
| Delta, 384 - reference | +0.030390 | -0.022779 | +0.886918 | -0.00252224 | +300.312s |

Best single result by metric:

| Metric | Best seed | Value | Render directory |
|---|---:|---:|---|
| SSIM, higher better | 42 | 0.590933 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_384/lookcloser/fixed_samples_384_seed42/renders_best_step-000030376` |
| LPIPS, lower better | 44 | 0.392803 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_384/lookcloser/fixed_samples_384_seed44/renders_best_step-000030376` |
| PSNR, higher better | 42 | 26.654209 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_384/lookcloser/fixed_samples_384_seed42/renders_best_step-000030376` |
| Eval loss, lower better | 44 | 0.03412760 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_384/lookcloser/fixed_samples_384_seed44/renders_best_step-000030376` |
| Train time, lower better | 43 | 2224.263s | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_384/lookcloser/fixed_samples_384_seed43/renders_best_step-000015188` |

## 512-Sample Follow-Up

Hypothesis: `fixed_num_samples_per_ray=512` may improve detail further than the accepted `384` setting, with possible diminishing returns or slower training.

Candidate config:
- `fixed_num_samples_per_ray=512`
- `grid_resolution=64`
- `max_res_base=2048`
- `num_frequency_levels=16`
- `grid_update_interval=512`
- `grid_update_batch_size=4096`
- fixed ray marching with Frequency Grid, FAS, and Feature Re-weighting enabled
- seeds `42`, `43`, `44`
- `train_num_rays_per_batch=1024`
- `eval_num_rays_per_batch=1024`
- `eval_num_rays_per_chunk=2048`
- final eval selected the checkpoint with lowest training-time eval loss

Experiment artifacts:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_512`

Per-run metrics from final eval on the selected best-eval-loss checkpoint:

| Seed | Selected checkpoint | Eval loss | PSNR | SSIM | LPIPS | Train time | Renders |
|---:|---|---:|---:|---:|---:|---:|---|
| 42 | `step-000030376` | 0.03405440 | 27.014740 | 0.595094 | 0.383091 | 4177.873s | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_512/lookcloser/fixed_samples_512_seed42/renders_best_step-000030376` |
| 43 | `step-000015188` | 0.03346990 | 26.960249 | 0.593761 | 0.408391 | 2704.904s | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_512/lookcloser/fixed_samples_512_seed43/renders_best_step-000015188` |
| 44 | `step-000030376` | 0.03322770 | 27.053366 | 0.596914 | 0.380180 | 4177.744s | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_512/lookcloser/fixed_samples_512_seed44/renders_best_step-000030376` |

Mean metrics:

| Candidate | SSIM | LPIPS | PSNR | Eval loss | Train time |
|---|---:|---:|---:|---:|---:|
| Original reference, 256 samples | 0.555427 | 0.425247 | 25.729128 | 0.03694857 | 2635.440s |
| Accepted reference, 384 samples | 0.585817 | 0.402468 | 26.616046 | 0.03442633 | 2935.752s |
| Fixed 512 samples | 0.595257 | 0.390554 | 27.009452 | 0.03358400 | 3686.840s |
| Delta, 512 - 384 | +0.009440 | -0.011914 | +0.393406 | -0.00084233 | +751.088s |
| Delta, 512 - 256 | +0.039830 | -0.034693 | +1.280324 | -0.00336457 | +1051.400s |

Best single result by metric:

| Metric | Best seed | Value | Render directory |
|---|---:|---:|---|
| SSIM, higher better | 44 | 0.596914 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_512/lookcloser/fixed_samples_512_seed44/renders_best_step-000030376` |
| LPIPS, lower better | 44 | 0.380180 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_512/lookcloser/fixed_samples_512_seed44/renders_best_step-000030376` |
| PSNR, higher better | 44 | 27.053366 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_512/lookcloser/fixed_samples_512_seed44/renders_best_step-000030376` |
| Eval loss, lower better | 44 | 0.03322770 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_512/lookcloser/fixed_samples_512_seed44/renders_best_step-000030376` |
| Train time, lower better | 43 | 2704.904s | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_improve_fixed_samples_512/lookcloser/fixed_samples_512_seed43/renders_best_step-000015188` |

Disk note: root free space dropped to `9.6G` after seed 44's first checkpoint. After final eval had already been written for seeds 42 and 43, their selected checkpoint files were removed to keep artifacts bounded; after seed 44 completed, its selected checkpoint was also removed. Eval JSONs, configs, compact metrics, logs, and rendered outputs were retained.

## Insights

`fixed_num_samples_per_ray=384` improved all three quality metrics and eval loss by a large margin across the 3-seed mean. The cost is higher mean training time, about `+300s` versus the carried reference. By the SSIM-first rule, this is an accepted non-paper improvement.

`fixed_num_samples_per_ray=512` further improved the 3-seed mean SSIM, LPIPS, PSNR, and eval loss versus both `384` and the original `256` reference. The cost is substantial: mean training time increased by about `+751s` versus `384`, or `+1051s` versus `256`.

Important correction after visual audit: this acceptance was only relative to earlier LookCloser/frequency-grid references. It was not a proven improvement over the bounded Instant-NGP baseline.

The same audit also found a setup mismatch: these LookCloser runs used `scene_scale=2.0` and `scale_factor=1.15`, while the bounded Instant-NGP baseline used `scene_scale=1.5` and `scale_factor=1.0`. Treat the fixed-sample metrics as LookCloser-internal only, not final baseline comparisons.

The visual audit in [lookcloser_visual_baseline_audit.md](lookcloser_visual_baseline_audit.md) shows that Instant-NGP preserves several target tiny details better than LookCloser despite LookCloser's higher PSNR/lower LPIPS:

- small writing on the stand;
- tangled cable / thin wires;
- fingers and hand boundaries;
- thin floor crack/detail.

The bounded Instant-NGP baseline also has higher final SSIM (`0.639772`) and much lower train-time eval loss (`0.00374830`) than the fixed-sample LookCloser runs. Therefore, do not treat `fixed_num_samples_per_ray=512` as an overall quality win. It can remain a LookCloser-internal reference for future debugging, but future acceptance must match the baseline dataparser scale and include visual crop checks against Instant-NGP.
