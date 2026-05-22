# 3k AABB Scene Scale Sweep

## What was tested

Short 600-step LookCloser fixed-sampler runs on the processed 3k split, with Frequency Grid enabled and Feature Re-weighting, FAS, and Adaptive RM disabled. The goal was to choose a practical `nerfstudio-data --scene-scale` before longer LookCloser training. After `2.5` won the first pass, `3.0` and `5.0` were added to check whether a looser box helped.

All runs used `--logging.csv-writer.enable True`, AABB collider, `fixed_num_samples_per_ray=256`, and eval-batch checks every 200 steps.

## Results

Sparse COLMAP points after nerfstudio orientation and auto-scale are not covered well by the default `scene_scale=1.0`: max-abs point quantiles are about 1.82 at 95%, 1.96 at 99.5%, and 2.10 at 99.9%, with one large outlier around 13.24.

| `scene_scale` | Final step | Train PSNR | Last eval-batch PSNR | Best eval PSNR | Iter time | Train rays/s | CSV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.0 | 590 | 21.4434 | 15.8853 | 15.8853 | 0.0796 | 51465.7 | [csv](aabb_scene_scale_lookcloser/outputs/scene_1p0/lookcloser/run_600/metrics_compact.csv) |
| 1.5 | 590 | 21.4335 | 15.5590 | 15.5590 | 0.0797 | 51429.9 | [csv](aabb_scene_scale_lookcloser/outputs/scene_1p5/lookcloser/run_600/metrics_compact.csv) |
| 2.0 | 590 | 24.0693 | 22.0915 | 22.0915 | 0.0817 | 50153.1 | [csv](aabb_scene_scale_lookcloser/outputs/scene_2p0/lookcloser/run_600/metrics_compact.csv) |
| 2.5 | 590 | 23.9688 | 22.8029 | 22.8029 | 0.0827 | 49503.5 | [csv](aabb_scene_scale_lookcloser/outputs/scene_2p5/lookcloser/run_600/metrics_compact.csv) |
| 3.0 | 590 | 23.9125 | 22.6068 | 22.6068 | 0.0825 | 49662.7 | [csv](aabb_scene_scale_lookcloser/outputs/scene_3p0/lookcloser/run_600/metrics_compact.csv) |
| 5.0 | 590 | 23.3934 | 22.3281 | 22.3281 | 0.0841 | 48694.3 | [csv](aabb_scene_scale_lookcloser/outputs/scene_5p0/lookcloser/run_600/metrics_compact.csv) |

## Insights

`scene_scale=1.0` and `1.5` are too tight and cause poor eval PSNR. `scene_scale=2.0` is the first acceptable setting, and `2.5` is the best early setting in this sweep while only slowing training by about 4% versus `1.0`. Looser boxes `3.0` and `5.0` did not improve early eval PSNR, and `5.0` was also slower.

Use `nerfstudio-data --scene-scale 2.5` for the next 3k LookCloser training run. Reconfirm with full-image eval and rendered inspection after longer training.
