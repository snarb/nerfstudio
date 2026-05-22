# 3k Hash Grid Hyperparameter Sweep

## What was tested

Short 600-step LookCloser fixed-sampler runs on the processed 3k split after fixing AABB to `nerfstudio-data --scene-scale 2.5`. Frequency Grid was enabled; Feature Re-weighting, FAS, and Adaptive RM were disabled.

This sweep varies `pipeline.model.max_res_base`, which controls LookCloser's hash-grid max resolution as `max_res_base * scene_size`.

## Results

| `max_res_base` | Final step | Train PSNR | Last eval-batch PSNR | Best eval PSNR | Iter time | Train rays/s | CSV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1024 | 590 | 24.0824 | 22.7755 | 22.7755 | 0.0808 | 50694.4 | [csv](hashgrid_hparams_lookcloser/outputs/maxbase_1024/lookcloser/run_600/metrics_compact.csv) |
| 2048 | 590 | 24.0712 | 22.8229 | 22.8229 | 0.0824 | 49728.4 | [csv](hashgrid_hparams_lookcloser/outputs/maxbase_2048/lookcloser/run_600/metrics_compact.csv) |
| 4096 | 590 | 23.9635 | 22.6533 | 22.6533 | 0.0844 | 48558.3 | [csv](hashgrid_hparams_lookcloser/outputs/maxbase_4096/lookcloser/run_600/metrics_compact.csv) |

## Insights

`max_res_base=2048` remains the quality-first default for current 3k LookCloser runs. `1024` is nearly tied and about 2% faster, so it is useful for fast debugging. `4096` is worse in this early sweep and should not be used unless a longer run contradicts this.
