# 3k NGP Baselines and Compact CSV Logging

## What was tested

Added a compact CSV scalar logger to nerfstudio and ran two Instant-NGP baselines on the same processed 3k dataset used by the prior baseline run:

- `instant-ngp`, 30,000 iterations.
- `instant-ngp-big`, 30,000 iterations, using `log2_hashmap_size=23`, `max_res=4096`, `train_num_rays_per_batch=8192`, and `eval_num_rays_per_chunk=16384`.

Both runs used:

- Dataset: `experiments/baselines_3k_colmap/processed`
- Eval split: filename mode with `eval_D004_A014.png`, `eval_E004_B014.png`, and `eval_I004_D014.png`.
- TensorBoard disabled via `--vis viewer`.
- Compact CSV enabled via `--logging.csv-writer.enable True`.
- CSV write interval: 100 steps.

## Results

Final aggregate `ns-eval` metrics:

| Model | PSNR | SSIM | LPIPS | Metrics |
|---|---:|---:|---:|---|
| `nerfacto` | 20.5122 | 0.4825 | 0.5365 | [metrics](baselines_3k_colmap/metrics/nerfacto.json) |
| `nerfacto-huge` | 18.3378 | 0.4250 | 0.5677 | [metrics](baselines_3k_colmap/metrics/nerfacto-huge.json) |
| `instant-ngp` | 22.7189 | 0.5906 | 0.5385 | [metrics](ngp_baselines_3k/metrics/instant-ngp.json) |
| `instant-ngp-big` | 22.6693 | 0.6013 | 0.4950 | [metrics](ngp_baselines_3k/metrics/instant-ngp-big.json) |
| `splatfacto-big` | 28.8538 | 0.8027 | 0.3265 | [metrics](baselines_3k_colmap/metrics/splatfacto-big.json) |

Compact CSV trend summary:

| Model | Train loss first | Train loss final | Eval loss first | Eval loss best | Eval loss final | Best eval image PSNR | Final eval image PSNR | CSV |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `instant-ngp` | 0.038172 | 0.001511 | 0.007354 | 0.004440 @ 14500 | 0.005443 | 24.3104 @ 6000 | 22.4356 | [csv](ngp_baselines_3k/outputs/007740_3k_ngp/instant-ngp/instant-ngp/metrics_compact.csv) |
| `instant-ngp-big` | 0.038115 | 0.001381 | 0.006214 | 0.004382 @ 10000 | 0.005317 | 24.4161 @ 8000 | 23.1707 | [csv](ngp_baselines_3k/outputs/007740_3k_ngp/instant-ngp-big/instant-ngp-big/metrics_compact.csv) |

Render outputs:

- `instant-ngp`: [renders](ngp_baselines_3k/renders/instant-ngp)
- `instant-ngp-big`: [renders](ngp_baselines_3k/renders/instant-ngp-big)
- `splatfacto-big`: [renders](baselines_3k_colmap/renders/splatfacto-big)

## Insights

The large gap is still present after adding NGP baselines: `splatfacto-big` is about 6.1 dB PSNR and 0.20 SSIM above both NGP variants.

The most suspicious implementation difference is data leakage through sparse COLMAP points. `splatfacto-big` uses `load_3D_points: true`, while `nerfacto`, `instant-ngp`, and `instant-ngp-big` use `load_3D_points: false`. The current COLMAP reconstruction was generated from all 69 images, including the three eval views, so the sparse point cloud used to initialize `splatfacto-big` can contain geometry and color observations from held-out eval frames. That makes the `splatfacto-big` number not directly comparable to the NeRF/NGP baselines.

The compact CSV logs show both NGP variants keep reducing train loss while eval loss and eval image PSNR peak much earlier and then degrade or plateau. Continuing the same NGP runs past about 10k-15k steps is unlikely to improve held-out metrics without changing regularization, split/pose setup, or model configuration.

Next fair comparison should rerun `splatfacto-big` with either:

- `--pipeline.datamanager.dataparser.load-3D-points False`, or
- a train-only COLMAP reconstruction plus separate eval camera registration, so held-out eval images do not contribute to the 3D point initialization.
