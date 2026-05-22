# 3k Baseline Metrics

## What was tested

Baseline-only training on the 3k dataset with `E004_B014.png`, `D004_A014.png`, and `I004_D014.png` held out for evaluation.

Most runs use the processed all-frame COLMAP dataset at `experiments/baselines_3k_colmap/processed`. The `splatfacto-big train-only COLMAP` rerun uses `experiments/splatfacto_train_only_colmap/processed_trainonly`, where COLMAP sparse points were built only from the 66 train frames; eval camera poses were copied from the earlier all-frame registration so the same held-out views can be evaluated.

## Results

Aggregate `ns-eval` metrics:

| Model / setup | Point init / pose setup | PSNR | SSIM | LPIPS | Metrics |
| --- | --- | ---: | ---: | ---: | --- |
| `nerfacto` | all-frame poses, no 3D point init | 20.5122 | 0.4825 | 0.5365 | [json](baselines_3k_colmap/metrics/nerfacto.json) |
| `nerfacto-huge` | all-frame poses, no 3D point init | 18.3378 | 0.4250 | 0.5677 | [json](baselines_3k_colmap/metrics/nerfacto-huge.json) |
| `instant-ngp` | all-frame poses, no 3D point init | 22.7189 | 0.5906 | 0.5385 | [json](ngp_baselines_3k/metrics/instant-ngp.json) |
| `instant-ngp-big` | all-frame poses, no 3D point init | 22.6693 | 0.6013 | 0.4950 | [json](ngp_baselines_3k/metrics/instant-ngp-big.json) |
| `splatfacto-big` | all-frame COLMAP points, includes eval frames | 28.8538 | 0.8027 | 0.3265 | [json](baselines_3k_colmap/metrics/splatfacto-big.json) |
| `splatfacto-big train-only COLMAP` | train-only COLMAP points, eval poses copied from all-frame registration | 14.1115 | 0.4685 | 0.7239 | [json](splatfacto_train_only_colmap/metrics/splatfacto-big-train-only-colmap.json) |

Selected per-image metrics for the original three-model baseline:

| Model | Image | PSNR | SSIM | Render |
| --- | --- | ---: | ---: | --- |
| `nerfacto` | `D004_A014` | 20.2351 | 0.4339 | [png](baselines_3k_colmap/renders/nerfacto/eval_img_0000.png) |
| `nerfacto` | `E004_B014` | 20.6247 | 0.4721 | [png](baselines_3k_colmap/renders/nerfacto/eval_img_0001.png) |
| `nerfacto` | `I004_D014` | 20.6749 | 0.5394 | [png](baselines_3k_colmap/renders/nerfacto/eval_img_0002.png) |
| `nerfacto-huge` | `D004_A014` | 17.9517 | 0.3987 | [png](baselines_3k_colmap/renders/nerfacto-huge/eval_img_0000.png) |
| `nerfacto-huge` | `E004_B014` | 18.3287 | 0.3850 | [png](baselines_3k_colmap/renders/nerfacto-huge/eval_img_0001.png) |
| `nerfacto-huge` | `I004_D014` | 18.7317 | 0.4891 | [png](baselines_3k_colmap/renders/nerfacto-huge/eval_img_0002.png) |
| `splatfacto-big` | `D004_A014` | 28.1153 | 0.7618 | [png](baselines_3k_colmap/renders/splatfacto-big/eval_img_0000.png) |
| `splatfacto-big` | `E004_B014` | 27.4185 | 0.7902 | [png](baselines_3k_colmap/renders/splatfacto-big/eval_img_0001.png) |
| `splatfacto-big` | `I004_D014` | 31.0141 | 0.8543 | [png](baselines_3k_colmap/renders/splatfacto-big/eval_img_0002.png) |

Compact CSV trends:

| Model | Best eval signal | Final eval signal | Status / note | CSV |
| --- | ---: | ---: | --- | --- |
| `instant-ngp` | eval loss 0.004440 @ 14500; eval image PSNR 24.3104 @ 6000 | eval loss 0.005443; eval image PSNR 22.4356 | overfit/plateau before 30k | [csv](ngp_baselines_3k/outputs/007740_3k_ngp/instant-ngp/instant-ngp/metrics_compact.csv) |
| `instant-ngp-big` | eval loss 0.004382 @ 10000; eval image PSNR 24.4161 @ 8000 | eval loss 0.005317; eval image PSNR 23.1707 | overfit/plateau before 30k | [csv](ngp_baselines_3k/outputs/007740_3k_ngp/instant-ngp-big/instant-ngp-big/metrics_compact.csv) |
| `splatfacto-big train-only COLMAP` | eval all PSNR 14.5789 @ 2000; eval image PSNR 15.9441 @ 600 | eval all PSNR 14.1290; eval image PSNR 14.9797 | `plateau_watch` | [csv](splatfacto_train_only_colmap/outputs/007740_3k_splatfacto_train_only/splatfacto/splatfacto-big/metrics_compact.csv) |

## Insights

The original `splatfacto-big` result is not directly comparable to NeRF/NGP baselines because it used `load_3D_points=True` with a COLMAP reconstruction built from all 69 frames, including the held-out eval images.

When COLMAP sparse points are rebuilt from train frames only, `splatfacto-big` drops from 28.8538 PSNR / 0.8027 SSIM to 14.1115 PSNR / 0.4685 SSIM, so the earlier large advantage was very likely inflated by point-cloud initialization leakage.

`instant-ngp` and `instant-ngp-big` outperform `nerfacto` on this split, but both plateau or overfit well before 30k steps; monitor best validation metrics rather than relying only on final-step metrics.

Next fair 3DGS check: localize eval cameras against the train-only COLMAP model, or run `splatfacto-big` with 3D point loading disabled.
