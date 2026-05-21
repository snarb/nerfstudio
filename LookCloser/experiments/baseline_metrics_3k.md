# 3k Baseline Metrics

## What was tested

Baseline-only training on the processed 3k COLMAP dataset at `experiments/baselines_3k_colmap/processed`, using filename eval split with `eval_D004_A014.png`, `eval_E004_B014.png`, and `eval_I004_D014.png` held out.

Models:
- `nerfacto`, 30,000 iterations
- `nerfacto-huge`, 30,000 iterations
- `splatfacto-big`, default 30,000 iterations

Aggregates are from `ns-eval` JSON outputs. Per-image metrics were recomputed from saved `eval_img_*.png` render pairs using the same metric functions used by the corresponding nerfstudio model code.

## Results

| Model | Image | PSNR | SSIM | Output |
|---|---:|---:|---:|---|
| nerfacto | D004_A014 | 20.2351 | 0.4339 | [render](baselines_3k_colmap/renders/nerfacto/eval_img_0000.png) |
| nerfacto | E004_B014 | 20.6247 | 0.4721 | [render](baselines_3k_colmap/renders/nerfacto/eval_img_0001.png) |
| nerfacto | I004_D014 | 20.6749 | 0.5394 | [render](baselines_3k_colmap/renders/nerfacto/eval_img_0002.png) |
| nerfacto | Aggregate | 20.5122 | 0.4825 | [metrics](baselines_3k_colmap/metrics/nerfacto.json) |
| nerfacto-huge | D004_A014 | 17.9517 | 0.3987 | [render](baselines_3k_colmap/renders/nerfacto-huge/eval_img_0000.png) |
| nerfacto-huge | E004_B014 | 18.3287 | 0.3850 | [render](baselines_3k_colmap/renders/nerfacto-huge/eval_img_0001.png) |
| nerfacto-huge | I004_D014 | 18.7317 | 0.4891 | [render](baselines_3k_colmap/renders/nerfacto-huge/eval_img_0002.png) |
| nerfacto-huge | Aggregate | 18.3378 | 0.4250 | [metrics](baselines_3k_colmap/metrics/nerfacto-huge.json) |
| splatfacto-big | D004_A014 | 28.1153 | 0.7618 | [render](baselines_3k_colmap/renders/splatfacto-big/eval_img_0000.png) |
| splatfacto-big | E004_B014 | 27.4185 | 0.7902 | [render](baselines_3k_colmap/renders/splatfacto-big/eval_img_0001.png) |
| splatfacto-big | I004_D014 | 31.0141 | 0.8543 | [render](baselines_3k_colmap/renders/splatfacto-big/eval_img_0002.png) |
| splatfacto-big | Aggregate | 28.8538 | 0.8027 | [metrics](baselines_3k_colmap/metrics/splatfacto-big.json) |

## Insights

`splatfacto-big` is the strongest 3k baseline by a large margin on both PSNR and SSIM. `nerfacto-huge` underperformed the standard `nerfacto` run despite the larger configuration, so it is not a useful baseline improvement for this split as trained here.

Rendered eval pairs are saved under `experiments/baselines_3k_colmap/renders/` for visual inspection.
