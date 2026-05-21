# Splatfacto-big with train-only COLMAP on 3k

## What was tested

Rebuilt COLMAP for `splatfacto-big` using only the 66 training frames from `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_data/007740/3k`, excluding the held-out evaluation images:

- `E004_B014.png`
- `D004_A014.png`
- `I004_D014.png`

The processed dataset is stored at `experiments/splatfacto_train_only_colmap/processed_trainonly`. Its sparse point cloud is train-only. The three eval image camera poses were copied from the earlier all-frame registration so that `ns-eval` can evaluate the same held-out views; this means the point-cloud initialization is fairer, but the eval poses are not yet independently localized against the train-only COLMAP model.

Training command:

```bash
conda run -p /home/ubuntu/anaconda3/envs/nerfstudio ns-train splatfacto-big \
  --output-dir experiments/splatfacto_train_only_colmap/outputs \
  --experiment-name 007740_3k_splatfacto_train_only \
  --timestamp splatfacto-big \
  --vis viewer --viewer.quit-on-train-completion True \
  --logging.local-writer.enable False \
  --logging.csv-writer.enable True \
  --logging.csv-writer.write-interval 100 \
  --logging.profiler none --logging.steps-per-log 100 \
  --steps-per-save 30000 --steps-per-eval-all-images 1000 \
  nerfstudio-data --data experiments/splatfacto_train_only_colmap/processed_trainonly \
  --eval-mode filename --downscale-factor 1
```

Evaluation command:

```bash
conda run -p /home/ubuntu/anaconda3/envs/nerfstudio ns-eval \
  --load-config experiments/splatfacto_train_only_colmap/outputs/007740_3k_splatfacto_train_only/splatfacto/splatfacto-big/config.yml \
  --output-path experiments/splatfacto_train_only_colmap/metrics/splatfacto-big-train-only-colmap.json \
  --render-output-path experiments/splatfacto_train_only_colmap/renders/splatfacto-big-train-only-colmap
```

## Results

| Model / setup | COLMAP point cloud | PSNR | SSIM | LPIPS | Metrics |
| --- | --- | ---: | ---: | ---: | --- |
| `splatfacto-big` previous baseline | all 69 frames, including eval frames | 28.8538 | 0.8027 | 0.3265 | `experiments/baselines_3k_colmap/metrics/splatfacto-big.json` |
| `splatfacto-big` rerun | train-only 66 frames | 14.1115 | 0.4685 | 0.7239 | `experiments/splatfacto_train_only_colmap/metrics/splatfacto-big-train-only-colmap.json` |
| `instant-ngp` | all-frame poses, no COLMAP point init | 22.7189 | 0.5906 | 0.5385 | `experiments/ngp_baselines_3k/metrics/instant-ngp.json` |
| `instant-ngp-big` | all-frame poses, no COLMAP point init | 22.6693 | 0.6013 | 0.4950 | `experiments/ngp_baselines_3k/metrics/instant-ngp-big.json` |
| `nerfacto` | all-frame poses, no COLMAP point init | 20.5122 | 0.4825 | 0.5365 | `experiments/baselines_3k_colmap/metrics/nerfacto.json` |

Compact logger:

| Signal | Best | Final | Status |
| --- | ---: | ---: | --- |
| `eval_all_psnr` | 14.5789 @ step 2000 | 14.1290 @ step 29000 | `plateau_watch` |
| `eval_image_psnr` | 15.9441 @ step 600 | 14.9797 @ step 29900 | `plateau_watch` |

CSV log: `experiments/splatfacto_train_only_colmap/outputs/007740_3k_splatfacto_train_only/splatfacto/splatfacto-big/metrics_compact.csv`

Rendered eval images: `experiments/splatfacto_train_only_colmap/renders/splatfacto-big-train-only-colmap`

## Insights

The earlier high `splatfacto-big` score was very likely inflated by using a COLMAP sparse point cloud generated from all frames, including the held-out evaluation images. Once the Gaussian initialization is rebuilt from train frames only, `splatfacto-big` is far below `instant-ngp` and `instant-ngp-big`.

The run plateaued early: best aggregate eval PSNR was at step 2000, while the final model was slightly worse. For this setup, continuing to 30k steps is not useful unless initialization, pose handling, or data split handling is changed.

Next stricter check: localize eval cameras against the train-only COLMAP model instead of copying eval poses from the all-frame registration, then rerun evaluation. Also compare `splatfacto-big` with `load_3D_points: false` to separate point-cloud leakage from other 3DGS behavior.
