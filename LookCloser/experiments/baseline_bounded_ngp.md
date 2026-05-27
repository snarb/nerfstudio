# Baseline Bounded Instant-NGP

Dataset:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns`

Run:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_focus_scene15/instant-ngp-bounded/20260527_122100`

## Setup

- Method: `instant-ngp-bounded`
- Dataparser: `nerfstudio-data`
- Eval split: `filename`
- Train images: 66
- Eval images: 3 (`frame_eval_00001.jpg`, `frame_eval_00002.jpg`, `frame_eval_00003.jpg`)
- Scene centering: `center-method=focus`
- Scene box half extent: `scene-scale=1.5`
- Pose scaling: `auto-scale-poses=True`
- Image downscale: `1`
- Train rays per batch: `8192`
- Eval/save interval: `15188` steps
- Max iterations: `60752`
- Logger: compact CSV only, local terminal writer disabled

Note: `aabb_scale=4` in `transforms.json` is not the same as `nerfstudio-data --scene-scale`. For this run the dataset is parsed with nerfstudio pose normalization and a post-normalization ROI half extent of `1.5`.

## Command

```bash
ns-train instant-ngp-bounded \
  --output-dir /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs \
  --experiment-name 007740_hd_aabb4_multicamera_eval3_ns_focus_scene15 \
  --timestamp 20260527_122100 \
  --vis tensorboard \
  --viewer.quit-on-train-completion True \
  --steps-per-eval-batch 15188 \
  --steps-per-eval-image 15188 \
  --steps-per-eval-all-images 15188 \
  --steps-per-save 15188 \
  --max-num-iterations 60752 \
  --save-only-latest-checkpoint False \
  --logging.local-writer.enable False \
  --logging.csv-writer.enable True \
  --logging.csv-writer.write-interval 15188 \
  --logging.csv-writer.improvement-tolerance 0.0 \
  --logging.profiler none \
  --pipeline.datamanager.cache-images-type uint8 \
  nerfstudio-data \
  --data /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns \
  --eval-mode filename \
  --eval-interval 8 \
  --orientation-method up \
  --center-method focus \
  --auto-scale-poses True \
  --scene-scale 1.5 \
  --downscale-factor 1
```

## Results

In-training eval checkpoints:

| Step | Eval loss | PSNR | SSIM |
|---:|---:|---:|---:|
| 15188 | 0.00425751 | 23.9691 | 0.610892 |
| 30376 | 0.00388616 | 24.2788 | 0.630985 |
| 45564 | 0.00374830 | 24.2847 | 0.631123 |

Final checkpoint evaluated with `ns-eval`:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_focus_scene15/instant-ngp-bounded/20260527_122100/nerfstudio_models/step-000060751.ckpt`

Final metrics:

| Metric | Value |
|---|---:|
| PSNR | 24.417955 |
| SSIM | 0.639772 |
| LPIPS | 0.460250 |
| PSNR std | 0.869802 |
| SSIM std | 0.060874 |
| LPIPS std | 0.096653 |

## Artifacts

- Config: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_focus_scene15/instant-ngp-bounded/20260527_122100/config.yml`
- CSV log: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_focus_scene15/instant-ngp-bounded/20260527_122100/metrics_compact.csv`
- Final eval JSON: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_focus_scene15/instant-ngp-bounded/20260527_122100/eval_last_step_60751.json`
- Render outputs: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_focus_scene15/instant-ngp-bounded/20260527_122100/renders_last_step_60751`

Render folder contains 9 PNG files: RGB, depth, and accumulation for each of the 3 eval views.
