# Baseline instant-ngp-bounded params

Reproduces PSNR 24.42 / SSIM 0.640 / LPIPS 0.460 on scene 007740.

```bash
ns-train instant-ngp-bounded \
  --output-dir /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs \
  --experiment-name 007740_hd_aabb4_multicamera_eval3_ns_focus_scene15 \
  --steps-per-eval-all-images 15188 \
  --steps-per-save 15188 \
  --max-num-iterations 60752 \
  --logging.local-writer.enable False \
  --logging.csv-writer.enable True \
  nerfstudio-data \
  --data /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns \
  --eval-mode filename \
  --orientation-method up \
  --center-method focus \
  --auto-scale-poses True \
  --scene-scale 1.5 \
  --downscale-factor 1
```

Or use the quiet runner (same defaults):

```bash
python scripts/run_bounded_ngp_quiet.py
```
