# Recommended Params

## Primary Metric-Leader Run

Use this isolated Interval Adjustment configuration:

```bash
python /home/ubuntu/repos/nerfstudio/LookCloser/scripts/run_lookcloser_quiet.py \
  --data /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns \
  --output-dir /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs \
  --experiment-name 007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning \
  --timestamp adaptive_fg_arm_metric_leader_repro \
  --scene-scale 2.0 \
  --scale-factor 1.15 \
  --center-method focus \
  --orientation-method up \
  --eval-mode filename \
  --max-num-iterations 36864 \
  --train-num-rays-per-batch 4096 \
  --eval-num-rays-per-batch 128 \
  --eval-num-rays-per-chunk 2048 \
  --eval-batch-interval 1024 \
  --eval-image-interval 100000 \
  --eval-all-interval 100000 \
  --save-interval 1024 \
  --grid-resolution 64 \
  --grid-update-interval 512 \
  --grid-update-batch-size 4096 \
  --adaptive-warmup-steps 2048 \
  --adaptive-max-frequency-level 12 \
  --adaptive-coarse-step-size 0.0125 \
  --fixed-num-samples-per-ray 256 \
  --max-steps-per-ray 1024 \
  --near-plane 0.02 \
  --render-step-size-mult 0.75 \
  --alpha-thre 0.0025 \
  --cone-angle 0.0 \
  --background-color black \
  --reconstruction-loss-type charbonnier \
  --distortion-loss-mult 0.01 \
  --depth-loss-mult 0.001 \
  --depth-loss-steps 5000 \
  --disable-feature-reweighting \
  --disable-fas \
  --no-stop-on-no-improve
```

Best metric-leader checkpoint observed:

```text
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_continue36864_r4096/nerfstudio_models/step-000034816.ckpt
```

Best metric-leader renders:

```text
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_continue36864_r4096/renders_full_step-000034816
```

Observed metrics:

```text
PSNR=28.8982
SSIM=0.6659
LPIPS=0.3653
```

## Secondary Visual-Balance Run

Use the same params as above, but set:

```text
--max-num-iterations 34816
--train-num-rays-per-batch 8192
```

Best visual-balance checkpoint observed:

```text
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h41_batch8192_coarse00125_continue34816_r8192/nerfstudio_models/step-000034815.ckpt
```

Best visual-balance renders:

```text
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h41_batch8192_coarse00125_continue34816_r8192/renders_full_step-000034815
```

Observed metrics:

```text
PSNR=28.8879
SSIM=0.6660
LPIPS=0.3664
```

## Next Tests

1. Test FAS next, with Feature Re-weighting still off.
2. Keep the metric-leader params fixed and enable only FAS by removing `--disable-fas`.
3. If FAS improves stand-label and tangled-cable crops without hurting global SSIM/LPIPS, then test Feature Re-weighting on top of that.
4. If FAS does not improve the crop gate, keep FAS off and test Feature Re-weighting alone.
5. Only after those two isolated tests should FAS + Feature Re-weighting be tested together.
