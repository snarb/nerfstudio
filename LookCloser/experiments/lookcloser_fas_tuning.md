# LookCloser FAS Tuning

## What Was Tested

Goal: isolate Frequency-Averaged Sampling (FAS) with Frequency Grid and Adaptive RM kept at the current recommended metric-leader settings, and Feature Re-weighting still disabled.

The starting point was `recomended_params.md`: `grid_resolution=64`, `grid_update_interval=512`, `grid_update_batch_size=4096`, `adaptive_warmup_steps=2048`, `adaptive_max_frequency_level=12`, `adaptive_coarse_step_size=0.0125`, `fixed_num_samples_per_ray=256`, `scene_scale=2.0`, `scale_factor=1.15`, Charbonnier RGB loss, distortion/depth losses enabled, and `--disable-feature-reweighting`.

Three FAS variants were tested:

- `fas_on_ramp1_3`: paper-style full FAS by removing `--disable-fas`, keeping `sampling_ramp_start=1`, `sampling_ramp_end=3`.
- `fas_mix035_w2048_r4096`: mixed uniform/FAS sampling with `--fas-strength 0.35 --fas-warmup-steps 2048 --fas-ramp-steps 4096`, keeping the same paper-style level ramp.
- `fas_ramp0_3`: aggressive full FAS with `sampling_ramp_start=0`, `sampling_ramp_end=3`; this was rejected early because eval loss was worse and oscillating.

All completed quality variants used seeds `42`, `43`, and `44` concurrently without OOM. Rendered best-checkpoint outputs were visually checked on high-frequency crops: floor crack, fingers, stand label, and tangled cable.

## Results

Current no-FAS reference from `recomended_params.md`: PSNR `28.8982`, SSIM `0.6659`, LPIPS `0.3653`.

### Full FAS, Ramp 1-3

| Seed | Selected step | Eval loss | PSNR | SSIM | LPIPS | Render dir |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 26624 | 0.0239693 | 28.942957 | 0.655462 | 0.318832 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_on_ramp1_3_seed42/renders_best_step-000026624` |
| 43 | 10240 | 0.0243826 | 28.446032 | 0.642079 | 0.355964 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_on_ramp1_3_seed43/renders_best_step-000010240` |
| 44 | 25600 | 0.0271662 | 28.592052 | 0.657932 | 0.313803 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_on_ramp1_3_seed44/renders_best_step-000025600` |

Mean: PSNR `28.660347`, SSIM `0.651824`, LPIPS `0.329533`, eval loss `0.02517270`.

### Mixed FAS, 0.35 Strength With Warmup/Ramp

| Seed | Selected step | Eval loss | PSNR | SSIM | LPIPS | Render dir |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 12288 | 0.0247757 | 28.830338 | 0.664944 | 0.398728 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_mix035_w2048_r4096_seed42/renders_best_step-000012288` |
| 43 | 34816 | 0.0244432 | 29.135916 | 0.681484 | 0.367407 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_mix035_w2048_r4096_seed43/renders_best_step-000034816` |
| 44 | 35840 | 0.0248972 | 29.189543 | 0.675507 | 0.369188 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_mix035_w2048_r4096_seed44/renders_best_step-000035840` |

Mean: PSNR `29.051932`, SSIM `0.673978`, LPIPS `0.378441`, eval loss `0.02470537`.

This is the new numeric PSNR/SSIM leader, but LPIPS is worse than both the no-FAS reference and full-FAS control.

### Crop Gate

Crop visual summaries:

- Full FAS seed 42: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_on_ramp1_3_seed42/crop_gate_best_stride4/all_crops.png`
- Full FAS seed 43: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_on_ramp1_3_seed43/crop_gate_best_stride4/all_crops.png`
- Full FAS seed 44: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_on_ramp1_3_seed44/crop_gate_best_stride4/all_crops.png`
- Mixed FAS seed 42: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_mix035_w2048_r4096_seed42/crop_gate_best_stride4/all_crops.png`
- Mixed FAS seed 43: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_mix035_w2048_r4096_seed43/crop_gate_best_stride4/all_crops.png`
- Mixed FAS seed 44: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_mix035_w2048_r4096_seed44/crop_gate_best_stride4/all_crops.png`

Mean crop SSIM, LookCloser candidate vs matched bounded Instant-NGP baseline:

| Variant | Floor crack | Fingers right | Stand label | Tangled cable | Fingers center |
|---|---:|---:|---:|---:|---:|
| Full FAS | 0.73771 / 0.87502 | 0.83002 / 0.95728 | 0.83305 / 0.96592 | 0.75758 / 0.96757 | 0.80642 / 0.95332 |
| Mixed FAS | 0.81923 / 0.87502 | 0.83532 / 0.95728 | 0.83299 / 0.96592 | 0.76455 / 0.96757 | 0.80852 / 0.95332 |

Visual inspection agrees with the crop metrics: mixed FAS improves global PSNR/SSIM and makes some broad crop structure cleaner, but labels remain unreadable and tangled cables/finger boundaries are still materially below the Instant-NGP reference. Full FAS improves LPIPS but does not pass the high-frequency visual gate.

### Aggressive Ramp Rejection

`sampling_ramp_start=0`, `sampling_ramp_end=3`, full FAS was stopped after early evals because it was clearly behind:

| Seed | Best early step | Best early eval loss | Last checked step | Last checked eval loss |
|---:|---:|---:|---:|---:|
| 42 | 5120 | 0.0295320 | 8192 | 0.0313641 |
| 43 | 5120 | 0.0292459 | 8192 | 0.0307756 |
| 44 | 6144 | 0.0298347 | 8192 | 0.0304991 |

## Insights

The HD frequency maps assign most patches to high frequency levels: about `77.3%` of cells are in levels `12-15`. Full paper-style FAS does not sample that tail heavily enough to fix local detail, while forcing a lower ramp start from step 0 destabilizes early training.

Mixed FAS at `0.35` strength with a `2048` step warmup and `4096` step ramp is the best FAS-only numeric result so far: mean PSNR and SSIM beat the previous no-FAS reference, and seeds 43/44 beat the previous single-run PSNR leader. However, it fails the visual-first crop gate and worsens LPIPS. Treat it as a numeric-leader candidate, not as a solved tiny-detail improvement.

Next FAS-focused tests should avoid more aggressive high-frequency-only sampling. More plausible directions are modest FAS mixtures with visual-gated selection, or a FAS probability schedule that better matches the observed high-frequency map distribution without starving uniform coverage.
