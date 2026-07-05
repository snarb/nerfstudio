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

## Artifact Follow-up

User visual inspection found a significant `eval_img_0000.png` artifact in the previous seed-44 PSNR leader: the left vertical metal stand was broken around the connector/wrist area. The exact crop comparison is saved at:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/problem_crop_compare/problem_crop_compare.png
```

The comparison shows seed 43 of the same `fas_strength=0.35`, `fas_warmup_steps=2048`, `fas_ramp_steps=4096` setup fixes the reported eval0 stand discontinuity better than seed 44 while retaining improved global metrics:

| Candidate | Step | PSNR | SSIM | LPIPS | Visual note |
|---|---:|---:|---:|---:|---|
| seed 43 | 34816 | 29.135916 | 0.681484 | 0.367407 | Preferred FAS visual candidate; eval0 stand is more continuous |
| seed 44 | 35840 | 29.189543 | 0.675507 | 0.369188 | Highest PSNR but rejected as primary due broken eval0 stand artifact |

The crop gate now includes `left_stand_eval0`. Updated crop sheets:

```text
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_mix035_w2048_r4096_seed43/crop_gate_with_left_stand_stride4/all_crops.png
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_mix035_w2048_r4096_seed44/crop_gate_with_left_stand_stride4/all_crops.png
```

Local artifact outlier scans compare sliding windows against GT and the no-FAS baseline:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/outliers_fas_s43_vs_nofas
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/outliers_fas_s44_vs_nofas
```

The scanner catches thin stand/wire regressions that global PSNR/SSIM can hide. Seed 43 has a better eval0 stand crop, but it still has thin-wire outliers on other eval views. Two FAS schedules were tested and rejected early while trying to reduce these artifacts:

| Variant | Status | Reason |
|---|---|---|
| `fas_strength=0.35`, `fas_warmup_steps=8192`, `fas_ramp_steps=8192` | Rejected early | Eval loss regressed after the delayed ramp and did not approach the metric leader |
| `fas_strength=0.20`, `fas_warmup_steps=2048`, `fas_ramp_steps=4096` | Rejected early | Ramp-complete evals were unstable; two seeds lagged badly |

After stricter user inspection, seed 43 is also rejected: it improves some broad high-frequency detail but the vertical stand is still not physically continuous.

### Strict Stand-Connector Gate

The crop gate now includes exact `left_stand_connector_eval0` and can render only that crop with `--crop-name left_stand_connector_eval0`. The outlier scanner now writes `named_crops.csv` and `named_crops.png` for the same target regions plus sliding-window outliers.

Baseline reference for the downsampled target crop: PSNR `26.5161`, SSIM `0.8709`.

Short FAS gate tests were run to `step=3071/3072`, then visually checked before any full metric run:

| Variant | Best target-crop PSNR | Best target-crop SSIM | Visual decision |
|---|---:|---:|---|
| `fas_strength=0.35`, `warmup=2048`, `ramp=4096`, `fas_level_count_alpha=0.5` | 26.1009 | 0.8117 | Rejected; stand remains weaker/broken |
| same plus `fas_patch_group_size=4` | 25.8693 | 0.8024 | Rejected; local grouping did not fix continuity |
| `fas_strength=0.25`, `warmup=2048`, `ramp=4096` | 25.7730 | 0.8068 | Rejected; softer mix still fails |
| `fas_strength=0.35`, ramp `1.0 -> 1.5` | 25.9709 | 0.8090 | Rejected; flatter level ramp still fails |
| `fas_strength=0.35`, `warmup=2048`, `ramp=4096`, `fas_max_sampling_level=12` | 25.9357 | 0.8099 | Rejected; cap matches model max level but crop still fails |
| `fas_strength=0.35`, no warmup/ramp, `fas_max_sampling_level=12` | 26.0458 | 0.8088 | Rejected; FAS active from start still fails |

Partial preprocessing checks with `ssim_threshold=0.93` were started but stopped because full map regeneration was too slow for this gate loop. A stronger direct finding was the schedule mismatch: training uses `adaptive_max_frequency_level=12` while the original FAS bucket schedule includes levels `13-15`; the new `fas_max_sampling_level` knob can test this explicitly. That cap alone did not fix the stand artifact.

Current FAS status: no FAS-enabled variant passes the strict eval0 stand-connector gate while also preserving the no-FAS metric baseline. Keep the no-FAS reference as the accepted stable baseline until a later FAS/preprocessing fix passes this visual-first gate.
