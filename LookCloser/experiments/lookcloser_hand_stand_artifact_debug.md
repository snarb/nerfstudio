# LookCloser Hand And Stand Artifact Debug

## What was tested

User visual inspection found that the two-stage FAS candidate improves the vertical stand but still has:

- a blotch/blurred background around the left hand in `eval_img_0000`;
- small stand breaks below the hand near the outlet;
- worse local visual quality than the Instant-NGP reference in the hand/stand crop.

This debug pass tracks exactly when the hand-background blotch and remaining stand breaks appear, starting from near-original controls and adding LookCloser components step by step.

Primary gates:

```text
left_stand_eval0: xyxy=(300, 0, 650, 650)
left_hand_background_eval0: xyxy=(300, 210, 560, 500)
left_hand_outlet_stand_eval0: xyxy=(300, 250, 500, 560)
```

The gate script now includes these crops:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/scripts/render_lookcloser_crop_gate.py
```

## Results

### Existing-run timeline

Rendered `left_stand_eval0` at stride 2 for existing checkpoints:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/hand_stand_artifact_timeline/left_stand_eval0_candidate_timeline_stride2.png
```

Key rows:

| Run/checkpoint | Modules | Crop PSNR | Crop SSIM | Visual read |
|---|---|---:|---:|---|
| `fixed_h12_s3584` | Frequency Grid, fixed renderer, no FAS/FR/ARM | `25.1695` | `0.7634` | no large black ARM hole, but undertrained/smoothed |
| `arm_h17_s4096` | FG + ARM, warmup 2048 | `22.8975` | `0.7103` | large hand/background blotch appears after ARM is active |
| `arm_h17_s8192` | FG + ARM | `23.0149` | `0.7285` | blotch remains severe |
| `arm_h17_s12287` | FG + ARM | `22.8139` | `0.7324` | blotch/stand weakness remains |
| `arm_h20_s16384` | FG + ARM, grid64/maxfreq12 | `23.8655` | `0.7549` | partially recovered but still worse than Instant-NGP |
| `arm_h20_s30375` | FG + ARM, grid64/maxfreq12 | `23.9790` | `0.7640` | still smoothed around hand/background |
| `arm_h28_s32768` | FG + ARM, coarse `0.0125` | `25.5972` | `0.7915` | improved, still below Instant-NGP |
| `arm_h40_s34816` | current no-FAS H40 reference | `25.6931` | `0.7936` | no severe black hole, but background/stand still worse than Instant-NGP |

Instant-NGP reference for this crop is `PSNR=26.9343`, `SSIM=0.8528`.

### ARM variant controls

Rendered existing variants:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/hand_stand_artifact_timeline/arm_variant_controls/arm_variants_candidate_left_stand_eval0.png
```

| Variant | Crop PSNR | Crop SSIM | Finding |
|---|---:|---:|---|
| H40 default | `25.6931` | `0.7936` | baseline no-FAS ARM artifact level |
| H35 `distortion_loss_mult=0` | `25.6718` | `0.7933` | no meaningful change |
| H38 MSE RGB loss | `25.6718` | `0.7933` | no meaningful change |
| H41 batch 8192 | `25.6724` | `0.7933` | no meaningful change |
| H33 maxfreq13 | `25.6947` | `0.7935` | no meaningful change |

These rule out distortion loss, RGB loss type, batch size, and maxfreq13 as the primary cause of the hand/stand crop artifact.

### Render-only ARM ablation

Same H40 checkpoint (`step-000034816`) was rendered with config-only overrides:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/hand_stand_artifact_timeline/render_only_arm_ablation
```

| Render config | Crop PSNR | Crop SSIM | Finding |
|---|---:|---:|---|
| default ARM | `25.6931` | `0.7936` | baseline |
| `alpha_thre=0.0` | `25.6931` | `0.7936` | no improvement |
| `alpha_thre=0.0`, coarse `0.00625` | `25.6823` | `0.7936` | no improvement |
| fixed renderer, same weights, `512` samples | `25.0340` | `0.7439` | worse |

The artifact is already in the learned weights after ARM training; it is not only an eval-time `alpha_thre` or coarse traversal issue.

### Fixed-render controls from existing runs

Existing fixed-step LookCloser renders do not show the same severe ARM hole, but remain below Instant-NGP and visibly over-smooth fine structure:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/hand_stand_artifact_timeline/render_dir_controls/fixed_render_controls_candidate_left_stand_eval0.png
```

| Run | Crop PSNR | Crop SSIM | Finding |
|---|---:|---:|---|
| FG fixed 256 seed43 | `24.9053` | `0.6143` | very smoothed |
| fixed 512 seed44 | `26.2867` | `0.6636` | no ARM hole, still worse than Instant-NGP |
| fixed 640 seed43 | `26.6783` | `0.6736` | no ARM hole, still worse than Instant-NGP |

### Current apples-to-apples check

Previous visual audit noted that LookCloser runs used `scene_scale=2.0`, `scale_factor=1.15`, while the bounded Instant-NGP baseline uses `scene_scale=1.5`, `scale_factor=1.0`.

An apples-to-apples no-FAS H40 control was run:

```text
run: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_hand_stand_debug/lookcloser/nofas_h40_scene15_scale10_maxres8192_seed42
scene_scale=1.5
scale_factor=1.0
max_res=8192
FAS disabled
Feature Re-weighting disabled
Frequency Grid + Adaptive RM enabled
```

The first attempt without explicit `max_res=8192` failed before training because the existing frequency metadata expected `max_res=8192` while the model grid derived `max_res=6144` from the changed scene scale. The current run preserves the existing frequency-map metadata and isolates the dataparser scale change.

Rendered gate montages:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/hand_stand_artifact_timeline/apples_scene15_scale10_maxres8192_seed42_s4096/all_gate_stride2/all_crops.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/hand_stand_artifact_timeline/apples_scene15_scale10_maxres8192_seed42_s8192/all_gate_stride2/all_crops.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/hand_stand_artifact_timeline/apples_scene15_scale10_maxres8192_seed42_s12288/all_gate_stride2/all_crops.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/hand_stand_artifact_timeline/apples_scene15_scale10_maxres8192_seed42_s16384/all_gate_stride2/all_crops.png
```

| Checkpoint | Eval loss | Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual read |
|---|---:|---|---:|---:|---:|---:|---|
| `s4096` | `0.0315111` | `left_stand_eval0` | `26.0529` | `0.7984` | `26.9343` | `0.8528` | no large hand/background blotch; stand/outlet much cleaner than old H17 at the same step |
| `s4096` | `0.0315111` | `left_hand_background_eval0` | `24.9498` | `0.7932` | `26.2863` | `0.8745` | blotch no longer matches the old ARM-hole failure |
| `s4096` | `0.0315111` | `left_hand_outlet_stand_eval0` | `25.2332` | `0.7879` | `26.1977` | `0.8533` | still below baseline, but no severe black patch |
| `s8192` | `0.0294324` | `left_stand_eval0` | `26.0418` | `0.8097` | `26.9343` | `0.8528` | visually stable; no large hand/background blotch |
| `s8192` | `0.0294324` | `left_hand_background_eval0` | `24.9520` | `0.8099` | `26.2863` | `0.8745` | background around hand remains rendered rather than hidden by a blotch |
| `s8192` | `0.0294324` | `left_hand_outlet_stand_eval0` | `24.5389` | `0.7884` | `26.1977` | `0.8533` | outlet/stand still weaker than Instant-NGP |
| `s12288` | `0.0263846` | `left_stand_eval0` | `25.9362` | `0.8032` | `26.9343` | `0.8528` | still no large blotch; fine stand structure below baseline |
| `s12288` | `0.0263846` | `left_hand_background_eval0` | `24.9436` | `0.8063` | `26.2863` | `0.8745` | no return of old ARM-hole artifact |
| `s12288` | `0.0263846` | `left_hand_outlet_stand_eval0` | `24.9800` | `0.7924` | `26.1977` | `0.8533` | small stand/outlet weaknesses remain |
| `s16384` | `0.0295020` | `left_stand_eval0` | `25.5046` | `0.7983` | `26.9343` | `0.8528` | eval-loss overfit; no large blotch return |
| `s16384` | `0.0295020` | `left_hand_background_eval0` | `24.6934` | `0.8052` | `26.2863` | `0.8745` | background remains visible; local quality lower |
| `s16384` | `0.0295020` | `left_hand_outlet_stand_eval0` | `24.6227` | `0.7897` | `26.1977` | `0.8533` | worse than `s12288`; not selected |

Best clean no-FAS checkpoint for this diagnostic was `s12288`. Later no-FAS training overfit by eval loss and did not improve the hand/stand gates.

### FAS continuation from clean apples-scale checkpoint

What was tested: continue from the clean no-FAS `s12288` checkpoint with the accepted flat/count-proportional FAS recipe, still keeping Feature Re-weighting disabled.

```text
run: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_hand_stand_debug/lookcloser/fas035_count1_from_apples_s12288_seed42
load checkpoint: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_hand_stand_debug/lookcloser/nofas_h40_scene15_scale10_maxres8192_seed42/nerfstudio_models/step-000012288.ckpt
scene_scale=1.5
scale_factor=1.0
max_res=8192
fas_strength=0.35
fas_level_count_alpha=1.0
sampling_ramp_start=1.0
sampling_ramp_end=1.0
Feature Re-weighting disabled
```

Eval rows:

| Checkpoint | Eval loss | Status |
|---:|---:|---|
| `16384` | `0.0261401` | selected best |
| `20480` | `0.0284477` | overfit-watch, stopped |

Rendered gate montages:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/hand_stand_artifact_timeline/fas035_count1_from_apples_s12288_s16384/all_gate_stride2/all_crops.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/hand_stand_artifact_timeline/fas035_count1_from_apples_s12288_s20480/all_gate_stride2/all_crops.png
```

| Checkpoint | Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual read |
|---:|---|---:|---:|---:|---:|---|
| `16384` | `left_stand_eval0` | `25.9387` | `0.8020` | `26.9343` | `0.8528` | no old blotch; residual fine-stand gap remains |
| `16384` | `left_hand_background_eval0` | `24.9038` | `0.8041` | `26.2863` | `0.8745` | background around hand is visible, not hidden by the old patch |
| `16384` | `left_hand_outlet_stand_eval0` | `24.8701` | `0.7873` | `26.1977` | `0.8533` | small outlet/stand weakness remains |
| `20480` | `left_stand_eval0` | `26.1922` | `0.8090` | `26.9343` | `0.8528` | visually still clean, but full eval loss worse |
| `20480` | `left_hand_background_eval0` | `24.8531` | `0.8099` | `26.2863` | `0.8745` | no blotch return |
| `20480` | `left_hand_outlet_stand_eval0` | `25.0203` | `0.7945` | `26.1977` | `0.8533` | locally mixed; not selected by eval loss |

Full eval for selected FAS checkpoint:

```text
checkpoint: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_hand_stand_debug/lookcloser/fas035_count1_from_apples_s12288_seed42/nerfstudio_models/step-000016384.ckpt
eval json: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_hand_stand_debug/lookcloser/fas035_count1_from_apples_s12288_seed42/eval_best_step-000016384.json
renders: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_hand_stand_debug/lookcloser/fas035_count1_from_apples_s12288_seed42/renders_best_step-000016384
PSNR=29.174545
SSIM=0.669102
LPIPS=0.378236
```

### Uniform frequency-map status

No full constant-frequency-map training control has been run yet. Previous "uniform" checks only covered FAS sampling mechanics (`fas_level_count_alpha=1.0` and flat ramp) while keeping the real per-pixel frequency map. A constant frequency map remains a useful control only if the apples-to-apples scale run still reproduces the artifact or if later checkpoints reintroduce it.

## Insights

- The first severe hand/background blotch appears when Adaptive RM becomes active, before FAS.
- Loss type, distortion loss, batch size, maxfreq13, render-time alpha threshold, and smaller render-time coarse traversal do not fix the H40 hand/stand crop.
- Fixed-step LookCloser avoids the most obvious ARM holes but is still too smooth versus Instant-NGP, so simply disabling ARM is not enough for the target quality/metrics.
- Matching Instant-NGP dataparser scale (`scene_scale=1.5`, `scale_factor=1.0`) removes the severe early hand/background blotch seen in old H17/H40 runs. The current remaining problem is not the old large ARM hole, but weaker fine stand/cable/background detail versus Instant-NGP.
- Flat/count-proportional FAS from the clean apples-scale checkpoint keeps the old hand blotch fixed and reaches full-eval PSNR/SSIM comparable to the previous accepted two-stage FAS seed-42/43/44 range. It does not fully close the local SSIM/detail gap to Instant-NGP on the hand/outlet/stand crops.
