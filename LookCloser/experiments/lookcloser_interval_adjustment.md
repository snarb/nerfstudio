# LookCloser Interval Adjustment Speed-Up

## What was tested

Implemented a packed adaptive ray marching path for the first isolated Interval Adjustment test:

- Frequency Grid: on
- Adaptive Ray Marching / Interval Adjustment: on
- Feature Re-weighting: off
- FAS: off

The old `LookCloserModel.adaptive_ray_marching()` stepped active rays in Python and queried the field MLP inside the loop. The new path uses nerfacc occupancy traversal plus vectorized frequency-aware subdivision in `nerfstudio/model_components/lookcloser_samplers.py`.

Important unit fix: the paper interval `1 / (2 * N_l)` is in normalized AABB/hash-grid coordinates, so it is converted to ray `t` units as `dt = dt_norm / ||ray_dir / aabb_size||`. A small probe verified the normalized displacement matches `1 / (2 * N_l)` for levels 0, 5, 10, and 15.

Baseline-matched smoke settings used `nerfstudio-data`, `scene_scale=2.0`, `scale_factor=1.15`, `center_method=focus`, `orientation_method=up`, `near_plane=0.02`, `render_step_size_mult=0.75`, `alpha_thre=0.0025`, `cone_angle=0.0`, `background_color=black`, and `use_gradient_scaling=false`.

## Results

| Run | Purpose | Result |
|---|---|---|
| `adaptive_interval_packed_metrics_smoke4_r64` | 4-step batch-eval smoke before coarse-step fix | Completed, but used about `650` samples/ray because coarse traversal still used the small baseline render step. Rejected. |
| `adaptive_interval_packed_coarse_smoke4_r64` | 4-step batch-eval smoke after coarse-step fix | Completed in `20.07s` including startup. Train step 0 used `2110` samples for 64 rays, mean `32.97`, max `41`, saturation `0`. Eval batches used about `33-34` samples/ray. |
| `adaptive_interval_packed_final_smoke16_r64` | Requested 16-step smoke, no full-image eval | Completed in `20.07s` including startup. No `eval_image_*` or `eval_all_*` metrics were populated. Iteration time fell to about `0.11s` by step 15. |
| `adaptive_interval_cropgate_128_r64` | Small crop-only visual gate checkpoint | Completed 128 iterations in `30.08s` including startup and batch eval. Best observed eval batch loss was `0.0979307` at step 48, but only latest checkpoint step 127 was saved. |

Packed distortion check against nerfstudio's dense distortion loss:

- dense: `[0.1186666787, 0.0832500011]`
- packed: `[0.1186666787, 0.0832499936]`
- max absolute difference: `7.45e-09`

Crop-only render artifacts from the 128-step checkpoint:

- directory: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_interval_adjustment_smoke/lookcloser/adaptive_interval_cropgate_128_r64/crop_gate_stride4`
- overview sheet: `all_crops.png`
- metrics: `metrics.csv`

| Crop | Rays | Seconds | PSNR | SSIM | Pixel std |
|---|---:|---:|---:|---:|---:|
| floor_crack_eval0 | 3525 | 0.993 | 25.8539 | 0.66548 | 17.0505 |
| fingers_right_eval1 | 6600 | 0.558 | 12.6620 | 0.24546 | 17.6062 |
| stand_label_eval2 | 6554 | 0.551 | 9.6297 | 0.21784 | 44.3435 |
| tangled_cable_eval2 | 6975 | 0.586 | 13.0295 | 0.25183 | 36.2829 |
| fingers_center_eval2 | 5110 | 0.432 | 13.1388 | 0.28556 | 24.7113 |

Visual inspection: crops are aligned and nonblank, but the 128-step candidate is heavily blurred and is not a quality improvement claim. It is only a rendering/speed sanity check.

## Insights

- The Python-loop bottleneck is removed for smoke-scale testing.
- The first packed implementation still accidentally sampled low-frequency regions at the small Instant-NGP render step; switching adaptive traversal to the adaptive max interval fixed this and reduced samples from about `650` per ray to about `33` per ray in the same 64-ray smoke.
- Full-image eval is still intentionally skipped. The next meaningful quality test should train a longer isolated Interval Adjustment run, save the lowest-eval-loss checkpoint, and then do the established crop gate before any full-HD final eval.

## Iteration 1 - Faster Runtime Updates At Smoke Scale

### What was tested

Hypothesis: `grid_update_interval=128` with `grid_update_batch_size=4096` would populate the frequency grid early enough for Interval Adjustment to affect a short isolated run.

Settings stayed isolated: Frequency Grid on, Adaptive RM on, Feature Re-weighting off, FAS off. No full-image eval was run.

### Results

Run: `adaptive_fg_arm_iso_h1_update128_profile512`

- Train budget: 512 iterations, 1024 rays/batch.
- Runtime: `50.06s` including startup.
- Best eval-batch loss: `0.0930171` at step 64; later eval-batch losses worsened.
- Latest checkpoint: step 511.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h1_update128_profile512/crop_gate_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 24.3178 | 0.66792 | 21.7506 | 0.88176 | smoother, weak crack detail |
| fingers_right_eval1 | 11.3881 | 0.23318 | 27.3119 | 0.96735 | blurred |
| stand_label_eval2 | 13.5147 | 0.20977 | 31.6403 | 0.97147 | blurred, unreadable |
| tangled_cable_eval2 | 15.5548 | 0.24146 | 31.5280 | 0.97162 | blurred wires |
| fingers_center_eval2 | 11.9277 | 0.24978 | 25.7366 | 0.96259 | blurred |

### Insights

Rejected as a quality result. The implementation is fast enough for smoke, but 512 iterations is not enough to judge improvement. The next hypothesis is a real training-budget run with larger ray batches and saved eval checkpoints so the crop gate can inspect the best-eval-loss checkpoint instead of an arbitrary latest checkpoint.

## Iteration 2 - Longer Budget With Carried Runtime Update Cadence

### What was tested

Hypothesis: the isolated adaptive setup needs a real training budget and saved eval checkpoints before crop quality can be judged. Used the carried runtime-update cadence `grid_update_interval=512`, `grid_update_batch_size=4096`, with Frequency Grid on, Adaptive RM on, Feature Re-weighting off, and FAS off.

### Results

Run: `adaptive_fg_arm_iso_h2_update512_train4096_r4096`

- Train budget: 4096 iterations, 4096 rays/batch.
- Runtime: `390.35s`.
- Best eval-batch loss: `0.0916356` at step 1024.
- Selected checkpoint: step 1024.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h2_update512_train4096_r4096/crop_gate_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 19.1520 | 0.59833 | 21.7506 | 0.88176 | blurry/noisy floor, worse crack |
| fingers_right_eval1 | 10.9084 | 0.23408 | 27.3119 | 0.96735 | blurred person/hand |
| stand_label_eval2 | 11.4499 | 0.13943 | 31.6403 | 0.97147 | unreadable, blocky |
| tangled_cable_eval2 | 12.4628 | 0.16585 | 31.5280 | 0.97162 | missing wires |
| fingers_center_eval2 | 12.2196 | 0.21196 | 25.7366 | 0.96259 | blurred, block artifacts |

### Insights

Rejected. More training budget alone did not recover high-frequency details. Eval-batch loss selected an early checkpoint, but the crop gate remains much worse than Instant-NGP. Next hypothesis: remove the distortion loss for isolated adaptive training, because packed sparse intervals plus a tiny-detail objective may make the current distortion regularizer oversmooth or destabilize geometry.

## Iteration 3 - Remove Distortion Loss

### What was tested

Hypothesis: the packed sparse adaptive intervals plus tiny-detail objective may be oversmoothed or destabilized by the current distortion regularizer, so `distortion_loss_mult=0.0` was tested. Other settings matched Iteration 2.

### Results

Run: `adaptive_fg_arm_iso_h3_no_distortion_train4096_r4096`

- Train budget: 4096 iterations, 4096 rays/batch.
- Runtime: `360.41s`.
- Best eval-batch loss: `0.0997715` at step 512, worse than Iteration 2's `0.0916356`.
- Selected checkpoint: step 512.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h3_no_distortion_train4096_r4096/crop_gate_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 20.1699 | 0.64121 | 21.7506 | 0.88176 | smoother, no visual win |
| fingers_right_eval1 | 11.0050 | 0.22335 | 27.3119 | 0.96735 | blurred/block artifact |
| stand_label_eval2 | 15.0399 | 0.20466 | 31.6403 | 0.97147 | slightly better PSNR than Iteration 2 but unreadable |
| tangled_cable_eval2 | 14.0130 | 0.18048 | 31.5280 | 0.97162 | missing wires/block artifacts |
| fingers_center_eval2 | 12.7978 | 0.25441 | 25.7366 | 0.96259 | blurred/block artifact |

### Insights

Rejected. Removing distortion loss did not improve the visual target and worsened eval-batch loss. Keep the default distortion loss. Next hypothesis: reduce `adaptive_max_step_size` so low/empty-grid regions do not train with only about 33 samples per ray before reliable high-frequency grid updates exist.

## Iteration 4 - Smaller Adaptive Max Step

### What was tested

Hypothesis: `adaptive_max_step_size=0.1` leaves empty/low-frequency grid regions too coarsely sampled early in training, causing persistent blur. Tested `adaptive_max_step_size=0.025` with the Iteration 2 settings, keeping distortion loss enabled.

### Results

Run: `adaptive_fg_arm_iso_h4_maxstep0025_train2048_r4096`

- Train budget: 2048 iterations, 4096 rays/batch.
- Runtime: `180.19s`.
- Best eval-batch loss: `0.0937316` at step 512, worse than Iteration 2's `0.0916356`.
- Selected checkpoint: step 512.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h4_maxstep0025_train2048_r4096/crop_gate_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 16.2356 | 0.53460 | 21.7506 | 0.88176 | worse floor/crack |
| fingers_right_eval1 | 11.0893 | 0.23009 | 27.3119 | 0.96735 | blurred |
| stand_label_eval2 | 12.0064 | 0.13810 | 31.6403 | 0.97147 | blocky/unreadable |
| tangled_cable_eval2 | 12.0997 | 0.13448 | 31.5280 | 0.97162 | blocky/missing wires |
| fingers_center_eval2 | 12.0758 | 0.22440 | 25.7366 | 0.96259 | blocky |

### Insights

Rejected. More low-frequency sampling did not fix detail recovery and worsened crops. Revert to `adaptive_max_step_size=0.1`. Next hypothesis: switch the reconstruction loss from Charbonnier to MSE so the isolated adaptive model is optimized closer to the bounded Instant-NGP metric target.

## Iteration 5 - MSE Reconstruction Loss

### What was tested

Hypothesis: MSE reconstruction loss may optimize closer to the bounded Instant-NGP metric target than Charbonnier. Added `reconstruction_loss_type` to LookCloser and tested `mse`, with all isolated module flags unchanged.

### Results

Run: `adaptive_fg_arm_iso_h5_mse_train4096_r4096`

- Train budget: 4096 iterations, 4096 rays/batch.
- Runtime: `510.47s`.
- Best eval-batch MSE loss: `0.0139232` at step 1024.
- Selected checkpoint: step 1024.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h5_mse_train4096_r4096/crop_gate_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 21.7575 | 0.43148 | 21.7506 | 0.88176 | PSNR tie but much worse structure |
| fingers_right_eval1 | 12.2079 | 0.20451 | 27.3119 | 0.96735 | blocky/missing hand detail |
| stand_label_eval2 | 11.6062 | 0.14535 | 31.6403 | 0.97147 | blocky/unreadable |
| tangled_cable_eval2 | 13.2955 | 0.19431 | 31.5280 | 0.97162 | missing wires |
| fingers_center_eval2 | 12.7490 | 0.22922 | 25.7366 | 0.96259 | blocky |

### Insights

Rejected. MSE did not improve visual detail and introduced obvious block artifacts. Keep Charbonnier for now. Next hypothesis: set `alpha_thre=0.0` for adaptive traversal to avoid pruning weak early densities and causing missing/blocky geometry.

## Iteration 6 - No Alpha Pruning

### What was tested

Hypothesis: adaptive traversal was pruning weak early densities too aggressively, causing missing/blocky geometry. Tested `alpha_thre=0.0` with the Iteration 2 settings and Charbonnier loss.

### Results

Run: `adaptive_fg_arm_iso_h6_alpha0_train4096_r4096`

- Train budget: 4096 iterations, 4096 rays/batch.
- Runtime: `390.37s`.
- Best eval-batch loss: `0.0901013` at step 512, numerically best so far but still an early, visually weak checkpoint.
- Selected checkpoint: step 512.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h6_alpha0_train4096_r4096/crop_gate_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 19.7276 | 0.60337 | 21.7506 | 0.88176 | blurred, no crack detail win |
| fingers_right_eval1 | 11.2439 | 0.23221 | 27.3119 | 0.96735 | blurred/blocky hand boundary |
| stand_label_eval2 | 10.9200 | 0.16582 | 31.6403 | 0.97147 | unreadable label |
| tangled_cable_eval2 | 12.7343 | 0.16749 | 31.5280 | 0.97162 | missing wires |
| fingers_center_eval2 | 12.2647 | 0.23297 | 25.7366 | 0.96259 | blurred/blocky |

### Insights

Rejected. Removing alpha pruning improved the compact eval-batch loss but did not pass the visual crop gate. Adaptive sample statistics also show a schedule problem: before the first frequency-grid update, occupancy pruning reduces many rays to very few samples; after the update, mean samples rise sharply and later approach the per-ray cap with nontrivial saturation. Next hypothesis: keep alpha pruning at the baseline-matched value, but impose a moderate minimum frequency level for interval adjustment so early/empty-grid voxels do not march with overly coarse intervals.

## Iteration 7 - Minimum Adaptive Frequency Level 5

### What was tested

Hypothesis: empty or not-yet-updated frequency-grid voxels are causing the adaptive marcher to use overly coarse intervals early in training. Added an interval-adjustment-only clamp and tested `adaptive_min_frequency_level=5`, while restoring `alpha_thre=0.0025`.

### Results

Run: `adaptive_fg_arm_iso_h7_minfreq5_train4096_r4096`

- Train budget: configured for 4096 iterations, stopped at 1024 by eval-loss guard.
- Runtime: `91.35s`.
- Best eval-batch loss: `0.106162` at step 512, worse than Iteration 2 and Iteration 6.
- Selected checkpoint: step 512.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h7_minfreq5_train4096_r4096/crop_gate_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 20.5219 | 0.63640 | 21.7506 | 0.88176 | blurred, no structural win |
| fingers_right_eval1 | 10.9271 | 0.23252 | 27.3119 | 0.96735 | hand mostly smeared |
| stand_label_eval2 | 12.0454 | 0.11948 | 31.6403 | 0.97147 | unreadable and blocky |
| tangled_cable_eval2 | 12.3887 | 0.13457 | 31.5280 | 0.97162 | wires missing |
| fingers_center_eval2 | 12.4064 | 0.22618 | 25.7366 | 0.96259 | blurred/blocky |

### Insights

Rejected. Forcing a moderate minimum level increased early sampling but hurt eval loss and did not improve the high-frequency crop sheet. Do not carry this setting. Next hypothesis: the current early-stop/best-eval checkpoint selection is not a reliable proxy for visual high-frequency detail this early; rerun the carried Iteration 2 settings longer without early stopping and inspect the latest crop checkpoint.

## Iteration 8 - Longer Latest Checkpoint

### What was tested

Hypothesis: early eval-loss checkpoint selection may reject checkpoints before adaptive marching has trained enough high-frequency structure. Reran the carried Iteration 2 settings with early stopping disabled and latest-checkpoint selection. The run was stopped after step 4660 once sample-cap saturation became severe.

### Results

Run: `adaptive_fg_arm_iso_h8_long_latest15188_r4096`

- Planned train budget: 15188 iterations, 4096 rays/batch.
- Stopped after step 4660 because about one third of rays were hitting `max_steps_per_ray=1024`.
- Eval-batch loss: `0.100886` at step 2048 and `0.125911` at step 4096, worsening with training.
- Inspected checkpoint: step 4096.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h8_long_latest15188_r4096/crop_gate_step4096_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 21.4623 | 0.57673 | 21.7506 | 0.88176 | smeared floor, crack not recovered |
| fingers_right_eval1 | 11.4960 | 0.22871 | 27.3119 | 0.96735 | hand boundary missing |
| stand_label_eval2 | 12.4348 | 0.21083 | 31.6403 | 0.97147 | no readable label |
| tangled_cable_eval2 | 17.1504 | 0.25307 | 31.5280 | 0.97162 | some contrast but wires still absent |
| fingers_center_eval2 | 12.2585 | 0.22499 | 25.7366 | 0.96259 | blurred/blocky |

### Insights

Rejected. Training longer without controlling interval sizes worsens eval loss and sample saturation, and the latest crop remains visually unacceptable. Next hypothesis: cap the adaptive frequency level used for interval sizing to reduce cap saturation after frequency-grid updates.

## Iteration 9 - Maximum Adaptive Frequency Level 12

### What was tested

Hypothesis: runtime frequency-grid updates over-assign high levels for interval adjustment, driving many rays into the `max_steps_per_ray=1024` cap. Added an interval-adjustment-only maximum level and tested `adaptive_max_frequency_level=12`, with the carried Iteration 2 settings.

### Results

Run: `adaptive_fg_arm_iso_h9_maxfreq12_train4096_r4096`

- Configured train budget: 4096 iterations, stopped at 1536 by eval-loss guard.
- Runtime: `121.34s`.
- Best eval-batch loss: `0.0866579` at step 1024, the best compact eval loss so far.
- Selected checkpoint: step 1024.
- Sampler stats: eval mean samples stayed around `44.6` to `57.8` per ray, max about `360`, saturation `0`. This fixed the previous late-run cap saturation.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h9_maxfreq12_train4096_r4096/crop_gate_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 19.3057 | 0.55767 | 21.7506 | 0.88176 | blurred/no crack win |
| fingers_right_eval1 | 11.5166 | 0.23324 | 27.3119 | 0.96735 | boundary still smeared |
| stand_label_eval2 | 10.9618 | 0.14771 | 31.6403 | 0.97147 | unreadable |
| tangled_cable_eval2 | 12.3353 | 0.12747 | 31.5280 | 0.97162 | wires absent |
| fingers_center_eval2 | 11.9968 | 0.21984 | 25.7366 | 0.96259 | blurred/blocky |

### Insights

Partially accepted as a stability fix, not a visual win. The level cap fixes runaway sample saturation and improves compact eval loss, but the selected 1024-step checkpoint still fails the crop gate. Next hypothesis: with cap saturation fixed, train longer without early stopping and inspect later checkpoints for visual recovery.

## Iteration 10 - Longer Training With Level-12 Cap

### What was tested

Hypothesis: the level-12 cap fixed the sampler instability, so longer training might recover visual detail. Reran `adaptive_max_frequency_level=12` with early stopping disabled and latest/checkpoint inspection.

### Results

Run: `adaptive_fg_arm_iso_h10_maxfreq12_long15188_r4096`

- Planned train budget: 15188 iterations, 4096 rays/batch.
- Stopped after the 8192 checkpoint because eval-batch loss kept worsening.
- Eval-batch losses: `0.0867312` at step 2048, `0.108533` at step 4096, `0.116851` at step 6144, `0.124670` at step 8192.
- Sampler stats remained stable; around step 4700, mean train samples were about `143` to `148` per ray with near-zero saturation.
- Inspected checkpoint: step 8192.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h10_maxfreq12_long15188_r4096/crop_gate_step8192_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 19.6585 | 0.60181 | 21.7506 | 0.88176 | blurred/no crack recovery |
| fingers_right_eval1 | 10.5259 | 0.22306 | 27.3119 | 0.96735 | hand boundary still lost |
| stand_label_eval2 | 13.6595 | 0.17205 | 31.6403 | 0.97147 | label still unreadable |
| tangled_cable_eval2 | 14.6676 | 0.13386 | 31.5280 | 0.97162 | wires missing |
| fingers_center_eval2 | 12.9746 | 0.23203 | 25.7366 | 0.96259 | blurred/blocky |

### Insights

Rejected as a quality fix. Level 12 is stable and fast but under-samples the target details. Next hypothesis: raise the interval-adjustment cap to level 13 to increase high-frequency sampling while still avoiding the uncapped level-15 sample explosion.

## Iteration 11 - Maximum Adaptive Frequency Level 13

### What was tested

Hypothesis: level 12 was too conservative for high-frequency detail, while uncapped level 15 saturated. Tested `adaptive_max_frequency_level=13`.

### Results

Run: `adaptive_fg_arm_iso_h11_maxfreq13_train4096_r4096`

- Configured train budget: 4096 iterations, stopped at 2048 by eval-loss guard.
- Runtime: `151.38s`.
- Best eval-batch loss: `0.0890107` at step 1536.
- Selected checkpoint: step 1536.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h11_maxfreq13_train4096_r4096/crop_gate_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 20.1878 | 0.56288 | 21.7506 | 0.88176 | blurred/no crack win |
| fingers_right_eval1 | 11.4061 | 0.22747 | 27.3119 | 0.96735 | hand boundary smeared |
| stand_label_eval2 | 12.6375 | 0.17084 | 31.6403 | 0.97147 | label unreadable |
| tangled_cable_eval2 | 13.4400 | 0.15000 | 31.5280 | 0.97162 | wires missing |
| fingers_center_eval2 | 12.5188 | 0.23364 | 25.7366 | 0.96259 | blurred/blocky |

### Insights

Rejected. Level 13 increases sampling compared with level 12 but does not produce a visual crop win, and its compact eval loss is worse than level 12. Next diagnostic: run a fixed-sample isolated control with the same Frequency Grid on and Feature Re-weighting/FAS off, to determine whether the crop failures are specific to Interval Adjustment or to short isolated LookCloser training.

## Iteration 12 - Fixed-Sample Isolated Control

### What was tested

Diagnostic hypothesis: the poor crop gates may be caused by short isolated LookCloser training rather than adaptive interval adjustment. Ran the same isolated module setup with Frequency Grid on, Feature Re-weighting off, FAS off, but disabled adaptive marching and used fixed `256` samples/ray.

### Results

Run: `fixed_fg_iso_control_h12_train4096_r4096_s256`

- Train budget: 4096 iterations, 4096 rays/batch.
- Runtime: `420.44s`.
- Best eval-batch loss: `0.0360221` at step 3584, much better than all adaptive runs.
- Selected checkpoint: step 3584.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/fixed_fg_iso_control_h12_train4096_r4096_s256/crop_gate_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 22.9861 | 0.57943 | 21.7506 | 0.88176 | sharper than adaptive but wrong bright artifact |
| fingers_right_eval1 | 12.3050 | 0.23058 | 27.3119 | 0.96735 | still weak, but less smeared than adaptive |
| stand_label_eval2 | 10.3376 | 0.08828 | 31.6403 | 0.97147 | structure sharper, label not readable |
| tangled_cable_eval2 | 12.6544 | 0.10595 | 31.5280 | 0.97162 | more structure than adaptive, still not baseline |
| fingers_center_eval2 | 12.3090 | 0.21074 | 25.7366 | 0.96259 | sharper edges than adaptive, still poor |

### Insights

Control confirms adaptive interval adjustment is the main bottleneck at this budget. Fixed 256 samples trains much better numerically and gives visibly sharper structure, though it is still not enough to beat the full Instant-NGP baseline. Next hypothesis: test `adaptive_max_frequency_level=14`, the last compromise before uncapped level-15 saturation, to increase effective adaptive sample density toward the fixed-control regime.

## Iteration 13 - Maximum Adaptive Frequency Level 14

### What was tested

Hypothesis: level 13 was still too conservative, and level 14 may increase adaptive sample density enough to recover more structure without returning to uncapped level-15 saturation.

### Results

Run: `adaptive_fg_arm_iso_h13_maxfreq14_train4096_r4096`

- Configured train budget: 4096 iterations, stopped at 1536 by eval-loss guard.
- Runtime: `121.33s`.
- Best eval-batch loss: `0.0863663` at step 1024, slightly better than level 12.
- Selected checkpoint: step 1024.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h13_maxfreq14_train4096_r4096/crop_gate_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 20.4584 | 0.61797 | 21.7506 | 0.88176 | still blurred |
| fingers_right_eval1 | 10.9968 | 0.23175 | 27.3119 | 0.96735 | boundary smeared |
| stand_label_eval2 | 12.6888 | 0.17507 | 31.6403 | 0.97147 | unreadable |
| tangled_cable_eval2 | 13.7842 | 0.18055 | 31.5280 | 0.97162 | wires not recovered |
| fingers_center_eval2 | 12.6922 | 0.23497 | 25.7366 | 0.96259 | blurred/blocky |

### Insights

Rejected as a standalone quality fix. It is the best adaptive cap numerically, but selected early checkpoints still fail visually. Next hypothesis: adaptive marching should not be used from step 0; use fixed samples as a geometry warmup, then switch to capped adaptive Interval Adjustment.

## Iteration 14 - Fixed-Sample Warmup Then Adaptive

### What was tested

Hypothesis: adaptive interval adjustment fails when used from scratch with an empty/noisy frequency grid and weak early density field. Added `adaptive_warmup_steps` and tested fixed-sample training for 2048 steps, then adaptive training with `adaptive_max_frequency_level=14`.

### Results

Run: `adaptive_fg_arm_iso_h14_warmup2048_maxfreq14_train4096_r4096`

- Train budget: 4096 iterations, 4096 rays/batch.
- Runtime: `390.40s`.
- Best eval-batch loss: `0.0398327` at step 3584, close to the fixed-control `0.0360221` and far better than all adaptive-from-scratch runs.
- Inspected checkpoint: step 3584.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h14_warmup2048_maxfreq14_train4096_r4096/crop_gate_best3584_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 23.0454 | 0.55184 | 21.7506 | 0.88176 | sharper but wrong bright artifact |
| fingers_right_eval1 | 12.3240 | 0.21638 | 27.3119 | 0.96735 | still weak hand boundary |
| stand_label_eval2 | 10.3806 | 0.10202 | 31.6403 | 0.97147 | structure returns, label unreadable |
| tangled_cable_eval2 | 13.2709 | 0.12452 | 31.5280 | 0.97162 | more structure than adaptive-from-scratch, wires still absent |
| fingers_center_eval2 | 12.0878 | 0.21015 | 25.7366 | 0.96259 | sharper than earlier adaptive, still poor |

### Insights

Accepted as the first real adaptive-speed/quality fix. Warmup plus capped adaptive recovers fixed-control-level eval loss and visible scene structure, proving the previous adaptive-from-scratch setup was broken for quality. It still does not pass the Instant-NGP visual crop gate. Next step: extend this warmup configuration to a longer checkpoint before any full-image eval.

## Iteration 15 - Longer Warmup-Adaptive Continuation

### What was tested

Hypothesis: the 2048-step warmup configuration was finally training stably, but needed a longer checkpoint before visual detail could be judged. Continued from the 4095-step warmup checkpoint to 15188 total iterations with adaptive Interval Adjustment active and `adaptive_max_frequency_level=14`.

### Results

Run: `adaptive_fg_arm_iso_h15_warmup2048_maxfreq14_continue15188_r4096`

- Continued from: `adaptive_fg_arm_iso_h14_warmup2048_maxfreq14_train4096_r4096/step-000004095.ckpt`.
- Additional runtime: `961.12s`.
- Best eval-batch loss: `0.0350567` at step 12288.
- Inspected checkpoint: step 12288.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h15_warmup2048_maxfreq14_continue15188_r4096/crop_gate_best12288_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 23.3429 | 0.56547 | 21.7506 | 0.88176 | sharper floor but wrong bright artifact; no clean crack win |
| fingers_right_eval1 | 12.2837 | 0.22728 | 27.3119 | 0.96735 | hand boundary still weak |
| stand_label_eval2 | 10.2181 | 0.09498 | 31.6403 | 0.97147 | label unreadable |
| tangled_cable_eval2 | 12.9284 | 0.11895 | 31.5280 | 0.97162 | wires still missing |
| fingers_center_eval2 | 11.9494 | 0.20625 | 25.7366 | 0.96259 | blurred/incorrect boundary |

### Insights

Rejected as a visual improvement over Instant-NGP. It is the best adaptive compact-loss run and proves the speed fix can support practical medium training, but it still fails the required high-frequency crop gate. Next hypothesis: use a longer fixed geometry warmup before switching to adaptive, because the 2048-step switch may still be too early for this scene.

## Iteration 16 - Delayed 8192-Step Warmup

### What was tested

Hypothesis: the 2048-step fixed warmup may switch to adaptive too early. Tested fixed-sample warmup for 8192 steps, then adaptive Interval Adjustment with `adaptive_max_frequency_level=14`.

### Results

Run: `adaptive_fg_arm_iso_h16_warmup8192_maxfreq14_train15188_r4096`

- Planned train budget: 15188 iterations, stopped after step 12288 because it failed to beat Iteration 15.
- Best eval-batch loss before stop: `0.0371527` at step 10240, worse than Iteration 15's `0.0350567`.
- Inspected checkpoint: step 10240.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h16_warmup8192_maxfreq14_train15188_r4096/crop_gate_best10240_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 23.7195 | 0.57002 | 21.7506 | 0.88176 | sharper but wrong artifact; no clean crack win |
| fingers_right_eval1 | 12.2798 | 0.21723 | 27.3119 | 0.96735 | weak boundary |
| stand_label_eval2 | 10.1998 | 0.09733 | 31.6403 | 0.97147 | unreadable |
| tangled_cable_eval2 | 13.2721 | 0.13311 | 31.5280 | 0.97162 | wires still absent |
| fingers_center_eval2 | 12.0095 | 0.18521 | 25.7366 | 0.96259 | worse hand structure |

### Insights

Rejected. Delaying the switch to 8192 steps is slower, worse in compact eval loss, and still fails visually. The 2048-step warmup remains the carried schedule. Next hypothesis: keep the 2048 warmup but lower the adaptive cap from 14 to 12 for the post-warmup phase to reduce saturation and artifacts.

## Iteration 17 - Warmup With Level-12 Cap

### What was tested

Hypothesis: the carried 2048-step warmup was correct, but level 14 caused too much late saturation/artifacting. Continued from the 2048-step fixed-warmup checkpoint with `adaptive_max_frequency_level=12`.

### Results

Run: `adaptive_fg_arm_iso_h17_warmup2048_maxfreq12_continue12288_r4096`

- Continued from: `adaptive_fg_arm_iso_h14_warmup2048_maxfreq14_train4096_r4096/step-000002048.ckpt`.
- Runtime: `660.74s`.
- Best eval-batch loss: `0.0337417` at step 10240, best adaptive compact loss so far.
- Inspected checkpoint: step 10240.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h17_warmup2048_maxfreq12_continue12288_r4096/crop_gate_best10240_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 22.4444 | 0.53813 | 21.7506 | 0.88176 | artifact remains; no clean crack win |
| fingers_right_eval1 | 12.0005 | 0.22045 | 27.3119 | 0.96735 | weak hand boundary |
| stand_label_eval2 | 10.1730 | 0.10115 | 31.6403 | 0.97147 | unreadable |
| tangled_cable_eval2 | 13.0975 | 0.13301 | 31.5280 | 0.97162 | wires missing |
| fingers_center_eval2 | 12.1911 | 0.21142 | 25.7366 | 0.96259 | boundary still poor |

### Insights

Accepted as the carried adaptive metric configuration, but not a visual success. It has the best compact eval loss and lower cap pressure than level 14, yet still fails the required high-frequency visual crop gate. Next step: extend this carried candidate to a 30k-scale checkpoint before deciding whether isolated Interval Adjustment can visually catch up.

## Iteration 18 - 30k-Scale Carried Candidate

### What was tested

Hypothesis: the carried `warmup2048 + adaptive_max_frequency_level=12` configuration needed a longer checkpoint to recover visual details. Continued the best 10k checkpoint to a 30k-scale budget.

### Results

Run: `adaptive_fg_arm_iso_h18_warmup2048_maxfreq12_continue30376_r4096`

- Continued from: `adaptive_fg_arm_iso_h17_warmup2048_maxfreq12_continue12288_r4096/step-000010240.ckpt`.
- Additional runtime: `1291.71s`.
- Best eval-batch loss: `0.0320170` at step 24576, best adaptive compact loss so far.
- Inspected checkpoint: step 24576.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h18_warmup2048_maxfreq12_continue30376_r4096/crop_gate_best24576_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 23.0921 | 0.55983 | 21.7506 | 0.88176 | artifact remains; no clean crack win |
| fingers_right_eval1 | 12.1303 | 0.21511 | 27.3119 | 0.96735 | hand boundary not recovered |
| stand_label_eval2 | 10.1904 | 0.09720 | 31.6403 | 0.97147 | label unreadable |
| tangled_cable_eval2 | 12.9826 | 0.11661 | 31.5280 | 0.97162 | wires absent |
| fingers_center_eval2 | 12.1583 | 0.20369 | 25.7366 | 0.96259 | boundary still poor |

### Insights

Rejected visually despite the best compact eval loss. The longer run improves broad eval loss but not the target high-frequency patches. Next hypothesis: use the previously selected Frequency Grid resolution `64` instead of the current paper/default `128`, keeping the warmup and cap schedule unchanged.

## Iteration 19 - Grid Resolution 64

### What was tested

Hypothesis: the previous LookCloser frequency-grid sweeps selected `grid_resolution=64`, while all adaptive Interval Adjustment tuning here used `128`. Tested `grid_resolution=64` with the carried `warmup2048 + adaptive_max_frequency_level=12` schedule.

### Results

Run: `adaptive_fg_arm_iso_h19_grid64_warmup2048_maxfreq12_train12288_r4096`

- Train budget: 12288 iterations, 4096 rays/batch.
- Runtime: `900.96s`.
- Best eval-batch loss: `0.0311500` at step 10240, best adaptive compact loss so far.
- Eval adaptive samples at best: mean `315.145` per ray, saturation `0`.
- Inspected checkpoint: step 10240.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h19_grid64_warmup2048_maxfreq12_train12288_r4096/crop_gate_best10240_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 23.4895 | 0.52713 | 21.7506 | 0.88176 | artifact remains |
| fingers_right_eval1 | 12.3160 | 0.21790 | 27.3119 | 0.96735 | hand boundary weak |
| stand_label_eval2 | 10.1194 | 0.09191 | 31.6403 | 0.97147 | unreadable |
| tangled_cable_eval2 | 12.9213 | 0.11737 | 31.5280 | 0.97162 | wires absent |
| fingers_center_eval2 | 12.1217 | 0.19970 | 25.7366 | 0.96259 | boundary poor |

### Insights

Accepted as the carried metric/speed configuration, but not a visual win. Grid 64 improves compact eval loss and keeps adaptive sampling stable, yet the 10k high-frequency crops remain below the Instant-NGP baseline. Next step: extend this grid-64 candidate to a 30k-scale checkpoint because FAS is intentionally off and high-frequency pixels may need more uniform-sampling exposure.

## Iteration 20 - 30k Grid-64 Candidate

### What was tested

Hypothesis: grid 64 with warmup and capped adaptive needed a longer checkpoint before high-frequency crops could catch up. Continued the best grid-64 checkpoint to a 30k-scale budget.

### Results

Run: `adaptive_fg_arm_iso_h20_grid64_warmup2048_maxfreq12_continue30376_r4096`

- Continued from: `adaptive_fg_arm_iso_h19_grid64_warmup2048_maxfreq12_train12288_r4096/step-000010240.ckpt`.
- Additional runtime: `1351.76s`.
- Best eval-batch loss: `0.0294421` at step 24576, best adaptive compact loss so far.
- Inspected checkpoint: step 24576.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h20_grid64_warmup2048_maxfreq12_continue30376_r4096/crop_gate_best24576_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 23.1437 | 0.49065 | 21.7506 | 0.88176 | artifact remains |
| fingers_right_eval1 | 12.2554 | 0.21587 | 27.3119 | 0.96735 | hand boundary not recovered |
| stand_label_eval2 | 10.1260 | 0.09034 | 31.6403 | 0.97147 | unreadable |
| tangled_cable_eval2 | 12.9474 | 0.12122 | 31.5280 | 0.97162 | wires absent |
| fingers_center_eval2 | 12.1144 | 0.19719 | 25.7366 | 0.96259 | boundary poor |

### Insights

Rejected visually. The sampler is now fast and compact eval loss is strong, but the isolated model still fails high-frequency foreground crops. Next hypothesis: align LookCloser's color MLP depth with bounded Instant-NGP by changing only `color_num_layers` from 2 to 3 while keeping Frequency Grid on, Adaptive RM on, Feature Re-weighting off, and FAS off.

## Iteration 21 - Color MLP Depth 3

### What was tested

Hypothesis: bounded Instant-NGP uses a deeper color MLP than the current LookCloser default. Tested `color_num_layers=3` with the carried grid64, 2048-step warmup, and level-12 adaptive cap.

### Results

Run: `adaptive_fg_arm_iso_h21_grid64_warmup2048_maxfreq12_color3_train12288_r4096`

- Train budget: 12288 iterations, 4096 rays/batch.
- Runtime: `901.08s`.
- Best eval-batch loss: `0.0306348` at step 10240.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h21_grid64_warmup2048_maxfreq12_color3_train12288_r4096/crop_gate_best10240_stride4/all_crops.png`.

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 23.4277 | 0.56510 | 21.7506 | 0.88176 | artifact/no clean crack win |
| fingers_right_eval1 | 12.4306 | 0.21953 | 27.3119 | 0.96735 | boundary weak |
| stand_label_eval2 | 10.1215 | 0.08436 | 31.6403 | 0.97147 | unreadable |
| tangled_cable_eval2 | 12.7898 | 0.10926 | 31.5280 | 0.97162 | wires absent |
| fingers_center_eval2 | 12.1213 | 0.19709 | 25.7366 | 0.96259 | boundary poor |

### Insights

Rejected visually. Color depth 3 slightly improves compact loss at 10k versus color depth 2 at 10k, but it still fails the high-frequency crop gate and is worse than the 30k color-depth-2 grid64 candidate. Do not carry this as a visual improvement.

## Iteration 22 - Appearance Embedding 32

### What was tested

Hypothesis: bounded Instant-NGP includes a 32D appearance embedding, while LookCloser had none. Added optional appearance embedding support, default-off, and tested `appearance_embedding_dim=32` with the carried grid64/warmup/cap settings and `color_num_layers=3`.

### Results

Run: `adaptive_fg_arm_iso_h22_grid64_warmup2048_maxfreq12_color3_app32_train12288_r4096`

- Planned train budget: 12288 iterations.
- Stopped after step 4096 because eval-batch loss was clearly worse than the carried setup.
- Eval-batch loss: `0.0510256` at step 2048 and `0.0514763` at step 4096.

### Insights

Rejected. Appearance embeddings worsened target-path eval loss early and were not carried. The implementation remains available behind `appearance_embedding_dim=0` default for future controlled tests, but it is not part of the current Interval Adjustment candidate.

## Current Candidate Summary

Best speed/metric candidate so far:

- `grid_resolution=64`
- `adaptive_warmup_steps=2048`
- `adaptive_max_frequency_level=12`
- `color_num_layers=2`
- Frequency Grid on
- Adaptive Ray Marching on
- Feature Re-weighting off
- FAS off

Best run before the crop-gate correction: `adaptive_fg_arm_iso_h20_grid64_warmup2048_maxfreq12_continue30376_r4096`, checkpoint step 24576, compact eval-batch loss `0.0294421`.

Decision before the crop-gate correction: not accepted for final comparison because visual crops still fail against Instant-NGP, especially stand label, tangled cable, fingers, and hand boundaries. Full final eval was not run because the visual gate did not pass.

## Iteration 23 - Corrected Crop Gate and Larger Ray Batch

### What was tested

Control fix: the crop renderer passed coordinates to `camera.generate_rays()` as `(x, y)`, but nerfstudio expects `(row/y, col/x)`. This made candidate crops sample the wrong rays while GT and baseline crops used the intended PIL rectangle. Fixed the crop renderer to pass `(y, x)` and switched the default baseline render directory to the current bounded Instant-NGP recommendation from `experiments/bounded_ngp_param_sweep.md`.

Hypothesis after the corrected gate: the adaptive candidate was ray-starved versus the bounded baseline because H20 used `train_num_rays_per_batch=4096`, while the current baseline recommendation uses `12288`. Tested a 12288-ray continuation from H20 step 24576 with all isolation flags unchanged.

### Results

Corrected H20 best-eval checkpoint, matched baseline render:

- Checkpoint: `adaptive_fg_arm_iso_h20_grid64_warmup2048_maxfreq12_continue30376_r4096/step-000024576.ckpt`
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h20_grid64_warmup2048_maxfreq12_continue30376_r4096/crop_gate_best24576_stride4_coordsfix_baseline12288s44/all_crops.png`

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 32.1321 | 0.83431 | 21.6884 | 0.87502 | candidate has high PSNR but less structural crack detail |
| fingers_right_eval1 | 22.1366 | 0.80447 | 25.5909 | 0.95728 | smoother hand/finger boundary |
| stand_label_eval2 | 23.9130 | 0.80382 | 32.6024 | 0.96592 | small writing still less readable |
| tangled_cable_eval2 | 22.3733 | 0.76318 | 32.4750 | 0.96757 | wires still weaker |
| fingers_center_eval2 | 22.6784 | 0.77575 | 27.7654 | 0.95332 | boundary remains weaker |

Larger-batch continuation:

- Run: `adaptive_fg_arm_iso_h23_batch12288_continue28672_r12288`
- Continued from: H20 step 24576.
- Runtime: `450.46s`.
- Best eval-batch loss: `0.0313084` at step 26624, worse than H20's `0.0294421`.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h23_batch12288_continue28672_r12288/crop_gate_best26624_stride4_coordsfix_baseline12288s44/all_crops.png`

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 32.3396 | 0.82174 | 21.6884 | 0.87502 | PSNR up, SSIM down |
| fingers_right_eval1 | 23.8352 | 0.82287 | 25.5909 | 0.95728 | improved versus H20 but still worse |
| stand_label_eval2 | 23.2875 | 0.77789 | 32.6024 | 0.96592 | regressed |
| tangled_cable_eval2 | 22.5443 | 0.76645 | 32.4750 | 0.96757 | tiny improvement but still weak |
| fingers_center_eval2 | 22.6020 | 0.76992 | 27.7654 | 0.95332 | regressed |

H20 latest checkpoint diagnostic:

- Checkpoint: H20 step 30375.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h20_grid64_warmup2048_maxfreq12_continue30376_r4096/crop_gate_latest30375_stride4_coordsfix_baseline12288s44/all_crops.png`

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 32.2900 | 0.83682 | 21.6884 | 0.87502 | best floor structure so far but still lower SSIM |
| fingers_right_eval1 | 24.1462 | 0.84266 | 25.5909 | 0.95728 | best adaptive hand crop so far |
| stand_label_eval2 | 23.1118 | 0.77149 | 32.6024 | 0.96592 | worse than best-eval checkpoint |
| tangled_cable_eval2 | 22.3907 | 0.76273 | 32.4750 | 0.96757 | not improved |
| fingers_center_eval2 | 23.3609 | 0.79615 | 27.7654 | 0.95332 | improved but still lower |

### Insights

The corrected crop gate invalidates the extremely low foreground metrics from earlier crop sheets, but it does not change the decision: isolated Frequency Grid + Adaptive RM still fails the high-frequency visual gate against the matched Instant-NGP baseline. Larger train batches are not a clean improvement and are rejected. The H20 latest checkpoint is a better visual checkpoint for hands than the compact-loss best checkpoint, but label and cable detail remain weak.

## Iteration 24 - Scalar Interval Extent Diagnostic

### What was tested

Hypothesis: the paper's normalized interval `dt = 1 / (2 * N_l)` may imply conversion through a scalar scene extent rather than current per-axis AABB-normalized ray speed. Temporarily added `adaptive_interval_extent_mode` and tested render-only scalar extent on the H20 latest checkpoint.

### Results

Render-only crop gate:

- Checkpoint: H20 step 30375.
- Mode: `adaptive_interval_extent_mode=max`.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h20_grid64_warmup2048_maxfreq12_continue30376_r4096/crop_gate_latest30375_stride4_scalar_extent_baseline12288s44/all_crops.png`

| Crop | Candidate PSNR | Candidate SSIM | AABB-mode PSNR | AABB-mode SSIM | Decision |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 32.2897 | 0.83681 | 32.2900 | 0.83682 | no change |
| fingers_right_eval1 | 24.1462 | 0.84266 | 24.1462 | 0.84266 | no change |
| stand_label_eval2 | 23.1118 | 0.77149 | 23.1118 | 0.77149 | no change |
| tangled_cable_eval2 | 22.3907 | 0.76273 | 22.3907 | 0.76273 | no change |
| fingers_center_eval2 | 23.3609 | 0.79615 | 23.3609 | 0.79615 | no change |

### Insights

Rejected as an explanation for the current failure. The scene AABB appears effectively isotropic for the crop rays, so scalar-longest-side interval conversion produces indistinguishable renders. The diagnostic switch was rolled back; the carried implementation remains the per-axis AABB conversion.

## Iteration 25 - Density Activation Parity Diagnostic

### What was tested

Hypothesis: bounded Instant-NGP uses `trunc_exp` density activation, while LookCloser used `softplus(h + 1)`. Temporarily added a `density_activation=trunc_exp` option and tested the same isolated Interval Adjustment setup with `grid_resolution=64`, `adaptive_warmup_steps=2048`, and `adaptive_max_frequency_level=12`.

### Results

Run: `adaptive_fg_arm_iso_h25_truncexp_train4096_r4096`

- Smoke: 16 iterations passed.
- Train budget: 4096 iterations.
- Runtime: `360.37s`.
- Best eval-batch loss: `0.094949` at step 2048; clearly worse than the carried warmup/cap path.
- The 4096 adaptive checkpoint did not beat step 2048, so the runner selected step 2048 and pruned the later checkpoint.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h25_truncexp_train4096_r4096/crop_gate_best2048_stride4_baseline12288s44/all_crops.png`

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 13.0404 | 0.59979 | 21.6884 | 0.87502 | strong artifacting |
| fingers_right_eval1 | 18.3249 | 0.70754 | 25.5909 | 0.95728 | worse body/hand structure |
| stand_label_eval2 | 18.6172 | 0.60828 | 32.6024 | 0.96592 | worse label/stand detail |
| tangled_cable_eval2 | 16.5596 | 0.57703 | 32.4750 | 0.96757 | worse wire/brick detail |
| fingers_center_eval2 | 22.5482 | 0.78677 | 27.7654 | 0.95332 | no useful gain |

### Insights

Rejected and rolled back. `trunc_exp` destabilized early LookCloser training under this setup and did not improve high-frequency crops. Keep `softplus(h + 1)` as the carried density activation.

## Iteration 26 - Geometry MLP Depth 2

### What was tested

Hypothesis: bounded Instant-NGP uses a deeper density/base MLP than LookCloser's current `geo_num_layers=1`, and the shallow geometry network may be blurring hand/cable/label boundaries. Tested only `geo_num_layers=2`, keeping color depth, grid64, warmup2048, maxfreq12, and all isolation flags unchanged.

### Results

Run: `adaptive_fg_arm_iso_h26_geo2_train4096_r4096`

- Train budget: 4096 iterations.
- Runtime: `360.31s`.
- Best eval-batch loss: `0.0477867` at step 2048; final adaptive checkpoint did not improve and was pruned.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h26_geo2_train4096_r4096/crop_gate_best2048_stride4_baseline12288s44/all_crops.png`

| Crop | Candidate PSNR | Candidate SSIM | Instant-NGP PSNR | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 26.6556 | 0.79232 | 21.6884 | 0.87502 | smoother floor than H20 latest |
| fingers_right_eval1 | 20.1504 | 0.76900 | 25.5909 | 0.95728 | worse than H20 latest |
| stand_label_eval2 | 22.5693 | 0.74157 | 32.6024 | 0.96592 | worse label detail |
| tangled_cable_eval2 | 21.1436 | 0.71562 | 32.4750 | 0.96757 | worse cable detail |
| fingers_center_eval2 | 22.8100 | 0.81459 | 27.7654 | 0.95332 | one-crop SSIM improvement only |

### Insights

Rejected. The deeper geometry MLP is not a clean improvement: it gives a small center-hand SSIM gain at an early warmup checkpoint but regresses the other high-frequency crops and does not produce an improved adaptive checkpoint by 4096 steps. Keep `geo_num_layers=1` as the carried isolated Interval Adjustment candidate.

## Corrected Recheck - Previous Max-Frequency-14 Candidate

### What was tested

Because the original crop gate had transposed candidate rays, re-rendered a previous high-frequency-cap run with the corrected crop gate and matched current Instant-NGP baseline.

### Results

Run: `adaptive_fg_arm_iso_h15_warmup2048_maxfreq14_continue15188_r4096`

- Checkpoint: step 15187.
- Config difference from H20: `grid_resolution=128`, `adaptive_max_frequency_level=14`.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h15_warmup2048_maxfreq14_continue15188_r4096/crop_gate_latest15187_stride4_coordsfix_baseline12288s44/all_crops.png`

| Crop | Candidate PSNR | Candidate SSIM | H20 Latest PSNR | H20 Latest SSIM | Decision |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 31.1173 | 0.71694 | 32.2900 | 0.83682 | worse |
| fingers_right_eval1 | 23.2275 | 0.79860 | 24.1462 | 0.84266 | worse |
| stand_label_eval2 | 22.5752 | 0.70850 | 23.1118 | 0.77149 | worse |
| tangled_cable_eval2 | 21.9116 | 0.73709 | 22.3907 | 0.76273 | worse |
| fingers_center_eval2 | 22.1219 | 0.76625 | 23.3609 | 0.79615 | worse |

### Insights

Do not reopen the maxfreq14/grid128 branch. Under the corrected crop gate it is worse than H20 latest on all target crop SSIMs.

## Iteration 27 - Finer Coarse Occupancy Traversal

### What was tested

Hypothesis: thin structures are partly missed before interval subdivision because adaptive traversal first asks nerfacc for coarse occupied intervals using `adaptive_max_step_size=0.1`. Tested only `adaptive_coarse_step_size=0.025`, keeping the actual adaptive interval clamp at `adaptive_max_step_size=0.1`, and carrying H20's grid64/warmup2048/maxfreq12 setup.

### Results

Smoke:

- Run: `adaptive_fg_arm_iso_h27_coarse0025_smoke16`
- Continued from H20 latest step 30375 for 16 iterations.
- Runtime: `25.03s`, no OOM.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h27_coarse0025_smoke16/crop_gate_latest30390_stride4_baseline12288s44/all_crops.png`
- Immediate crop signal versus H20 latest: stand-label SSIM improved from `0.77149` to `0.83242`; center fingers from `0.79615` to `0.81050`; floor from `0.83682` to `0.84193`; cable was nearly flat; right-hand dipped.

Continuation:

- Run: `adaptive_fg_arm_iso_h27_coarse0025_continue32768_r4096`
- Continued from H20 latest step 30375.
- Runtime: `121.32s`.
- Best eval-batch loss: `0.0284642` at step 30720, best adaptive compact loss so far.
- Latest checkpoint: step 31744, eval-batch loss `0.0289727`.
- Best-eval crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h27_coarse0025_continue32768_r4096/crop_gate_best30720_stride4_baseline12288s44/all_crops.png`
- Latest crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h27_coarse0025_continue32768_r4096/crop_gate_latest31744_stride4_baseline12288s44/all_crops.png`

Best-eval step 30720:

| Crop | Candidate PSNR | Candidate SSIM | H20 Latest PSNR | H20 Latest SSIM | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---:|---|
| floor_crack_eval0 | 32.4445 | 0.83736 | 32.2900 | 0.83682 | 0.87502 | slight metric gain, still lower structure SSIM |
| fingers_right_eval1 | 23.7759 | 0.83196 | 24.1462 | 0.84266 | 0.95728 | worse than H20 latest |
| stand_label_eval2 | 24.6749 | 0.83039 | 23.1118 | 0.77149 | 0.96592 | clear gain, still below baseline |
| tangled_cable_eval2 | 22.1847 | 0.76355 | 22.3907 | 0.76273 | 0.96757 | essentially flat |
| fingers_center_eval2 | 23.9698 | 0.80849 | 23.3609 | 0.79615 | 0.95332 | improved |

Latest step 31744:

| Crop | Candidate PSNR | Candidate SSIM | H20 Latest PSNR | H20 Latest SSIM | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---:|---|
| floor_crack_eval0 | 32.2019 | 0.82701 | 32.2900 | 0.83682 | 0.87502 | regressed versus H20 |
| fingers_right_eval1 | 24.0280 | 0.83637 | 24.1462 | 0.84266 | 0.95728 | slightly worse than H20 |
| stand_label_eval2 | 24.7840 | 0.83355 | 23.1118 | 0.77149 | 0.96592 | best adaptive label crop so far |
| tangled_cable_eval2 | 22.2311 | 0.76454 | 22.3907 | 0.76273 | 0.96757 | tiny SSIM gain |
| fingers_center_eval2 | 23.9332 | 0.80807 | 23.3609 | 0.79615 | 0.95332 | improved |

### Insights

Accepted as the new carried adaptive Interval Adjustment candidate. Reducing only coarse occupancy traversal from `0.1` to `0.025` improved compact eval loss and corrected-gate visuals on stand label and center fingers without changing Feature Re-weighting or FAS. It still does not satisfy the final visual gate: cable and hand boundaries remain well below Instant-NGP, and full final eval remains deferred.

## Iteration 28 - Coarse Traversal 0.0125

### What was tested

Hypothesis: since `adaptive_coarse_step_size=0.025` improved compact loss and some high-frequency crops, halving traversal again to `0.0125` might recover more cable/finger detail. Tested only `adaptive_coarse_step_size=0.0125`, starting from H27 latest step 31744.

### Results

Smoke:

- Run: `adaptive_fg_arm_iso_h28_coarse00125_smoke16`
- Continued from H27 latest step 31744 for 16 iterations.
- Runtime: `25.03s`.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h28_coarse00125_smoke16/crop_gate_latest31759_stride4_baseline12288s44/all_crops.png`
- Mixed signal versus H27 latest: cable, center fingers, and right hand improved slightly; stand label regressed.

Continuation:

- Run: `adaptive_fg_arm_iso_h28_coarse00125_continue33792_r4096`
- Continued from H27 latest step 31744.
- Runtime: `150.16s`.
- Best eval-batch loss: `0.0282583` at step 32768, best adaptive compact loss so far.
- Latest checkpoint: step 33791.
- Best-eval crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h28_coarse00125_continue33792_r4096/crop_gate_best32768_stride4_baseline12288s44/all_crops.png`
- Latest crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h28_coarse00125_continue33792_r4096/crop_gate_latest33791_stride4_baseline12288s44/all_crops.png`

Best-eval step 32768:

| Crop | Candidate PSNR | Candidate SSIM | H27 Latest PSNR | H27 Latest SSIM | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---:|---|
| floor_crack_eval0 | 32.3890 | 0.82588 | 32.2019 | 0.82701 | 0.87502 | similar, slightly lower SSIM |
| fingers_right_eval1 | 24.0714 | 0.83629 | 24.0280 | 0.83637 | 0.95728 | flat |
| stand_label_eval2 | 24.4862 | 0.83023 | 24.7840 | 0.83355 | 0.96592 | slightly worse |
| tangled_cable_eval2 | 22.2435 | 0.76551 | 22.2311 | 0.76454 | 0.96757 | tiny gain |
| fingers_center_eval2 | 23.8782 | 0.80934 | 23.9332 | 0.80807 | 0.95332 | tiny gain |

Latest step 33791:

| Crop | Candidate PSNR | Candidate SSIM | H27 Latest PSNR | H27 Latest SSIM | Instant-NGP SSIM | Visual |
|---|---:|---:|---:|---:|---:|---|
| floor_crack_eval0 | 32.1314 | 0.82694 | 32.2019 | 0.82701 | 0.87502 | flat |
| fingers_right_eval1 | 23.9472 | 0.83321 | 24.0280 | 0.83637 | 0.95728 | worse |
| stand_label_eval2 | 24.4763 | 0.83121 | 24.7840 | 0.83355 | 0.96592 | worse |
| tangled_cable_eval2 | 22.2356 | 0.76611 | 22.2311 | 0.76454 | 0.96757 | best adaptive cable SSIM so far, but tiny gain |
| fingers_center_eval2 | 23.8915 | 0.81000 | 23.9332 | 0.80807 | 0.95332 | small gain |

### Insights

Carry `0.0125` only as a metric/cable-center tradeoff, not as a clean visual replacement for H27. It improves compact eval loss and gives the best cable/center-finger SSIM so far, but it slightly regresses the stand label and right-hand crops versus H27 latest. The high-frequency visual gate is still not passed.

## Iteration 29 - Coarse Traversal 0.00625 Smoke

### What was tested

Hypothesis: one more halving of coarse traversal might further improve cable/finger detail. Tested a 16-iteration smoke at `adaptive_coarse_step_size=0.00625`, starting from H28 latest step 33791.

### Results

Run: `adaptive_fg_arm_iso_h29_coarse000625_smoke16`

- Runtime: `20.02s`.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h29_coarse000625_smoke16/crop_gate_latest33806_stride4_baseline12288s44/all_crops.png`

| Crop | Candidate PSNR | Candidate SSIM | H28 Latest PSNR | H28 Latest SSIM | Decision |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 32.3872 | 0.82858 | 32.1314 | 0.82694 | tiny gain |
| fingers_right_eval1 | 23.9299 | 0.83340 | 23.9472 | 0.83321 | flat |
| stand_label_eval2 | 24.4730 | 0.83058 | 24.4763 | 0.83121 | worse |
| tangled_cable_eval2 | 22.2403 | 0.76611 | 22.2356 | 0.76611 | flat |
| fingers_center_eval2 | 23.9239 | 0.81122 | 23.8915 | 0.81000 | tiny gain |

### Insights

Rejected for now. The 0.00625 smoke does not provide enough visual improvement over H28 to justify a longer run; it is mostly flat and slightly worse on the stand label. Keep the useful coarse traversal bracket at `0.025` for label/right-hand and `0.0125` for compact loss/cable-center.

## Current Candidate Summary

Current best visual adaptive checkpoint under the corrected crop gate:

- Runs: `adaptive_fg_arm_iso_h27_coarse0025_continue32768_r4096` and `adaptive_fg_arm_iso_h28_coarse00125_continue33792_r4096`.
- H27 latest step 31744 has the strongest stand-label/right-hand balance.
- H28 best step 32768 has the best compact eval loss (`0.0282583`) and H28 latest step 33791 has the best cable/center-finger SSIM so far.
- Shared config: `grid_resolution=64`, `adaptive_warmup_steps=2048`, `adaptive_max_frequency_level=12`, Frequency Grid on, Adaptive RM on, Feature Re-weighting off, FAS off. H27 uses `adaptive_coarse_step_size=0.025`; H28 uses `0.0125`.

Decision: improved and carried as a bracket, but not accepted for final comparison. Both still lose the matched Instant-NGP baseline on SSIM and visually on tangled cable, hand/finger boundaries, and remaining stand-label detail. Full final eval remains deferred until the crop gate improves further.

## Iteration 30 - Disable Alpha Pruning Smoke

### What was tested

Hypothesis: thin structures may be pruned too early by nerfacc visibility filtering before Interval Adjustment can add fine samples. Tested `alpha_thre=0.0` for a 16-iteration smoke from H28 latest step 33791, keeping `adaptive_coarse_step_size=0.0125` and all isolation flags unchanged.

### Results

Run: `adaptive_fg_arm_iso_h30_alpha0_coarse00125_smoke16`

- Runtime: `20.03s`.
- Latest checkpoint: step 33806.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h30_alpha0_coarse00125_smoke16/crop_gate_latest33806_stride4_baseline12288s44/all_crops.png`
- Sample cost increased: H30 logged `train_adaptive_samples_mean=228.035`, `train_adaptive_samples_max=1024`, `train_adaptive_saturation_rate=0.00170898`, versus H28 latest around `127` mean samples/ray and no saturation.

| Crop | Candidate PSNR | Candidate SSIM | H28 Latest PSNR | H28 Latest SSIM | Decision |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 32.2395 | 0.82736 | 32.1314 | 0.82694 | flat |
| fingers_right_eval1 | 23.9494 | 0.83306 | 23.9472 | 0.83321 | flat/slightly worse |
| stand_label_eval2 | 24.4744 | 0.83077 | 24.4763 | 0.83121 | slightly worse |
| tangled_cable_eval2 | 22.2356 | 0.76599 | 22.2356 | 0.76611 | flat/slightly worse |
| fingers_center_eval2 | 23.9070 | 0.81072 | 23.8915 | 0.81000 | tiny gain |

### Insights

Rejected. Removing alpha pruning roughly doubled sample count but did not visually recover the cable, label, or hand-boundary detail. Keep `alpha_thre=0.0025` and do not promote this direction without a stronger separate reason.

## Iteration 31 - Minimum Frequency Level 4 Smoke

### What was tested

Hypothesis: low-frequency occupied cells may be using intervals that are too large, producing smeared geometry before high-frequency cells can help. Tested `adaptive_min_frequency_level=4.0` for a 16-iteration smoke from H28 latest step 33791, keeping `adaptive_coarse_step_size=0.0125`, `adaptive_max_frequency_level=12`, and `alpha_thre=0.0025`.

### Results

Run: `adaptive_fg_arm_iso_h31_minfreq4_coarse00125_smoke16`

- Runtime: `20.12s`.
- Latest checkpoint: step 33806.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h31_minfreq4_coarse00125_smoke16/crop_gate_latest33806_stride4_baseline12288s44/all_crops.png`

| Crop | Candidate PSNR | Candidate SSIM | H28 Latest PSNR | H28 Latest SSIM | Decision |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 32.2880 | 0.82764 | 32.1314 | 0.82694 | tiny gain |
| fingers_right_eval1 | 23.9556 | 0.83314 | 23.9472 | 0.83321 | flat |
| stand_label_eval2 | 24.4893 | 0.83106 | 24.4763 | 0.83121 | flat |
| tangled_cable_eval2 | 22.2431 | 0.76614 | 22.2356 | 0.76611 | flat |
| fingers_center_eval2 | 23.9185 | 0.81085 | 23.8915 | 0.81000 | tiny gain |

### Insights

Rejected. The numeric changes are noise-level and the crop sheet shows the same visual failure mode as H28: cable and label detail are not recovered, and hand edges remain artifacted. Do not promote a minimum-frequency floor unless a later run shows a clear visual benefit.

## Iteration 32 - Corrected Grid-128 Recheck

### What was tested

Hypothesis: earlier grid-128 conclusions may have been distorted by the crop-coordinate bug. Rerendered the existing H18 `grid_resolution=128`, `adaptive_max_frequency_level=12` best-loss checkpoint with the corrected crop script and current matched Instant-NGP baseline path. No retraining or code changes.

### Results

Run: `adaptive_fg_arm_iso_h18_warmup2048_maxfreq12_continue30376_r4096`

- Checkpoint: step 24576.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h18_warmup2048_maxfreq12_continue30376_r4096/crop_gate_corrected_best24576_stride4_baseline12288s44/all_crops.png`

| Crop | Grid-128 PSNR | Grid-128 SSIM | H28 Latest PSNR | H28 Latest SSIM | Decision |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 32.7476 | 0.83195 | 32.1314 | 0.82694 | better floor |
| fingers_right_eval1 | 22.8879 | 0.78957 | 23.9472 | 0.83321 | worse |
| stand_label_eval2 | 22.8542 | 0.76183 | 24.4763 | 0.83121 | worse |
| tangled_cable_eval2 | 21.7633 | 0.74062 | 22.2356 | 0.76611 | worse |
| fingers_center_eval2 | 22.3033 | 0.77246 | 23.8915 | 0.81000 | worse |

### Insights

Rejected. The corrected crop gate confirms grid 128 is not the missing piece for the target foreground details. It improves the floor crop but loses substantially on hands, stand label, and cable versus the grid-64 H27/H28 bracket. Keep `grid_resolution=64`.

## Iteration 33 - Full-Resolution Crop Audit And Level-13 Cap

### What was tested

First, rerendered the H27/H28 bracket at stride 1 instead of stride 4 to avoid over-weighting downsample/resize artifacts in the visual decision. Then tested a one-level cap increase, `adaptive_max_frequency_level=13`, from H28 latest step 33791 with `grid_resolution=64` and `adaptive_coarse_step_size=0.0125`.

### Results

Full-resolution crop audit:

| Run | Step | floor SSIM | right-hand SSIM | stand SSIM | cable SSIM | center-hand SSIM | Notes |
|---|---:|---:|---:|---:|---:|---|
| Instant-NGP baseline | stage4 seed44 | 0.58658 | 0.77181 | 0.80566 | 0.77271 | 0.78641 | baseline render crop |
| H27 latest | 31744 | 0.62475 | 0.77616 | 0.78439 | 0.76305 | 0.79223 | wins floor/hands, loses label/cable |
| H28 best | 32768 | 0.62583 | 0.77789 | 0.78525 | 0.76342 | 0.79410 | best floor/right at stride 1 |
| H28 latest | 33791 | 0.62545 | 0.77472 | 0.78612 | 0.76466 | 0.79486 | best H28 label/cable/center |

The full-resolution crop sheets show a more favorable partial result than the stride-4 screening gates: the adaptive candidate is visually sharper than Instant-NGP on the floor crack and parts of the hand crops, but it still trails on the stand label and tangled cable.

Level-13 smoke:

- Run: `adaptive_fg_arm_iso_h33_maxfreq13_coarse00125_smoke16`
- Runtime: `20.03s`.
- Latest checkpoint: step 33806.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h33_maxfreq13_coarse00125_smoke16/crop_gate_latest33806_stride1_baseline12288s44/all_crops.png`
- Sample cost: `train_adaptive_samples_mean=196.267`, `max=1024`, `saturation_rate=0.000244141`, versus H28 latest around `127` mean samples/ray with no saturation.

| Crop | H33 PSNR | H33 SSIM | H28 Latest PSNR | H28 Latest SSIM | Baseline SSIM | Decision |
|---|---:|---:|---:|---:|---:|---|
| floor_crack_eval0 | 29.7088 | 0.62618 | 29.5951 | 0.62545 | 0.58658 | better |
| fingers_right_eval1 | 26.5588 | 0.77499 | 26.5493 | 0.77472 | 0.77181 | tiny gain |
| stand_label_eval2 | 26.9135 | 0.78620 | 26.9072 | 0.78612 | 0.80566 | tiny gain, still below baseline |
| tangled_cable_eval2 | 27.2061 | 0.76469 | 27.2150 | 0.76466 | 0.77271 | flat/tiny gain |
| fingers_center_eval2 | 26.8379 | 0.79500 | 26.8422 | 0.79486 | 0.78641 | tiny gain |

### Insights

Promoted only to a short continuation, not accepted as final. Level 13 is more expensive but still practical, and it nudges the weak label/cable crops in the right direction without losing the floor/hand wins. It needs a bounded continuation before deciding whether the extra high-frequency intervals are real or noise.

Continuation:

- Run: `adaptive_fg_arm_iso_h33_maxfreq13_coarse00125_continue35840_r4096`
- Continued from H33 smoke step 33806.
- Runtime: `160.19s`.
- Best eval-batch loss: `0.0263509` at step 34816, best compact adaptive loss so far.
- Latest checkpoint: step 35839.
- Best crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h33_maxfreq13_coarse00125_continue35840_r4096/crop_gate_best34816_stride1_baseline12288s44/all_crops.png`
- Latest crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h33_maxfreq13_coarse00125_continue35840_r4096/crop_gate_latest35839_stride1_baseline12288s44/all_crops.png`

| Crop | H33 Best SSIM | H33 Latest SSIM | H33 Smoke SSIM | H28 Latest SSIM | Baseline SSIM | Decision |
|---|---:|---:|---:|---:|---:|---|
| floor_crack_eval0 | 0.62573 | 0.62938 | 0.62618 | 0.62545 | 0.58658 | latest improves floor |
| fingers_right_eval1 | 0.77675 | 0.77585 | 0.77499 | 0.77472 | 0.77181 | small win |
| stand_label_eval2 | 0.78480 | 0.78491 | 0.78620 | 0.78612 | 0.80566 | worse than smoke/H28 |
| tangled_cable_eval2 | 0.76434 | 0.76466 | 0.76469 | 0.76466 | 0.77271 | flat |
| fingers_center_eval2 | 0.79387 | 0.79472 | 0.79500 | 0.79486 | 0.78641 | flat/slightly worse |

Continuation decision: not promoted as the visual candidate. The compact eval loss improved substantially, but the two blocking crops did not improve. Keep H33 smoke as a useful diagnostic and keep H27/H28 as the visual bracket.

## Iteration 34 - Decoupled Occupancy Grid Resolution

### What was tested

Hypothesis: tying nerfacc occupancy-grid resolution to the chosen `grid_resolution=64` may make empty-space pruning too coarse for wires and labels. Temporarily added an `occupancy_grid_resolution` model/runner option and a checkpoint-load filter for stale occupancy buffers, then tested frequency grid 64 with occupancy grid 128.

### Results

The 16-step smoke loaded and trained, but rendered black because the newly initialized occupancy grid had not rebuilt:

- Run: `adaptive_fg_arm_iso_h34_occ128_coarse00125_smoke16`
- Latest checkpoint: step 33806.
- Mean train samples at logged step: `0.394531/ray`.
- Crop pixel standard deviation was near zero; rejected as an invalid visual gate.

Reran a 512-step rebuild from H28 latest:

- Run: `adaptive_fg_arm_iso_h34_occ128_coarse00125_rebuild512`
- Runtime: `60.06s`.
- Latest checkpoint: step 34319.
- Sample count recovered by the end to `192.11/ray`, with `0.000976562` saturation.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h34_occ128_coarse00125_rebuild512/crop_gate_latest34319_stride1_baseline12288s44/all_crops.png`

| Crop | H34 SSIM | H28 Latest SSIM | Baseline SSIM | Decision |
|---|---:|---:|---:|---|
| floor_crack_eval0 | 0.32714 | 0.62545 | 0.58658 | much worse |
| fingers_right_eval1 | 0.00813 | 0.77472 | 0.77181 | failed |
| stand_label_eval2 | 0.40475 | 0.78612 | 0.80566 | much worse |
| tangled_cable_eval2 | 0.40152 | 0.76466 | 0.77271 | much worse |
| fingers_center_eval2 | 0.43600 | 0.79486 | 0.78641 | much worse |

### Insights

Rejected and rolled back. Decoupling occupancy resolution requires a nontrivial warm-start/rebuild strategy and did not improve visual crops in the tested form. The temporary model/runner code was removed after the failed hypothesis.

## Iteration 35 - Disable Distortion Loss

### What was tested

Hypothesis: the packed distortion regularizer may over-smooth thin structures and labels compared with the bounded Instant-NGP baseline. Tested `distortion_loss_mult=0.0` as a 1024-step continuation from H28 latest, keeping `grid_resolution=64`, `adaptive_coarse_step_size=0.0125`, and `adaptive_max_frequency_level=12`.

### Results

Run: `adaptive_fg_arm_iso_h35_dist0_coarse00125_continue34816_r4096`

- Runtime: `90.10s`.
- Eval-batch loss at step 33792: `0.0298357`, worse than H28's best `0.0282583` and H33's best `0.0263509`.
- Latest checkpoint: step 34815.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h35_dist0_coarse00125_continue34816_r4096/crop_gate_latest34815_stride1_baseline12288s44/all_crops.png`

| Crop | H35 Latest SSIM | H28 Latest SSIM | Baseline SSIM | Decision |
|---|---:|---:|---:|---|
| floor_crack_eval0 | 0.62524 | 0.62545 | 0.58658 | flat/slightly worse |
| fingers_right_eval1 | 0.77671 | 0.77472 | 0.77181 | small gain |
| stand_label_eval2 | 0.78614 | 0.78612 | 0.80566 | flat, still below baseline |
| tangled_cable_eval2 | 0.76460 | 0.76466 | 0.77271 | slightly worse |
| fingers_center_eval2 | 0.79357 | 0.79486 | 0.78641 | worse |

### Insights

Rejected. Removing distortion loss does not improve the two blocking high-frequency crops and worsens compact eval loss. Keep `distortion_loss_mult=0.01`.

## Iteration 36 - Max Steps 2048 With Level-13 Cap

### What was tested

Hypothesis: rare rays capped at `max_steps_per_ray=1024` in H33 may correspond to thin high-frequency structures. Tested H33's `adaptive_max_frequency_level=13` smoke again with `max_steps_per_ray=2048`.

### Results

Run: `adaptive_fg_arm_iso_h36_maxfreq13_maxsteps2048_smoke16`

- Runtime: `20.02s`.
- Mean train samples: `196.288/ray`, max `1111`, saturation `0`.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h36_maxfreq13_maxsteps2048_smoke16/crop_gate_latest33806_stride1_baseline12288s44/all_crops.png`

| Crop | H36 SSIM | H33 Smoke SSIM | Decision |
|---|---:|---:|---|
| floor_crack_eval0 | 0.62616 | 0.62618 | flat |
| fingers_right_eval1 | 0.77500 | 0.77499 | flat |
| stand_label_eval2 | 0.78622 | 0.78620 | noise-level gain |
| tangled_cable_eval2 | 0.76470 | 0.76469 | noise-level gain |
| fingers_center_eval2 | 0.79499 | 0.79500 | flat |

### Insights

Rejected as a meaningful change. The rare capped rays are not the limiting factor for the visible label/cable failure. Keep `max_steps_per_ray=1024` unless a later higher-frequency branch needs the headroom.

## Iteration 37 - Level-14 Cap With Max Steps 2048

### What was tested

Hypothesis: the current grid-64/coarse-0.0125 setup may tolerate `adaptive_max_frequency_level=14` better than the earlier grid-128 branch, and the extra frequency level may help label/cable detail. Tested a 16-step smoke with `adaptive_max_frequency_level=14` and `max_steps_per_ray=2048`.

### Results

Run: `adaptive_fg_arm_iso_h37_maxfreq14_maxsteps2048_smoke16`

- Runtime: `20.03s`.
- Mean train samples: `289.555/ray`, max `1639`, saturation `0`.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h37_maxfreq14_maxsteps2048_smoke16/crop_gate_latest33806_stride1_baseline12288s44/all_crops.png`

| Crop | H37 SSIM | H36 SSIM | Baseline SSIM | Decision |
|---|---:|---:|---:|---|
| floor_crack_eval0 | 0.62637 | 0.62616 | 0.58658 | tiny gain |
| fingers_right_eval1 | 0.77500 | 0.77500 | 0.77181 | flat |
| stand_label_eval2 | 0.78624 | 0.78622 | 0.80566 | noise-level gain, still below baseline |
| tangled_cable_eval2 | 0.76470 | 0.76470 | 0.77271 | flat |
| fingers_center_eval2 | 0.79499 | 0.79499 | 0.78641 | flat |

### Insights

Rejected for continuation. Level 14 increases sample count substantially but only moves crop metrics by noise-level amounts and still does not close the stand-label/cable gap.

## Iteration 38 - MSE Reconstruction Loss

### What was tested

Hypothesis: Charbonnier loss may downweight high-frequency residuals that MSE would keep pushing on. Tested `reconstruction_loss_type=mse` as a 1024-step continuation from H28 latest, with sampler settings unchanged.

### Results

Run: `adaptive_fg_arm_iso_h38_mse_coarse00125_continue34816_r4096`

- Runtime: `90.08s`.
- Eval-batch loss is on the MSE scale and not directly comparable to Charbonnier runs.
- Latest checkpoint: step 34815.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h38_mse_coarse00125_continue34816_r4096/crop_gate_latest34815_stride1_baseline12288s44/all_crops.png`

| Crop | H38 Latest SSIM | H28 Latest SSIM | Baseline SSIM | Decision |
|---|---:|---:|---:|---|
| floor_crack_eval0 | 0.62548 | 0.62545 | 0.58658 | flat |
| fingers_right_eval1 | 0.77705 | 0.77472 | 0.77181 | small gain |
| stand_label_eval2 | 0.78449 | 0.78612 | 0.80566 | worse |
| tangled_cable_eval2 | 0.76448 | 0.76466 | 0.77271 | worse |
| fingers_center_eval2 | 0.79346 | 0.79486 | 0.78641 | worse |

### Insights

Rejected. MSE does not improve the blocking high-frequency crops and regresses center-hand/stand detail. Keep Charbonnier.

## Full Eval - H33 Best Metric Candidate

### What was tested

Ran full `ns-eval` on the strongest compact-loss candidate after visual crop inspection. This is not a final visual acceptance because stand-label and cable crops still trail the matched Instant-NGP baseline locally.

### Results

Run: `adaptive_fg_arm_iso_h33_maxfreq13_coarse00125_continue35840_r4096`

- Checkpoint: step 34816.
- Eval JSON: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h33_maxfreq13_coarse00125_continue35840_r4096/eval_full_step-000034816.json`
- Renders: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h33_maxfreq13_coarse00125_continue35840_r4096/renders_full_step-000034816`

| Model | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Bounded Instant-NGP baseline | 24.83 | 0.63 | 0.46 |
| H33 step 34816 | 28.8834 | 0.6661 | 0.3654 |

### Insights

Accepted as a global metric improvement and speed proof for Interval Adjustment, but not as final visual completion. Full renders are coherent and the crop audit shows wins on floor crack and hand crops, while stand label and tangled cable still lag locally.

## Iteration 39 - Level-14 Cap With Finer Coarse Traversal

### What was tested

Hypothesis: H37's level-14 cap might need finer coarse occupancy traversal to catch thin intervals. Tested `adaptive_coarse_step_size=0.00625` with `adaptive_max_frequency_level=14` and `max_steps_per_ray=2048`.

### Results

Run: `adaptive_fg_arm_iso_h39_maxfreq14_maxsteps2048_coarse000625_smoke16`

- Runtime: `20.03s`.
- Mean train samples: `260.160/ray`, max `1510`, saturation `0`.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h39_maxfreq14_maxsteps2048_coarse000625_smoke16/crop_gate_latest33806_stride1_baseline12288s44/all_crops.png`

| Crop | H39 SSIM | H37 SSIM | Baseline SSIM | Decision |
|---|---:|---:|---:|---|
| floor_crack_eval0 | 0.62609 | 0.62637 | 0.58658 | slightly worse |
| fingers_right_eval1 | 0.77523 | 0.77500 | 0.77181 | tiny gain |
| stand_label_eval2 | 0.78608 | 0.78624 | 0.80566 | worse |
| tangled_cable_eval2 | 0.76456 | 0.76470 | 0.77271 | worse |
| fingers_center_eval2 | 0.79523 | 0.79499 | 0.78641 | tiny gain |

### Insights

Rejected. Finer coarse traversal with level 14 does not improve the blocking label/cable crops and costs more samples than H28/H33.

## Full Eval - H37 High-Frequency Smoke

### What was tested

Ran full `ns-eval` on H37 because it had the best local stand-label/cable crop numbers, even though the gains were tiny.

### Results

Run: `adaptive_fg_arm_iso_h37_maxfreq14_maxsteps2048_smoke16`

- Checkpoint: step 33806.
- Eval JSON: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h37_maxfreq14_maxsteps2048_smoke16/eval_full_step-000033806.json`
- Renders: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h37_maxfreq14_maxsteps2048_smoke16/renders_full_step-000033806`

| Model | PSNR | SSIM | LPIPS | Rays/sec |
|---|---:|---:|---:|---:|
| H33 step 34816 | 28.8834 | 0.6661 | 0.3654 | 173378.8 |
| H37 step 33806 | 28.8536 | 0.6661 | 0.3714 | 143155.6 |

### Insights

Rejected as the primary full-eval candidate. H37 keeps the global metric win over baseline but is slightly worse and slower than H33. Keep H33 step 34816 as the best global metric checkpoint, while continuing to search for label/cable visual gains.

## Iteration 40 - Longer Level-12 Continuation

### What was tested

Hypothesis: the original H28 level-12 branch was still improving label/cable with training time, even when compact eval loss was not the best visual selector. Continued H28 latest to step 36863 with the same sampler settings.

### Results

Run: `adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_continue36864_r4096`

- Runtime: `220.25s`.
- Best eval-batch loss: `0.0257004` at step 34816, best compact eval loss so far.
- Full eval at step 34816:

| Model | PSNR | SSIM | LPIPS | Rays/sec |
|---|---:|---:|---:|---:|
| Bounded Instant-NGP baseline | 24.83 | 0.63 | 0.46 | n/a |
| H33 step 34816 | 28.8834 | 0.6661 | 0.3654 | 173378.8 |
| H40 step 34816 | 28.8982 | 0.6659 | 0.3653 | 197993.2 |

Crop gates:

- Best: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_continue36864_r4096/crop_gate_best34816_stride1_baseline12288s44/all_crops.png`
- Latest: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_continue36864_r4096/crop_gate_latest36863_stride1_baseline12288s44/all_crops.png`

| Crop | H40 Best SSIM | H40 Latest SSIM | H28 Latest SSIM | Baseline SSIM | Decision |
|---|---:|---:|---:|---:|---|
| floor_crack_eval0 | 0.62530 | 0.62472 | 0.62545 | 0.58658 | flat/slightly worse |
| fingers_right_eval1 | 0.77666 | 0.77377 | 0.77472 | 0.77181 | best is small gain |
| stand_label_eval2 | 0.78601 | 0.78619 | 0.78612 | 0.80566 | flat |
| tangled_cable_eval2 | 0.76433 | 0.75176 | 0.76466 | 0.77271 | worse |
| fingers_center_eval2 | 0.79356 | 0.79423 | 0.79486 | 0.78641 | worse |

### Insights

Accepted as the current metric leader but rejected as a visual replacement for H28/H37. It improves PSNR/LPIPS over H33 and the bounded baseline, but SSIM is slightly below H33 and the cable crop regresses.

## Iteration 41 - Larger Batch Visual Continuation

### What was tested

Hypothesis: with FAS disabled, rare high-frequency pixels may need a larger uniform ray batch to keep appearing during continuation. Tested `train_num_rays_per_batch=8192` for 1024 steps from H28 latest, keeping Interval Adjustment settings unchanged.

### Results

Run: `adaptive_fg_arm_iso_h41_batch8192_coarse00125_continue34816_r8192`

- Runtime: `100.10s`.
- Eval-batch loss at step 33792: `0.0270023`; selected checkpoint by loss was effectively the start, but latest checkpoint was visually inspected.
- Latest checkpoint: step 34815.
- Crop gate: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h41_batch8192_coarse00125_continue34816_r8192/crop_gate_latest34815_stride1_baseline12288s44/all_crops.png`
- Full eval JSON: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h41_batch8192_coarse00125_continue34816_r8192/eval_full_step-000034815.json`
- Full renders: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h41_batch8192_coarse00125_continue34816_r8192/renders_full_step-000034815`

Full eval:

| Model | PSNR | SSIM | LPIPS | Rays/sec |
|---|---:|---:|---:|---:|
| Bounded Instant-NGP baseline | 24.83 | 0.63 | 0.46 | n/a |
| H40 metric leader | 28.8982 | 0.6659 | 0.3653 | 197993.2 |
| H41 visual candidate | 28.8879 | 0.6660 | 0.3664 | 199065.3 |

Crop gate:

| Crop | H41 SSIM | Best Prior Isolated SSIM | Baseline SSIM | Decision |
|---|---:|---:|---:|---|
| floor_crack_eval0 | 0.62513 | 0.62938 H33 latest | 0.58658 | beats baseline, not best isolated |
| fingers_right_eval1 | 0.77688 | 0.77789 H28 best | 0.77181 | beats baseline |
| stand_label_eval2 | 0.78633 | 0.78624 H37 | 0.80566 | best isolated, still below baseline |
| tangled_cable_eval2 | 0.76476 | 0.76470 H37 | 0.77271 | best isolated, still below baseline |
| fingers_center_eval2 | 0.79534 | 0.79523 H39 | 0.78641 | best isolated and beats baseline |

### Insights

Accepted as the best visual-balance candidate under the isolated settings. It preserves the global metric win, improves floor/hand/center high-frequency crops over the bounded baseline, and gives the best isolated stand-label/cable scores so far. However, stand-label and tangled-cable crops still do not beat the bounded Instant-NGP baseline, so this is a partial visual success rather than a complete win on every target patch.

## Current Best Candidates

- Best global metric checkpoint: `adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_continue36864_r4096`, step 34816. Metrics: PSNR `28.8982`, SSIM `0.6659`, LPIPS `0.3653`.
- Best visual-balance checkpoint: `adaptive_fg_arm_iso_h41_batch8192_coarse00125_continue34816_r8192`, step 34815. Metrics: PSNR `28.8879`, SSIM `0.6660`, LPIPS `0.3664`.
- Both beat the provided bounded Instant-NGP metrics: PSNR `24.83`, SSIM `0.63`, LPIPS `0.46`.
- Visual status: floor crack, hand boundaries, and center-hand crop improve over baseline; stand label and tangled cable are best among isolated adaptive runs but still below the bounded baseline.
