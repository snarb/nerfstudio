# LookCloser FAS Stand Artifact Debug

## What Was Debugged

Target artifact: with FAS enabled, `eval_img_0000.png` renders the vertical metal stand as broken / discontinuous around the connector area. The no-FAS baseline has weaker high-frequency detail and a wrist artifact, but the stand is less broken.

Visual gate used before waiting for long runs:

- Eval view: `eval_img_0000`.
- Crop name: `left_stand_connector_eval0`.
- Full-resolution crop: `xyxy=(320, 0, 617, 530)`.
- Early gate crop render: downsampled target crop from the first eval/checkpoint boundary. If the same stand artifact is visible, the run is rejected before continuing.

Primary no-FAS baseline reference:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_continue36864_r4096/renders_full_step-000034816`

FAS metric leader that remains visually rejected:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_mix035_w2048_r4096_seed43/renders_best_step-000034816`

## FAS Checks Run

Current best no-FAS target crop metric:

- PSNR: `26.5161`
- SSIM: `0.8709`

FAS candidates rejected by the target crop gate:

| Configuration | Best target crop PSNR | Best target crop SSIM | Visual status |
|---|---:|---:|---|
| `fas_strength=0.35`, warmup/ramp `2048/4096`, `fas_level_count_alpha=0.5` | `26.1009` | `0.8117` | rejected, stand still broken |
| same + `fas_patch_group_size=4` | `25.8693` | `0.8024` | rejected, blurrier / still broken |
| `fas_strength=0.25`, warmup/ramp `2048/4096` | `25.7730` | `0.8068` | rejected, still broken |
| flatter ramp `1.0 -> 1.5` | `25.9709` | `0.8090` | rejected, still broken |
| cap FAS sampling level to `12` | `25.9357` | `0.8099` | rejected, still broken |
| no warmup/ramp + cap `12` | `26.0458` | `0.8088` | rejected, still broken |

Useful target crop outputs:

- `/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/problem_crop_compare/problem_crop_compare.png`
- `/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/alpha05_seed43_step3072_target_crop/left_stand_connector_eval0.png`
- `/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/alpha05_group4_seed42_step3071_target_crop/left_stand_connector_eval0.png`
- `/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/mix025_seed42_step3071_target_crop/left_stand_connector_eval0.png`
- `/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/ramp15_seed44_step3071_target_crop/left_stand_connector_eval0.png`
- `/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/cap12_seed43_step3071_target_crop/left_stand_connector_eval0.png`
- `/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/nowarm_cap12_seed42_step3071_target_crop/left_stand_connector_eval0.png`

Code/debug controls added for this investigation:

- `LookCloserPixelSamplerConfig.fas_level_count_alpha`
- `LookCloserPixelSamplerConfig.fas_patch_group_size`
- `LookCloserPixelSamplerConfig.fas_max_sampling_level`
- crop gate support for `left_stand_connector_eval0` in `render_lookcloser_crop_gate.py`
- edge-F1 and named crop outputs in `find_render_artifact_outliers.py`

Checkpoint commit for current FAS debug state:

`5eb3716d debug: add FAS stand artifact gates and controls`

## Frequency Map / Preprocessing Check

Important limitation: there is no direct frequency map for `eval_img_0000`, because FAS maps are generated for train images only. The eval artifact must therefore be debugged through train frames where the same stand / connector appears.

The `.pt` files store scalar resolution values, not already-discrete levels. Correct conversion is:

`level = round(log(freq_map / min_res) / log(per_level_scale))`, clamped to `0..15`.

The target eval crop maps to patch coordinates:

- image crop: `xyxy=(320, 0, 617, 530)`
- frequency map crop: `yx=(0:67, 40:78)` with patch size `8`

Generated audit sheet:

`/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/frequency_map_stand_audit/train_eval0_stand_crop_frequency_level_overlay_sheet.jpg`

Generated stats CSV:

`/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/frequency_map_stand_audit/train_eval0_stand_crop_frequency_level_stats.csv`

Closest useful inspected train frames:

| Train frame | Why useful | `>=13` crop fraction | Mean crop level | Finding |
|---|---|---:|---:|---|
| `frame_train_00029` | vertical stand + connector resembles the eval artifact area | `0.5837` | `12.816` | no obvious missing-map hole on the pole, but levels are noisy and not object-continuous |
| `frame_train_00062` | clean vertical pole + ladder/metal structure | `0.6386` | `12.722` | stand is present, but map alternates high/mid patches along metal |
| `frame_train_00047` | labels, wires, metal stand, wall | `0.5656` | `12.690` | high-level mask is broad over wall and metal, weak object selectivity |
| `frame_train_00056` | dense metal structure / wall region | `0.5511` | `11.962` | mixed levels, large low/mid gaps around detailed structures |

Concrete paths for the most relevant frequency map:

- scalar-resolution frequency map: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns/lookcloser_frequencies/frame_train_00029.pt`
- metadata: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns/lookcloser_frequencies/frame_train_00029.json`
- RGB + level overlay diagnostic: `/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/frequency_map_stand_audit/frame_train_00029_stand_crop_frequency_diagnostic_side_by_side.jpg`
- patch-grid zoom: `/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/frequency_map_stand_audit/frame_train_00029_stand_crop_frequency_level_overlay_grid_red_ge13_zoom3x.png`
- full visual level map: `/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/frequency_map_stand_audit/frame_train_00029_full_frequency_level_map.png`

Additional useful map with a very clear metal pole:

- scalar-resolution frequency map: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns/lookcloser_frequencies/frame_train_00062.pt`
- metadata: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns/lookcloser_frequencies/frame_train_00062.json`
- RGB + level overlay diagnostic: `/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/frequency_map_stand_audit/frame_train_00062_stand_crop_frequency_diagnostic_side_by_side.jpg`
- patch-grid zoom: `/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/frequency_map_stand_audit/frame_train_00062_stand_crop_frequency_level_overlay_grid_red_ge13_zoom3x.png`
- full visual level map: `/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/frequency_map_stand_audit/frame_train_00062_full_frequency_level_map.png`

## Current Insight

The preprocessing/frequency map does not show a simple local "missing frequency" hole exactly on the stand in the closest train crop. The more likely issue is that the current FAS signal is very noisy and broad: high-frequency labels cover large brick-wall areas, while thin metal objects are represented by patchy, object-discontinuous levels. This can bias FAS toward many wall/hair/high-texture samples without reliably stabilizing thin-pole geometry.

Sampler-only changes did not fix the artifact. Next debugging should isolate whether the failure is caused by the frequency-map construction itself, the mapping from 2D FAS samples into the 3D frequency grid, or the density/geometry learned under FAS sample imbalance.

## 2026-06-05 Coverage and Early-Gate Recheck

The first-pass early gate at step ~3071 was too strict when interpreted without a same-step no-FAS control. A new same-seed no-FAS control was trained to the same checkpoint:

```text
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/nofas_control_gate3072_seed43/nerfstudio_models/step-000003071.ckpt
```

Strict crop output:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/nofas_seed43_step3071_target_crop_stride2/left_stand_connector_eval0.png
```

Same-step crop metrics:

| Run | FAS behavior | Step | Crop PSNR | Crop SSIM | Visual note |
|---|---|---:|---:|---:|---|
| `nofas_control_gate3072_seed43` | FAS disabled | 3071 | `25.9173` | `0.8112` | same early weak/broken stand appearance as FAS gates |
| `fas_uniform_mechanics_gate3072_seed43` | FAS path enabled, flat/count-proportional sampling | 3071 | `25.9982` | `0.8106` | visually similar to no-FAS early control |
| `fas_mix035_alpha10_gate3072_seed43` | ramp `1->3`, count alpha `1.0` | 3071 | `25.7822` | `0.8044` | slightly worse; still not a visual pass |

Interpretation: step ~3071 is not by itself evidence of a FAS-specific stand failure. It is an early-training weak-geometry state also present in no-FAS. Future "earliest failure" work must compare FAS and no-FAS at matched checkpoints, or use later checkpoints where the no-FAS reference has already recovered the stand.

Expected FAS coverage was audited with:

```bash
python scripts/audit_fas_sampling_coverage.py \
  --output-dir lookcloser_debug_outputs/fas_artifact_stand/fas_sampling_coverage_ramp_alpha0 \
  --fas-strength 0.35 \
  --sampling-ramp-start 1.0 \
  --sampling-ramp-end 3.0 \
  --fas-level-count-alpha 0.0
```

Key outputs:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_sampling_coverage_ramp_alpha0/level_sampling_weights.csv
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_sampling_coverage_ramp_alpha0/crop_sampling_coverage.csv
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_sampling_coverage_uniform_alpha1/crop_sampling_coverage.csv
```

Important coverage findings:

- Original ramp-only FAS (`fas_level_count_alpha=0`) samples common levels `12` and `13` at only `0.745x` and `0.769x` of uniform after the `0.35` mix, because the sampler spreads probability roughly equally over frequency levels rather than over pixels.
- The closest stand-like crop, `frame_train_00029` at the eval0 crop coordinates, receives `0.883x` uniform expected coverage under the original mixed-FAS distribution.
- Uniform-FAS (`sampling_ramp_start=end=1`, `fas_level_count_alpha=1`) gives exactly `1.0x` expected coverage for all audited levels and stand-like crops, confirming the audit math and isolating map weighting from sampler coordinate mechanics.
- Uniform-FAS still matched the no-FAS early crop visually at step 3071, so the early artifact is not caused solely by FAS coordinate/index mechanics.

Small sampler bug fixed while auditing: per-batch FAS rounding leftovers were always assigned to the last level. With `fas_max_sampling_level=12`, this sent leftover rays into empty level 15 and triggered fallback-uniform samples. Leftovers now go to non-empty levels by largest fractional remainder. This is a correctness fix, but it is too small to explain the full stand disappearance by itself.

Next evidence gap: timestamp the full-run divergence using matched no-FAS/FAS checkpoints after no-FAS stand recovery, then test whether count-proportional or flatter weighting prevents the late FAS-only disappearance without losing PSNR/SSIM gains.

## 2026-06-05 Later Continuation and Stride-1 Gate

The previous crop gates used `--stride 2` or `--stride 4`, which can hide thin-pole failure modes. The strict visual gate should use `--stride 1` when making final pass/fail calls for `left_stand_connector_eval0`; stride-2 is useful for quick triage but is not faithful enough for the floating/shifted vertical-bar artifact.

Known full-run references re-rendered with the same crop script at stride 1:

| Run | Checkpoint | Crop PSNR | Crop SSIM | Visual verdict |
|---|---|---:|---:|---|
| stable no-FAS H40 | `step-000034816` | `25.6556` | `0.7051` | stand connected, noisy but acceptable reference |
| known FAS seed43 | `step-000034816` | `25.7528` | `0.7256` | rejected; vertical stand segment appears as a shifted/floating dark bar |

Crop outputs:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/known_nofas_h40_step34816_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/known_fas_seed43_step34816_target_crop_stride1/left_stand_connector_eval0.png
```

The metrics do not catch the artifact: the visually rejected FAS crop has higher PSNR/SSIM than the stable no-FAS crop. The gate must remain visual-first.

Controlled continuation from the same no-FAS seed43 checkpoint:

1. Train no-FAS from `nofas_control_gate3072_seed43/step-000003071.ckpt` to step 6144, then resume to step 12287.
2. Train FAS from the no-FAS step-6144 checkpoint with FAS active immediately (`fas_strength=0.35`, `fas_warmup_steps=0`, `fas_ramp_steps=0`, ramp `1->3`, `fas_level_count_alpha=0`) to step 12287.
3. Continue that FAS branch to step 34815 with sparse checkpoints.

Key run paths:

```text
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/nofas_cont_from6144_to12288_seed43
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_ramp_alpha0_cont_from6144_to12288_seed43
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_ramp_alpha0_cont_from12287_to34816_seed43
```

Stride-1 target crop results:

| Run | Step | Eval loss | Crop PSNR | Crop SSIM | Visual verdict |
|---|---:|---:|---:|---:|---|
| FAS continuation | 33792 | `0.0273593` | `26.6739` | `0.7290` | stand connected; no severe floating bar |
| FAS continuation | 34815 | n/a | `25.5331` | `0.7177` | mild degradation; still not the same severe floating-bar failure |

Stride-2 target crop results from the same continuation showed similar direction but were less diagnostic:

| Run | Step | Crop PSNR | Crop SSIM |
|---|---:|---:|---:|
| no-FAS continuation | 6144 | `26.7743` | `0.8238` |
| no-FAS continuation | 9216 | `26.5176` | `0.8100` |
| no-FAS continuation | 12287 | `26.3686` | `0.8302` |
| FAS continuation | 9216 | `26.9810` | `0.8361` |
| FAS continuation | 12287 | `26.8913` | `0.8372` |
| FAS continuation | 22528 | `26.4301` | `0.8318` |
| FAS continuation | 33792 | `26.5389` | `0.8342` |
| FAS continuation | 34815 | `25.6495` | `0.8206` |

Interpretation:

- The original full FAS artifact is real and visible at stride 1.
- Continuing FAS from an already-trained no-FAS checkpoint does not reproduce the severe floating-bar artifact, even when FAS is active immediately for the rest of training.
- Therefore the smallest responsible cause is likely not the per-batch FAS sampler mechanics alone. The failure appears trajectory-dependent: FAS changes the early/mid optimization path enough that thin stand geometry can settle into a wrong detached solution while global eval metrics improve.
- The current rounding fix remains valid but is not sufficient as the final artifact fix.

Next test should start from scratch with a FAS schedule that preserves more uniform early/mid coverage, then validate at stride 1. The best candidate from current evidence is count-proportional or uniform-FAS for the FAS-selected portion (`fas_level_count_alpha=1`, optionally flatter `sampling_ramp_start=end=1`) with the existing `fas_strength=0.35`, while keeping Feature Re-weighting disabled.

## 2026-06-05 Uniform-FAS Full Seed-43 Run

Tested a from-scratch run with FAS enabled but the FAS-selected portion made count-proportional / pixel-uniform:

```text
run: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_w2048_r4096_seed43_full_stride1_gate
fas_strength=0.35
fas_warmup_steps=2048
fas_ramp_steps=4096
fas_level_count_alpha=1.0
sampling_ramp_start=1.0
sampling_ramp_end=1.0
Feature Re-weighting disabled
```

Sparse eval/checkpoint results:

| Step | Eval loss | Strict crop PSNR | Strict crop SSIM | Visual verdict |
|---:|---:|---:|---:|---|
| 11264 | `0.0258964` | `26.3716` | `0.7232` | pass; stand connected enough, no floating bar |
| 22528 | `0.0265006` | `27.2814` | `0.7396` | pass; stand and connector remain connected |
| 33792 | `0.0271881` | `24.8927` | `0.7158` | rejected; detached/floating vertical bar appears |
| 34815 | n/a | `24.6301` | `0.7130` | rejected; late floating-bar failure persists |

Strict crop outputs:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_full_seed43_step11264_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_full_seed43_step22528_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_full_seed43_step33792_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_full_seed43_step34815_target_crop_stride1/left_stand_connector_eval0.png
```

The normal eval-loss selection would choose step `11264`, which passes the strict crop gate. Full eval/render for that selected checkpoint:

```text
eval json: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_w2048_r4096_seed43_full_stride1_gate/eval_best_step-000011264.json
renders: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_w2048_r4096_seed43_full_stride1_gate/renders_best_step-000011264
PSNR=29.024340
SSIM=0.666960
LPIPS=0.399108
```

Single-seed comparison against the current no-FAS H40 reference and known rejected FAS run:

| Run | Selected checkpoint | PSNR | SSIM | LPIPS | Visual status |
|---|---|---:|---:|---:|---|
| no-FAS H40 reference | `step-000034816` | `28.898243` | `0.665879` | `0.365282` | pass |
| known mixed-FAS seed43 | `step-000034816` | `29.135916` | `0.681484` | `0.367407` | rejected |
| uniform-FAS seed43 best-loss | `step-000011264` | `29.024340` | `0.666960` | `0.399108` | pass on target crop, but worse LPIPS |

Interpretation:

- Flattening/count-proportional FAS does not prevent the late stand failure; the 33792 and latest checkpoints reproduce the detached vertical-bar artifact.
- The best eval-loss checkpoint occurs early, before the late artifact appears, and gives a small PSNR/SSIM gain over the no-FAS H40 single-seed reference, but LPIPS regresses.
- This is not a final FAS fix. It suggests the artifact is coupled to late training under FAS, likely an overfit / geometry-drift mode that eval loss detects earlier than PSNR/SSIM crop metrics.
- The next minimal fix should keep the sampler rounding correction, keep Feature Re-weighting disabled, and test whether normal early stopping plus a stricter visual checkpoint selector is sufficient, or whether FAS should be disabled/annealed after the best eval-loss window to avoid late thin-geometry drift.

### Seed-42 Early-Stop Recheck

The same uniform-FAS setup was rerun for seed 42 with normal early stopping and final renders enabled:

```text
run: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_w2048_r4096_seed42_earlystop_stride1_gate
selected checkpoint: step-000033792
renders: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_w2048_r4096_seed42_earlystop_stride1_gate/renders_best_step-000033792
eval json: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_w2048_r4096_seed42_earlystop_stride1_gate/eval_best_step-000033792.json
```

Eval rows:

| Step | Eval loss | Status |
|---:|---:|---|
| 11264 | `0.0307675` | `ok` |
| 22528 | `0.0301214` | `ok` |
| 33792 | `0.0266216` | `improving` |

Full eval at selected checkpoint:

```text
PSNR=29.203186
SSIM=0.679225
LPIPS=0.386479
```

Strict crop:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_earlystop_seed42_step33792_target_crop_stride1/left_stand_connector_eval0.png
PSNR=25.7357
SSIM=0.7260
```

Visual verdict: rejected. The selected seed-42 checkpoint shows the same detached/floating vertical-bar stand artifact. This rejects normal eval-loss early stopping as a robust fix: seed 43 selected an early visual-pass checkpoint, but seed 42's eval loss continued improving into a visually unsafe checkpoint.

Next test should avoid letting FAS keep steering the trajectory into late thin-geometry drift. A minimal direction is to stop or anneal FAS after the early/mid detail-learning window, then validate with keep-all checkpoints and the stride-1 crop before pruning.

### Seed-42 FAS Decay Test

Added a diagnostic sampler schedule that can decay FAS back to uniform sampling:

```text
fas_decay_start_steps=11264
fas_decay_steps=4096
```

This keeps the same seed-42 uniform-FAS setup, but starts turning FAS off after the first sparse eval/checkpoint and reaches fully uniform sampling by step `15360`.

Run:

```text
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_decay11264_4096_seed42_full_stride1_gate
```

Eval rows:

| Step | Eval loss | Status |
|---:|---:|---|
| 11264 | `0.0310269` | `ok` |
| 22528 | `0.0275160` | `improving` |
| 33792 | `0.0314439` | `overfit_watch` |

Strict crop results:

| Step | Crop PSNR | Crop SSIM | Visual verdict |
|---:|---:|---:|---|
| 22528 | `26.4057` | `0.7326` | rejected / borderline; left stand remains split and shifted |
| 33792 | `25.7550` | `0.7270` | rejected; similar split/shifted stand persists |

Crop outputs:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_decay11264_seed42_step22528_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_decay11264_seed42_step33792_target_crop_stride1/left_stand_connector_eval0.png
```

Interpretation: decaying FAS after step `11264` improves the eval-loss curve relative to the non-decay seed-42 run and avoids selecting the late `33792` checkpoint, but it does not repair the target stand artifact. The stand is already split/shifted by the selected `22528` checkpoint. This points away from "FAS persists too long" as the only cause and toward "FAS starts before thin-pole geometry is stable enough."

Next test should delay FAS start rather than only decay it. The prior continuation evidence suggests starting FAS from an already-recovered no-FAS checkpoint can avoid the severe artifact, so a from-scratch schedule with a longer uniform warmup, keep-all checkpoints, and stride-1 visual gates is the next minimal candidate.

### Seed-42 Delayed-FAS Start Test

Tested whether keeping the sampler uniform until step `11264` avoids the stand artifact:

```text
run: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_warm11264_r4096_seed42_full_stride1_gate
fas_strength=0.35
fas_warmup_steps=11264
fas_ramp_steps=4096
fas_level_count_alpha=1.0
sampling_ramp_start=1.0
sampling_ramp_end=1.0
Feature Re-weighting disabled
```

Eval rows before the run was stopped:

| Step | FAS state | Eval loss | Status |
|---:|---|---:|---|
| 11264 | pre-FAS / uniform | `0.0264088` | `ok` |
| 22528 | full delayed-FAS active | `0.0275902` | `overfit_watch` |

Strict crop results:

| Step | FAS state | Crop PSNR | Crop SSIM | Visual verdict |
|---:|---|---:|---:|---|
| 11264 | pre-FAS / uniform | `25.9758` | `0.7203` | rejected; stand already split/floating before FAS activates |
| 22528 | full delayed-FAS active | `25.8619` | `0.7254` | rejected; detached/floating vertical segments persist |

Crop outputs:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_warm11264_seed42_step11264_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_warm11264_seed42_step22528_target_crop_stride1/left_stand_connector_eval0.png
```

The run was stopped after the first meaningful FAS-active crop failed.

Interpretation: `11264` steps of uniform/no-FAS warmup is still too early for this seed. The pre-FAS checkpoint already fails the strict stand crop, so this run cannot prove that delayed FAS caused the artifact; it proves only that the delayed schedule did not wait long enough for the no-FAS geometry to recover. The stable no-FAS H40 reference is also seed 42 and passes at step `34816`.

Same-seed no-FAS H40 late reference:

| Step | Eval loss | Crop PSNR | Crop SSIM | Visual verdict |
|---:|---:|---:|---:|---|
| 33792 | `0.0299586` | `25.5695` | `0.7038` | connected/no severe floating bar, but noisier than selected step |
| 34816 | `0.0257004` | `25.6556` | `0.7051` | stable visual reference; connected stand |

The next useful test is a late continuation from the stable no-FAS H40 step `34816` checkpoint with FAS enabled. If late FAS is safe there, the smallest responsible cause is early/mid FAS pressure before thin-pole geometry has recovered, and a practical fix may need a two-stage or much-later activation schedule rather than modest warmup/decay.

### Seed-42 Late-FAS Continuation From Stable No-FAS H40

Tested FAS only after the same-seed no-FAS H40 reference had already reached the stable visual checkpoint:

```text
run: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_from_h40_34816_to43008_seed42_stride1_gate
load checkpoint: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_continue36864_r4096/nerfstudio_models/step-000034816.ckpt
fas_strength=0.35
fas_warmup_steps=0
fas_ramp_steps=0
fas_level_count_alpha=1.0
sampling_ramp_start=1.0
sampling_ramp_end=1.0
Feature Re-weighting disabled
```

Eval rows:

| Step | Eval loss | Status |
|---:|---:|---|
| 36864 | `0.0257669` | `ok` |
| 40960 | `0.0293686` | `overfit_watch` |

Strict crop results:

| Step | Crop PSNR | Crop SSIM | Visual verdict |
|---:|---:|---:|---|
| 36864 | `25.8220` | `0.7088` | pass; noisy but connected, no floating vertical segment |
| 40960 | `26.2174` | `0.7157` | pass; stand remains connected |
| 43007 | `26.6599` | `0.7204` | pass; stand remains connected |

Crop outputs:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_from_h40_seed42_step36864_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_from_h40_seed42_step40960_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_from_h40_seed42_step43007_target_crop_stride1/left_stand_connector_eval0.png
```

Full eval/renders:

```text
selected eval json: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_from_h40_34816_to43008_seed42_stride1_gate/eval_best_step-000036864.json
selected renders: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_from_h40_34816_to43008_seed42_stride1_gate/renders_best_step-000036864
latest eval json: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_from_h40_34816_to43008_seed42_stride1_gate/eval_latest_step-000043007.json
latest renders: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_from_h40_34816_to43008_seed42_stride1_gate/renders_latest_step-000043007
```

Metric comparison:

| Run | Checkpoint | PSNR | SSIM | LPIPS | Visual status |
|---|---|---:|---:|---:|---|
| no-FAS H40 reference | `34816` | `28.898243` | `0.665879` | `0.365282` | pass |
| late-FAS selected | `36864` | `28.904247` | `0.666365` | `0.367931` | pass |
| late-FAS latest | `43007` | `28.877571` | `0.668204` | `0.379719` | pass |

Interpretation:

- Starting FAS from the stable no-FAS H40 checkpoint does not reproduce the stand disappearance. The crop remains visually safe through the selected checkpoint and latest checkpoint.
- The selected late-FAS checkpoint slightly improves PSNR/SSIM over the no-FAS H40 reference, but the gain is tiny and LPIPS regresses.
- This is the strongest evidence so far that the FAS stand artifact is caused by early/mid optimization trajectory changes before thin-pole geometry is stable, not by FAS sampling at inference/eval time or by late FAS alone.
- This is not yet a final accepted fix: it is a one-seed two-stage continuation. Final validation still needs the same two-stage protocol across seeds and broader visual crops before recommending it as the LookCloser FAS training recipe.

### Existing Seed-43 FAS Continuation Re-eval

The prior seed-43 FAS continuation from a no-FAS checkpoint also supports the trajectory diagnosis, although it is not the exact same H40-late protocol:

```text
run: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_ramp_alpha0_cont_from12287_to34816_seed43
best checkpoint by eval loss: step-000022528
FAS behavior: ramp 1->3, fas_level_count_alpha=0
```

Strict crop:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_ramp_alpha0_cont_seed43_step22528_target_crop_stride1/left_stand_connector_eval0.png
PSNR=26.4053
SSIM=0.7249
```

Visual verdict: pass / no severe floating-bar artifact.

Full eval/renders:

```text
eval json: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_ramp_alpha0_cont_from12287_to34816_seed43/eval_best_step-000022528.json
renders: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_ramp_alpha0_cont_from12287_to34816_seed43/renders_best_step-000022528
PSNR=29.096552
SSIM=0.672171
LPIPS=0.356596
```

Interpretation: another continuation-style FAS run avoids the severe stand disappearance and has strong full-eval metrics. Because this run starts from a shorter no-FAS seed-43 continuation and uses ramp-only FAS rather than the flat/count-proportional late-H40 protocol, it is supporting evidence rather than final validation. The next validation run should reproduce the exact no-FAS-H40-to-FAS continuation protocol for additional seeds.

### Seed-43 Exact Two-Stage Validation

Trained a same-protocol no-FAS seed-43 H40 base, then continued flat/count-proportional FAS from its stable late checkpoint.

No-FAS base:

```text
run: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_seed43_exact2stage_base
FAS disabled
Feature Re-weighting disabled
```

No-FAS base eval rows:

| Step | Eval loss | Status |
|---:|---:|---|
| 11264 | `0.0260380` | `ok` |
| 22528 | `0.0277915` | `overfit_watch` |
| 33792 | `0.0276549` | `improving` |

No-FAS base strict crop:

| Step | Crop PSNR | Crop SSIM | Visual verdict |
|---:|---:|---:|---|
| 33792 | `27.1373` | `0.7381` | pass; connected stand |
| 34815 | `27.2838` | `0.7404` | pass; connected stand, used as continuation base |

FAS continuation:

```text
run: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_from_seed43_h40_34815_to43008_stride1_gate
load checkpoint: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_seed43_exact2stage_base/nerfstudio_models/step-000034815.ckpt
fas_strength=0.35
fas_warmup_steps=0
fas_ramp_steps=0
fas_level_count_alpha=1.0
sampling_ramp_start=1.0
sampling_ramp_end=1.0
Feature Re-weighting disabled
```

FAS continuation eval rows:

| Step | Eval loss | Status |
|---:|---:|---|
| 36864 | `0.0298654` | `ok` |
| 40960 | `0.0273780` | `improving` |

FAS continuation strict crop:

| Step | Crop PSNR | Crop SSIM | Visual verdict |
|---:|---:|---:|---|
| 36864 | `26.7804` | `0.7356` | pass; connected stand |
| 40960 | `24.8813` | `0.7175` | pass; no severe floating-bar artifact |
| 43007 | `25.2845` | `0.7226` | pass; no severe floating-bar artifact |

Crop outputs:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/nofas_h40_seed43_step33792_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/nofas_h40_seed43_step34815_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_from_seed43_h40_step36864_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_from_seed43_h40_step40960_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_from_seed43_h40_step43007_target_crop_stride1/left_stand_connector_eval0.png
```

Full eval/renders for selected FAS checkpoint:

```text
eval json: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_from_seed43_h40_34815_to43008_stride1_gate/eval_best_step-000040960.json
renders: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_from_seed43_h40_34815_to43008_stride1_gate/renders_best_step-000040960
PSNR=29.214664
SSIM=0.686451
LPIPS=0.377893
```

Interpretation: the exact two-stage protocol also passes on seed 43. The selected FAS checkpoint has very strong PSNR/SSIM and no severe stand failure, although the latest crop remains visually weaker than the no-FAS base and LPIPS is not yet compared against a full no-FAS seed-43 eval. This strengthens the conclusion that the responsible cause is FAS entering before the stand geometry is stable.

### Seed-44 Exact Two-Stage Validation

What was tested: repeat the exact no-FAS-H40 to uniform-FAS continuation protocol on seed 44. The validation explicitly checked the no-FAS base before selecting the FAS handoff checkpoint, because seed 44 showed late no-FAS geometry drift.

No-FAS base:

```text
run: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_seed44_exact2stage_base
FAS disabled
Feature Re-weighting disabled
```

No-FAS base eval rows:

| Step | Eval loss | Status |
|---:|---:|---|
| 11264 | `0.0289930` | `ok` |
| 22528 | `0.0272129` | `improving` |
| 33792 | `0.0293023` | `overfit_watch` |

No-FAS base strict crop:

| Step | Crop PSNR | Crop SSIM | Visual verdict |
|---:|---:|---:|---|
| 33792 | `27.2160` | `0.7408` | pass; connected stand, used as FAS handoff |
| 34815 | `25.9431` | `0.7285` | reject; stand already split without FAS |

FAS continuation:

```text
run: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_from_seed44_h40_33792_to43008_stride1_gate
load checkpoint: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_seed44_exact2stage_base/nerfstudio_models/step-000033792.ckpt
fas_strength=0.35
fas_warmup_steps=0
fas_ramp_steps=0
fas_level_count_alpha=1.0
sampling_ramp_start=1.0
sampling_ramp_end=1.0
Feature Re-weighting disabled
```

FAS continuation eval rows:

| Step | Eval loss | Status |
|---:|---:|---|
| 36864 | `0.0262159` | `ok` |
| 40960 | `0.0243743` | `improving` |

FAS continuation strict crop:

| Step | Crop PSNR | Crop SSIM | Visual verdict |
|---:|---:|---:|---|
| 36864 | `27.1483` | `0.7390` | pass; connected stand |
| 40960 | `26.2910` | `0.7350` | reject; eval-loss-selected checkpoint splits the stand |
| 43007 | `25.4914` | `0.7273` | reject; stand remains split/weak |

Crop outputs:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/nofas_h40_seed44_step33792_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/nofas_h40_seed44_step34815_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_from_seed44_h40_step36864_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_from_seed44_h40_step40960_target_crop_stride1/left_stand_connector_eval0.png
/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_from_seed44_h40_step43007_target_crop_stride1/left_stand_connector_eval0.png
```

Full eval/renders for gate-safe FAS checkpoint:

```text
eval json: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_from_seed44_h40_33792_to43008_stride1_gate/eval_gate_safe_step-000036864.json
renders: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_uniform_alpha1_from_seed44_h40_33792_to43008_stride1_gate/renders_gate_safe_step-000036864
PSNR=29.301405
SSIM=0.677420
LPIPS=0.370835
```

Broader stride-2 crop sheets for accepted two-stage checkpoints:

```text
seed42: /home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_seed42_step36864_all_crops_stride2/all_crops.png
seed43: /home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_seed43_step40960_all_crops_stride2/all_crops.png
seed44: /home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/fas_artifact_stand/fas_uniform_alpha1_seed44_step36864_all_crops_stride2/all_crops.png
```

Results summary:

| Seed | Accepted checkpoint | Selection rule | PSNR | SSIM | LPIPS | Strict stand verdict | Broader crop verdict |
|---:|---:|---|---:|---:|---:|---|---|
| 42 | `36864` | eval loss + strict crop pass | `28.904247` | `0.666365` | `0.367931` | pass | pass |
| 43 | `40960` | eval loss + strict crop pass | `29.214664` | `0.686451` | `0.377893` | pass | pass |
| 44 | `36864` | strict crop gate overrode lower eval loss at `40960` | `29.301405` | `0.677420` | `0.370835` | pass | pass |

Insights:

- The seed 44 no-FAS base proves the artifact can appear as late thin-geometry drift without FAS; step `34815` is already split, so it is not a valid handoff checkpoint.
- The seed 44 FAS continuation proves eval loss alone can select a visually invalid checkpoint: step `40960` has lower eval loss than `36864` but fails the stand crop.
- Across accepted seed 42/43/44 two-stage checkpoints, flat/count-proportional FAS from a visually stable no-FAS base avoids the severe stand disappearance with Feature Re-weighting disabled.
- The practical fix is training-protocol plus sampler cleanup: use the corrected FAS rounding, `fas_level_count_alpha=1.0`, flat sampling ramp, start FAS only after a no-FAS H40 checkpoint passes `left_stand_connector_eval0 --stride 1`, and require the same strict crop to approve the final checkpoint.

## Hand/Stand Follow-up: Baseline-Compatible Scale

User visual inspection later found a separate large blotch around the left hand in `eval_img_0000`, plus residual stand breaks near the outlet. That blotch appeared before FAS, so a separate step-by-step debug pass was run in:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/experiments/lookcloser_hand_stand_artifact_debug.md
```

Key finding: the severe hand/background blotch was reproduced in older LookCloser ARM checkpoints that used `scene_scale=2.0`, `scale_factor=1.15`, but it disappeared when LookCloser was rerun with the bounded Instant-NGP dataparser scale:

```text
scene_scale=1.5
scale_factor=1.0
max_res=8192
```

The explicit `max_res=8192` is required with the current frequency maps. A first scale-aligned attempt without it failed metadata validation because the model schedule derived `max_res=6144` while the precomputed maps were generated with `max_res=8192`.

Best clean no-FAS checkpoint from this pass:

```text
run: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_hand_stand_debug/lookcloser/nofas_h40_scene15_scale10_maxres8192_seed42
checkpoint: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_hand_stand_debug/lookcloser/nofas_h40_scene15_scale10_maxres8192_seed42/nerfstudio_models/step-000012288.ckpt
crop sheet: /home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/hand_stand_artifact_timeline/apples_scene15_scale10_maxres8192_seed42_s12288/all_gate_stride2/all_crops.png
eval loss: 0.0263846
```

FAS continuation from that clean checkpoint used the accepted flat/count-proportional recipe (`fas_strength=0.35`, `fas_level_count_alpha=1.0`, `sampling_ramp_start=end=1.0`, Feature Re-weighting disabled). It kept the hand-background blotch fixed and selected step `16384` by best eval loss:

```text
run: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_hand_stand_debug/lookcloser/fas035_count1_from_apples_s12288_seed42
checkpoint: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_hand_stand_debug/lookcloser/fas035_count1_from_apples_s12288_seed42/nerfstudio_models/step-000016384.ckpt
crop sheet: /home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/hand_stand_artifact_timeline/fas035_count1_from_apples_s12288_s16384/all_gate_stride2/all_crops.png
eval json: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_hand_stand_debug/lookcloser/fas035_count1_from_apples_s12288_seed42/eval_best_step-000016384.json
renders: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_hand_stand_debug/lookcloser/fas035_count1_from_apples_s12288_seed42/renders_best_step-000016384
PSNR=29.174545
SSIM=0.669102
LPIPS=0.378236
```

Uniform-frequency-map status: no full constant-frequency-map training control has been run. Earlier "uniform" results in this report are uniform/count-proportional FAS sampling (`fas_level_count_alpha=1.0`, flat ramp) with the real frequency maps still enabled, not a constant frequency map.

Updated interpretation:

- The large hand-background blotch was not caused by FAS; it was already present in old ARM training before FAS and is removed by matching the Instant-NGP dataparser scale.
- The accepted FAS recipe remains valid after scale alignment and gives metrics in the same range as the prior accepted two-stage FAS runs.
- The remaining problem is a local detail/SSIM gap versus Instant-NGP around the hand, outlet, stand, and cables, not the old severe stand-disappearance or hand-blotch failure.
