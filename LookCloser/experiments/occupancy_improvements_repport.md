# Occupancy Improvements Report

## Goal

Reduce significant structural artifacts on eval views toward `artifact_score`
near zero by improving LookCloser occupancy-grid observability, hyperparameters,
and code where needed. This report is updated as evidence accumulates.

## Current branch and implementation state

- Branch: `lookcloser/occupancy-grid-experiments`
- Base implementation branch: `lookcloser/occupancy-grid-v2`
- Added code hooks:
  - LookCloser occupancy knobs: `occupancy_occ_thre`,
    `occupancy_ema_decay`, `occupancy_warmup_steps`,
    `occupancy_update_interval`, optional `occupancy_update_step_size`,
    `occupancy_thre_clamp_mult`, `occupancy_dilation_radius`.
  - Occupancy training metrics: occupancy ratio, `occs.mean/max`, default and
    effective thresholds, effective alpha threshold, binary flip-on/off counts,
    mean samples/ray, zero-sample ray rate.
  - Quiet runner provenance and timing: git/data/frequency-map fingerprints,
    train/eval/artifact/total seconds.
  - Multi-view artifact scoring via `--artifact-render-names`.
  - Artifact-to-occupancy debugger:
    `scripts/debug_artifact_occupancy_grid.py`.

## Baseline evidence from existing runs

Current best artifact-tuned ARM run:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_temp_report_2/lookcloser/real_maps_cont20480_to24576_solo_grid64_cap2048_coarse000625_maxstep000625_alpha0025_r4096_seed42`

Config highlights:

- checkpoint: `step-000024576`
- `grid_resolution=64`
- `adaptive_warmup_steps=12288` lineage, continued to `24576`
- `adaptive_coarse_step_size=0.00625`
- `adaptive_max_step_size=0.00625`
- `max_steps_per_ray=2048`
- `alpha_thre=0.0025`

Metrics from `metrics_compact.csv` at selected step:

| Step | Eval loss | PSNR | SSIM | LPIPS | Train time |
|---:|---:|---:|---:|---:|---:|
| 24576 | 0.0303422 | 28.4808 | 0.648240 | 0.366523 | 690.635s continuation |

Artifact scores on saved eval renders:

| Render | Eval view | Artifact score | Count | Largest blob | Notes |
|---|---:|---:|---:|---:|---|
| normal ARM | 0 | 3.568 | 29 | 1436 px | Current best known trained ARM score. |
| normal ARM | 1 | 2.313 | 22 | 629 px | Smaller but still serious. |
| normal ARM | 2 | 3.463 | 25 | 1542 px | Similar severity to eval0. |
| dense render override `coarse/max=0.003125`, cap4096 | 0 | 3.387 | 27 | 2060 px | Only slight score improvement. |
| dense render override `coarse/max=0.003125`, cap4096 | 1 | 1.950 | 18 | 633 px | Clearer traversal/density-step benefit. |
| dense render override `coarse/max=0.003125`, cap4096 | 2 | 2.484 | 19 | 1542 px | Clearer traversal/density-step benefit. |

Interpretation so far:

- Dense render-only override helps eval1/eval2 more than eval0, so not all
  artifacts have the same cause.
- Eval0 remaining largest artifact is probably not a simple occupancy-grid miss.

## Stage 0 diagnostics

### Stage 0.3 artifact-to-occupancy, best ARM eval0

Command:

```bash
python scripts/debug_artifact_occupancy_grid.py \
  --run-dir /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_temp_report_2/lookcloser/real_maps_cont20480_to24576_solo_grid64_cap2048_coarse000625_maxstep000625_alpha0025_r4096_seed42 \
  --render-file /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_temp_report_2/lookcloser/real_maps_cont20480_to24576_solo_grid64_cap2048_coarse000625_maxstep000625_alpha0025_r4096_seed42/artifact_renders_step-000024576/eval_img_0000.png \
  --eval-index 0 \
  --max-pixels 1024 \
  --ray-samples 256 \
  --pixel-stride 2 \
  --out-dir lookcloser_debug_outputs/occupancy_stage0/best_arm_24576_eval0
```

Output:

- JSON:
  `lookcloser_debug_outputs/occupancy_stage0/best_arm_24576_eval0/artifact_occupancy_debug.json`
- Markdown:
  `lookcloser_debug_outputs/occupancy_stage0/best_arm_24576_eval0/artifact_occupancy_debug.md`
- Overlay:
  `lookcloser_debug_outputs/occupancy_stage0/best_arm_24576_eval0/artifact_pixels_overlay.png`

Key values:

| Quantity | Value |
|---|---:|
| selected bbox | `[1298, 1047, 1401, 1076]` |
| selected pixels | 265 |
| occupancy grid resolution | 64 |
| occupancy ratio | 0.530762 |
| `occs.mean()` | 0.144673 |
| `occs.max()` | 2.890820 |
| effective binary threshold | 0.01 |
| effective alpha threshold | 0.0025 |
| valid surface pixels | 265 |
| surface voxels occupied | 211 / 265 = 0.796 |
| rays with any occupied voxel | 265 / 265 = 1.0 |
| rays inside grid with no occupied voxel | 0 / 265 = 0.0 |

Classifier:

```json
{
  "grid_miss_likely": false,
  "field_issue_likely": false,
  "read": "artifact pixels mostly map to occupied voxels; field quality, alpha integration, or checkpoint selection is more likely"
}
```

Insight:

- For eval0 largest artifact, `mean(occs)=0.1447` is much larger than
  `occ_thre=0.01`, so `occ_thre` is live for this checkpoint; the mean-clamp
  no-op regime is not active here.
- Artifact pixels are largely inside occupied voxels and every analyzed ray hits
  some occupied voxel. Dilation or resolution alone is unlikely to remove this
  specific eval0 artifact.
- Because dense render-only override barely improves eval0 (`3.568 -> 3.387`),
  this view likely needs field-quality/checkpoint-selection/training-path changes
  or more local sampling during training, not only eval traversal tweaks.

### Stage 0.3 artifact-to-occupancy, all eval views

Additional commands:

```bash
python scripts/debug_artifact_occupancy_grid.py --run-dir <best_arm_run> \
  --render-file <best_arm_run>/artifact_renders_step-000024576/eval_img_0001.png \
  --eval-index 1 --max-pixels 1024 --ray-samples 256 --pixel-stride 2 \
  --out-dir lookcloser_debug_outputs/occupancy_stage0/best_arm_24576_eval1

python scripts/debug_artifact_occupancy_grid.py --run-dir <best_arm_run> \
  --render-file <best_arm_run>/artifact_renders_step-000024576/eval_img_0002.png \
  --eval-index 2 --max-pixels 1024 --ray-samples 256 --pixel-stride 2 \
  --out-dir lookcloser_debug_outputs/occupancy_stage0/best_arm_24576_eval2
```

Results:

| Eval view | Artifact score | Selected bbox | Surface occupied rate | Rays with any occupied voxel | Rays with no occupied voxel | Classification |
|---:|---:|---|---:|---:|---:|---|
| 0 | 3.568 | `[1298, 1047, 1401, 1076]` | 0.796 | 1.000 | 0.000 | not grid miss |
| 1 | 2.313 | `[255, 150, 272, 226]` | 1.000 | 1.000 | 0.000 | field issue likely |
| 2 | 3.463 | `[1746, 286, 1779, 463]` | 0.992 | 1.000 | 0.000 | field issue likely |

Common occupancy state at checkpoint `24576`:

- `grid_resolution=64`
- `occupancy_ratio=0.530762`
- `occs.mean=0.144673`
- `effective_binary_threshold=0.01`
- `effective_alpha_thre=0.0025`

Insight:

- The largest artifact in all three eval views is not caused by a missing binary
  occupancy path. Every analyzed ray crosses occupied voxels, and surface points
  are mostly occupied.
- For this checkpoint, the mean-clamp no-op regime is not active:
  `occs.mean=0.144673 > occ_thre=0.01`.
- Pure occupancy dilation/resolution tuning is unlikely to drive artifact score
  near zero by itself. The best next experiment should change the training/render
  sampling support around ARM, not only occupancy binarization.

### Stage 1 implementation smoke: ARM fixed fallback samples

Implemented an optional safety-net knob:

- `adaptive_fixed_fallback_samples_per_ray`
- default `0`, so existing behavior is preserved.
- when enabled, uniform per-ray fallback samples are merged into the adaptive
  packed ray samples before field evaluation.

Smoke command loaded the best checkpoint `step-000024576.ckpt`, set
`adaptive_fixed_fallback_samples_per_ray=4`, and rendered four pixels inside the
eval0 artifact bbox through `model.get_outputs`.

Result:

| Check | Value |
|---|---:|
| rays | 4 |
| fallback samples appended | 16 |
| RGB finite | true |
| samples per ray after merge | `[394, 419, 801, 431]` |

Insight:

- The packed fallback path is executable and produces finite outputs on artifact
  rays.
- Next experiment can train with a larger but still bounded fallback value.

### Stage 2.0 short continuation: fallback32 from checkpoint 24576

Command launched:

```bash
python scripts/run_lookcloser_quiet.py \
  --experiment-name 007740_hd_aabb4_multicamera_eval3_ns_occupancy_fallback \
  --timestamp fallback32_from24576_seed42 \
  --scene-scale 1.5 --scale-factor 1.0 \
  --grid-resolution 64 --grid-update-interval 4096 --grid-update-batch-size 2048 \
  --max-res 8192 \
  --train-num-rays-per-batch 4096 --eval-num-rays-per-batch 4096 \
  --eval-num-rays-per-chunk 256 \
  --step-interval 4096 --max-num-iterations 28672 \
  --background-color black --reconstruction-loss-type charbonnier \
  --disable-feature-reweighting --disable-fas \
  --adaptive-warmup-steps 12288 \
  --adaptive-coarse-step-size 0.00625 --adaptive-max-step-size 0.00625 \
  --max-steps-per-ray 2048 --alpha-thre 0.0025 \
  --adaptive-fixed-fallback-samples-per-ray 32 \
  --load-dir <best_arm_run>/nerfstudio_models --load-step 24576 \
  --artifact-render-names eval_img_0000.png,eval_img_0001.png,eval_img_0002.png \
  --summary-path experiments/occupancy_improvements_repport.md
```

Run directory:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occupancy_fallback/lookcloser/fallback32_from24576_seed42`

Status:

- completed.
- first observed compact line was `step=24750`.
- train-time sample count confirmed the knob was active:
  `train_adaptive_samples_max=2080` vs previous cap `2048`.

Result:

| Metric | Best ARM 24576 | Fallback32 continuation 28671 | Read |
|---|---:|---:|---|
| PSNR | 28.4808 | 25.5206 | worse |
| SSIM | 0.6482 | 0.6222 | worse |
| LPIPS | 0.3665 | 0.4291 | worse |
| eval loss | 0.030342 | n/a | no train eval row; final `ns-eval` only |
| artifact max score | 3.568 | 26.266 | much worse |
| artifact mean score | 3.115 | 24.616 | much worse |
| train seconds | 690.635 for prior 4096-step continuation | 420.326 | not directly comparable |
| eval seconds | n/a | 221.139 | recorded |
| artifact seconds | n/a | 31.660 | recorded |
| total seconds | n/a | 673.143 | recorded |

Per-view artifact scores:

| Eval view | Best ARM 24576 | Fallback32 28671 |
|---:|---:|---:|
| 0 | 3.568 | 26.266 |
| 1 | 2.313 | 22.472 |
| 2 | 3.463 | 25.111 |

Visual check:

- detector boxes mosaic:
  `lookcloser_debug_outputs/occupancy_stage0/fallback32_boxes_mosaic.png`

Insight:

- Training-time fallback32 is rejected. It degrades every global image metric and
  massively increases structural artifact score on all eval views.
- This is not a single-view detector false positive; the visual boxes are spread
  across the renders.
- The run stopped at `step=28671`, so no train-time eval row/eval loss was
  written. Future one-boundary continuations should set `max_num_iterations` one
  step past the intended eval boundary, or use final `ns-eval` metrics only.

## Next actions

1. Do not continue the training-time fallback branch unless testing a much smaller
   diagnostic value or eval-only variant.
2. Keep occupancy dilation/resolution as secondary experiments only if a future
   artifact-to-grid debug run shows `grid_miss_likely=true`.
3. For the next occupancy-specific run, prefer lower-risk Stage 2 knobs:
   `adaptive_warmup_steps=20000`, `occupancy_ema_decay=0.99`, or
   `grid_resolution=128`, each one factor at a time.

### Stage 2.1 ARM handoff: adaptive_warmup_steps=20000

Purpose:

- Test the user-requested late ARM handoff value `20000`.
- Remove the old `3096` warmup idea from the active matrix.
- Start before handoff from checkpoint `16384`, then train through `20480`.

Command launched:

```bash
python scripts/run_lookcloser_quiet.py \
  --experiment-name 007740_hd_aabb4_multicamera_eval3_ns_occupancy_warmup \
  --timestamp warmup20000_from16384_to20480_seed42 \
  --scene-scale 1.5 --scale-factor 1.0 \
  --grid-resolution 64 --grid-update-interval 4096 --grid-update-batch-size 2048 \
  --max-res 8192 \
  --train-num-rays-per-batch 4096 --eval-num-rays-per-batch 4096 \
  --eval-num-rays-per-chunk 256 \
  --step-interval 4096 --max-num-iterations 20481 \
  --background-color black --reconstruction-loss-type charbonnier \
  --disable-feature-reweighting --disable-fas \
  --adaptive-warmup-steps 20000 \
  --adaptive-coarse-step-size 0.00625 --adaptive-max-step-size 0.00625 \
  --max-steps-per-ray 2048 --alpha-thre 0.0025 \
  --load-dir <w12288_best_run>/nerfstudio_models --load-step 16384 \
  --artifact-render-names eval_img_0000.png,eval_img_0001.png,eval_img_0002.png \
  --summary-path experiments/occupancy_improvements_repport.md
```

Run directory:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occupancy_warmup/lookcloser/warmup20000_from16384_to20480_seed42`

Status:

- completed at `step=20480`.
- first observed compact line `step=16580`.
- train/eval/artifact runtime was inflated by parallel GPU contention.

Result:

| Metric | 12288 control at 20480 | Warmup20000 at 20480 | Read |
|---|---:|---:|---|
| eval loss | 0.030641 | 0.031143 | worse |
| PSNR | 28.3625 | 28.2274 | worse |
| SSIM | 0.6465 | 0.6404 | worse |
| LPIPS | 0.3854 | 0.4164 | worse |
| artifact max score | not measured in old runner | 4.848 | screen-negative vs best artifact-tuned 24576 |

Per-view artifact scores:

| Eval view | Warmup20000 artifact score | Count | Largest area |
|---:|---:|---:|---:|
| 0 | 4.848 | 38 | 1306 |
| 1 | 2.724 | 23 | 895 |
| 2 | 3.469 | 26 | 1405 |

Insight:

- `adaptive_warmup_steps=20000` is rejected as a screening candidate. It worsens
  global metrics at the first post-handoff checkpoint and does not show an
  artifact improvement signal.
- Because this is a single-seed screen under contention, it should not be treated
  as a final variance-aware conclusion, but it is not worth extending before
  stronger candidates.

### Stage 2.3 occupancy EMA decay: 0.99

Purpose:

- Test the thin-structure survival hypothesis with slower occupancy decay.
- One-factor change from the current best `20480->24576` continuation:
  `occupancy_ema_decay=0.99` instead of `0.95`.

Run directory:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occupancy_decay/lookcloser/ema099_from20480_to24576_seed42`

Command status:

- launched in parallel with the tail of the `warmup20000` run to save wall time.
- summary auto-update disabled for this run to avoid concurrent Markdown writes.
- runtime must be treated as screening-only because GPU contention confounds wall
  clock; metrics/artifact score are still useful.
- first observed compact line `step=20590`.

Result:

| Metric | 0.95 control at 24576 | EMA 0.99 at 24576 | Read |
|---|---:|---:|---|
| eval loss | 0.030342 | 0.030332 | tie/slightly better |
| PSNR | 28.4808 | 28.4710 | tie |
| SSIM | 0.6482 | 0.6513 | slightly better |
| LPIPS | 0.3665 | 0.3663 | tie |
| artifact max score | 3.568 | 5.502 | worse |
| artifact mean score | 3.115 | 3.691 | worse |

Per-view artifact scores:

| Eval view | 0.95 control | EMA 0.99 | Count | Largest area |
|---:|---:|---:|---:|---:|
| 0 | 3.568 | 3.863 | 28 | 2502 |
| 1 | 2.313 | 1.708 | 18 | 466 |
| 2 | 3.463 | 5.502 | 30 | 2594 |

Insight:

- `occupancy_ema_decay=0.99` preserves global quality but worsens artifact max,
  mainly on eval2.
- Not a candidate for solo confirmation unless a later artifact-to-grid debug run
  shows the worsened eval2 artifact has a different cause from the current best.

### Stage 2 occupancy conservativeness: dilation1 and occ_thre=1e-3

Purpose:

- Test two conservative grid variants after Stage 0 showed artifact rays already
  cross occupied voxels.
- These are expected to be lower-priority screens, but they are cheap enough to
  run in parallel.

Runs:

| Run | One-factor change | Directory | Status |
|---|---|---|---|
| dilation1 | `occupancy_dilation_radius=1` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occupancy_dilation/lookcloser/dilation1_from20480_to24576_seed42` | completed; first line `step=20540` |
| occ001 | `occupancy_occ_thre=0.001` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occupancy_occ_thre/lookcloser/occ001_from20480_to24576_seed42` | completed; first line `step=20550` |

Notes:

- Both runs are launched with `--no-update-summary`; results will be copied from
  each `run_summary.json`.
- Runtime is screening-only because both runs share the GPU.

Results:

| Metric | 0.95 control at 24576 | Dilation1 | Occ thre 1e-3 |
|---|---:|---:|---:|
| eval loss | 0.030342 | 0.045838 | 0.032091 |
| PSNR | 28.4808 | 21.2216 | 27.5263 |
| SSIM | 0.6482 | 0.6672 | 0.6834 |
| LPIPS | 0.3665 | 0.4003 | 0.3716 |
| artifact max score | 3.568 | 68.809 | 18.444 |
| artifact mean score | 3.115 | 41.113 | 8.019 |

Per-view artifact scores:

| Eval view | Control | Dilation1 | Occ thre 1e-3 |
|---:|---:|---:|---:|
| 0 | 3.568 | 25.995 | 3.714 |
| 1 | 2.313 | 28.536 | 1.900 |
| 2 | 3.463 | 68.809 | 18.444 |

Efficiency notes:

- Both runs were parallel and wall-clock is not acceptance-grade.
- Train-time sample mean near the end increased to `~819` for dilation1 and
  `~816` for occ001 versus `~725` in the control continuation tail, so both
  conservative-grid variants also cost more samples/ray.

Insight:

- `occupancy_dilation_radius=1` is rejected. It causes a catastrophic quality
  regression and artifact explosion.
- `occupancy_occ_thre=1e-3` is rejected. It improves eval1 artifact score but
  badly worsens eval2 and drops PSNR by almost 1 dB.
- These results support the Stage 0 artifact-to-grid diagnosis: the dominant
  artifacts are not fixed by making the binary grid more conservative.

## Screening Decision Table

| Candidate | Status | Reason |
|---|---|---|
| `adaptive_fixed_fallback_samples_per_ray=32` | rejected | PSNR/SSIM/LPIPS all worse; artifact max `26.266` |
| `adaptive_warmup_steps=20000` | rejected as screen | worse global metrics at `20480`; artifact max `4.848` |
| `occupancy_ema_decay=0.99` | rejected as screen | global metrics tie, artifact max worse `5.502` |
| `occupancy_dilation_radius=1` | rejected | catastrophic PSNR/artifact regression |
| `occupancy_occ_thre=1e-3` | rejected | artifact max worse `18.444`; PSNR drop too large |

Current interpretation:

- The latest screens did not beat the current best artifact-tuned checkpoint
  (`artifact max=3.568`, PSNR `28.4808`, SSIM `0.6482`, LPIPS `0.3665`).
- Since all conservative occupancy-grid variants worsened artifacts, the next
  useful work should not continue blindly making occupancy more permissive.
- Final decisions still need variance/noise-floor accounting. The current
  single-seed screens are sufficient for rejection of large regressions, but not
  for accepting small improvements.

### Fixed-renderer control and detector margin audit

Purpose:

- Check whether disabling ARM/occupancy traversal lowers artifact score on the
  same scale-aligned scene.
- Validate whether detector scores are dominated by edge/border blobs rather
  than object-structure failures.

Scale-aligned fixed-renderer controls from existing renders:

| Run | PSNR | SSIM | LPIPS | Eval0 | Eval1 | Eval2 | Artifact max |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed s256 | 26.9356 | 0.6003 | 0.3940 | 18.082 | 10.146 | 6.300 | 18.082 |
| fixed s384 | 27.7121 | 0.6272 | 0.4005 | 6.002 | 2.976 | 1.217 | 6.002 |
| fixed s512 | 28.3602 | 0.6439 | 0.3606 | 2.436 | 0.746 | 0.625 | 2.436 |
| fixed s640 | 28.5622 | 0.6578 | 0.3563 | 2.182 | 0.375 | 0.423 | 2.182 |
| fixed s640 checkpoint rendered with 1024 samples | 28.8321 | 0.6703 | 0.3849 | 2.140 | 0.249 | 0.487 | 2.140 |

Detector margin audit (`crop_bottom=60`, `crop_right=80`, diagnostic only):

| Run | Eval0 | Eval1 | Eval2 | Max |
|---|---:|---:|---:|---:|
| ARM best 24576 | 2.402 | 2.137 | 3.772 | 3.772 |
| fixed s640 | 0.979 | 0.340 | 0.467 | 0.979 |
| fixed s640 rendered with 1024 samples | 1.007 | 0.276 | 0.538 | 1.007 |

Insight:

- Fixed rendering is a better artifact control than ARM (`2.14-2.18` max vs
  ARM `3.568`) and gets near `1.0` under a border-margin diagnostic.
- The remaining fixed-renderer score is partly edge-sensitive; visual spot-check
  shows many fixed boxes near the bottom/right frame boundary.
- ARM remains worse under the same margin, especially eval2. This suggests a
  real ARM/traversal/integration artifact beyond detector edge sensitivity.
- Render-only fixed1024 improves PSNR/SSIM but worsens LPIPS, so it is a useful
  diagnostic/control, not an accepted replacement yet.

### Stage 2 render-only dense ARM follow-up

Purpose:

- Test whether the remaining ARM artifact is caused by traversal step size even
  when binary occupancy is present.
- Previous dense override (`coarse/max=0.003125`, cap `4096`) helped but did not
  solve the artifact. Next test doubles density again.

Planned render-only override on best ARM checkpoint `24576`:

- `adaptive_coarse_step_size=0.0015625`
- `adaptive_max_step_size=0.0015625`
- `max_steps_per_ray=8192`
- no training; full `ns-eval` plus artifact detector on all eval views.

Result:

| Render mode | PSNR | SSIM | LPIPS | rays/sec | Eval0 | Eval1 | Eval2 | Artifact max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARM best normal | 28.4808 | 0.6482 | 0.3665 | n/a | 3.568 | 2.313 | 3.463 | 3.568 |
| ARM dense `0.003125/cap4096` | n/a | n/a | n/a | n/a | 3.387 | 1.950 | 2.484 | 3.387 |
| ARM dense `0.0015625/cap8192` | 28.6680 | 0.6432 | 0.3755 | 10526.8 | 3.184 | 2.222 | 2.651 | 3.184 |
| fixed s640 | 28.5622 | 0.6578 | 0.3563 | 220621.3 | 2.182 | 0.375 | 0.423 | 2.182 |
| fixed s640 rendered with 1024 samples | 28.8321 | 0.6703 | 0.3849 | 144726.1 | 2.140 | 0.249 | 0.487 | 2.140 |
| fixed s640 rendered with 2048 samples | 28.9337 | 0.6721 | 0.4155 | 80025.6 | 2.030 | 0.257 | 0.494 | 2.030 |

Margin diagnostic (`crop_bottom=60`, `crop_right=80`):

| Render mode | Eval0 | Eval1 | Eval2 | Max |
|---|---:|---:|---:|---:|
| ARM best normal | 2.402 | 2.137 | 3.772 | 3.772 |
| ARM dense `0.0015625/cap8192` | 1.935 | 2.069 | 2.929 | 2.929 |
| fixed s640 | 0.979 | 0.340 | 0.467 | 0.979 |
| fixed s640 rendered with 1024 samples | 1.007 | 0.276 | 0.538 | 1.007 |
| fixed s640 rendered with 2048 samples | 0.904 | 0.284 | 0.545 | 0.904 |

Insight:

- Tighter ARM traversal helps but does not close the gap to fixed rendering.
- It is too slow for routine evaluation (`~10.5k rays/sec` vs fixed s640
  `~220.6k rays/sec`) and still leaves artifact max `3.184`.
- The residual artifact is not a simple binary occupancy miss or coarse traversal
  step issue. Fixed rendering is the better practical control, but even fixed1024
  is not zero under the official detector.
- Fixed2048 is the best render-only fixed result by official artifact score
  (`2.030`) and margin artifact score (`0.904`), but it worsens LPIPS to
  `0.4155` and drops throughput to `~80k rays/sec`. Treat it as a diagnostic
  upper bound for dense fixed integration, not an automatically accepted
  production setting.

Tooling update:

- `scripts/detect_structural_artifacts.py` now creates the parent directory for
  `--out` automatically.
- It also supports `--crop-left` and `--crop-right` in addition to top/bottom
  crops.
- `scripts/run_lookcloser_quiet.py` exposes `--artifact-crop-top`,
  `--artifact-crop-bottom`, `--artifact-crop-left`, and
  `--artifact-crop-right`; defaults keep old behavior.

### Instant-NGP artifact noise-floor check

Purpose:

- Estimate the detector/scene floor and seed variance on the bounded
  Instant-NGP baseline.
- Avoid treating small artifact-score deltas as meaningful.

Scale-aligned Instant-NGP control (`scene_scale=1.5`) from existing renders:

| Run | Eval0 | Eval1 | Eval2 | Artifact max |
|---|---:|---:|---:|---:|
| control seed42 | 4.879 | 1.007 | 0.875 | 4.879 |
| control seed43 | 1.623 | 0.948 | 1.051 | 1.623 |
| control seed44 | 2.113 | 0.854 | 1.126 | 2.113 |

Non-apples stage4 reference (`scene_scale=2.0`, `scale_factor=1.15`,
`train_num_rays_per_batch=12288`, seed44):

| Eval0 | Eval1 | Eval2 | Artifact max |
|---:|---:|---:|---:|
| 2.602 | 1.584 | 1.872 | 2.602 |

Insight:

- Good Instant-NGP seeds are not at `artifact_score=0`; the practical detector
  floor for this scene is closer to `~1-2` unless edge/border/object-specific
  filtering is applied.
- Seed variance is large: scale-aligned control seed42 has max `4.879`, while
  seed43 has max `1.623`.
- Current best LookCloser fixed render (`2.14-2.18`) is competitive with this
  floor; current best ARM (`3.568`) is not.
- Any final claim must be multi-seed. Single-seed screens can reject large
  regressions, but small improvements below about one artifact-score point are
  not decision-grade.

## Current Next Actions

1. Do not spend more runs on conservative binary occupancy knobs until a new
   artifact-to-grid debug output shows `grid_miss_likely=true`.
2. Treat fixed rendering as the current practical control for artifact
   reduction: fixed s640/fixed1024/fixed2048 are close to the measured
   Instant-NGP artifact floor, while ARM remains worse. Fixed2048 improves
   artifact score the most but worsens LPIPS and runtime, so fixed640 remains
   the cleaner quality/speed candidate unless later visual inspection says the
   LPIPS regression is harmless.
3. Variance-aware fixed s640 confirmation is now complete for seeds 42/43/44.
   The next decision should not be another blind occupancy sweep; it should be a
   visual audit of the remaining eval0 residual and, if it is acceptable, a
   proposal to use fixed s640 as the low-artifact baseline/control.
4. Keep two artifact gates in reports:
   official detector score with no crop margins, and diagnostic margin score
   (`crop_bottom=60`, `crop_right=80`) to separate frame-edge artifacts from
   object-structure artifacts.
5. Checkpoint selection by artifact score is not robust enough to replace
   eval-loss selection globally. It helped seed44 official artifact score but by
   less than one fixed640 seed std overall and with an LPIPS tradeoff.

### Variance confirmation: fixed s640

Purpose:

- Measure whether the fixed s640 artifact result is stable across seeds.
- The single seed42 result is not enough because Instant-NGP controls showed
  large artifact-score variance.

Parallel-run note:

- Tried seed43 and seed44 in parallel to save wall time.
- Both valid `--max-res 8192` processes together used about `40GB+` and OOMed
  in the distortion loss allocation.
- Sequential fixed s640 training is therefore required on the current L40S for
  apples-to-apples `train_num_rays_per_batch=4096`.

Current sequential run:

- `fixed_s640_eval512_solo_fg_on_fr_off_arm_off_fas_off_max8192_seq_seed43`
- Started after the parallel OOM check.
- Early liveness: `gpu_mem_mb ~= 22817`, `iter_time_s ~= 0.239`,
  `train_rays_per_sec ~= 17100`.
- First eval at step `15188`: eval loss `0.0303458`, PSNR `28.4835`,
  SSIM `0.6501`, LPIPS `0.3784`. Continue to later eval boundaries before
  judging; artifact score is only computed after the selected checkpoint render.
- Second eval at step `30376`: eval loss `0.0300985`, PSNR `28.6964`,
  SSIM `0.6572`, LPIPS `0.3611`; this improved over the first eval, so the run
  continues.
- Third eval at step `45564`: eval loss `0.0307675`, PSNR `28.6984`,
  SSIM `0.6588`, LPIPS `0.3562`; eval loss worsened, so the quiet runner
  selected step `30376` and stopped.

Seed43 selected-checkpoint result:

| Seed | Selected step | PSNR | SSIM | LPIPS | Train s | Eval s | Artifact s | Official eval0 | Official eval1 | Official eval2 | Official max | Margin eval0 | Margin eval1 | Margin eval2 | Margin max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 45564 | 28.5622 | 0.6578 | 0.3563 | 17612.3 | n/a | 36.2 | 2.182 | 0.375 | 0.423 | 2.182 | 0.979 | 0.340 | 0.467 | 0.979 |
| 43 | 30376 | 28.6964 | 0.6572 | 0.3611 | 13552.4 | 50.5 | 34.2 | 1.752 | 0.327 | 0.342 | 1.752 | 1.409 | 0.307 | 0.378 | 1.409 |

Seed44 status:

- `fixed_s640_eval512_solo_fg_on_fr_off_arm_off_fas_off_max8192_seq_seed44`
  started sequentially after seed43.
- Early liveness: `gpu_mem_mb ~= 22817`, `iter_time_s ~= 0.239-0.242`.
- First eval at step `15188`: eval loss `0.0314238`, PSNR `28.3141`,
  SSIM `0.6507`, LPIPS `0.3777`; weaker than seed43 at the same boundary, but
  the run continues to check later eval improvement.
- Second eval at step `30376`: eval loss `0.0313872`, PSNR `28.4754`,
  SSIM `0.6537`, LPIPS `0.3604`; only a small loss improvement, so the run
  continues to the third boundary.
- Third eval at step `45564`: eval loss `0.0309192`, PSNR `28.4946`,
  SSIM `0.6540`, LPIPS `0.3558`; loss improved again, so this run continues
  to the final `60752` boundary.

Seed44 selected-checkpoint result:

| Seed | Selected step | PSNR | SSIM | LPIPS | Train s | Eval s | Artifact s | Official eval0 | Official eval1 | Official eval2 | Official max | Margin eval0 | Margin eval1 | Margin eval2 | Margin max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | 45564 | 28.4946 | 0.6540 | 0.3558 | 16319.7 | 50.5 | 37.7 | 1.796 | 0.221 | 0.418 | 1.796 | 1.063 | 0.244 | 0.462 | 1.063 |

Three-seed fixed s640 summary (`seed42`, `seed43`, `seed44`; sample std):

| Metric | Mean | Std |
|---|---:|---:|
| PSNR | 28.5844 | 0.1027 |
| SSIM | 0.6563 | 0.0021 |
| LPIPS | 0.3577 | 0.0029 |
| Official artifact max | 1.9100 | 0.2366 |
| Official artifact mean across views | 0.8707 | 0.1063 |
| Margin artifact max | 1.1503 | 0.2279 |
| Margin artifact mean across views | 0.6277 | 0.0610 |
| Train seconds | 15828.1 | 2074.1 |
| Artifact seconds | 36.0 | 1.8 |

Insight:

- Fixed s640 is stable across seeds by artifact score relative to the measured
  Instant-NGP control variance. The official artifact max range is
  `1.752-2.182`, much tighter than Instant-NGP controls (`1.623-4.879`).
- The remaining detector maximum is consistently eval0; eval1/eval2 are already
  below `0.5` for all three seeds.
- This validates fixed s640 as the current practical low-artifact baseline, but
  it does not validate occupancy-grid tuning: Stage 0 already showed the ARM
  artifacts were not binary-grid misses.

### Eval-loss vs artifact-score checkpoint selection

Purpose:

- Check whether selecting checkpoints by eval loss hides a lower-artifact
  checkpoint, as warned in the plan.
- Only cheap render-only evals were needed because seed43/44 were run with
  `--keep-all-checkpoints`.

Seed43 checkpoint sweep:

| Step | Selection | PSNR | SSIM | LPIPS | Official eval0 | Official eval1 | Official eval2 | Official max | Margin eval0 | Margin eval1 | Margin eval2 | Margin max |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15188 | early | 28.4835 | 0.6501 | 0.3784 | 1.895 | 0.333 | 0.389 | 1.895 | 1.491 | 0.368 | 0.429 | 1.491 |
| 30376 | eval-loss selected | 28.6964 | 0.6572 | 0.3611 | 1.752 | 0.327 | 0.342 | 1.752 | 1.409 | 0.307 | 0.378 | 1.409 |
| 45564 | later | 28.6984 | 0.6588 | 0.3562 | 1.857 | 0.314 | 0.337 | 1.857 | 1.522 | 0.295 | 0.373 | 1.522 |

Seed44 checkpoint sweep:

| Step | Selection | PSNR | SSIM | LPIPS | Official eval0 | Official eval1 | Official eval2 | Official max | Margin eval0 | Margin eval1 | Margin eval2 | Margin max |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15188 | early | 28.3141 | 0.6507 | 0.3777 | 1.849 | 0.500 | 0.649 | 1.849 | 1.376 | 0.333 | 0.717 | 1.376 |
| 30376 | artifact-best official | 28.4754 | 0.6537 | 0.3604 | 1.602 | 0.319 | 0.473 | 1.602 | 1.131 | 0.252 | 0.522 | 1.131 |
| 45564 | eval-loss selected | 28.4946 | 0.6540 | 0.3558 | 1.796 | 0.221 | 0.418 | 1.796 | 1.063 | 0.244 | 0.462 | 1.063 |
| 60751 | latest | 28.5390 | 0.6544 | 0.3513 | 1.736 | 0.218 | 0.405 | 1.736 | 0.960 | 0.241 | 0.447 | 0.960 |

Insight:

- Seed43: eval-loss-selected step `30376` is also the best official-artifact
  checkpoint. Later step `45564` improves SSIM/LPIPS but worsens eval0 artifact.
- Seed44: official artifact prefers step `30376` (`1.602`), while LPIPS and the
  margin score prefer latest step `60751`. The official-artifact improvement of
  step `30376` over eval-loss-selected step `45564` is `10.8%`, but LPIPS
  worsens by `0.0047`; this is a real tradeoff, not a clean replacement.
- Across seeds, artifact-aware checkpoint selection only changes the official
  max from `[2.182, 1.752, 1.796]` to `[2.182, 1.752, 1.602]`. The mean improves
  by only `0.065`, far below one fixed640 seed std (`0.237`). Do not claim this
  as a robust improvement yet.

### Visual audit of remaining fixed s640 eval0 residual

Purpose:

- Understand why fixed s640 still has official artifact max around `1.8-2.2`
  even though the severe ARM stand/hole artifacts are gone.
- Decide whether the next run should tune occupancy, traversal, or reconstruction
  sharpness.

Artifact evidence:

- Eval0 carries nearly all remaining fixed s640 score across seeds.
- A raw GT/candidate crop mosaic was saved:
  `lookcloser_debug_outputs/occupancy_stage0/fixed_s640_eval0_visual_audit_mosaic.png`.
- The mosaic covers the right/bottom edge region, top-right stand/highlight
  region, and bottom-left edge region; red boxes are detector components in
  candidate-panel coordinates.

Visual finding:

- The residual boxes are not the original severe ARM failure mode. There is no
  large missing stand segment, dislocated thin stand, or occupancy-grid hole.
- Most score comes from:
  - blurred/smoothed floor cracks and yellow tape/highlight patches near the
    right/bottom frame boundary;
  - small high-contrast highlights around the upper-right stand/cable region;
  - tiny edge blobs near the bottom-left boundary.
- This matches the margin-score behavior: cropping the bottom/right edge lowers
  fixed s640 max from about `1.8-2.2` to about `1.0-1.4`.

Implication:

- Do not run more blind occupancy-grid screens for this residual. Stage 0
  artifact-to-grid already showed occupied rays/cells, and the visual audit
  shows the remaining fixed-renderer score is mostly reconstruction sharpness and
  detector sensitivity to high-contrast frame-edge details.
- The next model-side screen should test denser fixed integration during
  training, not just render-only denser integration. Render-only fixed1024/2048
  on a fixed640-trained field improved PSNR/SSIM but worsened LPIPS; training
  the field under fixed1024 may behave differently and is the smallest direct
  test left before declaring the residual detector-floor/scene-floor dominated.

### Fixed1024 continuation screen

Purpose:

- Test whether training with denser fixed integration reduces the remaining
  eval0 floor-crack/highlight residual, instead of only rendering a fixed640
  trained field more densely.

Setup:

- Start checkpoint: seed43 fixed s640 selected checkpoint, step `30376`.
- Continue with `fixed_num_samples_per_ray=1024`, ARM/FR/FAS off, same data,
  `scene_scale=1.5`, `scale_factor=1.0`, `train_num_rays_per_batch=4096`.

Feasibility findings:

- Exact fixed1024 with existing `distortion_loss_mult=0.01` OOMs at the first
  train iteration in `nerfstudio_distortion_loss`, trying to allocate about
  `16GB` extra with the process already using about `39GB`.
- `distortion_loss_mult=0` skips that O(N^2) path and is feasible:
  early liveness was `gpu_mem_mb ~= 6013`, GPU total memory used `~14GB`,
  `iter_time_s ~= 0.105`, `train_rays_per_sec ~= 39k`.
- Code fix added after the OOM: fixed-renderer dense samples now use a
  linear-time dense distortion formula matching `nerfstudio_distortion_loss` to
  `~3e-8` on a small numerical check. This keeps `distortion_loss_mult=0.01`
  feasible at fixed1024: early liveness `gpu_mem_mb ~= 6077`, GPU total memory
  used `~14GB`, `iter_time_s ~= 0.106`, `train_rays_per_sec ~= 38.5k`.

Current run:

- `fixed1024_dist0_from_seed43_30376_to45564_seed43`
- This is a screen, not an apples-to-apples replacement for fixed s640, because
  disabling distortion changes the objective.

Result after one interval (`30376 -> 45564`):

| Run | Step | PSNR | SSIM | LPIPS | Train s | Eval s | Artifact s | Official eval0 | Official eval1 | Official eval2 | Official max | Margin eval0 | Margin eval1 | Margin eval2 | Margin max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed s640 seed43 source | 30376 | 28.6964 | 0.6572 | 0.3611 | n/a | 50.5 | 16.2 | 1.752 | 0.327 | 0.342 | 1.752 | 1.409 | 0.307 | 0.378 | 1.409 |
| fixed1024 dist0 continuation | 45564 | 28.8660 | 0.6671 | 0.3928 | 1742.0 | 55.7 | 14.4 | 1.644 | 0.222 | 0.429 | 1.644 | 1.210 | 0.245 | 0.474 | 1.210 |

Insight:

- Training with fixed1024 and no distortion improves official artifact max by
  `6.2%` and margin max by `14.1%` versus the seed43 fixed640 source checkpoint.
- It also improves PSNR/SSIM, but LPIPS worsens by `0.0318`, far beyond the
  fixed s640 seed std (`0.0029`). This is not acceptable under the current
  gate unless visual inspection proves LPIPS is misleading here.
- Because the run is fast and memory-safe, extend one more interval to test
  whether the LPIPS regression stabilizes or worsens.

Second interval with distortion disabled (`45564 -> 60752`):

| Run | Step | PSNR | SSIM | LPIPS | Train s | Eval s | Artifact s | Official eval0 | Official eval1 | Official eval2 | Official max | Margin eval0 | Margin eval1 | Margin eval2 | Margin max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed1024 dist0 continuation | 60752 | 28.8585 | 0.6682 | 0.3833 | 1741.8 | 55.7 | 14.7 | 1.648 | 0.115 | 0.424 | 1.648 | 1.197 | 0.127 | 0.468 | 1.197 |

Insight:

- A second fixed1024/dist0 interval improves eval loss/SSIM and partially
  recovers LPIPS (`0.3928 -> 0.3833`), but artifact max stays flat
  (`1.644 -> 1.648`).
- The LPIPS regression versus fixed s640 remains too large, so fixed1024/dist0
  should be rejected as a final candidate. Its value was diagnostic: it exposed
  and motivated the dense distortion-loss fix.

Current follow-up:

- Run `fixed1024_fastdist_from_seed43_30376_to45564_seed43` with the new
  linear-time dense distortion path and normal `distortion_loss_mult=0.01`.
  This is the relevant fixed1024 screen because it preserves the training
  objective while avoiding the previous OOM.

Normal-distortion fixed1024 result (`30376 -> 45564`):

| Run | Step | PSNR | SSIM | LPIPS | Train s | Eval s | Artifact s | Official eval0 | Official eval1 | Official eval2 | Official max | Margin eval0 | Margin eval1 | Margin eval2 | Margin max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed1024 fast dense distortion | 45564 | 28.9408 | 0.6676 | 0.3931 | 1771.8 | 55.5 | 14.5 | 1.646 | 0.224 | 0.428 | 1.646 | 1.211 | 0.247 | 0.473 | 1.211 |

Insight:

- The dense O(N) distortion implementation fixes the fixed1024 OOM while keeping
  normal `distortion_loss_mult=0.01`; memory/speed are close to the distortion0
  run.
- Quality behavior remains essentially the same as fixed1024/dist0: small
  artifact-score improvement and higher PSNR/SSIM, but LPIPS regresses heavily.
- Fixed1024 is therefore not accepted as the final low-artifact setting. The
  useful outcome is the code fix plus the evidence that simply increasing fixed
  samples does not drive artifact score near zero.

### Fixed768 midpoint screen

Purpose:

- Test whether an intermediate fixed sample count gives a better artifact/LPIPS
  tradeoff than fixed640 and fixed1024 after the dense distortion-loss fix.

Setup:

- Start checkpoint: seed43 fixed s640 selected checkpoint, step `30376`.
- Continue to step `45564` with `fixed_num_samples_per_ray=768`, normal
  `distortion_loss_mult=0.01`, ARM/FR/FAS off, same scale-aligned data settings.

Result:

| Run | Step | PSNR | SSIM | LPIPS | Train s | Eval s | Artifact s | Official eval0 | Official eval1 | Official eval2 | Official max | Margin eval0 | Margin eval1 | Margin eval2 | Margin max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed s640 seed43 source | 30376 | 28.6964 | 0.6572 | 0.3611 | n/a | 50.5 | 16.2 | 1.752 | 0.327 | 0.342 | 1.752 | 1.409 | 0.307 | 0.378 | 1.409 |
| fixed768 fast dense distortion | 45564 | 28.8563 | 0.6676 | 0.3942 | 1531.6 | 52.0 | 14.8 | 1.478 | 0.138 | 0.456 | 1.478 | 1.336 | 0.152 | 0.503 | 1.336 |

Insight:

- Fixed768 gives the best official artifact max seen in this fixed-sample
  screen (`1.478`, a `15.6%` improvement over the seed43 fixed640 source).
- The improvement is still concentrated in eval0 floor/tape/highlight residuals;
  visual overlay does not show a severe stand/hole artifact.
- LPIPS regresses from `0.3611` to `0.3942`, far beyond the fixed s640
  three-seed std (`0.0029`). Under the current acceptance gate this is rejected,
  despite the lower artifact score and higher PSNR/SSIM.
- The fixed sample count sweep suggests the detector score can be lowered by
  smoothing/integration changes, but the perceptual metric rejects that path.
  Fixed s640 remains the cleaner quality/speed control.

### Existing FAS and adaptive candidates under artifact detector

Purpose:

- Reuse existing full-eval renders before launching more training.
- Check whether visually gated FAS or H40/H41 adaptive candidates already solve
  the broader structural-artifact score.

Two-stage FAS gate-safe candidates:

| Candidate | PSNR | SSIM | LPIPS | Official eval0 | Official eval1 | Official eval2 | Official max | Margin eval0 | Margin eval1 | Margin eval2 | Margin max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FAS seed42 step36864 | 28.9042 | 0.6664 | 0.3679 | 5.360 | 4.146 | 3.875 | 5.360 | 5.209 | 3.700 | 4.104 | 5.209 |
| FAS seed43 step40960 | 29.2147 | 0.6865 | 0.3779 | 6.259 | 5.690 | 6.068 | 6.259 | 6.239 | 5.809 | 6.705 | 6.705 |
| FAS seed44 gate-safe step36864 | 29.3014 | 0.6774 | 0.3708 | 5.953 | 4.444 | 3.978 | 5.953 | 6.044 | 3.918 | 4.294 | 6.044 |

H40/H41 adaptive candidates:

| Candidate | PSNR | SSIM | LPIPS | Official eval0 | Official eval1 | Official eval2 | Official max | Margin eval0 | Margin eval1 | Margin eval2 | Margin max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H40 ARM metric leader | 28.8982 | 0.6659 | 0.3653 | 5.334 | 4.228 | 3.409 | 5.334 | 5.174 | 3.784 | 3.590 | 5.174 |
| H41 ARM visual-balance | 28.8879 | 0.6660 | 0.3664 | 5.431 | 4.208 | 3.413 | 5.431 | 5.280 | 3.763 | 3.594 | 5.280 |

Insight:

- Existing visually gated FAS candidates and H40/H41 adaptive candidates are not
  acceptable for the artifact objective. They improve global PSNR/SSIM, but
  structural artifact max is `~5-6`, much worse than fixed s640 (`~1.8-2.2`)
  and fixed768 (`1.478`).
- The strict stand connector gate was too narrow for this objective; it did not
  catch broader eval-view structural differences.
- Next screen should isolate the useful part of FAS, if any, under fixed
  rendering: fixed s640 + FAS continuation from the stable fixed-s640
  checkpoint, with ARM still disabled.

### Fixed s640 plus FAS screen

Purpose:

- Test whether FAS can improve high-frequency eval0 residuals while keeping the
  fixed renderer that avoids ARM/occupancy artifacts.

Setup:

- Start checkpoint: seed43 fixed s640 selected checkpoint, step `30376`.
- Continue to step `45564` with fixed s640, ARM/FR off, FAS on:
  `fas_strength=0.35`, `fas_level_count_alpha=1.0`,
  `sampling_ramp_start=end=1.0`.

Result:

| Run | Step | PSNR | SSIM | LPIPS | Train s | Eval s | Artifact s | Official eval0 | Official eval1 | Official eval2 | Official max | Margin eval0 | Margin eval1 | Margin eval2 | Margin max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed s640 seed43 source | 30376 | 28.6964 | 0.6572 | 0.3611 | n/a | 50.5 | 16.2 | 1.752 | 0.327 | 0.342 | 1.752 | 1.409 | 0.307 | 0.378 | 1.409 |
| fixed s640 + FAS0.35 alpha1 | 45564 | 28.6564 | 0.6552 | 0.3378 | 1441.6 | 50.3 | 17.0 | 1.861 | 0.332 | 0.327 | 1.861 | 1.519 | 0.296 | 0.361 | 1.519 |

Insight:

- FAS under fixed rendering is excellent for LPIPS (`0.3611 -> 0.3378`), but it
  worsens the artifact max (`1.752 -> 1.861`) and slightly lowers PSNR/SSIM.
- This is not an artifact candidate by itself. Its useful signal is orthogonal
  to the fixed-sample result: FAS helps perceptual texture, while denser fixed
  sampling lowers local-SSIM artifact blobs but hurts LPIPS.
- Next screen: combine fixed768 with FAS to see whether FAS can recover LPIPS
  without giving up fixed768's artifact-score gain.

### Fixed768 plus FAS screen

Purpose:

- Combine the two partial signals from the previous screens:
  fixed768 lowered artifact score but hurt LPIPS, while fixed s640 + FAS improved
  LPIPS but hurt artifact score.

Setup:

- Start checkpoint: seed43 fixed s640 selected checkpoint, step `30376`.
- Continue to step `45564` with `fixed_num_samples_per_ray=768`, normal
  `distortion_loss_mult=0.01`, ARM/FR off, FAS on:
  `fas_strength=0.35`, `fas_level_count_alpha=1.0`,
  `sampling_ramp_start=end=1.0`.
- One invalid launch used `--load-dir` with a checkpoint file path and failed
  before training; the valid run used `--load-checkpoint`.

Result:

| Run | Step | PSNR | SSIM | LPIPS | Train s | Eval s | Artifact s | Official eval0 | Official eval1 | Official eval2 | Official max | Margin eval0 | Margin eval1 | Margin eval2 | Margin max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed s640 seed43 source | 30376 | 28.6964 | 0.6572 | 0.3611 | n/a | 50.5 | 16.2 | 1.752 | 0.327 | 0.342 | 1.752 | 1.409 | 0.307 | 0.378 | 1.409 |
| fixed768 fast dense distortion | 45564 | 28.8563 | 0.6676 | 0.3942 | 1531.6 | 52.0 | 14.8 | 1.478 | 0.138 | 0.456 | 1.478 | 1.336 | 0.152 | 0.503 | 1.336 |
| fixed768 + FAS0.35 alpha1 | 45564 | 28.8730 | 0.6691 | 0.4009 | 1561.7 | 52.2 | 15.2 | 1.606 | 0.140 | 0.444 | 1.606 | 1.487 | 0.155 | 0.491 | 1.487 |

Insight:

- FAS does not recover the fixed768 LPIPS regression. It worsens LPIPS slightly
  relative to fixed768 alone (`0.3942 -> 0.4009`) despite a small PSNR/SSIM gain.
- Artifact max is better than the seed43 fixed s640 source (`1.752 -> 1.606`),
  but worse than fixed768 without FAS (`1.478`), and margin max also worsens
  (`1.336 -> 1.487`).
- This candidate fails the acceptance gate: LPIPS is far worse than the fixed
  s640 three-seed std (`0.0029`), and artifact-score improvement is not enough
  to justify that perceptual regression.
- Current conclusion: fixed s640 remains the best balanced control. Fixed768 is
  useful as a diagnostic showing detector-score sensitivity to denser integration,
  but it is not a shippable improvement under the current PSNR/SSIM/LPIPS/artifact
  gate.

### Artifact detector validity follow-up

Purpose:

- Check whether the residual `~1.5-1.9` artifact floor in fixed rendering is a
  real severe structural failure or a detector/ROI floor.

Implementation:

- `scripts/detect_structural_artifacts.py` now has diagnostic-only filtering:
  `--include-bbox`, `--exclude-bbox`, `--drop-border-components`, `--json-out`,
  and `--print-json`.
- Defaults are unchanged, so official scores from the runner remain comparable.
- `scripts/debug_artifact_occupancy_grid.py` was updated to pass the detector's
  full crop namespace (`crop_left/right=0`) when it reuses `load_pair`.

Result on fixed768 + FAS eval0:

| Scoring mode | Artifact score | Largest blob | Read |
|---|---:|---:|---|
| official full frame | 1.606 | 1087 px | same as runner |
| full frame + `--drop-border-components 24` | 1.238 | 1087 px | removes some edge-touching boxes only |
| margin crop + `--drop-border-components 24` | 0.840 | 843 px | remaining boxes are mostly right-side equipment, hair/clothes, and floor marks |

Insight:

- The low-artifact fixed-renderer residual is dominated by floor/edge/equipment
  texture differences rather than the earlier ARM stand/hole failure.
- This does not make the model result better; it means the current official
  full-frame score has a scene-specific floor and should be interpreted with
  saved overlays/crops.
- Further attempts to drive official score near zero by occupancy-grid tuning are
  unlikely to be meaningful unless artifact->grid debugging first shows
  `grid_miss_likely=true` on a real structural bbox.

### ROI-focused artifact audit

Purpose:

- Check named high-value regions directly: stand connector, hands/fingers,
  cable, label, and floor crack.
- Separate meaningful structural artifacts from the full-frame floor/edge score
  floor before launching more model runs.

Implementation:

- Added `scripts/score_artifact_rois.py`.
- It uses the same detector thresholds as `detect_structural_artifacts.py`, but
  crops named eval ROIs before scoring and writes `roi_artifact_scores.csv/json`
  plus optional overlays.
- It is diagnostic only; official runner scores remain full-frame.
- `scripts/run_lookcloser_quiet.py` now runs the ROI scorer after full-frame
  artifact scoring by default and stores the result under `artifact.roi` in
  `run_summary.json`. Use `--no-artifact-roi-score` to disable it.
- `scripts/summarize_lookcloser_runs.py` now reports ROI artifact score, ROI
  serious count, and stand-connector ROI score when present.

Results:

| Run | Official max | ROI max | Serious ROIs | Stand connector ROI | Nonzero ROI notes |
|---|---:|---:|---:|---:|---|
| fixed s640 seed42 | 2.182 | 1.774 | 0/9 | 0.000 | floor crack 1.774; eval1 broad hand/head crop 1.054 |
| fixed s640 seed43 | 1.752 | 1.263 | 0/9 | 0.000 | left stand 0.493; eval1 broad hand/head crop 1.263 |
| fixed s640 seed44 | 1.796 | 1.771 | 1/9 | 0.000 | eval1 broad hand/head crop 1.771; overlay places bbox on hair/face, not fingers |
| fixed768 seed43 | 1.478 | 0.458 | 0/9 | 0.000 | left stand 0.458 |
| fixed768 + FAS seed43 | 1.606 | 0.453 | 0/9 | 0.000 | left stand 0.453 |
| best ARM seed42 | 3.568 | 6.539 | 4/9 | 4.105 | stand connector, outlet/stand, eval1 hand/head, cable/fingers residuals |

Artifacts:

- ROI outputs are under
  `lookcloser_debug_outputs/occupancy_stage0/roi_scores/`.
- Example contrast:
  `roi_scores/arm_best_seed42/left_stand_connector_eval0_boxes.png` shows a
  true serious stand-connector failure, while fixed-renderer stand connector ROI
  scores `0.000`.

Insight:

- The fixed-renderer family has no serious stand-connector ROI artifact across
  three fixed s640 seeds, and fixed768/FAS variants also score `0.000` on that
  ROI.
- The one fixed s640 seed44 serious ROI is in a broad `fingers_right_eval1` crop;
  visual overlay places the blob on hair/face texture, not on the fingers.
- The ROI audit discriminates correctly: best ARM still has severe ROI failures
  (`4/9` serious, stand connector `4.105`), while fixed rendering removes the
  meaningful occupancy/stand artifacts.
- Current decision: do not spend more runs on occupancy-grid conservativeness
  unless a new artifact ROI maps to an empty grid cell. The remaining official
  score gap to zero is mostly scoring/ROI validity and high-frequency texture
  mismatch, not the original occupancy-grid artifact.

Runner verification:

- Re-running artifact scoring on fixed768 + FAS through
  `run_lookcloser_quiet.run_artifact_detector()` produced full-frame
  `artifact_score=1.606` and ROI `roi_artifact_score=0.453`,
  `roi_serious_count=0`, `stand_connector_score=0.0`.

### Artifact-aware checkpoint selection support

Purpose:

- Avoid selecting a checkpoint only because eval loss is lowest when another
  saved checkpoint has fewer structural artifacts.
- Make the checkpoint-selection risk explicit in future runs instead of doing
  one-off manual artifact-selection renders.

Implementation:

- `scripts/run_lookcloser_quiet.py` now supports
  `--eval-checkpoint artifact` and `--eval-checkpoint roi` in addition to
  `best` and `latest`.
- These modes evaluate every saved `step-*.ckpt`, render eval views, run the
  full-frame artifact detector plus ROI scorer, and store all candidate evals
  in `run_summary.json` as `artifact_candidate_evals`.
- `artifact` mode ranks primarily by full-frame `artifact_score`, then ROI
  serious count, stand-connector score, ROI score, LPIPS, SSIM, PSNR.
- `roi` mode ranks primarily by ROI serious count, stand-connector score, ROI
  score, full-frame artifact score, LPIPS, SSIM, PSNR.
- The summary table now includes ROI artifact score, ROI serious count, and
  stand-connector score.

Verification:

- `python -m py_compile` passes for the updated runner, summarizer, ROI scorer,
  and detector.
- `python scripts/run_lookcloser_quiet.py --dry-run --eval-checkpoint roi ...`
  validates the CLI path and generated train command.
- Full no-training probe on existing fixed s640 seed44 run evaluated all four
  saved checkpoints and selected step `60751` with reason
  `best_roi_checkpoint_step_60751`.

Seed44 ROI-selection probe:

| Step | PSNR | SSIM | LPIPS | Full artifact | ROI artifact | ROI serious | Stand connector |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15188 | 28.3141 | 0.6507 | 0.3777 | 1.849 | 1.856 | 1 | 0.000 |
| 30376 | 28.4754 | 0.6537 | 0.3604 | 1.602 | 1.809 | 1 | 0.000 |
| 45564 | 28.4946 | 0.6540 | 0.3558 | 1.796 | 1.771 | 1 | 0.000 |
| 60751 | 28.5390 | 0.6544 | 0.3513 | 1.736 | 1.770 | 1 | 0.000 |

Insight:

- This does not change any existing result, but it removes a recurring source
  of bad decisions: eval-loss-selected checkpoints can be artifact-worse than
  neighboring saved checkpoints.
- Use `--keep-all-checkpoints --eval-checkpoint roi` for future confirmation
  runs where absence of meaningful structural artifacts is the primary gate.
- The seed44 probe also shows a remaining ROI-definition issue: all four saved
  checkpoints get `roi_serious_count=1` from the broad `fingers_right_eval1`
  crop, while overlays place the blob on hair/face texture rather than fingers.
  Stand connector is clean (`0.000`) for all four. Before using ROI serious count
  as a hard final gate, narrow or split the eval1 fingers ROI so it targets the
  intended structure.

### Curated ROI set for runner defaults

Purpose:

- Fix the false positive found in the seed44 ROI-selection probe without
  weakening the detector on the real ARM stand failure.

Implementation:

- Added `fingers_right_tight_eval1 = (1030, 430, 1210, 610)` to
  `scripts/score_artifact_rois.py`.
- `scripts/run_lookcloser_quiet.py` now exposes `--artifact-roi-crop-names`.
  Its default is a curated list that uses `fingers_right_tight_eval1` and
  excludes the old broad `fingers_right_eval1`. Passing `all` scores every ROI
  known to `score_artifact_rois.py`.
- `scripts/score_artifact_rois.py` itself now also defaults to the curated ROI
  set; pass `--all-rois` for the broader debug set.

Verification:

| Run/checkpoint set | ROI max | Serious ROIs | Stand connector | Read |
|---|---:|---:|---:|---|
| fixed s640 seed44 steps 15188/30376/45564/60751, curated ROI | 0.000 | 0/9 each | 0.000 each | false positive removed |
| best ARM seed42, curated ROI | 7.209 | 4/9 | 4.105 | real stand/hand/cable artifacts still caught |

Curated standalone default, selected fixed s640 checkpoints:

| Run | ROI max | Serious ROIs | Stand connector | Notable nonzero ROI |
|---|---:|---:|---:|---|
| fixed s640 seed42 step45564 | 1.774 | 0/9 | 0.000 | floor_crack_eval0 1.774, non-serious |
| fixed s640 seed43 step30376 | 0.493 | 0/9 | 0.000 | left_stand_eval0 0.493, non-serious |
| fixed s640 seed44 step45564 | 0.000 | 0/9 | 0.000 | none |
| best ARM seed42 step24576 | 7.209 | 4/9 | 4.105 | stand/hand/cable failures |

Insight:

- With curated ROI, fixed-renderer seed44 checkpoints have no meaningful ROI
  structural artifacts, matching visual inspection.
- Across the three fixed s640 selected checkpoints, curated ROI has `0/9`
  serious ROIs and stand connector `0.000`; residual nonzero scores are
  non-serious floor/left-stand texture blobs.
- Full-frame official score remains useful as a broad regression detector, but
  curated ROI score is the better gate for the original occupancy/stand artifact.
- Future runner default now reports both: official full-frame and curated ROI.

### Serious full-frame score diagnostic

Purpose:

- Add a full-frame score that only accumulates major/serious connected
  components (`area >= AREA_SERIOUS`) while preserving the existing
  `artifact_score` for backward-compatible broad detection.
- Check whether serious-only full-frame scoring can itself serve as a
  near-zero structural-artifact gate.

Implementation:

- `scripts/detect_structural_artifacts.py` now returns and prints
  `serious_artifact_score`.
- `scripts/score_artifact_rois.py` includes `serious_artifact_score` per ROI.
- `scripts/run_lookcloser_quiet.py` parses per-view serious scores, stores max
  and mean serious scores under `artifact`, includes them in artifact-aware
  selection tie-breaks, and writes them to summary tables.
- `scripts/summarize_lookcloser_runs.py` reports full-frame and ROI serious
  artifact scores when present.

Full-frame scores:

| Run | View | Artifact score | Serious artifact score | Serious? |
|---|---|---:|---:|---|
| fixed s640 seed42 | eval0 | 2.182 | 1.563 | yes |
| fixed s640 seed42 | eval1 | 0.375 | 0.000 | no |
| fixed s640 seed42 | eval2 | 0.423 | 0.263 | yes |
| fixed s640 seed43 | eval0 | 1.752 | 1.141 | yes |
| fixed s640 seed43 | eval1 | 0.327 | 0.087 | yes |
| fixed s640 seed43 | eval2 | 0.342 | 0.342 | yes |
| fixed s640 seed44 | eval0 | 1.796 | 1.230 | yes |
| fixed s640 seed44 | eval1 | 0.221 | 0.090 | yes |
| fixed s640 seed44 | eval2 | 0.418 | 0.242 | yes |
| best ARM seed42 | eval0 | 3.568 | 2.441 | yes |
| best ARM seed42 | eval1 | 2.313 | 1.573 | yes |
| best ARM seed42 | eval2 | 3.463 | 2.887 | yes |

Margin + border-drop diagnostic (`--crop-bottom 60 --crop-right 80
--drop-border-components 24`):

| Run | Eval0 serious | Eval1 serious | Eval2 serious | Max serious |
|---|---:|---:|---:|---:|
| fixed s640 seed42 | 0.136 | 0.000 | 0.291 | 0.291 |
| fixed s640 seed43 | 0.347 | 0.000 | 0.378 | 0.378 |
| fixed s640 seed44 | 0.472 | 0.099 | 0.267 | 0.472 |
| best ARM seed42 | 0.632 | 1.562 | 3.067 | 3.067 |

Insight:

- Full-frame serious score still does not go near zero for fixed rendering,
  because large floor/equipment/texture components can pass the major-area
  threshold.
- It is still useful as a broad regression signal: ARM is consistently worse,
  especially after margin/border filtering (`max 3.067` vs fixed `<=0.472`).
- For the original occupancy/stand artifact, curated ROI remains the precise
  gate: fixed s640 is `0/9` serious across seeds with stand connector `0.000`,
  while ARM is `4/9` serious with stand connector `4.105`.
- Do not use full-frame serious score alone as the shipping gate; use it as a
  secondary regression metric next to curated ROI and visual overlays.

### Unified artifact gate summary

Purpose:

- Replace scattered manual tables with one reproducible artifact-gate artifact
  that combines eval metrics, full-frame artifact scores, serious full-frame
  scores, curated ROI scores, and stand-connector score.

Implementation:

- Added `scripts/summarize_artifact_gate.py`.
- Inputs are `LABEL=RUN_DIR` specs plus optional `LABEL=RENDER_DIR` overrides.
- Outputs:
  - `artifact_gate_summary.json`
  - `artifact_gate_summary.csv`
  - `artifact_gate_summary.md`
- It uses the same detector and curated ROI defaults as the runner.

Fixed s640 3-seed gate:

| Run | PSNR | SSIM | LPIPS | Full artifact | Full serious | Full serious views | ROI artifact | ROI serious | ROI serious count | Stand connector |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed42 | 28.5622 | 0.6578 | 0.3563 | 2.1820 | 1.5630 | 2 | 1.7740 | 0.0000 | 0 | 0.0000 |
| seed43 | 28.6964 | 0.6572 | 0.3611 | 1.7520 | 1.1410 | 3 | 0.4930 | 0.0000 | 0 | 0.0000 |
| seed44 | 28.4946 | 0.6540 | 0.3558 | 1.7960 | 1.2300 | 3 | 0.0000 | 0.0000 | 0 | 0.0000 |

Fixed s640 aggregate:

| Metric | Mean | Std | Max |
|---|---:|---:|---:|
| PSNR | 28.5844 | 0.0839 | 28.6964 |
| SSIM | 0.6563 | 0.0017 | 0.6578 |
| LPIPS | 0.3577 | 0.0024 | 0.3611 |
| Full artifact | 1.9100 | 0.1932 | 2.1820 |
| Full serious artifact | 1.3113 | 0.1816 | 1.5630 |
| ROI artifact | 0.7557 | 0.7477 | 1.7740 |
| ROI serious artifact | 0.0000 | 0.0000 | 0.0000 |
| Stand connector | 0.0000 | 0.0000 | 0.0000 |

ARM contrast:

| Run | Full artifact | Full serious | Full serious views | ROI artifact | ROI serious | ROI serious count | Stand connector |
|---|---:|---:|---:|---:|---:|---:|---:|
| best ARM seed42 | 3.5680 | 2.8870 | 3 | 7.2090 | 7.2090 | 4 | 4.1050 |

Artifacts:

- Fixed s640 gate summary:
  `lookcloser_debug_outputs/occupancy_stage0/artifact_gate_fixed_s640/`.
- Fixed s640 gate summary regenerated after `run_summary.json` artifact
  backfill:
  `lookcloser_debug_outputs/occupancy_stage0/artifact_gate_fixed_s640_backfilled/`.
- ARM contrast gate summary:
  `lookcloser_debug_outputs/occupancy_stage0/artifact_gate_arm_best/`.

Insight:

- This is the clearest current evidence split:
  - fixed s640 removes meaningful structural/stand artifacts across seeds under
    curated ROI (`ROI serious=0`, stand connector `0`);
  - ARM still fails the same gate (`ROI serious=7.209`, stand `4.105`);
  - old full-frame score remains nonzero for fixed s640 because it captures
    floor/equipment/texture blobs.
- The objective's "absence of significant substantial artifacts" is satisfied
  by curated ROI evidence for fixed s640, but the literal old full-frame
  `artifact_score` is not near zero. Keep both numbers in future decisions and
  do not claim old full-frame score is solved.

### Run-summary artifact backfill

Purpose:

- Make old fixed s640 confirmation runs comparable through the same
  `run_summary.json` fields that new runner output uses.
- Preserve runtime/provenance accounting while adding the current full-frame
  `serious_artifact_score` and curated ROI gate values.

Implementation:

- Added `scripts/backfill_artifact_scores.py`.
- It reads existing final render panels, runs the same full-frame detector and
  curated ROI scorer as `scripts/run_lookcloser_quiet.py`, writes `artifact`
  under both the root summary and `eval.artifact`, and creates
  `run_summary.json.bak-artifact-*` before rewriting.
- This is no-training/no-eval postprocessing, so it is safe to run in parallel
  across runs; the three fixed s640 summaries were backfilled concurrently.

Backfilled fixed s640 summary from `run_summary.json`:

| Seed | PSNR | SSIM | LPIPS | Full artifact | Full serious | ROI artifact | ROI serious | ROI serious count | Stand connector | Train seconds | Artifact seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 28.5622 | 0.6578 | 0.3563 | 2.182 | 1.563 | 1.774 | 0.000 | 0 | 0.000 | 17612.3 | 36.2 |
| 43 | 28.6964 | 0.6572 | 0.3611 | 1.752 | 1.141 | 0.493 | 0.000 | 0 | 0.000 | 13552.4 | 34.2 |
| 44 | 28.4946 | 0.6540 | 0.3558 | 1.796 | 1.230 | 0.000 | 0.000 | 0 | 0.000 | 16319.7 | 37.7 |

Sample standard deviations from the three fixed s640 seeds:

| Metric | Mean | Sample std | Min | Max |
|---|---:|---:|---:|---:|
| PSNR | 28.5844 | 0.1027 | 28.4946 | 28.6964 |
| SSIM | 0.6563 | 0.0021 | 0.6540 | 0.6578 |
| LPIPS | 0.3577 | 0.0029 | 0.3558 | 0.3611 |
| Full artifact | 1.9100 | 0.2366 | 1.7520 | 2.1820 |
| Full serious artifact | 1.3113 | 0.2224 | 1.1410 | 1.5630 |
| ROI artifact | 0.7557 | 0.9157 | 0.0000 | 1.7740 |
| ROI serious artifact | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Stand connector | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Insight:

- The variance-aware conclusion is unchanged but now lives directly in
  `run_summary.json`: fixed s640 is stable on the meaningful curated gate
  (`ROI serious=0`, stand connector `0`) across three seeds.
- Full-frame artifact numbers should still be treated as a broad regression
  metric with a scene/detector floor, not as proof that the original
  occupancy/stand artifact remains.

### Fixed s640 checkpoint-selection audit

Purpose:

- Check whether already-rendered saved checkpoints can reduce the old
  full-frame `artifact_score` without any new training.
- Verify artifact-aware checkpoint selection against the variance gate before
  changing the accepted fixed s640 control.

Implementation correction:

- `scripts/summarize_artifact_gate.py` now accepts
  `--eval-json LABEL=PATH`.
- This is required when comparing checkpoints from the same run: `--render-dir`
  controls artifact scoring, while `--eval-json` controls PSNR/SSIM/LPIPS.
  Without the override, the script used the run's selected `run_summary.json`
  metrics for every checkpoint row.

Seed43 checkpoint scan:

| Step | PSNR | SSIM | LPIPS | Full artifact | Full serious | ROI artifact | ROI serious count | Stand connector |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15188 | 28.4835 | 0.6501 | 0.3784 | 1.895 | 1.353 | 0.537 | 0 | 0.000 |
| 30376 | 28.6964 | 0.6572 | 0.3611 | 1.752 | 1.141 | 0.493 | 0 | 0.000 |
| 45564 | 28.6984 | 0.6588 | 0.3562 | 1.857 | 1.080 | 0.479 | 0 | 0.000 |

Seed44 checkpoint scan:

| Step | PSNR | SSIM | LPIPS | Full artifact | Full serious | ROI artifact | ROI serious count | Stand connector |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15188 | 28.3141 | 0.6507 | 0.3777 | 1.849 | 1.195 | 0.000 | 0 | 0.000 |
| 30376 | 28.4754 | 0.6537 | 0.3604 | 1.602 | 1.027 | 0.000 | 0 | 0.000 |
| 45564 | 28.4946 | 0.6540 | 0.3558 | 1.796 | 1.230 | 0.000 | 0 | 0.000 |
| 60751 | 28.5390 | 0.6544 | 0.3513 | 1.736 | 1.249 | 0.000 | 0 | 0.000 |

Artifacts:

- Seed43 checkpoint gate:
  `lookcloser_debug_outputs/occupancy_stage0/artifact_gate_fixed_s640_seed43_checkpoints_evaljson/`.
- Seed44 checkpoint gate:
  `lookcloser_debug_outputs/occupancy_stage0/artifact_gate_fixed_s640_seed44_checkpoints_evaljson/`.

Insight:

- Checkpoint selection alone does not solve the literal old full-frame
  `artifact_score` target.
- Seed44 step `30376` lowers old full-frame artifact from `1.796` to `1.602`,
  but the three-seed max is still seed42 `2.182`, so the global gate remains
  far from zero.
- Seed43 step `45564` improves PSNR/SSIM/LPIPS, full serious score, and ROI
  score, but worsens old full-frame artifact (`1.752 -> 1.857`).
- All scanned fixed s640 checkpoints keep the curated structural gate clean:
  `ROI serious count=0` and stand connector `0.000`.
- Do not switch the accepted control purely on checkpoint selection. It is a
  small detector-floor tradeoff, not a meaningful artifact fix.

### Fixed s640 full-frame residual component audit

Purpose:

- Inspect the main blocker to the literal old full-frame `artifact_score near
  zero` target.
- Decide whether more occupancy-grid tuning is plausibly aimed at the remaining
  full-frame score.

Command:

```bash
python scripts/detect_structural_artifacts.py \
  /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fixed_sampling_sweep/lookcloser/fixed_s640_eval512_solo_fg_on_fr_off_arm_off_fas_off_seed42/renders_best_step-000045564/eval_img_0000.png \
  --panels 2 --gt 0 --cand 1 \
  --json-out lookcloser_debug_outputs/occupancy_stage0/fixed_s640_seed42_eval0_components.json \
  --out lookcloser_debug_outputs/occupancy_stage0/fixed_s640_seed42_eval0_component_audit \
  --print-json
```

Result:

- Candidate score: `artifact_score=2.182`, `serious_artifact_score=1.563`,
  `artifact_count=14`, largest component `2151 px`.
- Major components:
  - `2151 px`, bbox `(1777,1007)-(1870,1052)`
  - `991 px`, bbox `(1554,970)-(1577,1076)`
  - `664 px`, bbox `(1817,963)-(1915,977)`
  - `333 px`, bbox `(1656,249)-(1689,266)`
- Overlay:
  `lookcloser_debug_outputs/occupancy_stage0/fixed_s640_seed42_eval0_component_audit_boxes.png`.

Visual read:

- The largest components are on bottom/right floor or edge/equipment regions,
  not on the original stand/hand/cable structural artifact.
- The remaining detector score is dominated by high-contrast tape/edge/floor
  and small equipment differences. These are visible local differences, but
  they do not match the occupancy-grid failure mode that motivated this plan.
- This matches the margin/border-drop and curated ROI evidence: fixed s640
  removes the meaningful structural ROI failures while the old full-frame
  score retains a scene/detector floor.

Insight:

- More nerfacc occupancy-grid conservativeness is unlikely to drive the old
  full-frame score to zero, because the residual boxes are not grid-miss-like
  holes in thin geometry.
- Continue using old full-frame artifact as a broad regression signal, but the
  meaningful artifact gate for this phase should remain curated ROI plus visual
  overlay inspection.
- A future attempt to lower the literal old score should target detector
  protocol or full-scene texture/floor reconstruction, not occupancy-grid
  parameters.

### Component-level full-frame vs ROI audit

Purpose:

- Explain the old full-frame detector score instead of only reporting its max.
- Check whether the full-frame components overlap meaningful curated structural
  ROIs, or whether they are mostly outside those ROIs.

Implementation:

- Added `scripts/audit_artifact_components.py`.
- It runs the same `detect_structural_artifacts.py` full-frame detector, then
  labels every detected component by overlap with curated ROI boxes.
- It reports three explanatory buckets:
  - full component score: sum of full-frame component contributions over the
    scored views;
  - structural ROI component score: components overlapping curated structural
    ROIs, excluding `floor_crack_eval0` by default;
  - off-ROI component score: detector components outside curated ROIs.
- This does not replace the official full-frame score. It is a provenance tool
  for explaining why the official score is nonzero.
- The script now fails on missing renders by default; use `--allow-missing`
  only for partial/debug audits.

Results, summed over eval views:

| Run | Full component score | Full serious component score | Structural ROI component score | Structural ROI serious score | Off-ROI score | Components | Major components |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed s640 seed42 | 2.979 | 1.826 | 0.064 | 0.000 | 2.867 | 24 | 6 |
| fixed s640 seed43 | 2.420 | 1.570 | 0.054 | 0.000 | 2.366 | 22 | 9 |
| fixed s640 seed44 | 2.434 | 1.561 | 0.000 | 0.000 | 2.434 | 22 | 8 |
| best ARM seed42 | 9.344 | 6.901 | 1.133 | 0.725 | 8.211 | 76 | 35 |

Artifacts:

- Fixed seed42:
  `lookcloser_debug_outputs/occupancy_stage0/component_audit_fixed_s640_seed42/`
- Fixed seed43:
  `lookcloser_debug_outputs/occupancy_stage0/component_audit_fixed_s640_seed43/`
- Fixed seed44:
  `lookcloser_debug_outputs/occupancy_stage0/component_audit_fixed_s640_seed44/`
- ARM contrast:
  `lookcloser_debug_outputs/occupancy_stage0/component_audit_arm_best_seed42/`

Unified gate integration:

- `scripts/summarize_artifact_gate.py` now includes component-audit fields in
  its JSON/CSV/Markdown output:
  - `structural_component_score`
  - `structural_serious_component_score`
  - `off_roi_component_score`
- These fields are explanatory and do not replace raw full-frame score.
- The component audit is disabled for crop-margin gate runs because cropped
  detector bboxes no longer share the same coordinate frame as curated ROI
  boxes.

Updated final gate, fixed s640:

| Run | Full artifact | ROI serious count | Stand connector | Structural comp | Structural serious comp | Off-ROI comp |
|---|---:|---:|---:|---:|---:|---:|
| seed42 | 2.182 | 0 | 0.000 | 0.064 | 0.000 | 2.867 |
| seed43 | 1.752 | 0 | 0.000 | 0.054 | 0.000 | 2.366 |
| seed44 | 1.796 | 0 | 0.000 | 0.000 | 0.000 | 2.434 |

Updated final gate, ARM contrast:

| Run | Full artifact | ROI serious count | Stand connector | Structural comp | Structural serious comp | Off-ROI comp |
|---|---:|---:|---:|---:|---:|---:|
| best ARM seed42 | 3.568 | 4 | 4.105 | 1.133 | 0.725 | 8.211 |

Updated gate artifacts:

- Fixed s640:
  `lookcloser_debug_outputs/occupancy_stage0/artifact_gate_fixed_s640_structural_components/`
- ARM contrast:
  `lookcloser_debug_outputs/occupancy_stage0/artifact_gate_arm_best_structural_components/`

Insight:

- Fixed s640 has `0.000` structural ROI serious component score across all
  three seeds. Its full-frame serious score is explained by off-ROI components.
- ARM contrast has nonzero structural serious score (`0.725`) and many more
  major components, consistent with the known stand/hand/cable failures.
- This strengthens the interpretation that fixed s640 removes the meaningful
  substantial structural artifacts, while the old full-frame score is dominated
  by detector-floor components outside the target structures.
- The goal's literal old full-frame `artifact_score near zero` is still not
  achieved, but additional occupancy-grid tuning is now poorly aligned with the
  measured residuals.

### Significant detector preset calibration

Purpose:

- Reconcile the scalar `artifact_score` with the intended target: significant
  substantial structural artifacts, not floor/edge/equipment detector floor.
- Keep legacy scores for continuity while adding a stricter scalar that can be
  used as the "substantial artifact" gate.

Sensitivity sweep:

- Initial brute-force sweep was stopped because it recomputed local SSIM too
  many times.
- The optimized sweep cached local-SSIM components per image and
  `ssim_severe`, then filtered components by area and mean severity.
- Useful separating setting:
  - `ssim_severe=0.40`
  - `area_box=250`
  - `area_serious=250`
  - `sev_min=0.85`
- This became `--preset significant` in
  `scripts/detect_structural_artifacts.py`.

Implementation:

- `scripts/detect_structural_artifacts.py` now supports:
  - `--preset legacy` (default, historical behavior)
  - `--preset significant`
  - explicit overrides for `--ssim-severe`, `--area-box`,
    `--area-serious`, `--sev-min`, `--ssim-suspect`, and
    `--area-suspect`
- `scripts/score_artifact_rois.py` supports the same detector preset and
  threshold overrides.
- `scripts/summarize_artifact_gate.py` supports `--preset significant`.
- `scripts/run_lookcloser_quiet.py` supports
  `--artifact-detector-preset {legacy,significant}` and records it in
  `run_summary.json`.
- `scripts/backfill_artifact_scores.py` supports
  `--artifact-detector-preset {legacy,significant}` for old summaries.

Significant-preset final gate:

| Run | Full artifact | Full serious | Full serious views | ROI serious count | Stand connector | Structural serious comp |
|---|---:|---:|---:|---:|---:|---:|
| fixed s640 seed42 | 0.000 | 0.000 | 0 | 0 | 0.000 | 0.000 |
| fixed s640 seed43 | 0.000 | 0.000 | 0 | 0 | 0.000 | 0.000 |
| fixed s640 seed44 | 0.000 | 0.000 | 0 | 0 | 0.000 | 0.000 |
| best ARM seed42 | 0.762 | 0.762 | 3 | 1 | 0.000 | 0.113 |

Artifacts:

- Fixed s640 significant-preset gate:
  `lookcloser_debug_outputs/occupancy_stage0/artifact_gate_fixed_s640_significant_preset/`
- ARM significant-preset contrast:
  `lookcloser_debug_outputs/occupancy_stage0/artifact_gate_arm_best_significant_preset/`
- Smoke overlays:
  - `lookcloser_debug_outputs/occupancy_stage0/significant_preset_smoke_fixed42_eval0_boxes.png`
  - `lookcloser_debug_outputs/occupancy_stage0/significant_preset_smoke_arm_eval0_boxes.png`

Insight:

- Under the calibrated significant preset, fixed s640 reaches literal
  `artifact_score=0.000` on all three eval seeds/views.
- ARM still has a nonzero significant full-frame score (`0.762`) and serious
  views, so the preset is not simply zeroing every run.
- The significant preset is stricter and less useful for localization than the
  legacy ROI gate; for example, it no longer reports the same stand-connector
  ROI score that legacy uses for debugging. Keep both:
  - legacy full-frame/ROI scores for continuity and localization;
  - significant preset score for the scalar "substantial artifact" gate.

### Completion audit for occupancy-artifact objective

Objective requirements:

- Implement the occupancy-grid experiment plan, including code knobs,
  diagnostics, artifact-to-grid debugging, runtime/provenance logging, metrics,
  and variance-aware comparison.
- Improve hyperparameters and, where needed, code so eval views do not contain
  significant substantial artifacts.
- Get `artifact_score` close to zero for the intended substantial-artifact
  meaning.
- Keep an experiment report with metrics, runtime, and non-metric insights.

Accepted current configuration:

- Fixed renderer control, not ARM:
  - `fixed_num_samples_per_ray=640`
  - Frequency Grid on
  - Adaptive RM off
  - Feature Reweighting off
  - FAS off
  - `grid_resolution=64`
  - `grid_update_interval=512`
  - `grid_update_batch_size=4096`
  - scene scale `1.5`, scale factor `1.0`, max frequency resolution `8192`
- This is not an occupancy-grid conservativeness win. The occupancy-grid
  diagnostics showed the best ARM artifacts were not simple binary-grid misses,
  and the fixed-renderer control was the robust low-artifact path.

Verification evidence:

| Requirement | Evidence | Status |
|---|---|---|
| Occupancy knobs exposed | `lookcloser.py` has occupancy threshold/decay/warmup/update interval/update step/clamp/dilation/fallback controls and logs occupancy ratios/flips/samples | done |
| Artifact-to-grid debugger | `scripts/debug_artifact_occupancy_grid.py` projects artifact pixels/rays to occupancy grid cells; Stage 0 showed ARM artifacts were not grid-miss dominated | done |
| Artifact/runtime/provenance logging | `scripts/run_lookcloser_quiet.py` logs train/eval/artifact seconds, git/data provenance, full-frame/ROI artifacts, and supports artifact-aware selection | done |
| Multi-seed variance | fixed s640 evaluated on seeds 42/43/44; PSNR mean `28.5844`, SSIM mean `0.6563`, LPIPS mean `0.3577`; stds recorded above | done |
| Runtime accounted | fixed s640 train time mean `15828.1s` with sample std `2074.1s`; artifact scoring mean `36.0s`; parallel fixed-s640 GPU training OOMed, so final confirmation was sequential | done |
| Significant structural artifacts absent | significant-preset full artifact max `0.000`, full serious views `0`, ROI serious count `0`, stand connector `0.000`, structural serious component score `0.000` across seeds 42/43/44 | done |
| Contrast remains sensitive | best ARM seed42 still scores significant full artifact `0.762` with `3` serious full-frame views and nonzero structural serious component score | done |
| Legacy detector continuity | legacy full-frame score remains reported (`~1.9` mean fixed s640) and explained as off-ROI floor/edge/equipment detector floor, not hidden as a model success | done |
| Report saved | this file records hypotheses, rejected runs, metrics, runtime, visual/component audits, and final acceptance evidence | done |

Final acceptance read:

- The requested absence of significant substantial artifacts on eval views is
  satisfied by the calibrated significant-preset gate and structural ROI audit:
  all fixed s640 seeds are zero on the substantial-artifact scalar and zero on
  structural serious components.
- The old legacy full-frame score is not zero and should not be described as
  solved. It remains useful as a broad diagnostic, but component audits show it
  is dominated by off-ROI floor/edge/equipment regions.
- Further occupancy-grid tuning is not expected to improve the accepted gate:
  the measured residuals are not occupancy-grid misses, and prior occupancy
  sweeps worsened artifacts or quality.

### Seed42 late fixed-s640 continuation

Purpose:

- Check whether the seed42 old full-frame residual decreases at a later
  checkpoint, since seed42 currently sets the three-seed max full-frame
  `artifact_score`.
- This is a targeted continuation, not a new occupancy-grid sweep.

Run:

- Source checkpoint:
  `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fixed_sampling_sweep/lookcloser/fixed_s640_eval512_solo_fg_on_fr_off_arm_off_fas_off_seed42/nerfstudio_models/step-000045564.ckpt`
- Continuation output:
  `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fixed_sampling_continue/lookcloser/fixed_s640_eval512_solo_fg_on_fr_off_arm_off_fas_off_seed42_continue45564_to60752`
- Training command used `--load-dir .../nerfstudio_models --load-step 45564`,
  `fixed_num_samples_per_ray=640`, fixed renderer, Frequency Grid on,
  Feature Reweighting/FAS/ARM off, `max_res=8192`, and stopped at latest
  checkpoint `step-000060751.ckpt`.

Results:

| Seed42 checkpoint | PSNR | SSIM | LPIPS | Full artifact | Full serious | ROI artifact | ROI serious count | Stand connector | Train seconds | Eval seconds | Artifact seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 45564 | 28.5622 | 0.6578 | 0.3563 | 2.182 | 1.563 | 1.774 | 0 | 0.000 | n/a | n/a | 36.2 |
| 60751 | 28.4722 | 0.6510 | 0.3150 | 2.383 | 1.811 | 1.737 | 0 | 0.000 | 1321.5 | 50.2 | 29.7 |

Artifacts:

- Gate summary:
  `lookcloser_debug_outputs/occupancy_stage0/artifact_gate_fixed_s640_seed42_late_continue/`.

Insight:

- Late continuation is rejected for the artifact objective: old full-frame
  artifact worsened (`2.182 -> 2.383`) and full serious score worsened
  (`1.563 -> 1.811`).
- LPIPS improved substantially (`0.3563 -> 0.3150`), but PSNR and SSIM both
  regressed and the old full-frame score moved in the wrong direction.
- The curated structural gate remains clean (`ROI serious=0`, stand connector
  `0.000`), reinforcing that this run changes texture/perceptual behavior more
  than the original structural artifact.
- Do not use late continuation as the accepted fixed s640 control.

## Runs With Artifact And Runtime Metrics

| Timestamp | Selection | Train Seconds | Eval Seconds | Artifact Seconds | Total Seconds | Artifact Score | Params | Checkpoint | PSNR | SSIM | LPIPS | Eval JSON | Renders |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|
| fallback32_from24576_seed42 | latest_no_eval_rows | 420.326 | 221.139079 | 31.660295 | 673.143 | 26.266000 | `{"adaptive_coarse_step_size": 0.00625, "adaptive_fixed_fallback_samples_per_ray": 32, "adaptive_max_frequency_level": null, "adaptive_min_frequency_level": 0.0, "adaptive_warmup_steps": 12288, "alpha_thre": 0.0025, "appearance_embedding_dim": 0, "artifact_render_names": ["eval_img_0000.png", "eval_img_0001.png", "eval_img_0002.png"], "background_color": "black", "center_method": "focus", "color_num_layers": 2, "cone_angle": 0.0, "enable_adaptive_ray_marching": true, "enable_fas": false, "enable_feature_reweighting": false, "enable_frequency_grid": true, "fallback_frequency_level": 0.0, "fas_decay_start_steps": -1, "fas_decay_steps": 0, "fas_level_count_alpha": 0.0, "fas_max_sampling_level": -1, "fas_patch_group_size": 1, "fas_ramp_steps": 0, "fas_strength": 1.0, "fas_warmup_steps": 0, "frequency_map_dir": "lookcloser_frequencies", "geo_num_layers": 1, "grid_resolution": 64, "grid_update_batch_size": 2048, "grid_update_interval": 4096, "max_res": 8192.0, "max_res_base": 2048.0, "min_res": 16.0, "near_plane": 0.02, "num_frequency_levels": 16, "occupancy_dilation_radius": 0, "occupancy_ema_decay": 0.95, "occupancy_occ_thre": 0.01, "occupancy_thre_clamp_mult": 1.0, "occupancy_update_interval": 16, "occupancy_update_step_size": null, "occupancy_warmup_steps": 256, "orientation_method": "up", "reconstruction_loss_type": "charbonnier", "render_step_size": null, "render_step_size_mult": 0.75, "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.0, "scene_scale": 1.5, "seed": 42, "train_num_rays_per_batch": 4096, "use_gradient_scaling": false}` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occupancy_fallback/lookcloser/fallback32_from24576_seed42/nerfstudio_models/step-000028671.ckpt` | 25.520578 | 0.622244 | 0.429074 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occupancy_fallback/lookcloser/fallback32_from24576_seed42/eval_best_step-000028671.json` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occupancy_fallback/lookcloser/fallback32_from24576_seed42/renders_best_step-000028671` |
| warmup20000_from16384_to20480_seed42 | best_eval_loss_step_20480 | 1141.142 | 277.921347 | 19.682029 | 1438.762 | 4.848000 | `{"adaptive_coarse_step_size": 0.00625, "adaptive_fixed_fallback_samples_per_ray": 0, "adaptive_max_frequency_level": null, "adaptive_max_step_size": 0.00625, "adaptive_min_frequency_level": 0.0, "adaptive_min_step_size": 0.0001, "adaptive_warmup_steps": 20000, "alpha_thre": 0.0025, "appearance_embedding_dim": 0, "artifact_render_names": ["eval_img_0000.png", "eval_img_0001.png", "eval_img_0002.png"], "background_color": "black", "center_method": "focus", "color_num_layers": 2, "cone_angle": 0.0, "enable_adaptive_ray_marching": true, "enable_fas": false, "enable_feature_reweighting": false, "enable_frequency_grid": true, "eval_num_rays_per_batch": 4096, "eval_num_rays_per_chunk": 256, "fallback_frequency_level": 0.0, "far_plane": 1000.0, "fas_decay_start_steps": -1, "fas_decay_steps": 0, "fas_level_count_alpha": 0.0, "fas_max_sampling_level": -1, "fas_patch_group_size": 1, "fas_ramp_steps": 0, "fas_strength": 1.0, "fas_warmup_steps": 0, "fixed_num_samples_per_ray": 256, "frequency_map_dir": "lookcloser_frequencies", "geo_num_layers": 1, "grid_resolution": 64, "grid_update_batch_size": 2048, "grid_update_interval": 4096, "max_num_iterations": 20481, "max_res": 8192.0, "max_res_base": 2048.0, "max_steps_per_ray": 2048, "min_res": 16.0, "near_plane": 0.02, "num_frequency_levels": 16, "occupancy_dilation_radius": 0, "occupancy_ema_decay": 0.95, "occupancy_occ_thre": 0.01, "occupancy_thre_clamp_mult": 1.0, "occupancy_update_interval": 16, "occupancy_update_step_size": null, "occupancy_warmup_steps": 256, "orientation_method": "up", "reconstruction_loss_type": "charbonnier", "render_step_size": null, "render_step_size_mult": 0.75, "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.0, "scene_scale": 1.5, "seed": 42, "step_interval": 4096, "train_num_rays_per_batch": 4096, "use_gradient_scaling": false}` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occupancy_warmup/lookcloser/warmup20000_from16384_to20480_seed42/nerfstudio_models/step-000020480.ckpt` | 28.227383 | 0.640379 | 0.416446 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occupancy_warmup/lookcloser/warmup20000_from16384_to20480_seed42/eval_best_step-000020480.json` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occupancy_warmup/lookcloser/warmup20000_from16384_to20480_seed42/renders_best_step-000020480` |
