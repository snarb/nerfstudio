# Mix Improvements Report

## Technical summary

The old ARM / Interval Adjustment leaders did not become artifact-clean after the occupancy-grid cold-start fix. Both seed-42 retests used the new safe occupancy defaults, kept ARM enabled, kept `grid_resolution=64`, disabled FAS and Feature Reweighting, and selected checkpoints with `--eval-checkpoint artifact`.

H41 is the better quality candidate, but it still fails the full-frame significant artifact gate. H40 has the lower full-frame artifact score, but its artifact-selected checkpoint is much earlier and materially weaker on PSNR, SSIM, and LPIPS. Curated ROI scoring is clean for both selected checkpoints, while the official full-frame score remains nonzero.

The artifact-to-occupancy debugger does not support more conservative occupancy tuning as the next move: both H40 and H41 selected-checkpoint artifacts mapped mostly through occupied voxels (`grid_miss_likely=false`, `field_issue_likely=true`).

Current experiment policy: do not run new no-ARM controls in this phase. The no-ARM occupancy-grid path is already known to reach `artifact_score=0.000`, so remaining experiments must keep ARM enabled (`ray_sampling_mode=adaptive`) and keep occupancy-grid sampling enabled. Keep FAS and Feature Reweighting disabled until the ARM-only artifact path is stable and variance-safe.

Current full-run budget policy: use `max_num_iterations=200000` for new ARM-only experiments so slower cleanup/stabilization settings are not prematurely capped. Older sections with `90000` or short boundary caps are historical diagnostics, not the active budget.

## No-ARM occupancy baseline recheck

The old clean no-ARM result is documented in `experiments/occupancy_ngp_no_arm_sync.md`. The key recipe was:

- `ray_sampling_mode=occupancy`
- `grid_resolution=128`
- `occupancy_grid_levels=1`
- `occupancy_warmup_steps=4096`
- `occupancy_binary_warmup_steps=4096`
- `render_step_size_mult=1.0`
- `alpha_thre=0.0`
- `transmittance_threshold=0.0`
- `near_plane=0.01`
- `cone_angle=0.0`
- `train_num_rays_per_batch=4096`
- `max_num_iterations=15188`

Historical strict results:

| Run | Mode | Grid | Train rays | Selected checkpoint | PSNR | SSIM | LPIPS | Full artifact | ROI artifact | Train s | Total s | Read |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `occgrid_grid128_warm4096_s42` | occupancy | 128 | 4096 | 15187 | 28.6580 | 0.6465 | 0.4614 | 0.000 | 0.000 | 1742.2 | 1816.2 | clean |
| `occgrid_grid128_warm4096_s43` | occupancy | 128 | 4096 | 15187 | 27.3712 | 0.6569 | 0.4684 | 0.106 | 0.000 | 1742.3 | 1816.1 | tiny off-ROI residual without artifact selection |
| `occgrid_grid128_warm4096_s43_artifact_select_3797_v2` | occupancy | 128 | 4096 | 15187 | 27.1901 | 0.6568 | 0.4694 | 0.000 | 0.000 | 1081.4 | 1354.9 | clean with artifact-aware selection |
| `occgrid_grid128_warm4096_s44` | occupancy | 128 | 4096 | 15187 | 28.7747 | 0.6612 | 0.4618 | 0.000 | 0.000 | 1742.2 | 1814.3 | clean |

Incorrect fixed-sampler diagnostic runs started during this pass are not comparable to that historical clean occupancy recipe. They used `ray_sampling_mode=fixed`, `grid_resolution=64`, train batch `8192`, and checkpoint cadence `4096`; they are useful only as evidence that the fixed control can produce different detector behavior.

| Run | Mode | Grid | Train rays | Occ warmup | Selected checkpoint | PSNR | SSIM | LPIPS | Full artifact | ROI artifact | Train s | Total s | Read |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `noarm_h41_seed42` | fixed | 64 | 8192 | 4096 | 12288 | 26.8485 | 0.6019 | 0.3647 | 0.143 | 0.000 | 2705.3 | 3006.4 | not comparable to occupancy clean baseline |
| `noarm_h41_occ8192_seed42` | fixed | 64 | 8192 | 8192 | 4096 | 26.3649 | 0.5928 | 0.4533 | 0.106 | 2.917 | 3006.1 | 3293.5 | longer occupancy warmup helped but did not make fixed clean |
| `noarm_h41_seed44_rerun` | fixed | 64 | 8192 | 4096 | 12288 | 26.8983 | 0.6017 | 0.3613 | 0.000 | 0.000 | 3035.8 | 3318.6 | fixed can be clean on some seeds |

Historical note: these no-ARM rechecks were used only to separate the fixed-sampler diagnostic from the correct occupancy-mode baseline. They should not be repeated now; the active work is ARM-only artifact isolation with occupancy-grid sampling enabled.

Recheck status on the current branch:

| Run | Mode | Grid | Train rays | Stop policy | Selected checkpoint | PSNR | SSIM | LPIPS | Full artifact | ROI artifact | Train s | Total s | Read |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `occgrid_grid128_warm4096_recheck_s42` | occupancy | 128 | 4096 | default early stop | 7594 | 27.9918 | 0.6303 | 0.4851 | 0.000 | 0.000 | 2673.6 | 3178.9 | clean, selected earlier than historical final |
| `occgrid_grid128_warm4096_recheck_s44` | occupancy | 128 | 4096 | default early stop | 15187 | 28.8039 | 0.6547 | 0.4622 | 0.000 | 0.000 | 2673.6 | 3184.0 | clean |
| `occgrid_grid128_warm4096_recheck_s43` | occupancy | 128 | 4096 | default early stop | 11391 | 27.1056 | 0.6537 | 0.4742 | 0.208 | 2.744 | 2254.6 | 2634.3 | rejected; early stop skipped 15187 |
| `occgrid_grid128_warm4096_recheck_s43_nostop` | occupancy | 128 | 4096 | no early stop | 11391 | 27.1478 | 0.6558 | 0.4750 | 0.123 | 0.000 | 1231.5 | 1503.9 | rejected; tiny off-ROI full-frame residual remains |

Seed43 no-stop candidate scores:

| Checkpoint | PSNR | SSIM | LPIPS | Full artifact | ROI artifact | Read |
|---:|---:|---:|---:|---:|---:|---|
| 3797 | 26.6659 | 0.6068 | 0.5161 | 0.396 | 9.135 | early ROI/stand artifacts |
| 7594 | 26.9514 | 0.6432 | 0.4914 | 0.156 | 0.000 | off-ROI residual |
| 11391 | 27.1478 | 0.6558 | 0.4750 | 0.123 | 0.000 | selected; best artifact score but not clean |
| 15187 | 27.3254 | 0.6590 | 0.4704 | 0.145 | 0.000 | final is not clean in this recheck |

Seed43 residual details:

- Failing selected view: `eval_img_0000.png`.
- Overlay: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occgrid_no_arm_recheck/lookcloser/occgrid_grid128_warm4096_recheck_s43_nostop/artifact_renders_artifact_selection_step-000011391/eval_img_0000_boxes.png`
- Selected renders: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occgrid_no_arm_recheck/lookcloser/occgrid_grid128_warm4096_recheck_s43_nostop/renders_artifact_selection_step-000011391`
- Occupancy debug: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occgrid_no_arm_recheck/lookcloser/occgrid_grid128_warm4096_recheck_s43_nostop/artifact_occ_debug_eval0_step11391/artifact_occupancy_debug.md`
- Debug read: `grid_miss_likely=false`, `field_issue_likely=true`.

Current interpretation: the earlier fixed-sampler artifacts were caused by testing the wrong no-ARM control, not by a new occupancy warmup failure. The correct occupancy-mode no-ARM recipe is clean for seeds 42 and 44 in this recheck, but seed43 does not reproduce the old literal `0.000` full-frame result. The remaining seed43 issue is a tiny off-ROI component near the upper-left wall/cable area, not a stand/hand/cable ROI artifact and not an occupancy-grid miss. Do not proceed to ARM as "baseline clean" until seed43 is either reproduced clean or accepted as detector/variance noise with visual evidence.

## ARM occupancy-warmup retest

### What was tested

Retested the old best isolated ARM candidates after the occupancy-grid warmup fix:

- Occupancy warmup: `occupancy_warmup_steps=4096`
- Binary occupancy warmup: `occupancy_binary_warmup_steps=4096`
- Occupancy levels: `occupancy_grid_levels=1`
- Frequency grid resolution: `grid_resolution=64`
- Traversal/rendering safety: `render_step_size_mult=1.0`, `alpha_thre=0.0`, `transmittance_threshold=0.0`, `near_plane=0.01`, `cone_angle=0.0`
- ARM: enabled with `adaptive_warmup_steps=2048`, `adaptive_coarse_step_size=0.0125`, `adaptive_max_frequency_level=12`
- FAS: disabled
- Feature Reweighting: disabled
- Artifact scoring: significant preset over `eval_img_0000.png`, `eval_img_0001.png`, and `eval_img_0002.png`

### Results

| Run | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Curated ROI artifact | Stand connector | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| H40 `arm_h40_newocc_seed42` | 8192 | 28.5016 | 0.6522 | 0.4475 | 0.280 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h40_newocc_seed42/renders_artifact_selection_step-000008192` |
| H41 `arm_h41_newocc_seed42` | 16384 | 29.4104 | 0.6796 | 0.4037 | 0.469 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h41_newocc_seed42/renders_artifact_selection_step-000016384` |

Supporting summaries:

- Machine-readable summary: `experiments/arm_occwarmup_retest_summary.json`
- Markdown summary: `experiments/arm_occwarmup_retest_summary.md`
- H40 run summary: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h40_newocc_seed42/run_summary.json`
- H41 run summary: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h41_newocc_seed42/run_summary.json`

### Artifact details

H40 selected checkpoint:

- Selected by artifact-aware checkpoint selection: `best_artifact_checkpoint_step_8192`
- Full-frame view scores: `eval_img_0000=0.000`, `eval_img_0001=0.280`, `eval_img_0002=0.270`
- Largest artifact view: `eval_img_0001.png`
- Overlay: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h40_newocc_seed42/artifact_renders_artifact_selection_step-000008192/eval_img_0001_boxes.png`
- Occupancy debug: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h40_newocc_seed42/artifact_occ_debug_eval1/artifact_occupancy_debug.md`
- Occupancy read: `grid_miss_likely=false`, `field_issue_likely=true`

H41 selected checkpoint:

- Selected by artifact-aware checkpoint selection: `best_artifact_checkpoint_step_16384`
- Full-frame view scores: `eval_img_0000=0.168`, `eval_img_0001=0.177`, `eval_img_0002=0.469`
- Largest artifact view: `eval_img_0002.png`
- Overlay: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h41_newocc_seed42/artifact_renders_artifact_selection_step-000016384/eval_img_0002_boxes.png`
- Occupancy debug: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occwarmup_retest/lookcloser/arm_h41_newocc_seed42/artifact_occ_debug_eval2/artifact_occupancy_debug.md`
- Occupancy read: `grid_miss_likely=false`, `field_issue_likely=true`

### Insights

- The new occupancy warmup improves the safety of the cold start but does not make the old ARM metric leaders artifact-clean under the official full-frame significant detector.
- Artifact-aware checkpoint selection matters: H40's best artifact checkpoint is much earlier than its high-metric checkpoints, while H41 can keep clean curated ROIs at a stronger metric checkpoint but still fails full-frame scoring.
- The remaining selected-checkpoint artifacts do not look like binary occupancy misses. The debugger points toward field quality, alpha integration, training trajectory, or checkpoint-selection limits.
- Do not prioritize `ema`, lower threshold, dilation, grid 256, denser render-only steps, or fallback sampling based on these two runs; prior tests already found those risky, and the new debugger output does not implicate missing occupied cells.

## Next improvement candidate

The next low-hanging feature branch was Feature Reweighting in isolation on the stronger H41-style recipe, with FAS still disabled and artifact-aware checkpoint selection still enabled. Reason: the artifact debugger pointed away from occupancy pruning and toward field/training quality. Feature Reweighting is the remaining paper module most directly aimed at changing how frequency-conditioned features are emphasized without changing the occupancy grid.

First screen:

- Base: H41 settings (`train_num_rays_per_batch=8192`)
- Enable Feature Reweighting
- Keep FAS disabled
- Keep safe occupancy defaults
- Keep `grid_resolution=64`, `adaptive_warmup_steps=2048`, `adaptive_coarse_step_size=0.0125`, `adaptive_max_frequency_level=12`
- Keep `--eval-checkpoint artifact`, `--keep-all-checkpoints`, and all three artifact render names

Acceptance remains unchanged: full-frame artifact score `0.000`, curated ROI score `0.000`, no obvious broken stand/hand/cable artifact, and PSNR/SSIM/LPIPS close enough to the old ARM leaders to justify any artifact improvement.

### Feature Reweighting result

Direct isolated Feature Reweighting should be rejected for now. It improved mid/late SSIM but did not clear the artifact gate, and the higher-metric checkpoints developed serious ROI/stand failures.

| Run | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Curated ROI artifact | Stand connector | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| H41 + FR `arm_h41_fr_newocc_seed42` | 8192 | 28.6357 | 0.6650 | 0.4384 | 0.445 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_retest/lookcloser/arm_h41_fr_newocc_seed42/renders_artifact_selection_step-000008192` |

Supporting summaries:

- Machine-readable summary: `experiments/feature_reweighting_retest_summary.json`
- Markdown summary: `experiments/feature_reweighting_retest_summary.md`
- Run summary: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_retest/lookcloser/arm_h41_fr_newocc_seed42/run_summary.json`

Feature Reweighting checkpoint notes:

- Selected checkpoint: step `8192`, chosen by artifact-aware selection.
- Selected full-frame view scores: `eval_img_0000=0.000`, `eval_img_0001=0.445`, `eval_img_0002=0.127`.
- Selected overlay: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_retest/lookcloser/arm_h41_fr_newocc_seed42/artifact_renders_artifact_selection_step-000008192/eval_img_0001_boxes.png`
- Occupancy debug: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_retest/lookcloser/arm_h41_fr_newocc_seed42/artifact_occ_debug_eval1/artifact_occupancy_debug.md`
- Occupancy read: `grid_miss_likely=false`, `field_issue_likely=true`

Later checkpoints are not acceptable despite better global metrics:

| Checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Curated ROI artifact | Stand connector | Decision |
|---:|---:|---:|---:|---:|---:|---:|---|
| 12288 | 29.1925 | 0.6835 | 0.4145 | 0.814 | 3.461 | 3.461 | Reject: stand ROI failure |
| 16384 | 29.2207 | 0.6878 | 0.4043 | 1.525 | 12.058 | 5.776 | Reject: serious ROI/stand failure |
| 20480 | 29.5754 | 0.6913 | 0.3949 | 0.879 | 8.340 | 6.270 | Reject: serious ROI/stand failure |

### Feature Reweighting decision

Do not keep direct Feature Reweighting on the H41 ARM recipe. Its artifact-selected checkpoint is worse than no-FR H40 on full-frame artifact score and worse than no-FR H41 on PSNR/SSIM/LPIPS at the selected checkpoint. Its high-metric checkpoints reintroduce the stand failure that the ROI gate is meant to catch.

The next Feature Reweighting low-hanging fruit should be a reduced-strength or scheduled Feature Reweighting variant, not a full-strength rerun. A conservative next test is to add a configurable `feature_reweighting_strength` mixer and screen strengths such as `0.25`, `0.5`, and `0.75` on the same H41 ARM base, still with FAS disabled and artifact-aware selection.

### Reduced-strength Feature Reweighting

Implemented `feature_reweighting_strength` as a blend between identity features and the paper Eq. 6 weight curve:

```text
effective_weight = 1 + feature_reweighting_strength * (paper_weight - 1)
```

`1.0` preserves the current full-strength behavior. `0.0` is equivalent to identity weighting while still exercising the same code path.

| Run | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Curated ROI artifact | Stand connector | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| H41 + FR strength 0.25 `arm_h41_frs025_newocc_seed42` | 4096 | 19.6131 | 0.6400 | 0.5309 | 0.415 | 0.000 | 0.000 | Reject: selected checkpoint is undertrained |
| H41 + FR strength 0.5 `arm_h41_frs050_newocc_seed42` | 4096 | 19.7173 | 0.6406 | 0.5271 | 0.279 | 0.000 | 0.000 | Reject: selected checkpoint is undertrained |

Supporting paths:

- Summary: `experiments/feature_reweighting_strength_summary.md`
- Strength 0.25 run summary: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs025_newocc_seed42/run_summary.json`
- Strength 0.25 selected renders: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs025_newocc_seed42/renders_artifact_selection_step-000004096`
- Strength 0.25 selected overlay: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs025_newocc_seed42/artifact_renders_artifact_selection_step-000004096/eval_img_0002_boxes.png`
- Strength 0.25 occupancy debug: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs025_newocc_seed42/artifact_occ_debug_eval2/artifact_occupancy_debug.md`
- Run summary: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs050_newocc_seed42/run_summary.json`
- Selected renders: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs050_newocc_seed42/renders_artifact_selection_step-000004096`
- Selected overlay: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs050_newocc_seed42/artifact_renders_artifact_selection_step-000004096/eval_img_0000_boxes.png`
- Occupancy debug: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_strength/lookcloser/arm_h41_frs050_newocc_seed42/artifact_occ_debug_eval0/artifact_occupancy_debug.md`

Higher-quality reduced-strength checkpoints remained unsafe:

| Run | Checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Curated ROI artifact | Stand connector | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| strength 0.25 | 8192 | 28.6552 | 0.6597 | 0.4415 | 0.791 | 0.000 | 0.000 | Not artifact-clean |
| strength 0.25 | 12288 | 29.2109 | 0.6817 | 0.4178 | 1.197 | 8.257 | 6.130 | Reject: stand ROI failure |
| strength 0.25 | 16384 | 29.3949 | 0.6825 | 0.4052 | 1.013 | 6.924 | 2.054 | Reject: stand ROI failure |
| strength 0.25 | 20480 | 29.5767 | 0.6944 | 0.3981 | 1.005 | 8.244 | 5.888 | Reject: stand ROI failure |
| strength 0.25 | 24576 | 29.4747 | 0.6950 | 0.3961 | 1.271 | 12.501 | 5.813 | Reject: serious ROI/stand failure |
| strength 0.25 | 28672 | 29.4207 | 0.6908 | 0.3950 | 1.263 | 13.867 | 11.536 | Reject: serious ROI/stand failure |
| strength 0.5 | 8192 | 28.8206 | 0.6700 | 0.4394 | 0.431 | 0.000 | 0.000 | Not artifact-clean |
| strength 0.5 | 12288 | 29.2477 | 0.6892 | 0.4178 | 1.013 | 8.235 | 6.277 | Reject: stand ROI failure |
| strength 0.5 | 16384 | 29.2953 | 0.6883 | 0.4068 | 0.833 | 12.167 | 5.828 | Reject: serious ROI/stand failure |
| strength 0.5 | 20480 | 29.5818 | 0.6958 | 0.3991 | 0.659 | 8.218 | 6.276 | Reject: stand ROI failure |

Insight: static reduced strength is not enough. `0.5` improves the full-strength FR early instability but still fails later stand ROIs; `0.25` trains longer and reaches strong global metrics, but also fails the stand/ROI gate at all useful checkpoints. The selected low-artifact checkpoints for both strengths are undertrained and should not be kept.

Next Feature Reweighting option: test scheduled or two-stage FR instead of static FR. A reasonable next screen is to start from a stable no-FR checkpoint and enable weak FR only as a continuation, or add `feature_reweighting_warmup_steps` / `feature_reweighting_ramp_steps` so FR does not affect early/mid geometry formation.

### Two-stage Feature Reweighting continuation

Tested a two-stage variant without adding more scheduling code: load the no-FR H41 checkpoint at step `16384`, enable Feature Reweighting at strength `0.25`, keep FAS disabled, and continue to step `24576` with artifact-aware checkpoint selection.

| Run | Source checkpoint | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Curated ROI artifact | ROI serious count | Stand connector | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H41 -> FR 0.25 `arm_h41_fr025_from_h41s16384_seed42` | 16384 | 24575 | 29.5910 | 0.6832 | 0.3771 | 0.608 | 7.154 | 1 | 0.000 | Reject for artifact gate; keep as LPIPS signal |

Supporting paths:

- Summary: `experiments/feature_reweighting_twostage_summary.md`
- Run summary: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_twostage/lookcloser/arm_h41_fr025_from_h41s16384_seed42/run_summary.json`
- Selected renders: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_twostage/lookcloser/arm_h41_fr025_from_h41s16384_seed42/renders_artifact_selection_step-000024575`
- Full-frame overlay: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_twostage/lookcloser/arm_h41_fr025_from_h41s16384_seed42/artifact_renders_artifact_selection_step-000024575/eval_img_0002_boxes.png`
- Failing ROI overlay: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_twostage/lookcloser/arm_h41_fr025_from_h41s16384_seed42/artifact_renders_artifact_selection_step-000024575/roi_scores/stand_label_eval2_boxes.png`
- Occupancy debug: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_feature_reweighting_twostage/lookcloser/arm_h41_fr025_from_h41s16384_seed42/artifact_occ_debug_eval2/artifact_occupancy_debug.md`

View and ROI details:

- Full-frame selected view scores: `eval_img_0000=0.508`, `eval_img_0001=0.298`, `eval_img_0002=0.608`.
- The curated ROI failure is `stand_label_eval2`: ROI artifact `7.154`, serious count `1`, largest blob `862 px`.
- `left_stand_connector_eval0` remained clean: stand connector score `0.000`.
- Occupancy read: `grid_miss_likely=false`, `field_issue_likely=true`.

Insight: two-stage FR is the best Feature Reweighting signal so far for perceptual quality, improving selected LPIPS to `0.3771` versus no-FR H41's `0.4037`, while also improving PSNR to `29.5910`. It still fails both full-frame and curated ROI artifact gates, so it should not be kept as an accepted recipe. The failure is not explained by occupancy misses.

### Current Feature Reweighting conclusion

Do not keep static or two-stage Feature Reweighting as tested:

- Full-strength FR: rejected; useful checkpoints fail ROI/stand gate.
- Strength `0.5`: rejected; selected checkpoint is undertrained and useful checkpoints fail ROI/stand gate.
- Strength `0.25`: rejected; selected checkpoint is undertrained and useful checkpoints fail ROI/stand gate.
- Two-stage H41 -> FR `0.25`: rejected for artifact acceptance, but useful as an LPIPS/PSNR improvement signal.

Next low-hanging fruit should focus on artifact-aware continuation selection or targeted visual gating around `stand_label_eval2`, not more occupancy conservativeness. A practical next screen is a two-stage FR continuation with shorter continuation and denser checkpoint cadence, e.g. load H41 step `16384`, enable FR strength `0.25`, save/eval every `1024` or `2048` steps through `24576`, and select by ROI/artifact gate. The current two-stage run only saved `20480` and `24575`, so it may have skipped a cleaner midpoint.

## ARM H40 grid128 short-budget retest

### What was tested

Retested the old metric leader `adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_continue36864_r4096` with the current artifact-safe occupancy setup and the shorter no-ARM recheck budget:

- ARM on: `ray_sampling_mode=adaptive`
- FAS off, Feature Reweighting off
- `grid_resolution=128`
- `train_num_rays_per_batch=4096` for two seeds, plus one `8192` batch comparison
- `occupancy_warmup_steps=4096`
- `occupancy_binary_warmup_steps=4096`
- `occupancy_grid_levels=1`
- `render_step_size_mult=1.0`
- `alpha_thre=0.0`
- `transmittance_threshold=0.0`
- `near_plane=0.01`
- `cone_angle=0.0`
- H40 ARM settings preserved: `adaptive_warmup_steps=2048`, `adaptive_coarse_step_size=0.0125`, `adaptive_max_frequency_level=12`, `max_steps_per_ray=1024`
- `max_num_iterations=15188`
- checkpoint/eval cadence `3797`, producing candidates `3797`, `7594`, `11391`, `15187`
- default early stop on eval loss, `--eval-checkpoint artifact`, all three eval views scored

Experiment root:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_retest/lookcloser`

### Results

None of the three runs passed the full-frame significant artifact gate. The two primary batch-4096 runs selected the final checkpoint with clean curated ROI scores but nonzero full-frame artifacts. The batch-8192 comparison was worse: full-frame artifact was higher and curated ROI was not clean.

| Run | Seed | Train rays | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Train time (s) | Total time (s) | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arm_h40_grid128_warm4096_s42` | 42 | 4096 | 15187 | 28.6217 | 0.6579 | 0.4447 | 0.314 | 0.000 | 3004.2 | 3426.4 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_retest/lookcloser/arm_h40_grid128_warm4096_s42/renders_artifact_selection_step-000015187` |
| `arm_h40_grid128_warm4096_s43` | 43 | 4096 | 15187 | 28.4340 | 0.6551 | 0.4537 | 0.337 | 0.000 | 3004.3 | 3455.4 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_retest/lookcloser/arm_h40_grid128_warm4096_s43/renders_artifact_selection_step-000015187` |
| `arm_h40_grid128_warm4096_batch8192_s42` | 42 | 8192 | 15187 | 27.8980 | 0.6587 | 0.4443 | 0.438 | 2.199 | 3034.4 | 3414.3 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_retest/lookcloser/arm_h40_grid128_warm4096_batch8192_s42/renders_artifact_selection_step-000015187` |

Candidate artifact trend:

| Run | Step 3797 | Step 7594 | Step 11391 | Step 15187 | Read |
|---|---:|---:|---:|---:|---|
| `s42`, 4096 rays | 0.799 / ROI 9.368 | 0.569 / ROI 4.392 | 0.422 / ROI 1.600 | 0.314 / ROI 0.000 | Improving but not clean |
| `s43`, 4096 rays | 0.658 / ROI 9.574 | 0.772 / ROI 6.156 | 0.586 / ROI 3.304 | 0.337 / ROI 0.000 | Final is best artifact checkpoint |
| `s42`, 8192 rays | 1.483 / ROI 15.372 | 0.990 / ROI 10.374 | 0.492 / ROI 3.310 | 0.438 / ROI 2.199 | Worse than 4096 |

Selected-checkpoint view failures:

| Run | Failing view(s) | Largest blob | Overlay |
|---|---|---:|---|
| `s42`, 4096 rays | `eval_img_0000=0.314`; `0001=0.000`; `0002=0.000` | 413 px | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_retest/lookcloser/arm_h40_grid128_warm4096_s42/artifact_renders_artifact_selection_step-000015187/eval_img_0000_boxes.png` |
| `s43`, 4096 rays | `eval_img_0000=0.141`; `0001=0.337`; `0002=0.000` | 430 px | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_retest/lookcloser/arm_h40_grid128_warm4096_s43/artifact_renders_artifact_selection_step-000015187/eval_img_0001_boxes.png` |
| `s42`, 8192 rays | `eval_img_0000=0.286`; `0001=0.267`; `0002=0.438` | 470 px | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_retest/lookcloser/arm_h40_grid128_warm4096_batch8192_s42/artifact_renders_artifact_selection_step-000015187/eval_img_0002_boxes.png` |

Visual read: the accepted-by-selection batch-4096 artifacts are small wall/cable-region vertical blobs, not the earlier obvious broken stand/hand/cable failures. They are still serious under the significant detector, so these runs are rejected.

Occupancy-debug read:

| Run | Debugged view | `grid_miss_likely` | `field_issue_likely` | Debug summary |
|---|---|---|---|---|
| `s42`, 4096 rays | eval0 step 15187 | false | true | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_retest/lookcloser/arm_h40_grid128_warm4096_s42/artifact_occ_debug_eval0_step15187/artifact_occupancy_debug.md` |
| `s43`, 4096 rays | eval1 step 15187 | false | true | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_retest/lookcloser/arm_h40_grid128_warm4096_s43/artifact_occ_debug_eval1_step15187/artifact_occupancy_debug.md` |
| `s42`, 8192 rays | eval2 step 15187 | false | false | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_retest/lookcloser/arm_h40_grid128_warm4096_batch8192_s42/artifact_occ_debug_eval2_step15187/artifact_occupancy_debug.md` |

### Insights

- Changing old H40 from `grid_resolution=64` to `128`, shortening to `15188` iterations, and using occupancy/binary warmup `4096` improves the artifact trend over time, but it does not make ARM artifact-clean.
- The two seed batch-4096 runs are consistent: final selected checkpoint, ROI clean, full-frame artifact around `0.31-0.34`.
- Batch `8192` is not a useful fix here. It has lower PSNR than batch `4096`, higher full-frame artifact, and nonzero curated ROI artifact.
- The debugger again points away from binary occupancy misses for the primary runs. The artifact pixels mostly project through occupied voxels, so blindly increasing occupancy conservativeness is not supported by this test.
- Do not resume Feature Reweighting yet. The next step should stay FR-off and isolate why ARM leaves these small full-frame wall/cable blobs while no-ARM occupancy runs can be clean.

## ARM H40 grid128 long-budget retest

### What was tested

Retested the same H40-style ARM recipe after replacing the short diagnostic cap with the current long-budget policy:

- ARM on: `ray_sampling_mode=adaptive`
- FAS off, Feature Reweighting off
- `grid_resolution=128`
- `train_num_rays_per_batch=4096` for seeds `42` and `43`
- additional `train_num_rays_per_batch=8192` seed-42 comparison
- `occupancy_warmup_steps=4096`
- `occupancy_binary_warmup_steps=4096`
- `occupancy_grid_levels=1`
- `render_step_size_mult=1.0`
- `alpha_thre=0.0`
- `transmittance_threshold=0.0`
- `near_plane=0.01`
- `cone_angle=0.0`
- H40 ARM settings preserved: `adaptive_warmup_steps=2048`, `adaptive_coarse_step_size=0.0125`, `adaptive_max_frequency_level=12`, `max_steps_per_ray=1024`
- `max_num_iterations=90000`
- default early stop on eval-loss plateau
- checkpoint/eval cadence `4096`
- `--eval-checkpoint artifact`, all three eval views scored

Experiment root:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_long90000/lookcloser`

### Results

The long budget confirms that the `15188` cap was too short for ARM. Metrics keep improving past the old short run, and one primary seed becomes fully artifact-clean. The result is not yet final because seed `42` still fails the artifact/ROI gate on a thin wall/cable segment.

| Run | Seed | Train rays | Stop / selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_long90k_s42` | 42 | 4096 | stop after 32768 / select 32768 | 29.2119 | 0.6738 | 0.4274 | 0.269 | 1.863 | 1.863 | 5441.2 | 5979.1 | Reject: nonzero full-frame and ROI artifact | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_long90000/lookcloser/arm_h40_grid128_long90k_s42/renders_artifact_selection_step-000032768` |
| `arm_h40_grid128_long90k_s43` | 43 | 4096 | stop after 24576 / select 24576 | 29.0268 | 0.6569 | 0.4370 | 0.000 | 0.000 | 0.000 | 4719.7 | 5410.6 | Accept as artifact-clean candidate; needs visual/variance confirmation | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_long90000/lookcloser/arm_h40_grid128_long90k_s43/renders_artifact_selection_step-000024576` |
| `arm_h40_grid128_long90k_s44` | 44 | 4096 | stop after 28672 / select 28672 | 29.0364 | 0.6586 | 0.4345 | 0.122 | 1.604 | 1.604 | 2074.3 | 2555.7 | Reject: nonzero full-frame and ROI artifact | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_long90000/lookcloser/arm_h40_grid128_long90k_s44/renders_artifact_selection_step-000028672` |
| `arm_h40_grid128_long90k_batch8192_s42` | 42 | 8192 | stop after 20480 / select 12288 | 27.6435 | 0.6545 | 0.4554 | 0.182 | 5.375 | 2.117 | 4118.3 | 4856.3 | Reject: ROI/stand failure and weak metrics | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_long90000/lookcloser/arm_h40_grid128_long90k_batch8192_s42/renders_artifact_selection_step-000012288` |

Primary 3-seed batch-4096 summary, excluding the batch-8192 comparison:

| Metric | Mean | Median |
|---|---:|---:|
| PSNR | 29.0917 | 29.0364 |
| SSIM | 0.6631 | 0.6586 |
| LPIPS | 0.4330 | 0.4345 |
| Full-frame artifact | 0.130 | 0.122 |
| ROI artifact | 1.156 | 1.604 |
| Stand connector | 1.156 | 1.604 |
| Train time (s) | 4078.4 | 4719.7 |
| Total time (s) | 4648.5 | 5410.6 |
| Clean primary seeds | 1 / 3 | - |

Comparison to the old H40 metric leader:

| Run | PSNR delta vs old H40 | SSIM delta vs old H40 | LPIPS delta vs old H40 | Artifact gate |
|---|---:|---:|---:|---|
| `s42`, 4096 rays | +0.3137 | +0.0079 | +0.0621 worse | Fail |
| `s43`, 4096 rays | +0.1286 | -0.0090 | +0.0717 worse | Pass |
| `s44`, 4096 rays | +0.1382 | -0.0073 | +0.0692 worse | Fail |
| `s42`, 8192 rays | -1.2547 | -0.0114 | +0.0901 worse | Fail |

Candidate artifact trend:

| Run | Candidate timeline | Read |
|---|---|---|
| `s42`, 4096 rays | `4096: 0.736 / ROI 9.400`; `8192: 0.983 / ROI 4.099`; `12288: 0.513 / ROI 5.253`; `16384: 0.448 / ROI 4.128`; `20480: 0.445 / ROI 2.040`; `24576: 0.428 / ROI 3.988`; `28672: 0.384 / ROI 1.690`; `32768: 0.269 / ROI 1.863` | Improves but never clears artifact/ROI gate |
| `s43`, 4096 rays | `4096: 0.839 / ROI 15.159`; `8192: 0.467 / ROI 3.178`; `12288: 0.172 / ROI 1.551`; `16384: 0.109 / ROI 1.437`; `20480: 0.127 / ROI 0.000`; `24576: 0.000 / ROI 0.000` | Clean only at the final saved checkpoint |
| `s44`, 4096 rays | `4096: 0.778 / ROI 9.232`; `8192: 0.449 / ROI 4.945`; `12288: 0.660 / ROI 0.000`; `16384: 0.228 / ROI 2.849`; `20480: 0.253 / ROI 2.760`; `24576: 0.668 / ROI 3.938`; `28672: 0.122 / ROI 1.604` | Improves but never clears artifact/ROI gate |
| `s42`, 8192 rays | `4096: 13.885 / ROI 0.000`; `8192: 0.868 / ROI 7.504`; `12288: 0.182 / ROI 5.375`; `16384: 0.965 / ROI 0.000`; `20480: 0.738 / ROI 0.000` | Artifact-selected checkpoint still fails ROI/stand gate |

Selected-checkpoint view failures:

| Run | Per-view full-frame scores | Largest failing blob | Overlay / debug |
|---|---|---:|---|
| `s42`, 4096 rays | `eval_img_0000=0.141`, `eval_img_0001=0.269`, `eval_img_0002=0.000` | 375 px | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_long90000/lookcloser/arm_h40_grid128_long90k_s42/artifact_renders_artifact_selection_step-000032768/eval_img_0001_boxes.png` |
| `s43`, 4096 rays | `eval_img_0000=0.000`, `eval_img_0001=0.000`, `eval_img_0002=0.000` | 0 px | clean selected checkpoint |
| `s44`, 4096 rays | `eval_img_0000=0.122`, `eval_img_0001=0.110`, `eval_img_0002=0.000` | 292 px | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_long90000/lookcloser/arm_h40_grid128_long90k_s44/artifact_renders_artifact_selection_step-000028672/eval_img_0000_boxes.png` |
| `s42`, 8192 rays | `eval_img_0000=0.161`, `eval_img_0001=0.122`, `eval_img_0002=0.182` | 428 px | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_long90000/lookcloser/arm_h40_grid128_long90k_batch8192_s42/artifact_renders_artifact_selection_step-000012288/eval_img_0002_boxes.png` |

Visual read:

- `s43` selected render looks artifact-clean on the checked full eval0 view: no obvious broken stand, hand, or cable failure on the global render. This still needs focused crop review before treating the recipe as final.
- `s42` selected render fails on a small vertical wall/cable segment above the left actor's head. The ROI overlay also flags `left_stand_connector_eval0` with score `1.863`, so the run must be rejected even though PSNR/SSIM are strong.
- `s44` selected render fails in the same family: a small vertical top-left wall/cable/stand segment. Full-frame score is lower than s42 but still nonzero, and ROI/stand score is `1.604`.
- `batch8192` is rejected: it does not improve artifact behavior and its selected checkpoint is much weaker on PSNR/LPIPS.

Occupancy-debug read:

| Run | Debugged view | `grid_miss_likely` | Debug summary |
|---|---|---:|---|
| `s42`, 4096 rays | eval1 step 32768 | false | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_long90000/lookcloser/arm_h40_grid128_long90k_s42/artifact_occ_debug_eval1_step32768/artifact_occupancy_debug.md` |
| `s44`, 4096 rays | eval0 step 28672 | false | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_long90000/lookcloser/arm_h40_grid128_long90k_s44/artifact_occ_debug_eval0_step28672/artifact_occupancy_debug.md` |

The debug reads say the artifact pixels mostly map to occupied voxels. This again points away from occupancy-grid conservativeness and toward ARM sampling / field trajectory / checkpoint variance for thin wall-cable structures.

### Insights

- Long training plus artifact-aware checkpoint selection can make ARM artifact-clean: seed `43` reaches `artifact_score=0.000`, ROI `0.000`, and PSNR `29.0268`.
- The same recipe is not variance-safe: only `1/3` primary seeds are clean. Seeds `42` and `44` reach strong metrics but still fail full-frame and ROI gates on thin top-left wall/cable/stand segments.
- The long-budget clean seed supports the hypothesis that previous short ARM/grid128 results were undertrained. It does not prove the recipe is final because the strict objective requires reliable artifact-free behavior, not one clean seed.
- Do not test no-ARM controls here; occupancy-only/no-ARM is already known artifact-clean and does not address the ARM failure mode.
- Do not resume Feature Reweighting. The next tests should stay ARM-on, FR-off.

### Next step

The uniform max-frequency diagnostic below rejected a blunt frequency-map explanation. The next ARM-only fix is code synchronization: make the adaptive ARM occupancy traversal pass `transmittance_threshold=0.0` through to nerfacc as `early_stop_eps`, matching the occupancy-only safe-default path. After that, retest failing ARM seeds before touching Feature Reweighting. Do not run additional no-ARM controls for this stage; occupancy-only/no-ARM is already known to be the clean reference and does not isolate the ARM failure.

## ARM uniform max-frequency diagnostic

### What was tested

Tested whether the failing seed `42` and `44` cable/stand artifacts were caused by the runtime frequency map assigning intervals that were too coarse for thin structures. ARM stayed enabled and Feature Reweighting/FAS stayed disabled. The only intended interval-sizing change from the long-budget H40/grid128 recipe was:

- `adaptive_min_frequency_level=15`
- `adaptive_max_frequency_level=15`

This forces uniform max-frequency interval sizing while preserving ARM and occupancy traversal.

Experiment root:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_uniformmax15/lookcloser`

### Results

Uniform max-frequency sizing should be rejected. Both runs saturated `max_steps_per_ray=1024` for nearly every ray, overfit-stopped at step `8192`, and had catastrophic image metrics. It did not fix artifacts.

| Run | Seed | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_uniformmax15_s42` | 42 | 8192 | 14.2599 | 0.3864 | 0.9820 | 1.879 | 7.658 | 0.000 | 2734.4 | 3237.4 | Reject: sample-cap saturation and severe underfit | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_uniformmax15/lookcloser/arm_h40_grid128_uniformmax15_s42/renders_artifact_selection_step-000008192` |
| `arm_h40_grid128_uniformmax15_s44` | 44 | 8192 | 13.8420 | 0.3795 | 0.9742 | 1.235 | 7.071 | 0.000 | 2734.8 | 3236.9 | Reject: sample-cap saturation and severe underfit | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_uniformmax15/lookcloser/arm_h40_grid128_uniformmax15_s44/renders_artifact_selection_step-000008192` |

Checkpoint trend:

| Run | Step 4096 | Step 8192 | Read |
|---|---|---|---|
| `s42` | PSNR `14.3913`, SSIM `0.4040`, LPIPS `0.9603`, artifact `2.436`, ROI `7.568` | PSNR `14.2599`, SSIM `0.3864`, LPIPS `0.9820`, artifact `1.879`, ROI `7.658` | Overfit-stop; selected artifact checkpoint still unusable |
| `s44` | PSNR `13.8655`, SSIM `0.3998`, LPIPS `0.9665`, artifact `2.748`, ROI `7.110` | PSNR `13.8420`, SSIM `0.3795`, LPIPS `0.9742`, artifact `1.235`, ROI `7.071` | Overfit-stop; selected artifact checkpoint still unusable |

Sampling read:

- `train_samples_per_ray` stayed effectively at `1024`.
- `max_steps_per_ray` stayed `1024`.
- saturation stayed near `1.0`.
- GPU memory for two parallel runs rose to about `36G`.
- Final eval per checkpoint took about `218-220s`, much slower than the normal long-budget H40/grid128 final eval around `38-78s`.

### Insights

- The residual cable/stand artifact is not fixed by forcing every ARM interval to max frequency level.
- Uniform level 15 is too aggressive: it saturates the sample cap, makes optimization collapse, and worsens both metrics and artifacts.
- The next ARM-only direction should be less blunt than uniform level 15. Reasonable candidates are a moderate interval floor such as `adaptive_min_frequency_level=4` or `8`, or a finer nerfacc coarse traversal check, but new runs should wait until FSX free space is safer or rejected checkpoints are pruned.

## ARM transmittance-threshold sync

### What was changed

Code audit found an ARM-only mismatch in the occupancy traversal call. The normal occupancy sampler passed `transmittance_threshold` into nerfacc as `early_stop_eps`, but `FrequencyAwareVolumetricSampler` did not expose or pass that argument. That meant ARM runs using `transmittance_threshold=0.0` still used nerfacc's default early-stop epsilon during coarse occupancy traversal.

Changed files:

- `/home/ubuntu/repos/nerfstudio/nerfstudio/model_components/lookcloser_samplers.py`
- `/home/ubuntu/repos/nerfstudio/nerfstudio/models/lookcloser.py`

The ARM sampler now accepts `early_stop_eps` and passes it to `occupancy_grid.sampling`; `LookCloserModel.adaptive_ray_marching` passes `float(self.config.transmittance_threshold)`.

### Verification

- `python -m py_compile ../nerfstudio/model_components/lookcloser_samplers.py ../nerfstudio/models/lookcloser.py`

### Retest result: seed 42

The first failing-seed retest used the H40/grid128 long-budget recipe with ARM enabled, FAS disabled, Feature Reweighting disabled, safe occupancy warmup `4096/4096`, and artifact-aware selection over `eval_img_0000.png,eval_img_0001.png,eval_img_0002.png`.

The initial FSX run reached step `8192`, then FSX quota/inodes prevented writing the step-12288 checkpoint. Non-selected checkpoints from completed old ARM runs were moved, not deleted, to local archive:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/fsx_ckpt_archive_20260617`

The completed retest resumed from the saved step-8192 checkpoint and wrote outputs locally to avoid FSX write failures:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix/lookcloser/arm_h40_grid128_transfix_s42_localresume8192`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix/lookcloser/arm_h40_grid128_transfix_s42_localresume8192/renders_artifact_selection_step-000032768`

Focused ROI evidence:

- `left_stand_connector_eval0`: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix/lookcloser/arm_h40_grid128_transfix_s42_localresume8192/artifact_renders_artifact_selection_step-000032768/roi_scores/left_stand_connector_eval0_boxes.png`
- `tangled_cable_eval2`: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix/lookcloser/arm_h40_grid128_transfix_s42_localresume8192/artifact_renders_artifact_selection_step-000032768/roi_scores/tangled_cable_eval2_boxes.png`
- `stand_label_eval2`: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix/lookcloser/arm_h40_grid128_transfix_s42_localresume8192/artifact_renders_artifact_selection_step-000032768/roi_scores/stand_label_eval2_boxes.png`

| Run | Seed | Start | Stop / selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arm_h40_grid128_transfix_s42_localresume8192` | 42 | resume from step 8192 | stop after 32768 / select 32768 | 29.2768 | 0.6710 | 0.4235 | 0.000 | 0.000 | 0.000 | 0.000 | 1563.1 resume-only; about 2133 including source step-8192 training | 1944.9 resume-only | Accept as clean single-seed candidate; needs variance confirmation |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8192 | 27.7624 | 0.6294 | 0.4799 | 0.372 | 0.000 | 0.000 | Salvage eval from the failed FSX source run; too early |
| 12288 | 28.4072 | 0.6472 | 0.4492 | 0.270 | 0.000 | 0.000 | Improving but not clean |
| 16384 | 28.7176 | 0.6567 | 0.4223 | 0.143 | 0.000 | 0.000 | Better artifact score but not clean |
| 20480 | 29.0036 | 0.6683 | 0.4291 | 0.242 | 0.000 | 0.000 | Metrics improve, artifact regresses |
| 24576 | 29.1078 | 0.6658 | 0.4274 | 0.592 | 0.000 | 0.000 | Reject; artifact spike |
| 28672 | 29.2415 | 0.6657 | 0.4244 | 0.125 | 0.000 | 0.000 | Best eval loss, but not artifact-clean |
| 32768 | 29.2768 | 0.6710 | 0.4235 | 0.000 | 0.000 | 0.000 | Selected by artifact gate |

Per-view artifact scores at selected step `32768`:

| View | Artifact score | Largest blob |
|---|---:|---:|
| `eval_img_0000.png` | 0.000 | 0 px |
| `eval_img_0001.png` | 0.000 | 0 px |
| `eval_img_0002.png` | 0.000 | 0 px |

Visual read:

- Full `eval_img_0000` does not show the previous broken stand/hand/cable failure.
- `left_stand_connector_eval0` is visually clean: the vertical stand and connector are continuous, with no obvious holes or bitten-looking segment.
- `tangled_cable_eval2` and `stand_label_eval2` are also clean under the ROI detector and visual inspection.

### Current conclusion

The ARM transmittance-threshold sync is a real low-hanging fix candidate. On the previously failing seed `42`, it changed the H40/grid128 long-budget retest from nonzero artifact (`0.269` at old selected step 32768) to `artifact_score=0.000`, ROI `0.000`, stand `0.000`, while improving metrics versus old H40 on PSNR and SSIM:

- PSNR: `29.2768` vs old H40 `28.8982`
- SSIM: `0.6710` vs old H40 `0.6659`
- LPIPS: `0.4235`, still worse than old H40 `0.3653`

This is not final because it is only one post-fix seed and LPIPS is still weak. Next step is variance confirmation with ARM enabled and Feature Reweighting/FAS disabled. Start with the other previously failing seed `44`; if it also clears, rerun seed `43` under the same local-output setup for the 3-seed post-fix estimate.

### Retest result: seed 44

Seed `44` was rerun from scratch with the same post-fix H40/grid128 recipe and local output:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix/lookcloser/arm_h40_grid128_transfix_s44_local`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix/lookcloser/arm_h40_grid128_transfix_s44_local/renders_artifact_selection_step-000020480`

| Run | Seed | Stop / selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arm_h40_grid128_transfix_s44_local` | 44 | stop after 32768 / select 20480 | 28.9101 | 0.6615 | 0.4389 | 0.241 | 0.241 | 0.000 | 0.000 | 2104.4 | 2631.6 | Reject: full-frame artifact remains |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 22.9455 | 0.6257 | 0.5556 | 0.630 | 14.584 | 0.000 | Early ROI failure |
| 8192 | 27.7088 | 0.6257 | 0.4854 | 0.309 | 2.964 | 0.000 | Still early/dirty |
| 12288 | 28.3167 | 0.6423 | 0.4597 | 0.403 | 0.000 | 0.000 | ROI clean, full-frame nonzero |
| 16384 | 28.7101 | 0.6585 | 0.4479 | 0.259 | 0.000 | 0.000 | Full-frame nonzero |
| 20480 | 28.9101 | 0.6615 | 0.4389 | 0.241 | 0.000 | 0.000 | Selected by artifact gate but rejected |
| 24576 | 29.0682 | 0.6655 | 0.4359 | 0.245 | 0.000 | 0.000 | Similar artifact |
| 28672 | 29.1371 | 0.6674 | 0.4341 | 0.359 | 0.000 | 0.000 | Best eval loss, artifact worse |
| 32768 | 29.2126 | 0.6687 | 0.4328 | 0.242 | 0.000 | 0.000 | Metrics improve, artifact remains |

Selected-checkpoint per-view artifact:

| View | Artifact score | Largest blob |
|---|---:|---:|
| `eval_img_0000.png` | 0.241 | 580 px |
| `eval_img_0001.png` | 0.189 | 444 px |
| `eval_img_0002.png` | 0.000 | 0 px |

Overlay:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix/lookcloser/arm_h40_grid128_transfix_s44_local/artifact_renders_artifact_selection_step-000020480/eval_img_0000_boxes.png`

Artifact-to-occupancy debug:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix/lookcloser/arm_h40_grid128_transfix_s44_local/artifact_occ_debug_eval0_step20480/artifact_occupancy_debug.md`

Result: `grid_miss_likely=false`, `field_issue_likely=true`. The remaining failure is the same top-left thin wall/cable segment family and not a binary occupancy pruning miss.

### Updated conclusion

The transmittance-threshold sync is necessary but not sufficient. It fixed seed `42` but seed `44` still fails full-frame artifact selection with ROI/stand clean. Since the failing pixels already project through occupied voxels, do not make the occupancy grid more conservative and do not enable Feature Reweighting. The next ARM-only low-hanging test is finer coarse traversal on the failing seed, e.g. `adaptive_coarse_step_size=0.00625` with the same grid128/safe-occupancy recipe, to reduce midpoint/coarse-interval misses before frequency subdivision.

## ARM finer coarse traversal diagnostic

### What was tested

Tested the next ARM-only low-hanging fix on failing seed `44`: reduce `adaptive_coarse_step_size` from `0.0125` to `0.00625`, keeping ARM enabled, FAS disabled, Feature Reweighting disabled, grid128, safe occupancy warmup `4096/4096`, and artifact-aware selection over all three eval views.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625/lookcloser/arm_h40_grid128_transfix_coarse00625_s44_local`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625/lookcloser/arm_h40_grid128_transfix_coarse00625_s44_local/renders_artifact_selection_step-000012288`

Focused ROI:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625/lookcloser/arm_h40_grid128_transfix_coarse00625_s44_local/artifact_renders_artifact_selection_step-000012288/roi_scores/left_stand_connector_eval0_boxes.png`

### Results

| Run | Seed | Coarse step | Stop / selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arm_h40_grid128_transfix_coarse00625_s44_local` | 44 | 0.00625 | stop after 20480 / select 12288 | 29.0247 | 0.6541 | 0.4363 | 0.000 | 0.000 | 0.000 | 0.000 | 1382.8 | 1709.3 | Clean but metrics not enough; tune checkpoint density |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 23.2623 | 0.6513 | 0.5297 | 0.368 | 8.638 | 0.000 | Early ROI failure |
| 8192 | 28.3689 | 0.6379 | 0.4589 | 0.000 | 0.000 | 0.000 | Clean but undertrained |
| 12288 | 29.0247 | 0.6541 | 0.4363 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |
| 16384 | 29.3025 | 0.6634 | 0.4275 | 0.264 | 0.000 | 0.000 | Metrics improve, artifact returns |
| 20480 | 29.4016 | 0.6662 | 0.4201 | 0.106 | 0.000 | 0.000 | Strong metrics, still nonzero artifact |

Per-view selected artifact scores:

| View | Artifact score | Largest blob |
|---|---:|---:|
| `eval_img_0000.png` | 0.000 | 0 px |
| `eval_img_0001.png` | 0.000 | 0 px |
| `eval_img_0002.png` | 0.000 | 0 px |

Visual read:

- Full eval0 selected render shows no obvious broken top-left cable/stand segment.
- `left_stand_connector_eval0` is clean by detector and visual inspection.

### Insight

Finer coarse traversal changes the seed44 failure mode in the right direction: it creates artifact-clean checkpoints, unlike the post-fix `0.0125` run. However, artifact-clean selection lands early at step `12288`, with SSIM below old H40. Because step `16384` and `20480` have much better metrics but nonzero artifact, the next low-risk test is not another hyperparameter: save/evaluate denser checkpoints between `12288` and `16384` with the same `0.00625` recipe to find the latest artifact-clean point before the thin-segment artifact appears.

## ARM coarse00625 dense checkpoint scan

### What was tested

Two short continuation scans narrowed the clean/dirty transition for seed `44` with `adaptive_coarse_step_size=0.00625`, ARM enabled, FAS/Feature Reweighting disabled.

Scan 1 started from the clean step `12288` checkpoint and saved/evaluated every `1024` steps:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_dense/lookcloser/arm_h40_grid128_transfix_coarse00625_s44_dense12288_16384`

Scan 2 started from the clean step `14336` checkpoint and saved/evaluated every `512` steps:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_dense2/lookcloser/arm_h40_grid128_transfix_coarse00625_s44_dense14336_15360`

### Results

| Run | Seed | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `coarse00625_s44_dense12288_16384` | 44 | 14336 | 29.1356 | 0.6569 | 0.4235 | 0.000 | 0.000 | 0.000 | 211.4 | 338.4 | Best clean seed44 so far, but SSIM below target | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_dense/lookcloser/arm_h40_grid128_transfix_coarse00625_s44_dense12288_16384/renders_artifact_selection_step-000014336` |
| `coarse00625_s44_dense14336_15360` | 44 | 14848 | 29.1191 | 0.6590 | 0.3952 | 0.156 | 0.000 | 0.000 | 120.1 | 249.5 | Reject: nonzero artifact despite strong LPIPS | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_dense2/lookcloser/arm_h40_grid128_transfix_coarse00625_s44_dense14336_15360/renders_artifact_selection_step-000014848` |

Dense candidate timeline:

| Step | Source scan | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 13312 | dense 1024 | 29.1366 | 0.6565 | 0.4295 | 0.126 | 0.000 | 0.000 | Dirty |
| 14336 | dense 1024 | 29.1356 | 0.6569 | 0.4235 | 0.000 | 0.000 | 0.000 | Best clean seed44 checkpoint |
| 14848 | dense 512 | 29.1191 | 0.6590 | 0.3952 | 0.156 | 0.000 | 0.000 | Dirty, but LPIPS much better |
| 15359 | dense 512 | 29.1302 | 0.6587 | 0.4040 | 0.248 | 0.000 | 0.000 | Dirty |

### Insight

The artifact boundary is narrow and non-monotonic: step `13312` is dirty, `14336` is clean, and `14848+` is dirty again. The best current seed44 artifact-clean candidate is step `14336`. It improves PSNR over old H40 and LPIPS versus the coarse00625 step `12288` clean checkpoint, but still misses the old H40 SSIM target (`0.6569` vs `0.6659`) and LPIPS remains worse than the old metric leader (`0.4235` vs `0.3653`).

Local disk is now too tight for more scans (`~6.8G` free after dense2). Next work requires cleanup/pruning or moving old checkpoints. After cleanup, the best next options are:

- Micro-scan `14336 -> 14848` at `256` or `128` step spacing to find the latest clean seed44 point.
- Run the same `coarse00625` recipe on seed `42` and seed `43` for variance; seed42 may retain its clean 32768 behavior, but metrics/LPIPS need confirmation under the finer traversal.

## ARM coarse00625 micro checkpoint scans

### What was tested

After pruning local non-selected checkpoints from completed local runs, two seed44 micro scans were run with the same `adaptive_coarse_step_size=0.00625` recipe:

- `14336 -> 14848`, checkpoint/eval every `128`, normal early-stop.
- `14592 -> 14848`, checkpoint/eval every `128`, `--no-stop-on-no-improve`.

Runs:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_s44_micro14336_14848`

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_micro2/lookcloser/arm_h40_grid128_transfix_coarse00625_s44_micro14592_14848`

### Results

| Step | Source scan | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 14464 | micro 128 | 29.1130 | 0.6562 | 0.4079 | 0.000 | 0.000 | 0.000 | Clean; runner selected by LPIPS tie-break |
| 14592 | micro 128 | 29.1482 | 0.6588 | 0.4082 | 0.000 | 0.000 | 0.000 | Clean; best seed44 PSNR/SSIM clean point so far |
| 14720 | micro2 128 | 28.9089 | 0.6545 | 0.3857 | 0.000 | 0.000 | 0.000 | Clean; best seed44 LPIPS clean point so far |
| 14847 | micro2 128 | 28.9907 | 0.6543 | 0.3816 | 0.148 | 0.000 | 0.000 | Dirty despite best LPIPS |

Selected clean render examples:

- Step `14592` run-selected neighbor: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_s44_micro14336_14848/renders_artifact_selection_step-000014464`
- Step `14720`: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_micro2/lookcloser/arm_h40_grid128_transfix_coarse00625_s44_micro14592_14848/renders_artifact_selection_step-000014720`

Visual read:

- Step `14720` full eval0 and `left_stand_connector_eval0` crop were visually clean: no obvious top-left cable/stand hole and no broken stand connector.

### Insight

The clean/dirty boundary remains narrow. The current best seed44 clean candidates are:

- Quality-balanced: step `14592`, PSNR `29.1482`, SSIM `0.6588`, LPIPS `0.4082`.
- LPIPS-balanced: step `14720`, PSNR `28.9089`, SSIM `0.6545`, LPIPS `0.3857`.

Neither reaches the old H40 SSIM `0.6659` and LPIPS `0.3653` together. The artifact fix path is promising, but metric recovery still needs variance and/or another ARM-only hyperparameter. Next run should test the same `coarse00625` recipe on seed42 or seed43 before changing modules, because seed44 may be the weak seed for SSIM.

## ARM coarse00625 seed42 variance check

### What was tested

Ran the same ARM-only `adaptive_coarse_step_size=0.00625` recipe on seed `42`, from scratch, with grid128, safe occupancy warmup `4096/4096`, FAS disabled, Feature Reweighting disabled, and artifact-aware selection over all three eval views.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625/lookcloser/arm_h40_grid128_transfix_coarse00625_s42_local`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625/lookcloser/arm_h40_grid128_transfix_coarse00625_s42_local/renders_artifact_selection_step-000028672`

### Results

| Run | Seed | Stop / selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arm_h40_grid128_transfix_coarse00625_s42_local` | 42 | stop after 32768 / select 28672 | 29.4857 | 0.6824 | 0.4095 | 0.000 | 0.000 | 0.000 | 0.000 | 2194.3 | 2804.8 | Strong clean candidate; LPIPS still worse than old H40 |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 23.6290 | 0.6501 | 0.5306 | 0.000 | 0.000 | 0.000 | Clean but undertrained |
| 8192 | 28.1939 | 0.6498 | 0.4576 | 0.000 | 0.000 | 0.000 | Clean but under target |
| 12288 | 28.8499 | 0.6632 | 0.4327 | 0.485 | 0.000 | 0.000 | Dirty |
| 16384 | 29.2695 | 0.6725 | 0.4203 | 0.236 | 0.000 | 0.000 | Dirty |
| 20480 | 29.3918 | 0.6800 | 0.4146 | 0.150 | 0.000 | 0.000 | Dirty |
| 24576 | 29.4716 | 0.6797 | 0.4113 | 0.000 | 0.000 | 0.000 | Clean |
| 28672 | 29.4857 | 0.6824 | 0.4095 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |
| 32768 | 29.4948 | 0.6825 | 0.4092 | 0.123 | 1.623 | 1.623 | Dirty, including stand connector |

Per-view selected artifact scores:

| View | Artifact score | Largest blob |
|---|---:|---:|
| `eval_img_0000.png` | 0.000 | 0 px |
| `eval_img_0001.png` | 0.000 | 0 px |
| `eval_img_0002.png` | 0.000 | 0 px |

Visual read:

- Full eval0 selected render looks clean on the known top-left wall/cable/stand failure area.
- `left_stand_connector_eval0` is clean by detector and visual inspection.

### Insight

The `coarse00625` recipe is now clean on seed42 and seed44, but the clean checkpoint windows differ and are non-monotonic. Seed42 reaches excellent PSNR/SSIM while clean, but LPIPS remains worse than the old H40 metric leader. Seed44 can reach better clean LPIPS (`0.3857`) but lower SSIM. This points to checkpoint selection/trajectory variance as the current limiting factor, not occupancy misses or Feature Reweighting.

Next required variance step: run seed43 with the same ARM-only `coarse00625` recipe before accepting or changing the recipe.

## ARM coarse00625 seed43 variance completion

### What was tested

Ran the same ARM-only `adaptive_coarse_step_size=0.00625` recipe on seed `43`, from scratch, with grid128, safe occupancy warmup `4096/4096`, FAS disabled, Feature Reweighting disabled, early stop on eval-loss regression, and artifact-aware selection over all three eval views.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625/lookcloser/arm_h40_grid128_transfix_coarse00625_s43_local`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625/lookcloser/arm_h40_grid128_transfix_coarse00625_s43_local/renders_artifact_selection_step-000016384`

### Results

| Run | Seed | Stop / selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arm_h40_grid128_transfix_coarse00625_s43_local` | 43 | stop after 24576 / select 16384 | 29.2552 | 0.6691 | 0.4298 | 0.000 | 0.000 | 0.000 | 0.000 | 1653.4 | 2045.5 | Clean; PSNR/SSIM beat old H40, LPIPS still worse |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 23.4084 | 0.6501 | 0.5351 | 0.378 | 8.683 | 0.000 | Undertrained, ROI dirty |
| 8192 | 28.4289 | 0.6450 | 0.4658 | 0.147 | 0.000 | 0.000 | Off-ROI dirty |
| 12288 | 28.9275 | 0.6596 | 0.4420 | 0.152 | 0.000 | 0.000 | Off-ROI dirty |
| 16384 | 29.2552 | 0.6691 | 0.4298 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |
| 20480 | 29.4279 | 0.6748 | 0.4256 | 0.323 | 0.000 | 0.000 | Dirty despite better metrics |
| 24576 | 29.4736 | 0.6767 | 0.4212 | 0.175 | 1.598 | 1.598 | Dirty, including stand connector |

Visual read:

- Full eval0 selected render has no obvious broken top-left stand/cable hole.
- `left_stand_connector_eval0` detector crop is clean with score `0.000` and largest blob `0 px`.
- The selected render still looks soft/smoothed compared with the old metric leader, consistent with LPIPS remaining high.

### Three-seed ARM-only coarse00625 summary

Artifact-selected checkpoints for the current ARM-only recipe:

| Seed | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 28672 | 29.4857 | 0.6824 | 0.4095 | 0.000 | 0.000 | 0.000 | 2194.3 | 2804.8 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625/lookcloser/arm_h40_grid128_transfix_coarse00625_s42_local/renders_artifact_selection_step-000028672` |
| 43 | 16384 | 29.2552 | 0.6691 | 0.4298 | 0.000 | 0.000 | 0.000 | 1653.4 | 2045.5 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625/lookcloser/arm_h40_grid128_transfix_coarse00625_s43_local/renders_artifact_selection_step-000016384` |
| 44 | 12288 | 29.0247 | 0.6541 | 0.4363 | 0.000 | 0.000 | 0.000 | 1382.8 | 1709.3 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625/lookcloser/arm_h40_grid128_transfix_coarse00625_s44_local/renders_artifact_selection_step-000012288` |

Mean over the three artifact-selected clean checkpoints: PSNR `29.2552`, SSIM `0.6685`, LPIPS `0.4252`, train time `1743.5s`, total time `2186.5s`, full-frame/ROI/stand artifact scores all `0.000`.

### Insight

The current ARM-only fix is validated for the artifact gate across seeds 42/43/44: 3/3 artifact-selected checkpoints have full-frame significant artifact `0.000`, curated ROI `0.000`, and stand connector `0.000`. The important caveat is checkpoint timing. Later checkpoints often improve PSNR/SSIM/LPIPS but become dirty again, so eval-loss selection is unsafe for ARM.

This points away from `occupancy_binary_warmup_steps` and away from binary-grid misses as the primary remaining issue. The likely failure mode is ARM trajectory/checkpoint timing: the field can form clean geometry under finer coarse traversal, but later optimization can introduce structural blobs or stand/cable damage. The next ARM-only work should focus on recovering LPIPS/detail while preserving the clean checkpoint window, not on Feature Reweighting and not on no-ARM controls.

## ARM coarse00625 seed43 dense checkpoint refinement

### What was tested

Continued the clean seed43 checkpoint `16384` to `20480` with the same ARM-only `coarse00625` recipe, but saved/evaluated every `512` steps. This was a checkpoint-window diagnostic to recover detail/LPIPS while keeping ARM enabled and Feature Reweighting/FAS disabled. Early stop was disabled only for this short boundary scan.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_dense_seed43/lookcloser/arm_h40_grid128_transfix_coarse00625_s43_dense16384_20480`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_dense_seed43/lookcloser/arm_h40_grid128_transfix_coarse00625_s43_dense16384_20480/renders_artifact_selection_step-000019968`

### Results

| Run | Seed | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arm_h40_grid128_transfix_coarse00625_s43_dense16384_20480` | 43 | 19968 | 29.3795 | 0.6706 | 0.4186 | 0.000 | 0.000 | 0.000 | 0.000 | 480.4 | 1079.9 | Keep as refined seed43 clean checkpoint |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16896 | 29.2255 | 0.6714 | 0.4258 | 0.000 | 0.000 | 0.000 | Clean, slight LPIPS improvement |
| 17408 | 29.3041 | 0.6717 | 0.4251 | 0.121 | 0.000 | 0.000 | Dirty full-frame |
| 17920 | 29.3331 | 0.6726 | 0.4218 | 0.132 | 1.735 | 1.735 | Dirty, stand connector |
| 18432 | 29.3434 | 0.6706 | 0.4203 | 0.000 | 0.000 | 0.000 | Clean |
| 18944 | 29.3300 | 0.6706 | 0.4206 | 0.000 | 0.000 | 0.000 | Clean |
| 19456 | 29.3980 | 0.6709 | 0.4210 | 0.000 | 0.000 | 0.000 | Clean, best PSNR clean |
| 19968 | 29.3795 | 0.6706 | 0.4186 | 0.000 | 0.000 | 0.000 | Selected clean, best LPIPS clean |
| 20479 | 29.3550 | 0.6701 | 0.4177 | 0.170 | 0.000 | 0.000 | Dirty despite best LPIPS |

Visual read:

- Full eval0 selected render is clean in the known top-left stand/cable failure area.
- `left_stand_connector_eval0` detector crop is clean with score `0.000` and largest blob `0 px`.
- The image remains visibly softer than the old H40 metric leader, so this is an incremental checkpoint-selection improvement, not full LPIPS/detail recovery.

### Insight

Dense checkpointing improves seed43 without changing the ARM recipe: selected clean LPIPS improves from `0.4298` at step `16384` to `0.4186` at step `19968`, while PSNR improves from `29.2552` to `29.3795` and artifact/ROI/stand remain `0.000`. The clean/dirty pattern is non-monotonic: `17408` and `17920` are dirty, `18432` through `19968` are clean, then `20479` is dirty. This reinforces that ARM artifact control is mostly checkpoint-window selection under the finer coarse traversal, not a simple monotonic early-stop rule.

Replacing seed43 with the dense clean checkpoint gives a 3-seed artifact-clean set with approximate mean PSNR `29.2966`, SSIM `0.6690`, LPIPS `0.4215`, and artifact/ROI/stand `0.000`. LPIPS remains far from old H40 (`0.3653`), so the next ARM-only improvement should target detail recovery without Feature Reweighting.

## ARM coarse00625 batch8192 seed42 screen

### What was tested

Tested whether the old H41-style larger train batch can recover detail/LPIPS under the current fixed ARM recipe. This kept ARM enabled, FAS disabled, Feature Reweighting disabled, grid128, safe occupancy warmup `4096/4096`, `adaptive_coarse_step_size=0.00625`, `adaptive_max_frequency_level=12`, and artifact-aware selection over all three eval views. The only intended recipe change from the current batch4096 clean run was `train_num_rays_per_batch=8192`.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_batch8192/lookcloser/arm_h40_grid128_transfix_coarse00625_batch8192_s42`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_batch8192/lookcloser/arm_h40_grid128_transfix_coarse00625_batch8192_s42/renders_artifact_selection_step-000036864`

Artifact occupancy debug:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_batch8192/lookcloser/arm_h40_grid128_transfix_coarse00625_batch8192_s42/artifact_occ_debug_eval0_step36864/artifact_occupancy_debug.md`

### Results

| Run | Seed | Stop / selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arm_h40_grid128_transfix_coarse00625_batch8192_s42` | 42 | stop after 36864 / select 36864 | 29.8715 | 0.6912 | 0.4048 | 0.131 | 0.131 | 0.000 | 0.000 | 3035.8 | 3405.3 | Reject: full-frame artifact gate fails |

Candidate timeline retained for artifact selection:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16384 | 29.5322 | 0.6824 | 0.4170 | 0.170 | 0.000 | 0.000 | Dirty full-frame |
| 20480 | 29.7572 | 0.6893 | 0.4121 | 0.272 | 0.000 | 0.000 | Dirty full-frame |
| 24576 | 29.7750 | 0.6881 | 0.4082 | 0.263 | 0.000 | 0.000 | Dirty full-frame |
| 28672 | 29.9133 | 0.6930 | 0.4058 | 0.149 | 0.000 | 0.000 | Dirty full-frame |
| 32768 | 29.8812 | 0.6901 | 0.4048 | 0.154 | 0.000 | 0.000 | Dirty full-frame |
| 36864 | 29.8715 | 0.6912 | 0.4048 | 0.131 | 0.000 | 0.000 | Selected by artifact, still dirty |

Visual/debug read:

- The detector box is on `eval_img_0000.png`, in the top-left thin vertical stand/cable segment.
- Curated ROI and stand connector remain `0.000`, but the full-frame acceptance gate fails.
- Artifact-to-occupancy debug reports `grid_miss_likely=false`, `field_issue_likely=true`; artifact pixels mostly project through occupied voxels.

### Insight

Batch8192 directly tests the "not trained long enough / old H41 visual-balance" hypothesis. It strongly improves PSNR/SSIM and slightly improves LPIPS versus the batch4096 clean seed42, but it does not solve artifact cleanliness: every retained checkpoint has nonzero full-frame significant artifact. Since the debugger says the selected artifact is not an occupancy miss, the failure should not be addressed by making the occupancy grid more conservative. Reject batch8192 as the next default until there is a separate field/checkpoint fix for the top-left cable artifact.

## ARM maxfreq13 seed42 continuation

### What was tested

Tested whether the LPIPS/detail gap is caused by clamping ARM interval subdivision at `adaptive_max_frequency_level=12`. Continued the best clean seed42 batch4096 checkpoint at step `28672` with the same ARM-only `coarse00625` recipe, but changed `adaptive_max_frequency_level` from `12` to `13`. FAS and Feature Reweighting stayed disabled. This short diagnostic ran to step `32768` with `1024`-step checkpoints and artifact-aware selection over all three eval views.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq13_cont/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq13_s42_cont28672_32768`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq13_cont/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq13_s42_cont28672_32768/renders_artifact_selection_step-000032767`

### Results

| Run | Seed | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arm_h40_grid128_transfix_coarse00625_maxfreq13_s42_cont28672_32768` | 42 | 32767 | 29.5286 | 0.6832 | 0.4042 | 0.000 | 0.000 | 0.000 | 0.000 | 420.4 | 673.6 | Keep as small clean improvement; not enough for LPIPS target |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 29696 | 29.5249 | 0.6832 | 0.4085 | 0.118 | 0.000 | 0.000 | Dirty full-frame |
| 30720 | 29.4876 | 0.6835 | 0.4066 | 0.118 | 0.000 | 0.000 | Dirty full-frame |
| 31744 | 29.5102 | 0.6830 | 0.4067 | 0.121 | 0.000 | 0.000 | Dirty full-frame |
| 32767 | 29.5286 | 0.6832 | 0.4042 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |

Visual read:

- Full eval0 selected render is clean in the known top-left stand/cable failure area.
- `left_stand_connector_eval0` detector crop is clean with score `0.000` and largest blob `0 px`.
- Image still looks smoother than the old H40 metric leader; LPIPS improved only modestly.

### Insight

Raising the max frequency cap from `12` to `13` can be kept as a small ARM-only improvement on seed42: clean LPIPS improves from `0.4095` to `0.4042` while PSNR/SSIM remain strong and artifact/ROI/stand stay `0.000`. However, the gain is too small to explain the old H40 LPIPS gap, and intermediate maxfreq13 checkpoints were dirty before the final clean window. Treat maxfreq13 continuation as promising but not sufficient; it needs variance confirmation before becoming the default.

## ARM maxfreq13 seed42 longer continuation

### What was tested

Continued the clean maxfreq13 seed42 checkpoint `32767` to `36864` with the same ARM-only recipe and `1024`-step checkpoints. This directly tested whether the maxfreq13 branch simply needed more training to recover LPIPS/detail while staying artifact-clean.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq13_cont2/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq13_s42_cont32767_36864`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq13_cont2/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq13_s42_cont32767_36864/renders_artifact_selection_step-000036863`

### Results

| Run | Seed | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arm_h40_grid128_transfix_coarse00625_maxfreq13_s42_cont32767_36864` | 42 | 36863 | 29.5555 | 0.6848 | 0.4002 | 0.000 | 0.000 | 0.000 | 0.000 | 420.4 | 737.9 | Best clean seed42 detail so far; still short of old LPIPS |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32768 | 29.5328 | 0.6815 | 0.4044 | 0.000 | 0.000 | 0.000 | Clean |
| 33792 | 29.5112 | 0.6820 | 0.4040 | 0.000 | 0.000 | 0.000 | Clean |
| 34816 | 29.4728 | 0.6812 | 0.4035 | 0.000 | 0.000 | 0.000 | Clean |
| 35840 | 29.5490 | 0.6839 | 0.4031 | 0.000 | 0.000 | 0.000 | Clean |
| 36863 | 29.5555 | 0.6848 | 0.4002 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |

Visual read:

- Full eval0 selected render is clean around the known stand/cable artifact area.
- No obvious broken stand/cable artifact in the selected render, but the result is still visibly softer than GT and old H40 LPIPS target.

### Insight

The maxfreq13 branch is now the best clean seed42 high-quality candidate: it improves LPIPS from the original clean seed42 `0.4095` to `0.4002` while improving PSNR/SSIM to `29.5555` / `0.6848` and keeping all artifact gates zero. This supports the hypothesis that longer training plus a slightly finer ARM frequency cap helps, but the improvement is still not enough to match old H40 LPIPS `0.3653`. Next useful work is variance confirmation of maxfreq13 or a stronger ARM-only detail lever; batch8192 is rejected because it reintroduced full-frame artifacts.

## ARM maxfreq13 seed42 continuation to 40960

### What was tested

Continued the clean maxfreq13 seed42 checkpoint `36863` to `40960` with the same ARM-only recipe and `1024`-step checkpoints. This further tested whether the maxfreq13 branch improves detail/LPIPS with longer training.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq13_cont3/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq13_s42_cont36863_40960`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq13_cont3/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq13_s42_cont36863_40960/renders_artifact_selection_step-000038912`

### Results

| Run | Seed | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arm_h40_grid128_transfix_coarse00625_maxfreq13_s42_cont36863_40960` | 42 | 38912 | 29.5353 | 0.6823 | 0.3983 | 0.000 | 0.000 | 0.000 | 0.000 | 450.4 | 768.0 | New best clean LPIPS for high-quality seed42 branch |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 36864 | 29.5585 | 0.6830 | 0.4002 | 0.000 | 0.000 | 0.000 | Clean |
| 37888 | 29.5490 | 0.6834 | 0.3999 | 0.128 | 0.000 | 0.000 | Dirty full-frame |
| 38912 | 29.5353 | 0.6823 | 0.3983 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |
| 39936 | 29.5308 | 0.6839 | 0.3988 | 0.000 | 0.000 | 0.000 | Clean |
| 40959 | 29.5354 | 0.6831 | 0.3969 | 0.148 | 0.000 | 0.000 | Dirty despite best LPIPS |

Visual read:

- Full eval0 selected render is clean around the known top-left stand/cable failure area.
- The final checkpoint's lower LPIPS is not usable because it fails the full-frame artifact gate.

### Insight

Longer maxfreq13 continuation continues to improve clean LPIPS, but the clean/dirty boundary remains non-monotonic. The current best clean seed42 branch is step `38912`: PSNR `29.5353`, SSIM `0.6823`, LPIPS `0.3983`, artifact/ROI/stand `0.000`. This is now a meaningful improvement over the original post-fix clean seed42 LPIPS `0.4095`, but still does not reach old H40 LPIPS `0.3653`. The result supports keeping maxfreq13 as the next candidate for variance confirmation, while continuing to look for another ARM-only detail lever.

## ARM maxfreq13 seed43 variance check

### What was tested

Continued the clean seed43 maxfreq12 checkpoint `19968` to `24576` with `adaptive_max_frequency_level=13`, keeping ARM enabled and FAS/Feature Reweighting disabled. This tests whether the seed42 maxfreq13 LPIPS/detail improvement transfers to another seed.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq13_seed43/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq13_s43_cont19968_24576`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq13_seed43/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq13_s43_cont19968_24576/renders_artifact_selection_step-000021504`

### Results

| Run | Seed | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arm_h40_grid128_transfix_coarse00625_maxfreq13_s43_cont19968_24576` | 43 | 21504 | 29.3628 | 0.6730 | 0.4085 | 0.000 | 0.000 | 0.000 | 0.000 | 480.4 | 804.8 | Keep as clean seed43 improvement; variance not as strong as seed42 |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 20480 | 29.3631 | 0.6722 | 0.4074 | 0.168 | 0.000 | 0.000 | Dirty despite best LPIPS |
| 21504 | 29.3628 | 0.6730 | 0.4085 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |
| 22528 | 29.3209 | 0.6748 | 0.4100 | 0.256 | 0.000 | 0.000 | Dirty full-frame |
| 23552 | 29.4059 | 0.6758 | 0.4085 | 0.114 | 0.000 | 0.000 | Dirty full-frame |
| 24575 | 29.4505 | 0.6750 | 0.4102 | 0.000 | 0.000 | 0.000 | Clean but worse LPIPS |

Visual read:

- Full eval0 selected render is clean around the known top-left cable/stand area.
- Clean seed43 LPIPS improves from maxfreq12 step `19968` LPIPS `0.4186` to maxfreq13 step `21504` LPIPS `0.4085`.

### Insight

Maxfreq13 transfers to seed43 as a real but seed-dependent improvement. It improves clean seed43 LPIPS by about `0.010`, but the clean checkpoints still sit around LPIPS `0.408` rather than the seed42 clean `0.3983`, and several lower-LPIPS or higher-SSIM checkpoints are dirty. This supports maxfreq13 as the current best ARM-only direction, but variance confirmation is only partially positive; seed44 is still needed before accepting it as a recipe.

## ARM maxfreq13 seed44 partial attempt

### What was tested

Started seed44 maxfreq13 continuation from clean quality-balanced checkpoint `14592` to `18944`, with `512`-step checkpoints because seed44 has narrow artifact windows. ARM stayed enabled, FAS and Feature Reweighting stayed disabled.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq13_seed44/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq13_s44_cont14592_18944`

### Partial Results

This run failed with `No space left on device` before artifact-aware selection, so it is not a valid artifact-gated result. The large checkpoints were deleted to recover disk; logs and `metrics_compact.csv` remain.

Observed eval checkpoints before failure:

| Step | PSNR | SSIM | LPIPS | Read |
|---:|---:|---:|---:|---|
| 14848 | 29.0075 | 0.6548 | 0.3861 | Good LPIPS, weak SSIM; artifact unknown |
| 15360 | 29.1245 | 0.6599 | 0.4045 | Quality-balanced but LPIPS worse |
| 15872 | 29.1352 | 0.6606 | 0.4027 | Slight quality-balanced improvement |
| 16384 | 29.1211 | 0.6591 | 0.4017 | LPIPS improves slightly |
| 16896 | 29.1829 | 0.6577 | 0.4111 | LPIPS regresses |
| 17408 | 29.2481 | 0.6617 | 0.4136 | PSNR/SSIM improve, LPIPS worse |
| 17920 | 29.2690 | 0.6627 | 0.4117 | Best PSNR/SSIM observed, still below old SSIM and LPIPS weak |

### Insight

Seed44 maxfreq13 needs a rerun with more disk or fewer checkpoints before making an artifact decision. The partial metric trend is not compelling: the early LPIPS-friendly point has weak SSIM, and the later PSNR/SSIM-improving points regress LPIPS. For now, maxfreq13 is confirmed on seed42 and partially confirmed on seed43, but seed44 remains unresolved.

## ARM maxfreq13 seed44 micro artifact-gated rerun

### What was tested

Reran only the narrow seed44 maxfreq13 window from the clean maxfreq12 checkpoint `14592` to `14848`, saving checkpoints every `128` steps. ARM stayed enabled, FAS and Feature Reweighting stayed disabled. This was designed to validate the early LPIPS-friendly seed44 point without filling disk again.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq13_seed44_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq13_s44_micro14592_14848`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq13_seed44_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq13_s44_micro14592_14848/renders_artifact_selection_step-000014720`

Visual-check crops:

- `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/visual_checks/seed44_maxfreq13_step14720_eval0_render_left_cables_crop.png`
- `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/visual_checks/seed44_maxfreq13_step14720_eval0_render_center_stand_crop.png`
- `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/visual_checks/seed44_maxfreq13_step14720_eval0_render_right_stand_crop.png`

### Results

| Run | Seed | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `arm_h40_grid128_transfix_coarse00625_maxfreq13_s44_micro14592_14848` | 44 | 14720 | 28.9274 | 0.6549 | 0.3864 | 0.000 | 0.000 | 0.000 | 0.000 | 90.1 | 223.7 | Keep as clean seed44 LPIPS/detail checkpoint |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14720 | 28.9274 | 0.6549 | 0.3864 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |
| 14847 | 29.0015 | 0.6543 | 0.3818 | 0.209 | 0.000 | 0.000 | Dirty despite better LPIPS |

Clean maxfreq13 variance set:

| Seed | Selected clean maxfreq13 checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Train time used for continuation (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `s42_cont36863_40960` step `38912` | 29.5353 | 0.6823 | 0.3983 | 0.000 | 450.4 |
| 43 | `s43_cont19968_24576` step `21504` | 29.3628 | 0.6730 | 0.4085 | 0.000 | 480.4 |
| 44 | `s44_micro14592_14848` step `14720` | 28.9274 | 0.6549 | 0.3864 | 0.000 | 90.1 |
| Mean | - | 29.2752 | 0.6701 | 0.3977 | 0.000 | 340.3 |

Visual read:

- The selected eval0 render has no obvious broken stand/cable/hand artifact in the central and right stand/cable crops.
- The left cable crop is artifact-clean by detector, but very thin vertical elements are still soft/ragged; this looks like field/detail reconstruction rather than the earlier hard occupancy hole.
- The clean/dirty boundary is narrow: step `14847` has better LPIPS but fails the full-frame artifact gate.

### Insight

Maxfreq13 is now validated as artifact-clean across three seeds when selected by artifact gate, with mean PSNR `29.2752`, SSIM `0.6701`, LPIPS `0.3977`, and all full-frame/ROI/stand scores `0.000`. This improves LPIPS versus the maxfreq12 clean set, especially for seed44, but it still does not recover the old H40 LPIPS target around `0.365`. The remaining quality gap is not explained by occupancy-grid misses: dirty checkpoints keep showing nonzero full-frame artifacts while ROI/stand stay clean, and previous artifact-to-occupancy debugging points to `field_issue_likely=true`. Keep ARM on and continue with ARM-only field/traversal/detail diagnostics before any Feature Reweighting work.

## ARM maxfreq14 cap diagnostic on seed42

### What was tested

Continued clean seed42 maxfreq13 checkpoint `38912` with `adaptive_max_frequency_level=14`, keeping ARM enabled and FAS/Feature Reweighting disabled. This tests whether a slightly higher frequency cap can recover detail/LPIPS. Two caps were tested:

- `max_steps_per_ray=1024`: same cap as maxfreq13.
- `max_steps_per_ray=2048`: because the cap1024 run showed nonzero cap pressure and dirty artifacts.

Runs:

- cap1024: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq14_seed42_diag/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq14_s42_cont38912_40960`
- cap2048: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq14_cap2048_seed42_diag/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq14_cap2048_s42_cont38912_40960`

Selected cap2048 renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq14_cap2048_seed42_diag/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq14_cap2048_s42_cont38912_40960/renders_artifact_selection_step-000039936`

Visual-check crops:

- `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/visual_checks/seed42_maxfreq14_cap2048_step39936_eval0_render_left_cables_crop.png`
- `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/visual_checks/seed42_maxfreq14_cap2048_step39936_eval0_render_center_stand_crop.png`
- `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/visual_checks/seed42_maxfreq14_cap2048_step39936_eval0_render_right_stand_crop.png`

### Results

| Run | Seed | Max freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `maxfreq14_s42_cont38912_40960` | 42 | 14 | 1024 | 40959 | 29.4562 | 0.6884 | 0.3962 | 0.178 | 0.178 | 0.000 | 0.000 | 240.2 | 373.4 | Reject: dirty full-frame |
| `maxfreq14_cap2048_s42_cont38912_40960` | 42 | 14 | 2048 | 39936 | 29.4888 | 0.6852 | 0.3972 | 0.000 | 0.000 | 0.000 | 0.000 | 210.1 | 343.5 | Keep as seed42 diagnostic; needs variance |

Candidate timeline:

| Run | Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cap1024 | 39936 | 29.5064 | 0.6863 | 0.3979 | 0.213 | 0.000 | 0.000 | Dirty |
| cap1024 | 40959 | 29.4562 | 0.6884 | 0.3962 | 0.178 | 0.000 | 0.000 | Selected by artifact but still dirty |
| cap2048 | 39936 | 29.4888 | 0.6852 | 0.3972 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |
| cap2048 | 40959 | 29.4359 | 0.6845 | 0.3966 | 0.135 | 0.000 | 0.000 | Dirty despite slightly better LPIPS |

Debug:

`cap1024` selected dirty step `40959` was checked with `scripts/debug_artifact_occupancy_grid.py` on eval0:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq14_seed42_diag/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq14_s42_cont38912_40960/artifact_occ_debug_eval0_step40959/artifact_occupancy_debug.md`

The debugger reported `grid_miss_likely=false`, `field_issue_likely=true`.

Visual read:

- The cap2048 selected eval0 render has no obvious broken stand/cable/hand artifact.
- Center stand and right stand crops are visually continuous.
- Left thin cables remain soft/ragged, but there is no hard structural hole at the significant detector threshold.

### Insight

Maxfreq14 alone is too risky under the old `max_steps_per_ray=1024` cap: both saved checkpoints failed the full-frame artifact gate. Raising the cap to `2048` makes the first seed42 checkpoint artifact-clean and gives a small detail/LPIPS improvement versus maxfreq13 seed42 step `38912` (`0.3983 -> 0.3972`) while improving SSIM (`0.6823 -> 0.6852`). The effect is small and not enough to recover old H40 LPIPS `0.3653`, but it is the first evidence that maxfreq14 can be usable if the sample cap is raised. Next step is variance confirmation on seed43/seed44 before accepting maxfreq14+cap2048 as a recipe.

## ARM maxfreq14 cap2048 seed44 micro check

### What was tested

Continued the clean seed44 maxfreq13 checkpoint `14720` to `14976` with `adaptive_max_frequency_level=14` and `max_steps_per_ray=2048`, keeping ARM enabled and FAS/Feature Reweighting disabled. This targeted the seed44 LPIPS-friendly narrow window where maxfreq13 had reached good LPIPS but nearby checkpoints became dirty.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq14_cap2048_seed44_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq14_cap2048_s44_micro14720_14976`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq14_cap2048_seed44_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq14_cap2048_s44_micro14720_14976/renders_artifact_selection_step-000014975`

### Results

| Run | Seed | Max freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `maxfreq14_cap2048_s44_micro14720_14976` | 44 | 14 | 2048 | 14975 | 28.9482 | 0.6520 | 0.3753 | 0.165 | 0.165 | 0.000 | 0.000 | 90.1 | 229.5 | Reject: dirty full-frame |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14848 | 28.7557 | 0.6498 | 0.3677 | 0.205 | 0.000 | 0.000 | Near old-H40 LPIPS, but dirty |
| 14975 | 28.9482 | 0.6520 | 0.3753 | 0.165 | 0.000 | 0.000 | Selected by artifact, still dirty |

Debug:

Selected dirty step `14975` was checked with `scripts/debug_artifact_occupancy_grid.py` on eval0:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq14_cap2048_seed44_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq14_cap2048_s44_micro14720_14976/artifact_occ_debug_eval0_step14975/artifact_occupancy_debug.md`

The debugger reported `grid_miss_likely=false`, `field_issue_likely=true`.

### Insight

Maxfreq14+cap2048 is not variance-safe. It can produce LPIPS close to the old H40 metric leader on seed44 (`0.3677`), but the full-frame significant artifact gate fails on eval0/eval1 while curated ROI and stand stay `0.000`. This keeps pointing away from occupancy-grid misses and toward field/training/checkpoint instability. Do not accept maxfreq14+cap2048 yet; if explored further, it needs a way to keep the LPIPS-friendly window clean, not more occupancy conservativeness.

## ARM maxfreq13 cap2048 seed44 micro check

### What was tested

Repeated the seed44 micro cap test with the already-confirmed `adaptive_max_frequency_level=13`, raising only `max_steps_per_ray` from `1024` to `2048`. This isolates whether the dirty LPIPS-friendly maxfreq13 seed44 window was caused by sample-cap truncation rather than by the higher maxfreq14 interval schedule.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq13_cap2048_seed44_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq13_cap2048_s44_micro14720_14976`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq13_cap2048_seed44_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq13_cap2048_s44_micro14720_14976/renders_artifact_selection_step-000014848`

### Results

| Run | Seed | Max freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `maxfreq13_cap2048_s44_micro14720_14976` | 44 | 13 | 2048 | 14848 | 28.7502 | 0.6497 | 0.3681 | 0.148 | 0.148 | 0.000 | 0.000 | 90.1 | 224.6 | Reject: dirty full-frame |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14848 | 28.7502 | 0.6497 | 0.3681 | 0.148 | 0.000 | 0.000 | Near old-H40 LPIPS, but dirty |
| 14975 | 28.9567 | 0.6523 | 0.3751 | 0.148 | 0.000 | 0.000 | Dirty |

Artifact localization:

- Official failing view for selected step `14848` is `eval_img_0001.png`, not eval0: score `0.148`, largest blob `355 px`.
- Overlay: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_maxfreq13_cap2048_seed44_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_maxfreq13_cap2048_s44_micro14720_14976/artifact_renders_artifact_selection_step-000014848/eval_img_0001_boxes.png`
- The box is on the left thin vertical cable, so this is relevant to the visual wire gate, not just an irrelevant floor/equipment detector floor.
- No valid occupancy-grid debug was run on the official offending eval1 view before the rejected checkpoint was pruned. A mistaken eval0 debug exists in the run folder but should not be used as evidence for this official failure.

### Insight

Increasing `max_steps_per_ray` alone does not make the seed44 near-H40 LPIPS window artifact-clean. It lowers the dirty score versus maxfreq14+cap2048 at step `14848` (`0.205 -> 0.148`) but still fails the gate. The official failure is a left vertical cable artifact in eval1 while ROI/stand remain `0.000`. Current accepted clean recipe remains maxfreq13 with cap1024 selected at step `14720` for seed44; near-H40 LPIPS checkpoints are not yet usable.

## ARM minfreq4 cap2048 seed44 micro check

### What was tested

Continued the clean seed44 maxfreq13 checkpoint `14720` to `14976` with `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, and `max_steps_per_ray=2048`, keeping ARM enabled and FAS/Feature Reweighting disabled. This directly tested whether the left-cable artifacts in the near-H40 LPIPS window came from frequency-grid underlabeling or too-coarse intervals in low-frequency cells.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed44_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s44_micro14720_14976`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed44_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s44_micro14720_14976/renders_artifact_selection_step-000014848`

Visual-check crops:

- `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/visual_checks/seed44_minfreq4_cap2048_step14848_eval1_render_left_cable_crop.png`
- `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/visual_checks/seed44_minfreq4_cap2048_step14848_eval0_render_center_stand_crop.png`

### Results

| Run | Seed | Min freq | Max freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `minfreq4_cap2048_s44_micro14720_14976` | 44 | 4 | 13 | 2048 | 14848 | 28.7638 | 0.6501 | 0.3675 | 0.000 | 0.000 | 0.000 | 0.000 | 90.1 | 225.4 | Strong LPIPS/detail candidate; needs variance |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14848 | 28.7638 | 0.6501 | 0.3675 | 0.000 | 0.000 | 0.000 | Selected clean, near old-H40 LPIPS |
| 14975 | 28.9462 | 0.6521 | 0.3752 | 0.165 | 0.000 | 0.000 | Dirty later checkpoint |

Visual read:

- The previously failing eval1 left vertical cable is now clean by significant detector.
- The eval1 cable crop still looks thin and somewhat ragged, but there is no obvious hard hole at the detector gate.
- The eval0 center stand crop is continuous without broken stand/cable artifacts.

### Insight

`adaptive_min_frequency_level=4` is the strongest ARM-only signal so far. It turns the seed44 near-H40 LPIPS window from dirty to clean: compared with maxfreq13+cap2048 step `14848`, artifact improves `0.148 -> 0.000` while LPIPS stays essentially the same (`0.3681 -> 0.3675`). This supports the hypothesis that some thin-wire regions are under-sampled by low frequency levels or low-frequency intervals, not by missing occupancy. The tradeoff is lower PSNR/SSIM (`28.7638` / `0.6501`) than old H40 (`28.8982` / `0.6659`) and the later checkpoint becomes dirty again, so this is not final. Next step is variance confirmation of minfreq4+cap2048 on seed42/seed43 and possibly finer checkpointing around clean windows.

## ARM minfreq4 cap2048 seed42 variance check

### What was tested

Continued clean seed42 maxfreq13 checkpoint `38912` to `40960` with the same minfreq4+cap2048 settings that cleaned the seed44 cable window. ARM stayed enabled; FAS and Feature Reweighting stayed disabled.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed42_diag/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s42_cont38912_40960`

### Results

| Run | Seed | Min freq | Max freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `minfreq4_cap2048_s42_cont38912_40960` | 42 | 4 | 13 | 2048 | 39936 | 29.5156 | 0.6832 | 0.3977 | 0.213 | 0.213 | 0.000 | 0.000 | 210.2 | 337.5 | Reject: dirty full-frame |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 39936 | 29.5156 | 0.6832 | 0.3977 | 0.213 | 0.000 | 0.000 | Selected by artifact, still dirty |
| 40959 | 29.4306 | 0.6859 | 0.3964 | 0.225 | 0.000 | 0.000 | Dirty |

Artifact localization:

- Selected step `39936` official failing view is `eval_img_0002.png`: score `0.213`, largest blob `515 px`.
- `eval_img_0000.png` and `eval_img_0001.png` were clean under the significant full-frame gate.

### Insight

Minfreq4+cap2048 is not variance-safe yet. It cleaned the seed44 left-cable LPIPS window, but on seed42 it introduced or failed to avoid a full-frame eval2 artifact while giving only a tiny LPIPS change versus the clean seed42 maxfreq13 baseline (`0.3983 -> 0.3977`). Keep minfreq4 as a targeted promising direction for wire under-sampling, but do not promote it to a general ARM recipe until a seed42/seed43-safe selection or schedule is found.

## ARM minfreq2/minfreq3 cap2048 seed44 threshold checks

### What was tested

Repeated the seed44 near-H40 LPIPS micro window with lower frequency floors, `adaptive_min_frequency_level=2` and `3`, to find the least aggressive floor that cleans the eval1 left-cable artifact. ARM stayed enabled; FAS and Feature Reweighting stayed disabled.

Runs:

- minfreq2: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq2_cap2048_seed44_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq2_cap2048_s44_micro14720_14976`
- minfreq3: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq3_cap2048_seed44_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq3_cap2048_s44_micro14720_14976`

### Results

| Run | Seed | Min freq | Max freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `minfreq2_cap2048_s44_micro14720_14976` | 44 | 2 | 13 | 2048 | 14975 | 28.9537 | 0.6526 | 0.3751 | 0.148 | 0.148 | 0.000 | 0.000 | 90.1 | 224.7 | Reject: dirty eval1 cable |
| `minfreq3_cap2048_s44_micro14720_14976` | 44 | 3 | 13 | 2048 | 14975 | 28.9512 | 0.6523 | 0.3753 | 0.148 | 0.148 | 0.000 | 0.000 | 90.1 | 225.6 | Reject: dirty eval1 cable |

Candidate timeline:

| Run | Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| minfreq2 | 14848 | 28.7581 | 0.6496 | 0.3674 | 0.205 | 0.000 | 0.000 | Near-H40 LPIPS, dirty |
| minfreq2 | 14975 | 28.9537 | 0.6526 | 0.3751 | 0.148 | 0.000 | 0.000 | Dirty |
| minfreq3 | 14848 | 28.7574 | 0.6498 | 0.3675 | 0.205 | 0.000 | 0.000 | Near-H40 LPIPS, dirty |
| minfreq3 | 14975 | 28.9512 | 0.6523 | 0.3753 | 0.148 | 0.000 | 0.000 | Dirty |

### Insight

The seed44 cable cleanup threshold is not gradual in this window. Floors `2` and `3` behave like the dirty no-floor/cap2048 runs, while floor `4` is the first tested value that clears the eval1 cable artifact at the near-H40 LPIPS checkpoint. This strengthens the frequency-underlabeling/interval-size hypothesis for that cable, but it also explains why minfreq4 can be disruptive on seed42: it is a real sampling change, not a small no-op. The next useful test is not minfreq2/3; it is either a seed43 check for minfreq4 or a scheduled/localized floor that applies only where needed.

## ARM minfreq4 cap2048 seed43 variance check

### What was tested

Continued clean seed43 maxfreq13 checkpoint `21504` to `24576` with `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, and `max_steps_per_ray=2048`, keeping ARM enabled and FAS/Feature Reweighting disabled. This checks whether the seed44 minfreq4 improvement is seed-specific or transfers to the third seed.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed43_diag/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s43_cont21504_24576`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed43_diag/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s43_cont21504_24576/renders_artifact_selection_step-000022528`

### Results

| Run | Seed | Min freq | Max freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `minfreq4_cap2048_s43_cont21504_24576` | 43 | 4 | 13 | 2048 | 22528 | 29.3134 | 0.6737 | 0.4015 | 0.000 | 0.000 | 0.000 | 0.000 | 300.3 | 495.6 | Keep as clean seed43 LPIPS improvement |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 22528 | 29.3134 | 0.6737 | 0.4015 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |
| 23552 | 29.3250 | 0.6709 | 0.4026 | 0.000 | 0.000 | 0.000 | Clean but worse LPIPS |
| 24575 | 29.4065 | 0.6728 | 0.4067 | 0.149 | 0.000 | 0.000 | Dirty later checkpoint |

### Insight

Seed43 partially confirms minfreq4: it stays artifact-clean at the useful early checkpoints and improves clean seed43 LPIPS from the maxfreq13 selected `0.4085` to `0.4015`. It does not recover old-H40 LPIPS, and the later checkpoint becomes dirty again. Combined with seed44, minfreq4 is useful for detail, but seed42 remains the blocker. The next practical direction is not a global minfreq4 default; it is either denser seed42 checkpointing around the transition, a scheduled frequency floor, or a localized/frequency-map correction for under-labeled thin-wire regions.

## ARM minfreq4 cap2048 seed42 dense checkpoint check

### What was tested

Reran the first seed42 minfreq4+cap2048 window densely from clean maxfreq13 checkpoint `38912` to `39936`, saving/evaluating every `128` steps. The earlier seed42 minfreq4 check saved only 1024-step boundaries and selected dirty step `39936`; this test verifies whether a clean window exists before that boundary.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed42_dense/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s42_dense38912_39936`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed42_dense/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s42_dense38912_39936/renders_artifact_selection_step-000039424`

Visual-check crops:

- `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/visual_checks/seed42_minfreq4_cap2048_step39424_eval0_render_center_stand_crop.png`
- `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/visual_checks/seed42_minfreq4_cap2048_step39424_eval1_render_left_cable_crop.png`

### Results

| Run | Seed | Min freq | Max freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `minfreq4_cap2048_s42_dense38912_39936` | 42 | 4 | 13 | 2048 | 39424 | 29.5004 | 0.6817 | 0.3976 | 0.000 | 0.000 | 0.000 | 0.000 | 330.2 | 959.2 | Keep; seed42 clean window recovered |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 39040 | 29.5577 | 0.6843 | 0.3982 | 0.000 | 0.000 | 0.000 | Clean |
| 39168 | 29.5213 | 0.6843 | 0.3978 | 0.000 | 0.000 | 0.000 | Clean |
| 39296 | 29.5062 | 0.6839 | 0.3978 | 0.000 | 0.000 | 0.000 | Clean |
| 39424 | 29.5004 | 0.6817 | 0.3976 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |
| 39552 | 29.5163 | 0.6811 | 0.3978 | 0.000 | 0.000 | 0.000 | Clean |
| 39680 | 29.5326 | 0.6816 | 0.3977 | 0.000 | 0.000 | 0.000 | Clean |
| 39808 | 29.5868 | 0.6814 | 0.3979 | 0.000 | 0.000 | 0.000 | Clean |
| 39935 | 29.5695 | 0.6831 | 0.3977 | 0.000 | 0.000 | 0.000 | Clean |

Visual read:

- Eval0 center stand crop is continuous with no obvious broken stand/cable artifact.
- Eval1 left cable crop is clean by detector and has no hard missing segment, though the cable remains softer than GT.

### Insight

The earlier seed42 minfreq4 rejection was caused by coarse checkpoint spacing, not by the minfreq4 recipe itself. Dense checkpointing found a fully clean seed42 window and selected step `39424`. This makes minfreq4+cap2048 a real 3-seed ARM-only candidate, but it requires dense artifact-aware checkpointing because later coarse checkpoints can become dirty.

Clean minfreq4+cap2048 variance set:

| Seed | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Train time used for continuation (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `s42_dense38912_39936` step `39424` | 29.5004 | 0.6817 | 0.3976 | 0.000 | 330.2 |
| 43 | `s43_cont21504_24576` step `22528` | 29.3134 | 0.6737 | 0.4015 | 0.000 | 300.3 |
| 44 | `s44_micro14720_14976` step `14848` | 28.7638 | 0.6501 | 0.3675 | 0.000 | 90.1 |
| Mean | - | 29.1925 | 0.6685 | 0.3889 | 0.000 | 240.2 |

Compared with the clean maxfreq13 variance set, minfreq4+cap2048 improves mean LPIPS from about `0.3977` to `0.3889`, while mean PSNR/SSIM drop slightly from about `29.2752`/`0.6701` to `29.1925`/`0.6685`. It beats the old H40 PSNR/SSIM means but still trails old H40 LPIPS `0.3653`. Current best next step is to improve the seed42/43 LPIPS side without losing artifact-clean windows, likely through denser checkpoint selection, a scheduled min-frequency floor, or targeted frequency-map/interval debugging rather than occupancy-grid conservativeness.

## ARM minfreq4 cap2048 seed43 dense checkpoint check

### What was tested

Reran seed43 minfreq4+cap2048 densely from clean maxfreq13 checkpoint `21504` to `22528`, saving/evaluating every `128` steps. This checked whether the earlier coarse selected step `22528` missed a better LPIPS/detail checkpoint.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed43_dense/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s43_dense21504_22528`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed43_dense/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s43_dense21504_22528/renders_artifact_selection_step-000022016`

### Results

| Run | Seed | Min freq | Max freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `minfreq4_cap2048_s43_dense21504_22528` | 43 | 4 | 13 | 2048 | 22016 | 29.3296 | 0.6741 | 0.4025 | 0.000 | 0.000 | 0.000 | 0.000 | 360.3 | 991.6 | Reject as replacement; worse LPIPS than coarse step 22528 |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 21632 | 29.3550 | 0.6736 | 0.4035 | 0.000 | 0.000 | 0.000 | Clean |
| 21760 | 29.3302 | 0.6745 | 0.4035 | 0.000 | 0.000 | 0.000 | Clean |
| 21888 | 29.3230 | 0.6735 | 0.4030 | 0.000 | 0.000 | 0.000 | Clean |
| 22016 | 29.3296 | 0.6741 | 0.4025 | 0.000 | 0.000 | 0.000 | Selected, but worse than coarse run |
| 22144 | 29.3042 | 0.6738 | 0.4030 | 0.000 | 0.000 | 0.000 | Clean |
| 22272 | 29.2835 | 0.6734 | 0.4036 | 0.000 | 0.000 | 0.000 | Clean |
| 22400 | 29.3006 | 0.6701 | 0.4033 | 0.000 | 0.000 | 0.000 | Clean |
| 22527 | 29.2975 | 0.6711 | 0.4036 | 0.000 | 0.000 | 0.000 | Clean |

### Insight

Dense seed43 checkpointing does not improve the accepted seed43 minfreq4 result. It confirms the early window is artifact-clean, but the best dense LPIPS `0.4025` is worse than the previous coarse step `22528` LPIPS `0.4015`. Keep the previous seed43 minfreq4 checkpoint `22528` in the clean variance set. The dense run's checkpoint was pruned after recording metrics because it is not needed as a source.

## ARM minfreq4 cap2048 seed44 ultra-dense checkpoint check

### What was tested

Reran the seed44 minfreq4+cap2048 window more densely from the same clean maxfreq13 source checkpoint `14720`, saving/evaluating every `32` steps from `14720` to `14912`. ARM stayed enabled, FAS and Feature Reweighting stayed disabled. This tested whether a clean checkpoint between the previous clean step `14848` and dirty step `14975` could improve seed44 LPIPS toward the old H40 leader.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed44_ultradense/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s44_ultradense14720_14912`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed44_ultradense/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s44_ultradense14720_14912/renders_artifact_selection_step-000014816`

Dirty diagnostic render:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed44_ultradense/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s44_ultradense14720_14912/renders_artifact_selection_step-000014848/eval_img_0001.png`

Occupancy debug:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed44_ultradense/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s44_ultradense14720_14912/artifact_occ_debug_step14848_eval1_ckpt14848/artifact_occupancy_debug.md`

### Results

| Run | Seed | Min freq | Max freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `minfreq4_cap2048_s44_ultradense14720_14912` | 44 | 4 | 13 | 2048 | 14816 | 29.0228 | 0.6527 | 0.3839 | 0.000 | 0.000 | 0.000 | 0.000 | 210.1 | 683.3 | Reject as replacement; worse LPIPS than accepted seed44 step 14848 |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14752 | 28.9337 | 0.6551 | 0.3819 | 0.141 | 0.000 | 0.000 | Dirty |
| 14784 | 28.9500 | 0.6553 | 0.3842 | 0.000 | 0.000 | 0.000 | Clean but weak LPIPS |
| 14816 | 29.0228 | 0.6527 | 0.3839 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |
| 14848 | 28.8494 | 0.6530 | 0.3834 | 0.206 | 0.000 | 0.000 | Dirty; debugged on eval1 |
| 14880 | 29.0321 | 0.6530 | 0.3876 | 0.149 | 0.000 | 0.000 | Dirty |
| 14911 | 29.0069 | 0.6539 | 0.3897 | 0.149 | 0.000 | 0.000 | Dirty |

Occupancy-debug read for dirty step `14848` / `eval_img_0001.png`:

- `grid_miss_likely=false`
- `field_issue_likely=true`
- rays with any occupied voxel: `1.000`
- occupied surface voxel rate: `0.9801`
- occupancy ratio: `0.4869`

### Insight

This ultra-dense rerun is not an improvement and should not replace the accepted seed44 minfreq4 checkpoint. It also revealed an important reproducibility/trajectory issue: compared with the accepted seed44 micro run, the only recorded parameter differences are `step_interval=32` versus `128` and `max_num_iterations=14912` versus `14976`, yet the shared nominal step `14848` changed from clean LPIPS `0.3675` to dirty LPIPS `0.3834`. The most likely explanations are eval/save cadence affecting training state/RNG during continuation, or remaining nondeterminism in the ARM continuation path. Treat checkpoint cadence as part of the experimental recipe for now.

The dirty `14848` artifact is not a binary occupancy miss: the artifact rays mostly pass through occupied grid cells. This reinforces the current root-cause direction: ARM artifacts in these windows are dominated by field/training/checkpoint trajectory and artifact-aware selection, not by making the occupancy grid more conservative. Continue ARM-only debugging; do not add no-ARM controls in this phase, and do not enable Feature Reweighting until the ARM artifact-clean baseline is stable.

## ARM minfreq4 maxfreq14 cap2048 seed44 micro check

### What was tested

Tested whether combining the seed44 cable-cleaning floor (`adaptive_min_frequency_level=4`) with the near-target but previously dirty `adaptive_max_frequency_level=14` branch could improve LPIPS while keeping the artifact gate clean. ARM stayed enabled; FAS and Feature Reweighting stayed disabled. The run used the stable 128-step micro cadence from clean maxfreq13 source step `14720`.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_maxfreq14_cap2048_seed44_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_maxfreq14_cap2048_s44_micro14720_14976`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_maxfreq14_cap2048_seed44_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_maxfreq14_cap2048_s44_micro14720_14976/renders_artifact_selection_step-000014848`

### Results

| Run | Seed | Min freq | Max freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `minfreq4_maxfreq14_cap2048_s44_micro14720_14976` | 44 | 4 | 14 | 2048 | 14848 | 28.7630 | 0.6501 | 0.3675 | 0.000 | 0.000 | 0.000 | 0.000 | 90.1 | 229.2 | Reject as replacement; no improvement over maxfreq13 |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14848 | 28.7630 | 0.6501 | 0.3675 | 0.000 | 0.000 | 0.000 | Clean, same as current seed44 minfreq4 checkpoint |
| 14975 | 28.9571 | 0.6525 | 0.3747 | 0.166 | 0.000 | 0.000 | Dirty later checkpoint |

### Insight

`maxfreq14` adds no useful signal when `minfreq4` is active in this seed44 window. The selected clean checkpoint is effectively the same as the accepted minfreq4/maxfreq13 checkpoint, and the later checkpoint remains dirty. Do not spend seed42/seed43 variance budget on this combination unless a later frequency-map audit shows that level-14 cells are actually relevant to the failing thin structures.

## ARM minfreq4 cap2048 seed42 late continuation

### What was tested

Continued the accepted clean seed42 minfreq4+cap2048 checkpoint `39424` to `40960`, saving/evaluating every `256` steps. This directly tested the hypothesis that the current best clean ARM branch simply needed longer training.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed42_late/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s42_late39424_40960_i256`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed42_late/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s42_late39424_40960_i256/renders_artifact_selection_step-000039680`

### Results

| Run | Seed | Min freq | Max freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `minfreq4_cap2048_s42_late39424_40960_i256` | 42 | 4 | 13 | 2048 | 39680 | 29.5066 | 0.6818 | 0.3970 | 0.000 | 0.000 | 0.000 | 0.000 | 330.2 | 778.8 | Keep; small seed42 LPIPS improvement |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 39680 | 29.5066 | 0.6818 | 0.3970 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |
| 39936 | 29.4688 | 0.6821 | 0.3969 | 0.213 | 0.000 | 0.000 | Dirty despite slightly better LPIPS |
| 40192 | 29.4601 | 0.6805 | 0.3974 | 0.140 | 0.000 | 0.000 | Dirty |
| 40448 | 29.5496 | 0.6838 | 0.3976 | 0.140 | 0.000 | 0.000 | Dirty |
| 40704 | 29.5576 | 0.6860 | 0.3975 | 0.140 | 0.000 | 0.000 | Dirty |
| 40959 | 29.5195 | 0.6862 | 0.3969 | 0.140 | 0.000 | 0.000 | Dirty |

### Insight

Longer training helps seed42 only slightly. It moves the accepted seed42 LPIPS from `0.3976` to `0.3970`, but all later higher-SSIM or slightly lower-LPIPS checkpoints fail the full-frame artifact gate. This weakens the simple "just train longer" hypothesis: the useful clean window extends a little past step `39424`, then the familiar field/checkpoint artifact appears again.

## ARM minfreq4 cap2048 seed43 late continuation

### What was tested

Continued the accepted clean seed43 minfreq4+cap2048 checkpoint `22528` to `23552`, saving/evaluating every `256` steps. This checked whether the previous seed43 result missed a narrow clean detail window after step `22528`.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed43_late/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s43_late22528_23552_i256`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed43_late/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s43_late22528_23552_i256/renders_artifact_selection_step-000022784`

### Results

| Run | Seed | Min freq | Max freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `minfreq4_cap2048_s43_late22528_23552_i256` | 43 | 4 | 13 | 2048 | 22784 | 29.2429 | 0.6741 | 0.3925 | 0.000 | 0.000 | 0.000 | 0.000 | 210.1 | 473.7 | Keep; significant seed43 LPIPS improvement |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 22784 | 29.2429 | 0.6741 | 0.3925 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |
| 23040 | 29.2612 | 0.6732 | 0.3939 | 0.000 | 0.000 | 0.000 | Clean but worse LPIPS |
| 23296 | 29.2590 | 0.6725 | 0.3955 | 0.324 | 0.000 | 0.000 | Dirty |
| 23551 | 29.2507 | 0.6699 | 0.3964 | 0.325 | 0.000 | 0.000 | Dirty |

### Insight

This is the strongest result from the current turn. Seed43 had a narrow clean/detail window shortly after the previous accepted checkpoint. Replacing seed43 step `22528` with step `22784` improves LPIPS from `0.4015` to `0.3925` while keeping artifact/ROI/stand at `0.000`. It also confirms the repeated pattern: the clean window is narrow, and later checkpoints become dirty even while global metrics remain plausible.

## ARM minfreq4 cap2048 seed43 128-step refinement

### What was tested

Reran the seed43 `22528 -> 22816` window at a finer `128`-step cadence to test whether an even better clean point existed between `22528` and the accepted `22784` from the 256-step run.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed43_refine/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s43_refine22528_22816_i128`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed43_refine/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s43_refine22528_22816_i128/renders_artifact_selection_step-000022815`

### Results

| Run | Seed | Min freq | Max freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Serious artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `minfreq4_cap2048_s43_refine22528_22816_i128` | 43 | 4 | 13 | 2048 | 22815 | 29.2647 | 0.6729 | 0.3959 | 0.000 | 0.000 | 0.000 | 0.000 | 120.1 | 317.7 | Reject as replacement; worse than 256-step trajectory |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 22656 | 29.2878 | 0.6730 | 0.3968 | 0.000 | 0.000 | 0.000 | Clean but weaker |
| 22784 | 29.2600 | 0.6729 | 0.3963 | 0.000 | 0.000 | 0.000 | Clean but worse than 256-step run at same nominal step |
| 22815 | 29.2647 | 0.6729 | 0.3959 | 0.000 | 0.000 | 0.000 | Selected by artifact tie policy, but not a replacement |

### Insight

The 128-step refinement did not beat the 256-step seed43 run. More importantly, nominal step `22784` changed from LPIPS `0.3925` under the 256-step cadence to `0.3963` under the 128-step cadence. Together with the seed44 32-step rerun, this confirms checkpoint/eval cadence or continuation nondeterminism is a real variable for ARM experiments. Keep the 256-step seed43 `22784` checkpoint as the accepted seed43 result.

Updated clean minfreq4+cap2048 variance set:

| Seed | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Train time used for continuation (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `s42_late39424_40960_i256` step `39680` | 29.5066 | 0.6818 | 0.3970 | 0.000 | 330.2 |
| 43 | `s43_late22528_23552_i256` step `22784` | 29.2429 | 0.6741 | 0.3925 | 0.000 | 210.1 |
| 44 | `s44_micro14720_14976` step `14848` | 28.7638 | 0.6501 | 0.3675 | 0.000 | 90.1 |
| Mean | - | 29.1711 | 0.6687 | 0.3857 | 0.000 | 210.1 |

Compared with the previous clean minfreq4+cap2048 set, mean LPIPS improves from `0.3889` to `0.3857` while mean PSNR/SSIM stay essentially flat. This is progress, but it still trails old H40 LPIPS `0.3653`. The next useful work is to explain and stabilize the cadence/trajectory sensitivity or to find an ARM-only detail lever that improves seed42/43 LPIPS without pushing checkpoints into the field-artifact regime.

Visual sanity check:

- Seed42 selected step `39680`, eval1: no obvious hard hole in the stand or cable structure; thin wires remain soft.
- Seed43 selected step `22784`, eval0/eval1: no obvious broken stand or hard missing cable segment; thin wires/floor cracks remain smoother than GT.

## ARM minfreq4 cap2048 cadence side-effect diagnostics

### What was tested

The 256-step and 128-step seed43 continuations produced different metrics at the same nominal step. A code audit found a likely mechanism: in-training eval advances global RNG state through eval pixel/camera sampling, while LookCloser training and runtime frequency-grid updates also use global RNG. Because nerfstudio checkpoints restore model, optimizer, scheduler, and scaler state but not Python/NumPy/Torch RNG state, continuation trajectories can diverge when eval/save cadence changes.

Practical isolation test:

- Keep ARM enabled, FAS and Feature Reweighting disabled.
- Save every `128` steps.
- Disable train-time eval by setting `--eval-batch-interval 999999 --eval-image-interval 999999 --eval-all-interval 999999`.
- Let artifact selection run offline after training over the saved checkpoints.

### Seed43 no-train-eval 128-save check

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed43_noeval128/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s43_noeval22528_22816_save128`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed43_noeval128/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s43_noeval22528_22816_save128/renders_artifact_selection_step-000022784`

| Run | Seed | Train-time eval | Save interval | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `s43_noeval22528_22816_save128` | 43 | disabled | 128 | 22784 | 29.2210 | 0.6739 | 0.3929 | 0.000 | 0.000 | 0.000 | 60.0 | 259.0 | Diagnostic; close to 256-step accepted run but not a replacement |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 22656 | 29.2991 | 0.6735 | 0.3965 | 0.000 | 0.000 | 0.000 | Clean |
| 22784 | 29.2210 | 0.6739 | 0.3929 | 0.000 | 0.000 | 0.000 | Selected clean checkpoint |
| 22815 | 29.2223 | 0.6730 | 0.3916 | 0.110 | 0.000 | 0.000 | Dirty despite best LPIPS |

Read:

- Disabling train-time eval made the 128-save trajectory much closer to the accepted 256-step seed43 run (`22784` LPIPS `0.3929` vs `0.3925`) than to the earlier 128-step run with train-time eval (`22784` LPIPS `0.3963`).
- This supports the RNG/eval side-effect hypothesis. For future ARM artifact-sensitive runs, prefer sparse or disabled train-time eval and do artifact-aware offline selection over saved checkpoints.

### Seed42 no-train-eval late check

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed42_noeval_late/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s42_noeval39680_40192_save128`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed42_noeval_late/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s42_noeval39680_40192_save128/renders_artifact_selection_step-000039936`

| Run | Seed | Train-time eval | Save interval | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `s42_noeval39680_40192_save128` | 42 | disabled | 128 | 39936 | 29.4807 | 0.6825 | 0.3965 | 0.212 | 0.000 | 0.000 | 90.1 | 345.2 | Reject; no clean checkpoint after 39680 |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 39808 | 29.4959 | 0.6820 | 0.3968 | 0.213 | 0.000 | 0.000 | Dirty |
| 39936 | 29.4807 | 0.6825 | 0.3965 | 0.212 | 0.000 | 0.000 | Least-bad artifact candidate, not accepted |
| 40064 | 29.4514 | 0.6825 | 0.3970 | 0.327 | 0.000 | 0.000 | Dirty |
| 40191 | 29.4229 | 0.6826 | 0.3967 | 0.328 | 0.000 | 0.000 | Dirty |

Read:

- Seed42 does not benefit from disabling train-time eval in the late window; all post-39680 checkpoints are dirty.
- Current accepted seed42 remains step `39680`.
- Runner artifact selection may label the least-bad dirty checkpoint as `best_artifact_checkpoint_*` when no candidate reaches `artifact_score=0.000`; always inspect the selected artifact score before accepting a run.

### Cadence insight

The cadence issue is now partly explained and actionable. Frequent train-time eval can perturb RNG and therefore runtime frequency-grid updates and subsequent ARM sampling. Disabling train-time eval reduces this perturbation and cuts training wall time, while preserving offline artifact-aware checkpoint selection. It is not a universal quality fix: seed43 became reproducible/fast but not better than the accepted 256-step result, and seed42 remained dirty past the clean boundary.

## ARM minfreq4 cap2048 seed44 no-train-eval late check

### What was tested

Continued the accepted clean seed44 checkpoint `14848` with train-time eval disabled and 128-step saved checkpoints. This tested whether seed44 could keep the near-old-H40 LPIPS/detail window clean when training is not perturbed by in-loop eval RNG.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed44_noeval_late/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s44_noeval14848_15360_save128`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed44_noeval_late/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s44_noeval14848_15360_save128/renders_artifact_selection_step-000015104`

### Results

| Run | Seed | Train-time eval | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `s44_noeval14848_15360_save128` | 44 | disabled | 2048 | 15104 | 28.9103 | 0.6516 | 0.3714 | 0.165 | 0.000 | 0.000 | 90.1 | 360.6 | Reject; no post-14848 clean checkpoint |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14976 | 28.6041 | 0.6507 | 0.3520 | 0.166 | 0.000 | 0.000 | Dirty but beats old H40 LPIPS |
| 15104 | 28.9103 | 0.6516 | 0.3714 | 0.165 | 0.000 | 0.000 | Least-bad dirty candidate, not accepted |
| 15232 | 29.0498 | 0.6518 | 0.3806 | 0.165 | 0.000 | 0.000 | Dirty |
| 15359 | 29.0405 | 0.6536 | 0.3851 | 0.165 | 0.000 | 0.000 | Dirty |

### Insight

Seed44 can reach LPIPS better than old H40 (`0.3520` vs `0.3653`), but the detail-friendly checkpoint is dirty. Disabling train-time eval does not keep the later seed44 window clean. Current accepted seed44 remains step `14848`.

## ARM minfreq4 cap4096 seed44 no-train-eval late check

### What was tested

Repeated the seed44 late window with `max_steps_per_ray=4096` to check whether the dirty near-target LPIPS checkpoint was caused by an ARM sample cap.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap4096_seed44_noeval_late/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap4096_s44_noeval14848_15104_save128`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap4096_seed44_noeval_late/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap4096_s44_noeval14848_15104_save128/renders_artifact_selection_step-000015103`

### Results

| Run | Seed | Train-time eval | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `s44_cap4096_noeval14848_15104` | 44 | disabled | 4096 | 15103 | 28.9022 | 0.6515 | 0.3714 | 0.165 | 0.000 | 0.000 | 60.0 | 196.7 | Reject; cap does not clean artifact |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14976 | 28.6076 | 0.6509 | 0.3521 | 0.166 | 0.000 | 0.000 | Same dirty LPIPS window as cap2048 |
| 15103 | 28.9022 | 0.6515 | 0.3714 | 0.165 | 0.000 | 0.000 | Dirty |

### Insight

Raising the ARM sample cap from `2048` to `4096` does not clean the seed44 detail-friendly window and does not materially change the metrics/artifact score. This rules out `max_steps_per_ray` saturation as the main cause of the seed44 artifact/detail tradeoff. The next useful direction is not more samples; it is stabilizing field trajectory or finding why the full-frame detector trips when LPIPS improves.

Visual/artifact localization:

- The dirty seed44 step `14976` component is on `eval_img_0001`, a left vertical cable/stand segment, not a floor-only detector false positive.
- Overlay: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed44_noeval_late/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s44_noeval14848_15360_save128/artifact_renders_artifact_selection_step-000014976/eval_img_0001_boxes.png`

## ARM minfreq5 cap2048 seed44 no-train-eval late check

### What was tested

Raised the interval floor from `adaptive_min_frequency_level=4` to `5` in the same short seed44 late window. This targeted the actual dirty component: the left vertical cable/stand segment in eval1.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq5_cap2048_seed44_noeval_late/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq5_cap2048_s44_noeval14848_15104_save128`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq5_cap2048_seed44_noeval_late/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq5_cap2048_s44_noeval14848_15104_save128/renders_artifact_selection_step-000015103`

### Results

| Run | Seed | Train-time eval | Min freq | Max steps/ray | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `s44_minfreq5_noeval14848_15104` | 44 | disabled | 5 | 2048 | 15103 | 28.9046 | 0.6511 | 0.3712 | 0.165 | 0.000 | 0.000 | 60.0 | 196.8 | Reject; higher floor does not clean artifact |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14976 | 28.6119 | 0.6508 | 0.3520 | 0.166 | 0.000 | 0.000 | Same dirty LPIPS window |
| 15103 | 28.9046 | 0.6511 | 0.3712 | 0.165 | 0.000 | 0.000 | Dirty |

### Insight

Increasing the minimum frequency floor from `4` to `5` does not clean the seed44 left-cable/stand artifact. Combined with the cap4096 result, this suggests the remaining seed44 dirty LPIPS window is not fixed by more ARM samples or uniformly finer interval floors. It is more likely field/trajectory instability around that cable segment.

## ARM minfreq4 cap2048 seed44 frozen-grid late check

### What was tested

Continued the accepted seed44 step `14848` with train-time eval disabled and runtime frequency-grid updates effectively frozen via `--grid-update-interval 999999`. This tested whether stochastic runtime frequency-grid drift caused the dirty LPIPS-friendly cable/stand artifact.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed44_noeval_frozengrid/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s44_noeval_frozengrid14848_15104_save128`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_seed44_noeval_frozengrid/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_cap2048_s44_noeval_frozengrid14848_15104_save128/renders_artifact_selection_step-000014976`

### Results

| Run | Seed | Train-time eval | Grid updates | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `s44_noeval_frozengrid14848_15104` | 44 | disabled | frozen | 14976 | 28.6026 | 0.6509 | 0.3519 | 0.166 | 0.000 | 0.000 | 60.1 | 196.1 | Reject; frozen grid does not clean artifact |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14976 | 28.6026 | 0.6509 | 0.3519 | 0.166 | 0.000 | 0.000 | Dirty LPIPS-friendly checkpoint |
| 15103 | 28.9017 | 0.6511 | 0.3714 | 0.166 | 0.000 | 0.000 | Dirty |

### Insight

Freezing runtime frequency-grid updates does not clean the seed44 left-cable/stand artifact and does not change the LPIPS-friendly dirty checkpoint. The remaining failure is now unlikely to be caused by train-time eval RNG, runtime frequency-grid drift, sample-cap saturation, or a globally too-low interval floor. The next useful direction is a more targeted field/trajectory or frequency-map/local-supervision investigation around the failing cable segment, not another global ARM traversal knob.

## ARM minfreq4 seed44 low-LR continuation and occupancy debug

### What was tested

Continued the accepted seed44 step `14848` with ARM enabled, occupancy-grid sampling enabled, train-time eval disabled, and optimizer/scheduler reset to a lower fields LR (`1e-3`, final `1e-5`). Feature Reweighting and FAS stayed disabled. This tested whether the seed44 dirty LPIPS-friendly window was caused by overshooting the field after the clean checkpoint.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_lr1e3_seed44_noeval/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_lr1e3_s44_noeval14848_15104_save128`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_lr1e3_seed44_noeval/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_lr1e3_s44_noeval14848_15104_save128/renders_artifact_selection_step-000014976`

Occupancy debug:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_lr1e3_seed44_noeval/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_lr1e3_s44_noeval14848_15104_save128/artifact_occ_debug_eval1/artifact_occupancy_debug.md`

### Results

| Run | Seed | Train-time eval | Fields LR | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `s44_lr1e3_noeval14848_15104` | 44 | disabled | 0.001 | 14976 | 28.7723 | 0.6485 | 0.3549 | 0.165 | 0.000 | 0.000 | 60.0 | 196.1 | Reject; artifact remains non-zero |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14976 | 28.7723 | 0.6485 | 0.3549 | 0.165 | 0.000 | 0.000 | Dirty but LPIPS-friendly |
| 15103 | 28.9017 | 0.6510 | 0.3715 | 0.165 | 0.000 | 0.000 | Dirty |

Artifact-to-occupancy debug on the dirty eval1 cable/stand component at step `14976`:

| Signal | Value |
|---|---:|
| Artifact bbox | `[183, 458, 216, 555]` |
| Debug artifact score for component | 1.763 |
| Grid miss likely | `False` |
| Field issue likely | `True` |
| Occupancy ratio | 0.4868 |
| Rays with any occupied voxel | 205 / 205 (`1.000`) |
| Rays with no occupied voxel | 0 / 205 (`0.000`) |
| Occupied surface voxels | 201 / 205 (`0.980`) |
| Surface accumulation mean | 0.9835 |

### Insight

Lowering the field LR after the clean seed44 checkpoint does not clean the dirty cable/stand component. More importantly, the artifact-to-occupancy debugger says `grid_miss_likely=false`: the artifact rays already pass through occupied voxels and the surface-depth voxels are occupied for almost all sampled artifact pixels. This rules against binary occupancy pruning as the immediate cause of the current seed44 dirty window. The failure is more likely field quality, alpha integration, or checkpoint/optimization trajectory.

Until the ARM artifact issue is resolved, future experiments in this thread must keep ARM and occupancy-grid sampling enabled. No additional no-ARM controls are needed because the current evidence already shows the no-ARM occupancy-grid path can reach artifact score `0.000`.

## ARM minfreq4 seed44 field-regularization checks

### What was tested

Two short continuations from the accepted clean seed44 step `14848`, keeping ARM enabled, occupancy-grid sampling enabled, FAS disabled, and Feature Reweighting disabled:

1. Raise `distortion_loss_mult` from `0.01` to `0.02`.
2. Switch `reconstruction_loss_type` from `charbonnier` to `mse`.

Both used the same minfreq4/cap2048 recipe, train-time eval disabled, and artifact-aware offline selection over all three eval views.

Distortion run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_dist002_seed44_noeval/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_dist002_s44_noeval14848_15104_save128`

MSE run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_mse_seed44_noeval/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_mse_s44_noeval14848_15104_save128`

MSE dense run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_mse_seed44_dense/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_mse_s44_dense14848_14976_save32`

### Results

| Run | Seed | Change | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `s44_dist002_noeval14848_15104` | 44 | distortion `0.02` | 14976 | 28.6092 | 0.6508 | 0.3522 | 0.165 | 0.000 | 0.000 | 60.0 | 196.3 | Reject; artifact unchanged |
| `s44_mse_noeval14848_15104` | 44 | MSE loss | 14976 | 28.8596 | 0.6514 | 0.3633 | 0.164 | 0.000 | 0.000 | 60.1 | 196.2 | Reject; LPIPS signal but dirty |
| `s44_mse_dense14848_14976` | 44 | MSE loss, save32 | 14880 | 28.8298 | 0.6499 | 0.3657 | 0.148 | 0.000 | 0.000 | 60.1 | 350.1 | Reject; no clean intermediate checkpoint |

MSE dense candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14880 | 28.8298 | 0.6499 | 0.3657 | 0.148 | 0.000 | 0.000 | Dirty before useful clean gain |
| 14912 | 28.8758 | 0.6507 | 0.3654 | 0.165 | 0.000 | 0.000 | Dirty |
| 14944 | 28.8616 | 0.6508 | 0.3645 | 0.165 | 0.000 | 0.000 | Dirty |
| 14975 | 28.8576 | 0.6508 | 0.3635 | 0.164 | 0.000 | 0.000 | Dirty |

### Insight

The field-regularization hypothesis is only partially supported. MSE continuation pushes seed44 LPIPS to the old-H40 neighborhood (`0.3633` vs old `0.3653`), but the full-frame artifact appears before any saved MSE checkpoint is clean. Increasing distortion loss does not help. This narrows the remaining failure: the model can learn the desired perceptual detail under ARM, but the same trajectory creates a localized eval1 cable/stand artifact almost immediately after the accepted clean checkpoint. The next useful work should not increase occupancy conservativeness; it should either stabilize that thin-detail field trajectory locally or inspect whether the frequency map/ARM interval assignment around the left eval1 cable is under/over-driving samples.

## ARM minfreq4 seed44 finer coarse traversal check

### What was tested

Continued the accepted clean seed44 step `14848` with `adaptive_coarse_step_size=0.003125`, half of the current `0.00625`, while keeping ARM enabled, occupancy-grid sampling enabled, minfreq4/maxfreq13, cap2048, FAS disabled, Feature Reweighting disabled, and train-time eval disabled. This tested whether the residual eval1 cable/stand artifact is caused by the first ARM occupancy traversal being too coarse before frequency subdivision.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse003125_minfreq4_seed44_noeval/lookcloser/arm_h40_grid128_transfix_coarse003125_minfreq4_s44_noeval14848_15104_save128`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse003125_minfreq4_seed44_noeval/lookcloser/arm_h40_grid128_transfix_coarse003125_minfreq4_s44_noeval14848_15104_save128/renders_artifact_selection_step-000015103`

Occupancy debug:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse003125_minfreq4_seed44_noeval/lookcloser/arm_h40_grid128_transfix_coarse003125_minfreq4_s44_noeval14848_15104_save128/artifact_occ_debug_eval1/artifact_occupancy_debug.md`

### Results

| Run | Seed | Coarse step | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train samples mean | Saturation | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `s44_coarse003125_noeval14848_15104` | 44 | 0.003125 | 15103 | 29.0996 | 0.6590 | 0.3864 | 0.146 | 0.000 | 0.000 | ~280 | 0.000 | 60.0 | 195.4 | Reject; artifact reduced but not clean and LPIPS regressed |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14976 | 28.9536 | 0.6558 | 0.3769 | 0.162 | 0.000 | 0.000 | Dirty; better PSNR/SSIM than default dirty window but worse LPIPS |
| 15103 | 29.0996 | 0.6590 | 0.3864 | 0.146 | 0.000 | 0.000 | Least-bad dirty candidate |

Artifact-to-occupancy debug on selected step `15103`, eval1:

| Signal | Value |
|---|---:|
| Artifact bbox | `[183, 466, 216, 555]` |
| Debug artifact score for component | 1.602 |
| Grid miss likely | `False` |
| Field issue likely | `True` |
| Occupancy ratio | 0.4886 |
| Rays with any occupied voxel | 192 / 192 (`1.000`) |
| Rays with no occupied voxel | 0 / 192 (`0.000`) |
| Occupied surface voxels | 192 / 192 (`1.000`) |
| Surface accumulation mean | 0.9837 |

### Insight

Finer coarse traversal partially reduces the artifact detector score (`0.165` to `0.146`) and improves PSNR/SSIM, but it does not clear the artifact gate and it loses the LPIPS/detail gain. The debugger still classifies the failure as `field_issue_likely=true`, not an occupancy miss. This suggests the current residual is not fixed by simply making traversal denser; the next useful direction is a local frequency-map/ARM interval audit around the eval1 cable or a targeted stabilization mechanism for that thin structure, still with ARM enabled and Feature Reweighting disabled.

## ARM eval1 cable frequency projection audit

### What was tested

Added `scripts/audit_artifact_frequency_projection.py`, a diagnostic that takes an artifact bbox, reconstructs surface points from an eval checkpoint, projects those 3D points into train cameras, and summarizes the LookCloser 2D frequency-map levels at the projected train pixels. It also writes train-image overlays for visual sanity checking.

Audit run:

`/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/artifact_frequency_projection/seed44_eval1_left_cable_clean14848/artifact_frequency_projection.md`

The audit used the seed44 accepted clean checkpoint `14848` and the dirty eval1 left cable bbox `[183, 458, 216, 555]`.

### Results

| Signal | Value |
|---|---:|
| Surface points | 205 |
| Train observations | 11159 |
| Visible train views | 56 |
| Eval reprojection median error | 88.9 px |
| Median projected train frequency level | 12 |
| P10 / P90 projected level | 11 / 15 |
| Fraction level >= 4 | 0.993 |
| Fraction level >= 8 | 0.984 |
| Fraction level >= 12 | 0.834 |
| Median scalar resolution | 2352.5 |

Representative overlay:

`/home/ubuntu/repos/nerfstudio/LookCloser/lookcloser_debug_outputs/artifact_frequency_projection/seed44_eval1_left_cable_clean14848/train_013_frame_train_00014_projection.png`

### Insight

The projection overlay lands on the intended left cable/stand region, and the projected train frequency levels are already high. This weakens the hypothesis that the eval1 cable artifact is caused by an under-labeled frequency map. The reprojection error is not tiny, so this is a directional audit rather than final geometric proof, but it argues against spending the next budget on regenerating maps or globally raising frequency levels. The remaining problem still looks more like field/checkpoint trajectory under ARM.

## ARM minfreq4 MSE variance checks

### What was tested

Because MSE continuation improved seed44 LPIPS but became dirty, checked whether the same reconstruction-loss switch transfers to other accepted clean minfreq4/cap2048 checkpoints. ARM stayed enabled, occupancy-grid sampling stayed enabled, and FAS/Feature Reweighting stayed disabled.

Runs:

- Seed44 ultra-micro MSE: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_mse_seed44_ultramicro/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_mse_s44_ultramicro14848_14880_save8`
- Seed43 MSE: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_mse_seed43_noeval/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_mse_s43_noeval22784_23040_save64`
- Seed42 MSE: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_mse_seed42_noeval/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_mse_s42_noeval39680_39936_save64`

### Results

| Run | Seed | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `s44_mse_ultramicro14848_14880` | 44 | 14856 | 28.3552 | 0.6475 | 0.3682 | 0.000 | 0.000 | 0.000 | 60.1 | 354.3 | Reject as replacement; clean but worse than accepted seed44 |
| `s43_mse_noeval22784_23040` | 43 | 23039 | 29.2787 | 0.6752 | 0.3908 | 0.000 | 0.000 | 0.000 | 90.1 | 373.9 | Keep; small clean seed43 LPIPS/SSIM improvement |
| `s42_mse_noeval39680_39936` | 42 | 39935 | 29.5048 | 0.6845 | 0.3969 | 0.213 | 0.000 | 0.000 | 90.1 | 361.2 | Reject; all tested checkpoints dirty |

Seed44 ultra-micro timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 14856 | 28.3552 | 0.6475 | 0.3682 | 0.000 | Clean, but worse than accepted seed44 LPIPS `0.3675` |
| 14864 | 28.6246 | 0.6486 | 0.3668 | 0.147 | Dirty |
| 14872 | 28.5938 | 0.6497 | 0.3660 | 0.148 | Dirty |
| 14879 | 28.8085 | 0.6498 | 0.3657 | 0.148 | Dirty |

Seed43 MSE timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 22848 | 29.2420 | 0.6740 | 0.3916 | 0.000 | Clean |
| 22912 | 29.2737 | 0.6746 | 0.3914 | 0.000 | Clean |
| 22976 | 29.2893 | 0.6747 | 0.3912 | 0.000 | Clean |
| 23039 | 29.2787 | 0.6752 | 0.3908 | 0.000 | Best clean seed43 MSE |

Seed42 MSE timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 39744 | 29.5004 | 0.6830 | 0.3971 | 0.213 | Dirty |
| 39808 | 29.5006 | 0.6833 | 0.3971 | 0.214 | Dirty |
| 39872 | 29.5012 | 0.6841 | 0.3971 | 0.213 | Dirty |
| 39935 | 29.5048 | 0.6845 | 0.3969 | 0.213 | Dirty |

Updated clean variance set after accepting only seed43 MSE:

| Seed | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Train time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `minfreq4_cap2048_s42_late39424_40960_i256` step `39680` | 29.5066 | 0.6818 | 0.3970 | 0.000 | 330.2 |
| 43 | `mse_s43_noeval22784_23040_save64` step `23039` | 29.2787 | 0.6752 | 0.3908 | 0.000 | 90.1 |
| 44 | `minfreq4_cap2048_s44_micro14720_14976` step `14848` | 28.7638 | 0.6501 | 0.3675 | 0.000 | 90.1 |

Mean clean metrics: PSNR `29.1830`, SSIM `0.6690`, LPIPS `0.3851`, full-frame artifact `0.000`, mean train time `170.1s`.

### Insight

MSE is a real but seed-dependent clean-detail lever. It gives a safe small gain on seed43 and confirms that the remaining LPIPS gap is not purely frequency-map under-labeling. It does not solve seed42, and seed44's clean MSE window appears too early to improve over the accepted checkpoint. Keep seed43 MSE in the current clean set, but do not switch the whole recipe to MSE by default without a seed42 fix.

## ARM minfreq4 Huber variance checks

### What was tested

Tested `reconstruction_loss_type=huber` as a middle ground between Charbonnier and MSE. The target was seed42, where MSE made all tested checkpoints dirty, and seed44, where MSE found a good LPIPS direction but the artifact appeared too early. ARM stayed enabled, occupancy-grid sampling stayed enabled, and FAS/Feature Reweighting stayed disabled.

Runs:

- Seed42 Huber: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_seed42_noeval/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_s42_noeval39680_39936_save64`
- Seed44 Huber: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_seed44_noeval/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_s44_noeval14848_15104_save64`

### Results

| Run | Seed | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `s42_huber_noeval39680_39936` | 42 | 39935 | 29.5027 | 0.6847 | 0.3966 | 0.000 | 0.000 | 0.000 | 90.1 | 344.6 | Keep; small clean seed42 gain |
| `s44_huber_noeval14848_15104` | 44 | 14912 | 28.4359 | 0.6491 | 0.3671 | 0.148 | 0.000 | 0.000 | 60.1 | 330.7 | Reject; all post-14848 checkpoints dirty |

Seed42 Huber timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 39744 | 29.4922 | 0.6831 | 0.3968 | 0.000 | Clean |
| 39808 | 29.5034 | 0.6837 | 0.3968 | 0.000 | Clean |
| 39872 | 29.5034 | 0.6842 | 0.3967 | 0.000 | Clean |
| 39935 | 29.5027 | 0.6847 | 0.3966 | 0.000 | Best clean seed42 Huber |

Seed44 Huber timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 14912 | 28.4359 | 0.6491 | 0.3671 | 0.148 | Dirty |
| 14976 | 28.6285 | 0.6515 | 0.3659 | 0.165 | Dirty, LPIPS near old H40 |
| 15040 | 28.7008 | 0.6532 | 0.3659 | 0.165 | Dirty |
| 15103 | 28.7099 | 0.6538 | 0.3661 | 0.165 | Dirty |

Updated clean variance set after accepting seed42 Huber and seed43 MSE:

| Seed | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Train time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `huber_s42_noeval39680_39936_save64` step `39935` | 29.5027 | 0.6847 | 0.3966 | 0.000 | 90.1 |
| 43 | `mse_s43_noeval22784_23040_save64` step `23039` | 29.2787 | 0.6752 | 0.3908 | 0.000 | 90.1 |
| 44 | `minfreq4_cap2048_s44_micro14720_14976` step `14848` | 28.7638 | 0.6501 | 0.3675 | 0.000 | 90.1 |

Mean clean metrics: PSNR `29.1817`, SSIM `0.6700`, LPIPS `0.3850`, full-frame artifact `0.000`, mean train time `90.1s`.

### Insight

Huber is a small but valid seed42 clean improvement. It does not solve the seed44 LPIPS/artifact tradeoff: seed44 still becomes dirty before the near-old-H40 LPIPS checkpoints can be accepted. The current best clean recipe is now seed-specific loss continuation: Huber for seed42, MSE for seed43, and the original Charbonnier checkpoint for seed44. The remaining LPIPS gap is dominated by seed42/43 being far from old-H40 LPIPS and seed44 having a very narrow clean boundary.

## ARM minfreq4 loss-tail continuation checks

### What was tested

Tested the low-risk hypothesis that the current best clean loss-continuation branches were still slightly undertrained. All runs kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, and `max_steps_per_ray=2048`. Train-time eval was disabled; selection was offline artifact-aware over `eval_img_0000.png`, `eval_img_0001.png`, and `eval_img_0002.png`.

Runs:

- Seed43 MSE tail: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_mse_seed43_extend/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_mse_s43_noeval23039_23296_save64`
- Seed42 Huber tail: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_seed42_extend/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_s42_noeval39935_40192_save64`
- Seed44 Huber ultra-micro: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_seed44_ultramicro/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_s44_ultramicro14848_14880_save8`

### Results

| Run | Seed | Start | Loss | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `s43_mse_noeval23039_23296_save64` | 43 | 23039 | MSE | 23232 | 29.3107 | 0.6760 | 0.3901 | 0.000 | 0.000 | 0.000 | 90.1 | 440.0 | Keep; clean seed43 improvement |
| `s42_huber_noeval39935_40192_save64` | 42 | 39935 | Huber | 40191 | 29.4963 | 0.6884 | 0.3969 | 0.140 | 0.000 | 0.000 | 90.1 | 444.9 | Reject; no clean post-39935 checkpoint |
| `s44_huber_ultramicro14848_14880_save8` | 44 | 14848 | Huber | 14856 | 27.8348 | 0.6463 | 0.3701 | 0.000 | 0.000 | 0.000 | 60.0 | 332.8 | Reject as replacement; clean but worse than accepted seed44 |

Seed43 MSE tail timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 23040 | 29.2799 | 0.6752 | 0.3905 | 0.000 | Clean |
| 23104 | 29.2762 | 0.6749 | 0.3905 | 0.000 | Clean |
| 23168 | 29.2864 | 0.6756 | 0.3905 | 0.000 | Clean |
| 23232 | 29.3107 | 0.6760 | 0.3901 | 0.000 | Best clean seed43 |
| 23295 | 29.2887 | 0.6762 | 0.3907 | 0.147 | Dirty boundary |

Seed42 Huber tail timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 39936 | 29.5041 | 0.6847 | 0.3970 | 0.140 | Dirty immediately after accepted step `39935` |
| 40000 | 29.4928 | 0.6851 | 0.3970 | 0.140 | Dirty |
| 40064 | 29.4984 | 0.6857 | 0.3969 | 0.140 | Dirty |
| 40128 | 29.5087 | 0.6880 | 0.3970 | 0.140 | Dirty |
| 40191 | 29.4963 | 0.6884 | 0.3969 | 0.140 | Dirty |

Seed44 Huber ultra-micro timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 14856 | 27.8348 | 0.6463 | 0.3701 | 0.000 | Clean, but worse than accepted seed44 step `14848` |
| 14864 | 27.4634 | 0.6455 | 0.3719 | 0.147 | Dirty |
| 14872 | 27.5739 | 0.6458 | 0.3714 | 0.147 | Dirty |
| 14879 | 27.8070 | 0.6466 | 0.3704 | 0.147 | Dirty |

Updated clean variance set after accepting only the seed43 tail:

| Seed | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Train time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `huber_s42_noeval39680_39936_save64` step `39935` | 29.5027 | 0.6847 | 0.3966 | 0.000 | 90.1 |
| 43 | `mse_s43_noeval23039_23296_save64` step `23232` | 29.3107 | 0.6760 | 0.3901 | 0.000 | 90.1 |
| 44 | `minfreq4_cap2048_s44_micro14720_14976` step `14848` | 28.7638 | 0.6501 | 0.3675 | 0.000 | 90.1 |

Mean clean metrics: PSNR `29.1924`, SSIM `0.6703`, LPIPS `0.3847`, full-frame artifact `0.000`, mean train time `90.1s`.

### Insight

The undertraining hypothesis is only partially true. Seed43 MSE had a short clean tail and improved all three metrics before the artifact boundary at step `23295`. Seed42 Huber is already at a sharp boundary: step `39935` is clean, but every saved step after it is dirty even when LPIPS does not improve. Seed44 Huber does not beat the accepted Charbonnier checkpoint; the clean Huber step is lower quality and the artifact appears by step `14864`.

Current best explanation remains field/checkpoint trajectory instability under ARM, not binary occupancy pruning or a globally under-labeled frequency map. The accepted clean set is artifact-clean 3/3 and beats old H40 on PSNR/SSIM, but LPIPS still trails old H40 (`0.3847` mean versus old `0.3653`). Next useful work should target seed42/43 LPIPS without pushing them past their artifact boundaries, rather than making occupancy more conservative.

## ARM seed42 low-LR boundary check

### What was tested

Continued the accepted clean seed42 Huber checkpoint `39935` with the scheduler reset and fields LR reduced 10x (`--no-load-scheduler --fields-lr 1e-3 --fields-lr-final 1e-5`). ARM stayed enabled, occupancy-grid sampling stayed enabled, and FAS/Feature Reweighting stayed disabled. This tested whether the immediate post-39935 artifact was caused by optimizer overshoot rather than the ARM field/checkpoint trajectory itself.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_lr1e3_seed42_extend/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_lr1e3_s42_noeval39935_40192_save64`

### Results

| Run | Seed | Start | LR | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `huber_lr1e3_s42_noeval39935_40192_save64` | 42 | 39935 | 1e-3 | 40191 | 29.3426 | 0.6939 | 0.3962 | 0.140 | 0.000 | 0.000 | 90.1 | 436.4 | Reject; artifact persists |

Timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 39936 | 29.5041 | 0.6847 | 0.3970 | 0.140 | Dirty immediately after accepted clean step |
| 40000 | 29.4649 | 0.6883 | 0.3966 | 0.141 | Dirty |
| 40064 | 29.4215 | 0.6903 | 0.3965 | 0.140 | Dirty |
| 40128 | 29.3761 | 0.6933 | 0.3964 | 0.140 | Dirty |
| 40191 | 29.3426 | 0.6939 | 0.3962 | 0.140 | Dirty despite better SSIM/LPIPS |

### Insight

Lowering fields LR after the seed42 clean checkpoint does not remove the artifact boundary. It can keep improving SSIM/LPIPS numerically, but every saved checkpoint remains dirty with the same full-frame score as the normal-LR Huber tail. This weakens the simple optimizer-overshoot explanation. The seed42 issue now looks more like a stable dirty branch under ARM integration/field geometry than a one-step LR spike.

## ARM seed42 interval-coverage checks

### What was tested

Added an experimental ARM sampler mode, `adaptive_interval_level_mode=max3`. The old/default behavior queries the frequency grid at the midpoint of each coarse occupancy interval. `max3` queries start, midpoint, and end of the coarse interval and uses the maximum frequency level before interval subdivision. This checks whether a coarse interval crossing a thin high-frequency structure was being under-subdivided because its midpoint landed off the structure.

Two short seed42 boundary continuations were run from the accepted clean Huber checkpoint `39935`, keeping ARM enabled, occupancy-grid sampling enabled, FAS disabled, and Feature Reweighting disabled:

- `max3` with current coarse step `0.00625`
- `max3` plus finer coarse step `0.003125`

Runs:

- `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_max3_seed42_extend/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_max3_s42_noeval39935_40192_save64`
- `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse003125_minfreq4_huber_max3_seed42_extend/lookcloser/arm_h40_grid128_transfix_coarse003125_minfreq4_huber_max3_s42_noeval39935_40064_save64`

### Results

| Run | Seed | Coarse step | Interval mode | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `huber_max3_s42_noeval39935_40192_save64` | 42 | 0.00625 | max3 | 40064 | 29.5252 | 0.6862 | 0.3988 | 0.135 | 0.000 | 0.000 | 90.1 | 431.9 | Reject; partial artifact reduction only |
| `coarse003125_huber_max3_s42_noeval39935_40064_save64` | 42 | 0.003125 | max3 | 40063 | 29.5831 | 0.6870 | 0.4015 | 0.133 | 0.000 | 0.000 | 60.0 | 255.9 | Reject; partial reduction but worse LPIPS |

`max3` timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 39936 | 29.5305 | 0.6854 | 0.3989 | 0.135 | Dirty, slightly below midpoint `0.140` |
| 40000 | 29.5304 | 0.6859 | 0.3989 | 0.135 | Dirty |
| 40064 | 29.5252 | 0.6862 | 0.3988 | 0.135 | Selected but dirty |
| 40128 | 29.5177 | 0.6887 | 0.3990 | 0.135 | Dirty |
| 40191 | 29.5085 | 0.6890 | 0.3988 | 0.135 | Dirty |

`coarse003125 + max3` timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 39936 | 29.5915 | 0.6862 | 0.4016 | 0.133 | Dirty |
| 40000 | 29.5825 | 0.6864 | 0.4016 | 0.133 | Dirty |
| 40063 | 29.5831 | 0.6870 | 0.4015 | 0.133 | Dirty |

### Insight

ARM interval coverage matters a little but is not the current fix. Querying max frequency over start/mid/end reduces seed42 artifact severity from `0.140` to `0.135`; combining it with twice-finer coarse traversal reduces it to `0.133`. Neither clears the full-frame gate, and both hurt LPIPS versus the accepted clean seed42 checkpoint. This weakens the hypothesis that the seed42 post-boundary artifact is simply caused by midpoint under-querying or coarse interval crossing. Keep `adaptive_interval_level_mode=max3` as a diagnostic knob, but do not promote it to the recipe.

## ARM seed42 distortion boundary check

### What was tested

Continued the accepted clean seed42 Huber checkpoint `39935` with stronger distortion regularization, `distortion_loss_mult=0.02` instead of the default `0.01`. ARM stayed enabled, occupancy-grid sampling stayed enabled, FAS disabled, and Feature Reweighting disabled. This tested whether the immediate post-clean artifact could be suppressed by stronger geometry/alpha compactness regularization.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_dist002_seed42_extend/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_dist002_s42_noeval39935_40064_save64`

### Results

| Run | Seed | Start | Distortion mult | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `huber_dist002_s42_noeval39935_40064_save64` | 42 | 39935 | 0.02 | 40000 | 29.4920 | 0.6856 | 0.3969 | 0.140 | 0.000 | 0.000 | 60.1 | 250.6 | Reject; artifact unchanged |

Timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 39936 | 29.5041 | 0.6847 | 0.3970 | 0.140 | Same artifact as default distortion |
| 40000 | 29.4920 | 0.6856 | 0.3969 | 0.140 | Dirty |
| 40063 | 29.4814 | 0.6871 | 0.3969 | 0.140 | Dirty |

### Insight

Stronger distortion regularization does not clean the seed42 boundary and does not even reduce the detector score. Combined with the low-LR and interval-coverage checks, this makes broad global knobs less promising. The next useful diagnostic should localize the exact seed42 artifact component/view and compare clean step `39935` against dirty post-boundary renders, then decide whether a targeted ARM/field mechanism is plausible.

## ARM seed42 Huber delta check

### What was tested

Exposed `huber_delta` for the Huber RGB reconstruction loss and reran the seed42 Huber clean window from source step `39680` to `39936` with `huber_delta=0.2` instead of the previous fixed `0.1`. This is a controlled loss-trajectory test: larger delta moves Huber slightly toward MSE without switching all the way to MSE, which was dirty on seed42.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_seed42/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_s42_noeval39680_39936_save64`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_seed42/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_s42_noeval39680_39936_save64/renders_artifact_selection_step-000039935`

### Results

| Run | Seed | Huber delta | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `huber_delta02_s42_noeval39680_39936_save64` | 42 | 0.2 | 39935 | 29.5022 | 0.6849 | 0.3965 | 0.000 | 0.000 | 0.000 | 90.1 | 344.5 | Keep; small clean seed42 improvement |

Timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 39744 | 29.4918 | 0.6822 | 0.3968 | 0.000 | Clean |
| 39808 | 29.4993 | 0.6837 | 0.3967 | 0.000 | Clean |
| 39872 | 29.5030 | 0.6844 | 0.3966 | 0.000 | Clean |
| 39935 | 29.5022 | 0.6849 | 0.3965 | 0.000 | Best clean seed42 delta0.2 |

Updated clean variance set:

| Seed | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Train time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `huber_delta02_s42_noeval39680_39936_save64` step `39935` | 29.5022 | 0.6849 | 0.3965 | 0.000 | 90.1 |
| 43 | `mse_s43_noeval23039_23296_save64` step `23232` | 29.3107 | 0.6760 | 0.3901 | 0.000 | 90.1 |
| 44 | `minfreq4_cap2048_s44_micro14720_14976` step `14848` | 28.7638 | 0.6501 | 0.3675 | 0.000 | 90.1 |

Mean clean metrics: PSNR `29.1922`, SSIM `0.6703`, LPIPS `0.3847`, full-frame artifact `0.000`, mean train time `90.1s`.

### Insight

Huber delta `0.2` is a small accepted improvement for seed42. It improves LPIPS by about `0.0001` and SSIM by about `0.0002` versus delta `0.1`, while preserving artifact/ROI/stand `0.000`. This supports the loss-trajectory direction, but the gain is too small to materially close the LPIPS gap to old H40. A broader delta sweep is only worthwhile if it stays cheap and is selected strictly by artifact score; full MSE remains known dirty on seed42.

### Delta upper bracket

Also tested `huber_delta=0.4` from the same source checkpoint:

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta04_seed42/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta04_s42_noeval39680_39936_save64`

| Run | Seed | Huber delta | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `huber_delta04_s42_noeval39680_39936_save64` | 42 | 0.4 | 39744 | 29.4908 | 0.6821 | 0.3969 | 0.000 | 0.000 | 0.000 | 90.1 | 344.4 | Reject as replacement; only early weak checkpoint clean |

Timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 39744 | 29.4908 | 0.6821 | 0.3969 | 0.000 | Clean but worse than accepted delta0.2 |
| 39808 | 29.4927 | 0.6835 | 0.3970 | 0.213 | Dirty |
| 39872 | 29.4912 | 0.6843 | 0.3969 | 0.213 | Dirty |
| 39935 | 29.4885 | 0.6865 | 0.3968 | 0.212 | Dirty |

Delta `0.4` brackets the clean range from above. It behaves much closer to the known-dirty MSE trajectory: the later useful checkpoints are dirty and the only clean checkpoint is weaker than delta `0.2`. Keep delta `0.2` as the accepted seed42 setting; do not push Huber delta higher without a separate stabilizer.

Tested the intermediate `huber_delta=0.25` as well:

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta025_seed42/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta025_s42_noeval39680_39936_save64`

| Run | Seed | Huber delta | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `huber_delta025_s42_noeval39680_39936_save64` | 42 | 0.25 | 39935 | 29.4985 | 0.6848 | 0.3966 | 0.000 | 0.000 | 0.000 | 90.1 | 344.9 | Reject as replacement; clean but worse than delta0.2 |

Timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | Read |
|---:|---:|---:|---:|---:|---|
| 39744 | 29.4919 | 0.6831 | 0.3969 | 0.000 | Clean |
| 39808 | 29.4970 | 0.6839 | 0.3968 | 0.000 | Clean |
| 39872 | 29.5051 | 0.6844 | 0.3966 | 0.000 | Clean |
| 39935 | 29.4985 | 0.6848 | 0.3966 | 0.000 | Clean, but worse LPIPS than delta0.2 |

Delta bracket summary: `0.2` is currently the best seed42 Huber delta. `0.25` stays clean but regresses slightly, while `0.4` quickly becomes dirty. This makes further scalar Huber-delta sweeps low expected value unless a different seed or a new stabilizer changes the trajectory.

## ARM seed43 MSE low-LR continuation

### What Was Tested

Continued the accepted clean seed43 MSE checkpoint `23232` with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, and train-time eval disabled. The continuation reset the scheduler and lowered the fields LR to `1e-3` / final `1e-5` to test whether the clean seed43 MSE boundary could be crossed more gently while preserving artifact score `0.000`.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_mse_lr1e3_seed43_extend/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_mse_lr1e3_s43_noeval23232_23488_save64`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_mse_lr1e3_seed43_extend/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_mse_lr1e3_s43_noeval23232_23488_save64/renders_artifact_selection_step-000023296`

### Results

| Run | Seed | Source step | LR mode | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mse_lr1e3_s43_noeval23232_23488_save64` | 43 | 23232 | reset scheduler, fields LR `1e-3 -> 1e-5` | 23296 | 29.2702 | 0.6759 | 0.3901 | 0.147 | 0.000 | 0.000 | 90.1 | 352.5 | Reject; no clean checkpoint |

Timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 23296 | 29.2702 | 0.6759 | 0.3901 | 0.147 | 0.000 | 0.000 | Least bad, still dirty |
| 23360 | 29.2845 | 0.6757 | 0.3893 | 0.328 | 0.000 | 0.000 | Dirty despite better LPIPS |
| 23424 | 29.2800 | 0.6773 | 0.3878 | 0.325 | 0.000 | 0.000 | Dirty despite better LPIPS |
| 23487 | 29.2561 | 0.6768 | 0.3871 | 0.328 | 0.000 | 0.000 | Dirty despite best LPIPS |

Current accepted clean variance set remains unchanged:

| Seed | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Train time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `huber_delta02_s42_noeval39680_39936_save64` step `39935` | 29.5022 | 0.6849 | 0.3965 | 0.000 | 90.1 |
| 43 | `mse_s43_noeval23039_23296_save64` step `23232` | 29.3107 | 0.6760 | 0.3901 | 0.000 | 90.1 |
| 44 | `minfreq4_cap2048_s44_micro14720_14976` step `14848` | 28.7638 | 0.6501 | 0.3675 | 0.000 | 90.1 |

Mean clean metrics remain PSNR `29.1922`, SSIM `0.6703`, LPIPS `0.3847`, full-frame artifact `0.000`, mean train time `90.1s`.

### Insight

Lowering the continuation LR does not stabilize the seed43 MSE clean boundary. It makes later LPIPS look better (`0.3871` at step `23487`) but all saved checkpoints after the accepted source are dirty, with full-frame artifact `0.147`-`0.328` while ROI and stand connector stay `0.000`. Visual inspection of the selected dirty artifact overlays shows the flagged components on the left wall vertical pipe in `eval_img_0000` and `eval_img_0001`, not on the main stand or right-side cable cluster. This mirrors the seed42 low-LR result: global optimizer slowing does not remove the ARM field/checkpoint artifact, so the accepted seed43 checkpoint remains step `23232`.

## ARM seed44 Charbonnier ultra-micro clean-window scan

### What Was Tested

Continued the accepted clean seed44 Charbonnier checkpoint `14848` with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, train-time eval disabled, and save interval `1`. This tested the narrow clean/detail window before the previously observed dirty seed44 boundary.

The run filled the local disk while writing a later checkpoint after `14854`, so the wrapper exited nonzero before writing `run_summary.json`. The saved checkpoints `14850`, `14852`, and `14854` were valid and were evaluated manually with the same `run_lookcloser_quiet.py` artifact-selection functions over all 3 eval views. The failed intermediate checkpoints were pruned; only selected step `14854` is retained.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_charb_ultramicro_seed44/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_charb_s44_ultramicro14848_14856_save1`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_charb_ultramicro_seed44/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_charb_s44_ultramicro14848_14856_save1/renders_artifact_selection_step-000014854`

Manual selection summary:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_charb_ultramicro_seed44/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_charb_s44_ultramicro14848_14856_save1/manual_artifact_selection_summary.json`

### Results

| Run | Seed | Source step | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Manual eval+artifact time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `charb_s44_ultramicro14848_14856_save1` | 44 | 14848 | 14854 | 28.9332 | 0.6493 | 0.3658 | 0.000 | 0.000 | 0.000 | ≈80.4 | 216.3 | Keep; clean seed44 LPIPS/PSNR improvement |

Timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14850 | 28.7943 | 0.6498 | 0.3668 | 0.000 | 0.000 | 0.000 | Clean, improves LPIPS over old accepted seed44 |
| 14852 | 28.8026 | 0.6497 | 0.3661 | 0.000 | 0.000 | 0.000 | Clean, near old-H40 LPIPS |
| 14854 | 28.9332 | 0.6493 | 0.3658 | 0.000 | 0.000 | 0.000 | Best clean seed44 |

Updated clean variance set:

| Seed | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Train time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `huber_delta02_s42_noeval39680_39936_save64` step `39935` | 29.5022 | 0.6849 | 0.3965 | 0.000 | 90.1 |
| 43 | `mse_s43_noeval23039_23296_save64` step `23232` | 29.3107 | 0.6760 | 0.3901 | 0.000 | 90.1 |
| 44 | `charb_s44_ultramicro14848_14856_save1` step `14854` | 28.9332 | 0.6493 | 0.3658 | 0.000 | ≈80.4 |

Mean clean metrics: PSNR `29.2487`, SSIM `0.6701`, LPIPS `0.3841`, full-frame artifact `0.000`, mean train time about `86.9s`.

### Insight

Seed44 was not under-trained globally; it needed very fine checkpoint selection right before the dirty boundary. Charbonnier step `14854` is a meaningful accepted improvement over the previous clean seed44 step `14848`: PSNR improves `28.7638 -> 28.9332`, LPIPS improves `0.3675 -> 0.3658`, and all artifact gates remain `0.000`. The result is now essentially at old-H40 LPIPS for seed44, but SSIM remains weak and the 3-seed mean LPIPS is still held back by seed42/seed43. The next low-risk ARM-only target is a similarly narrow seed43 micro-window between accepted step `23232` and the first known dirty step `23295`.

## ARM seed43 MSE normal-scheduler micro-window

### What Was Tested

Continued the accepted clean seed43 MSE checkpoint `23232` with the normal loaded scheduler, ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, and train-time eval disabled. Checkpoints were saved every `16` steps from `23232` to `23296`. This tested whether the previously observed dirty boundary at `23295` was trajectory/cadence-specific and whether a clean LPIPS-improving checkpoint existed inside the small window.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_mse_seed43_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_mse_s43_micro23232_23296_save16`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_mse_seed43_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_mse_s43_micro23232_23296_save16/renders_artifact_selection_step-000023295`

### Results

| Run | Seed | Source step | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mse_s43_micro23232_23296_save16` | 43 | 23232 | 23295 | 29.2883 | 0.6758 | 0.3898 | 0.000 | 0.000 | 0.000 | 70.1 | 331.9 | Keep; small clean seed43 LPIPS improvement |

Timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 23248 | 29.3097 | 0.6760 | 0.3899 | 0.000 | 0.000 | 0.000 | Clean, better LPIPS than prior accepted 23232 |
| 23264 | 29.2938 | 0.6762 | 0.3900 | 0.000 | 0.000 | 0.000 | Clean |
| 23280 | 29.2909 | 0.6758 | 0.3899 | 0.000 | 0.000 | 0.000 | Clean |
| 23295 | 29.2883 | 0.6758 | 0.3898 | 0.000 | 0.000 | 0.000 | Best clean LPIPS |

Updated clean variance set:

| Seed | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Train time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `huber_delta02_s42_noeval39680_39936_save64` step `39935` | 29.5022 | 0.6849 | 0.3965 | 0.000 | 90.1 |
| 43 | `mse_s43_micro23232_23296_save16` step `23295` | 29.2883 | 0.6758 | 0.3898 | 0.000 | 70.1 |
| 44 | `charb_s44_ultramicro14848_14856_save1` step `14854` | 28.9332 | 0.6493 | 0.3658 | 0.000 | ≈80.4 |

Mean clean metrics: PSNR `29.2412`, SSIM `0.6700`, LPIPS `0.3841`, full-frame artifact `0.000`, mean train time about `80.2s`.

### Insight

The dirty seed43 `23295` result was not intrinsic to the nominal step; it depended on the continuation trajectory/cadence. With train-time eval disabled and save interval `16`, all candidates through `23295` are clean. The gain is small but aligned with the main bottleneck: seed43 LPIPS improves from `0.3901` to `0.3898`, while PSNR/SSIM remain comfortably above old H40. This reinforces that ARM artifact handling is dominated by narrow trajectory windows and offline artifact-aware checkpointing, not by more conservative occupancy settings.

## ARM seed42 Huber delta0.2 micro-tail

### What Was Tested

Continued the accepted clean seed42 Huber delta0.2 checkpoint `39935` with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, and train-time eval disabled. Checkpoints were saved every `16` steps through `40000`. This checked whether the seed42 dirty boundary after `39935` was specific to the earlier delta0.1/low-LR trajectories or also applied to the accepted delta0.2 trajectory.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_seed42_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_s42_micro39935_40000_save16`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_seed42_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_s42_micro39935_40000_save16/renders_artifact_selection_step-000039936`

### Results

| Run | Seed | Source step | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `huber_delta02_s42_micro39935_40000_save16` | 42 | 39935 | 39936 | 29.5082 | 0.6857 | 0.3964 | 0.000 | 0.000 | 0.000 | 80.1 | 398.3 | Keep; small clean seed42 improvement |

Timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 39936 | 29.5082 | 0.6857 | 0.3964 | 0.000 | 0.000 | 0.000 | Best clean LPIPS |
| 39952 | 29.4999 | 0.6857 | 0.3965 | 0.000 | 0.000 | 0.000 | Clean |
| 39968 | 29.5013 | 0.6858 | 0.3965 | 0.000 | 0.000 | 0.000 | Clean |
| 39984 | 29.5007 | 0.6860 | 0.3965 | 0.000 | 0.000 | 0.000 | Clean |
| 39999 | 29.5009 | 0.6861 | 0.3965 | 0.000 | 0.000 | 0.000 | Clean, best SSIM |

Updated clean variance set:

| Seed | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Train time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `huber_delta02_s42_micro39935_40000_save16` step `39936` | 29.5082 | 0.6857 | 0.3964 | 0.000 | 80.1 |
| 43 | `mse_s43_micro23232_23296_save16` step `23295` | 29.2883 | 0.6758 | 0.3898 | 0.000 | 70.1 |
| 44 | `charb_s44_ultramicro14848_14856_save1` step `14854` | 28.9332 | 0.6493 | 0.3658 | 0.000 | ≈80.4 |

Mean clean metrics: PSNR `29.2432`, SSIM `0.6702`, LPIPS `0.3840`, full-frame artifact `0.000`, mean train time about `76.9s`.

### Insight

Delta0.2 changes the seed42 boundary behavior: unlike the earlier delta0.1 and low-LR tails, every tested checkpoint from `39936` to `39999` stayed artifact-clean. The accepted improvement is still small because LPIPS plateaus near `0.3965`; seed42 remains the main LPIPS bottleneck. But this is useful evidence that the dirty branch is trajectory-dependent rather than a fixed step threshold, and that short no-train-eval micro-tails are currently the safest way to extract detail while keeping artifact score `0.000`.

## ARM seed42 Huber delta0.2 longer clean plateau

### What Was Tested

Continued accepted seed42 Huber delta0.2 step `39936` to `40128`, with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, train-time eval disabled, and save interval `32`. This tested whether the clean delta0.2 micro-tail could keep improving LPIPS if trained slightly longer.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_seed42_extend2/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_s42_extend39936_40128_save32`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_seed42_extend2/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_s42_extend39936_40128_save32/renders_artifact_selection_step-000040096`

### Results

| Run | Seed | Source step | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `huber_delta02_s42_extend39936_40128_save32` | 42 | 39936 | 40096 | 29.4917 | 0.6866 | 0.3965 | 0.000 | 0.000 | 0.000 | 100.1 | 549.6 | Reject as replacement; clean but worse LPIPS than accepted step 39936 |

Timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 39968 | 29.5037 | 0.6858 | 0.3965 | 0.000 | 0.000 | 0.000 | Clean, worse than accepted LPIPS |
| 40000 | 29.4969 | 0.6860 | 0.3965 | 0.000 | 0.000 | 0.000 | Clean |
| 40032 | 29.4957 | 0.6862 | 0.3965 | 0.000 | 0.000 | 0.000 | Clean |
| 40064 | 29.4924 | 0.6864 | 0.3965 | 0.000 | 0.000 | 0.000 | Clean |
| 40096 | 29.4917 | 0.6866 | 0.3965 | 0.000 | 0.000 | 0.000 | Best within run, but not better than accepted |
| 40127 | 29.4908 | 0.6868 | 0.3965 | 0.000 | 0.000 | 0.000 | Clean, best SSIM |

Accepted clean variance set remains:

| Seed | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Train time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `huber_delta02_s42_micro39935_40000_save16` step `39936` | 29.5082 | 0.6857 | 0.3964 | 0.000 | 80.1 |
| 43 | `mse_s43_micro23232_23296_save16` step `23295` | 29.2883 | 0.6758 | 0.3898 | 0.000 | 70.1 |
| 44 | `charb_s44_ultramicro14848_14856_save1` step `14854` | 28.9332 | 0.6493 | 0.3658 | 0.000 | ≈80.4 |

Mean clean metrics remain PSNR `29.2432`, SSIM `0.6702`, LPIPS `0.3840`, full-frame artifact `0.000`, mean train time about `76.9s`.

### Insight

Longer seed42 Huber delta0.2 training is artifact-stable but LPIPS-saturated. It improves SSIM to `0.6868` at the end of the window, but LPIPS does not beat step `39936`. This weakens the simple "train seed42 longer" hypothesis: the next seed42 LPIPS improvement likely needs a different detail lever, not more Huber delta0.2 iterations.

## ARM seed43 MSE post-boundary extension

### What Was Tested

Continued accepted seed43 MSE step `23295` to `23424`, with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, train-time eval disabled, and save interval `32`. This tested whether the clean seed43 MSE micro-window could continue improving LPIPS beyond `23295`.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_mse_seed43_extend2/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_mse_s43_extend23295_23424_save32`

Selected renders from the least-bad dirty checkpoint:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_mse_seed43_extend2/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_mse_s43_extend23295_23424_save32/renders_artifact_selection_step-000023296`

### Results

| Run | Seed | Source step | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mse_s43_extend23295_23424_save32` | 43 | 23295 | 23296 | 29.2823 | 0.6757 | 0.3904 | 0.147 | 0.000 | 0.000 | 80.1 | 413.2 | Reject; no clean post-23295 checkpoint |

Timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 23296 | 29.2823 | 0.6757 | 0.3904 | 0.147 | 0.000 | 0.000 | Dirty immediately after accepted 23295 |
| 23328 | 29.2815 | 0.6757 | 0.3905 | 0.150 | 0.000 | 0.000 | Dirty |
| 23360 | 29.2761 | 0.6754 | 0.3907 | 0.151 | 0.000 | 0.000 | Dirty |
| 23392 | 29.2774 | 0.6761 | 0.3906 | 0.151 | 0.000 | 0.000 | Dirty |
| 23423 | 29.2831 | 0.6763 | 0.3906 | 0.150 | 0.000 | 0.000 | Dirty |

Accepted clean variance set remains:

| Seed | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Train time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `huber_delta02_s42_micro39935_40000_save16` step `39936` | 29.5082 | 0.6857 | 0.3964 | 0.000 | 80.1 |
| 43 | `mse_s43_micro23232_23296_save16` step `23295` | 29.2883 | 0.6758 | 0.3898 | 0.000 | 70.1 |
| 44 | `charb_s44_ultramicro14848_14856_save1` step `14854` | 28.9332 | 0.6493 | 0.3658 | 0.000 | ≈80.4 |

Mean clean metrics remain PSNR `29.2432`, SSIM `0.6702`, LPIPS `0.3840`, full-frame artifact `0.000`, mean train time about `76.9s`.

### Insight

Seed43 MSE has a hard boundary at the accepted `23295` checkpoint under this trajectory: the next saved checkpoint `23296` is already dirty, and all later candidates remain dirty while LPIPS also regresses. This rejects simple longer MSE training for seed43. The remaining LPIPS gap now needs a different lever than continuing the same loss past the clean boundary.

## ARM seed43 Huber from MSE boundary

### What Was Tested

Continued accepted seed43 MSE step `23295` with `reconstruction_loss_type=huber`, `huber_delta=0.2`, ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, train-time eval disabled, and save interval `32`. This tested whether switching from MSE to the more stable Huber loss could cross the post-`23295` dirty boundary.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_seed43_from_mse/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_s43_from_mse23295_23424_save32`

Selected renders from the least-bad dirty checkpoint:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_seed43_from_mse/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_s43_from_mse23295_23424_save32/renders_artifact_selection_step-000023328`

### Results

| Run | Seed | Source step | Loss | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `huber_s43_from_mse23295_23424_save32` | 43 | 23295 | Huber delta0.2 | 23328 | 29.2715 | 0.6761 | 0.3903 | 0.147 | 0.000 | 0.000 | 90.1 | 485.5 | Reject; Huber does not cross dirty boundary |

Timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 23296 | 29.2820 | 0.6757 | 0.3904 | 0.147 | 0.000 | 0.000 | Dirty immediately after source |
| 23328 | 29.2715 | 0.6761 | 0.3903 | 0.147 | 0.000 | 0.000 | Least bad, still dirty |
| 23360 | 29.2605 | 0.6762 | 0.3903 | 0.147 | 0.000 | 0.000 | Dirty |
| 23392 | 29.2430 | 0.6764 | 0.3905 | 0.150 | 0.000 | 0.000 | Dirty |
| 23423 | 29.2356 | 0.6764 | 0.3905 | 0.150 | 0.000 | 0.000 | Dirty |

### Insight

Huber from the accepted seed43 MSE boundary does not stabilize the post-`23295` dirty branch. It reproduces the same immediate full-frame artifact score (`0.147`) while ROI and stand stay clean, and it does not improve LPIPS over accepted MSE step `23295`. This rejects loss softening as a local seed43 boundary fix; the next seed43 attempt should change the trajectory before the boundary or target the local artifact mechanism directly.

## ARM seed44 Charbonnier leader extension

### What Was Tested

Continued the current seed44 visual/detail leader step `14854` with Charbonnier loss, ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, train-time eval disabled, and save interval `2`. This tested whether the clean seed44 detail window continues for a few more steps and can match or beat the old H40 LPIPS target.

Run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_charb_seed44_leader_extend/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_charb_s44_extend14854_14862_save2`

Selected renders:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_charb_seed44_leader_extend/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_charb_s44_extend14854_14862_save2/renders_artifact_selection_step-000014861`

### Results

| Run | Seed | Source step | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `charb_s44_extend14854_14862_save2` | 44 | 14854 | 14861 | 28.8794 | 0.6503 | 0.3653 | 0.000 | 0.000 | 0.000 | 70.0 | 380.3 | Keep as LPIPS/detail leader; balanced step 14858 is higher PSNR |

Timeline:

| Step | PSNR | SSIM | LPIPS | Full-frame artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 14856 | 28.9281 | 0.6498 | 0.3658 | 0.000 | 0.000 | 0.000 | Clean, small LPIPS gain over 14854 |
| 14858 | 28.9089 | 0.6514 | 0.3656 | 0.000 | 0.000 | 0.000 | Clean, best balanced PSNR/SSIM/LPIPS in this window |
| 14860 | 28.8833 | 0.6508 | 0.3654 | 0.000 | 0.000 | 0.000 | Clean, near old-H40 LPIPS |
| 14861 | 28.8794 | 0.6503 | 0.3653 | 0.000 | 0.000 | 0.000 | Clean LPIPS/detail leader |

Updated clean variance set using the LPIPS/detail leader:

| Seed | Selected checkpoint | PSNR | SSIM | LPIPS | Full-frame artifact | Train time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | `huber_delta02_s42_micro39935_40000_save16` step `39936` | 29.5082 | 0.6857 | 0.3964 | 0.000 | 80.1 |
| 43 | `mse_s43_micro23232_23296_save16` step `23295` | 29.2883 | 0.6758 | 0.3898 | 0.000 | 70.1 |
| 44 | `charb_s44_extend14854_14862_save2` step `14861` | 28.8794 | 0.6503 | 0.3653 | 0.000 | 70.0 |

Mean clean metrics: PSNR `29.2253`, SSIM `0.6706`, LPIPS `0.3839`, full-frame artifact `0.000`, mean train time about `73.4s`.

### Insight

Seed44 can now match the old H40 LPIPS/detail level while staying artifact-clean under ARM and occupancy-grid sampling. Step `14861` reaches LPIPS `0.365286`, slightly better than old H40 `0.3653`, with full-frame/ROI/stand artifact `0.000`. The cost is seed44 PSNR dropping below old H40, so this is a visual/detail leader rather than a per-seed all-metric leader. The 3-seed clean mean still beats old H40 PSNR/SSIM and improves mean LPIPS slightly, but seed42 and seed43 remain the real LPIPS bottleneck.


## ARM post-leader follow-up checks

### What Was Tested

Three short ARM-on, occupancy-on, FAS-off, Feature-Reweighting-off checks were run after the current seed42 Huber delta0.2 leader. These were diagnostic and do not replace the current leader.

### Results

| Run | Seed | Source step | Selected step | PSNR | SSIM | LPIPS | Full-frame artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `charb_s44_refine14861_14865_save1` | 44 | 14861 | 14863 | 28.8553 | 0.6497 | 0.3649 | 0.000 | 0.000 | 340.6 | 635.9 | Keep as comparison only; not current leader after visual review | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_charb_seed44_refine_boundary/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_charb_s44_refine14861_14865_save1/renders_artifact_selection_step-000014863` |
| `charb_s42_from_huber39936_40064_save16` | 42 | 39936 | 40063 | 29.4900 | 0.6858 | 0.3963 | 0.000 | 0.000 | 490.4 | 1230.1 | Reject as replacement; tiny LPIPS gain but lower PSNR/SSIM and micro ROI score worsens | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_charb_seed42_from_huber/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_charb_s42_from_huber39936_40064_save16/renders_artifact_selection_step-000040063` |
| `charb_s43_from_mse23295_23360_save16` | 43 | 23295 | 23296 | 29.2827 | 0.6757 | 0.3904 | 0.147 | 0.000 | 390.5 | 923.9 | Reject; dirty immediately after accepted boundary | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_charb_seed43_from_mse/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_charb_s43_from_mse23295_23360_save16/renders_artifact_selection_step-000023296` |

### Small-region artifact diagnostic

Added detector preset `micro` to `scripts/detect_structural_artifacts.py` and exposed it through the quiet runner/backfill scripts. This is diagnostic only and does not replace the historical `significant` score. The preset lowers connected-component thresholds to catch small hard holes/obstructions that can be visually obvious but too small for `artifact_score=0.000` under the significant preset.

ROI micro rescoring over all named ROIs:

| Candidate | Max ROI micro score | Serious ROIs | Read |
|---|---:|---:|---|
| Current leader seed42 Huber step 39936 | 0.255 | 0/10 | Best current balance; only small non-serious component |
| Seed42 Charbonnier step 40063 | 1.709 | 3/10 | Reject despite significant artifact `0.000`; stricter score catches new small defects |
| Old pre-fix H40 step 34816 | 11.612 | 7/10 | Better LPIPS but much worse small-region artifact score |

Micro ROI outputs:

`/home/ubuntu/repos/nerfstudio/LookCloser/experiments/micro_artifact_scores/current_leader`

`/home/ubuntu/repos/nerfstudio/LookCloser/experiments/micro_artifact_scores/seed42_charb40063`

`/home/ubuntu/repos/nerfstudio/LookCloser/experiments/micro_artifact_scores/old_h40`

### Insight

The stricter micro score agrees with the visual decision to keep seed42 Huber step `39936` as the single current leader over seed42 Charbonnier step `40063`, even though both have significant artifact `0.000`. Old H40 remains useful for LPIPS/detail comparison, but its small-region artifact score is worse than the current leader. Future checkpoint selection should report both significant and micro scores until the small-hole visual issue is solved.

## Occupancy warmup 8000 check

### What Was Tested

Two valid ARM-on, occupancy-on, FAS-off, Feature-Reweighting-off Charbonnier from-scratch runs raised both `occupancy_warmup_steps` and `occupancy_binary_warmup_steps` from `4096` to `8000`. This tested whether the residual small artifacts are caused by too-short occupancy cold-start protection.

### Results

| Run | Seed | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `occwarm8000_charb_s42` | 42 | 24576 | 29.3893 | 0.6815 | 0.4128 | 0.585 | 0.477 | 0.283 | 6011.4 | 6710.2 | Reject; worse LPIPS and no micro cleanup | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occwarm8000_charb_seed42/lookcloser/arm_h40_grid128_occwarm8000_charb_s42/renders_artifact_selection_step-000024576` |
| `occwarm8000_charb_s43` | 43 | 32768 | 29.4677 | 0.6670 | 0.4211 | 0.640 | 0.497 | 1.301 | 6011.3 | 6728.5 | Reject; worse LPIPS and worse stand micro score | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occwarm8000_charb_seed43/lookcloser/arm_h40_grid128_occwarm8000_charb_s43/renders_artifact_selection_step-000032768` |

### Insight

Raising occupancy/binary warmup to `8000` did not improve the current leader. The result is consistent with earlier artifact-to-occupancy debugging: the remaining failures are not primarily binary occupancy-grid misses, so further conservative occupancy warmup is not the next low-risk direction.

## Training-time occupancy bypass check

### What Was Tested

Temporarily added a reversible diagnostic that, on training batches only, forced the ARM occupancy grid binaries to fully occupied with probability `0.10`, `0.30`, or `0.40`, then restored the real grid immediately after ARM sampling. Eval/render stayed on the normal occupancy grid. All runs continued from the current seed42 Huber leader step `39936` to `40000` with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, and artifact-aware selection over all 3 eval views. The `40/50` idea was normalized to `40%` bypass and `60%` normal batches.

### Results

| Run | Bypass prob | Source step | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Current leader reference | 0.00 | 39936 | 39936 | 29.5082 | 0.6857 | 0.3964 | 0.691 | 0.652 | 0.255 | 80.1 | 398.3 | Keep leader | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_seed42_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_s42_micro39935_40000_save16/renders_artifact_selection_step-000039936` |
| `occdrop010_huber_delta02_s42` | 0.10 | 39936 | 39999 | 29.5093 | 0.6854 | 0.3964 | 0.676 | 0.653 | 0.255 | 120.1 | 735.3 | Reject; tiny LPIPS/PSNR change, no real micro cleanup | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occdrop010_huber_delta02_seed42_micro/lookcloser/arm_h40_grid128_occdrop010_huber_delta02_s42_39936_40000_save16_r1/renders_artifact_selection_step-000039999` |
| `occdrop030_huber_delta02_s42` | 0.30 | 39936 | 39968 | 29.5041 | 0.6854 | 0.3965 | 0.675 | 0.650 | 0.259 | 210.1 | 801.0 | Reject; lower PSNR/SSIM and no cleanup | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occdrop030_huber_delta02_seed42_micro/lookcloser/arm_h40_grid128_occdrop030_huber_delta02_s42_39936_40000_save16_r1/renders_artifact_selection_step-000039968` |
| `occdrop040_huber_delta02_s42` | 0.40 | 39936 | 39968 | 29.5069 | 0.6856 | 0.3965 | 0.675 | 0.650 | 0.259 | 180.1 | 736.4 | Reject; no improvement over 10/30 or current leader | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occdrop040_huber_delta02_seed42_micro/lookcloser/arm_h40_grid128_occdrop040_huber_delta02_s42_39936_40000_save16_r1/renders_artifact_selection_step-000039968` |

### Insight

Training-time occupancy bypass did not materially change the continuation. The best micro full-frame score moved only from the current leader's diagnostic `0.691` to `0.675`, while ROI/stand stayed about the same and LPIPS did not improve meaningfully. Because this is not a clear improvement, the diagnostic code was removed after the runs to keep the model and runner clean. This again points away from binary occupancy misses as the main cause of the remaining small holes/obstructions.

## Low distortion regularization checks

### What Was Tested

Tested whether reducing geometry compactness regularization could recover LPIPS/detail while keeping ARM enabled, occupancy-grid sampling enabled, FAS disabled, and Feature Reweighting disabled. Prior stronger distortion (`0.02`) did not clean artifacts; this checked the opposite direction with `distortion_loss_mult=0.005` and `0.0`. Seed42 continued from the current Huber delta0.2 leader step `39936`; seed43 continued from the accepted MSE step `23295`.

### Results

| Run | Seed | Source step | Distortion mult | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `huber_delta02_s42_dist0005_39936_40064_save16` | 42 | 39936 | 0.005 | 39952 | 29.5050 | 0.6857 | 0.3965 | 0.691 | 0.598 | 0.255 | 330.2 | 1498.5 | Reject; no LPIPS/detail gain and micro score not better | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_lowdist_seed42_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_dist0005_39936_40064_save16/renders_artifact_selection_step-000039952` |
| `huber_delta02_s42_dist0000_39936_40064_save16` | 42 | 39936 | 0.000 | 39984 | 29.4981 | 0.6877 | 0.3965 | 0.690 | 0.652 | 0.255 | 330.2 | 1514.5 | Reject; SSIM up but PSNR/LPIPS worse than leader | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_lowdist_seed42_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_dist0000_39936_40064_save16/renders_artifact_selection_step-000039984` |
| `mse_s43_dist0005_23295_23360_save16` | 43 | 23295 | 0.005 | 23296 | 29.2823 | 0.6757 | 0.3904 | 1.221 | 1.045 | 3.450 | 300.2 | 1053.9 | Reject; immediately worsens micro/ROI/stand artifacts | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_lowdist_seed43_micro/lookcloser/arm_h40_grid128_mse_s43_dist0005_23295_23360_save16/renders_artifact_selection_step-000023296` |

### Insight

Reducing distortion regularization is not a useful ARM-only detail lever here. Seed42 stays near the same LPIPS plateau while losing PSNR or micro score, and seed43's first post-boundary checkpoint becomes much dirtier. Keep default `distortion_loss_mult=0.01`.

## Seed42 Huber delta bracket

### What Was Tested

Tested a narrow Huber delta bracket around the accepted `huber_delta=0.2`, starting from the saved clean seed42 minfreq4/cap2048 checkpoint `39680`. All runs kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, train-time eval disabled, and artifact-aware selection over all 3 eval views with the stricter `micro` detector.

### Results

| Run | Seed | Source step | Huber delta | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `huber_delta016_s42_39680_39936_save64` | 42 | 39680 | 0.16 | 39744 | 29.4923 | 0.6821 | 0.3969 | 0.741 | 0.665 | 1.075 | 180.2 | 732.7 | Reject; worse LPIPS and micro artifacts than accepted delta0.2 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta_bracket_seed42/lookcloser/arm_h40_grid128_huber_delta016_s42_39680_39936_save64/renders_artifact_selection_step-000039744` |
| `huber_delta018_s42_39680_39936_save64` | 42 | 39680 | 0.18 | 39808 | 29.4915 | 0.6826 | 0.3970 | 0.855 | 0.806 | 1.075 | 180.1 | 786.7 | Reject; worse across LPIPS and micro artifacts | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta_bracket_seed42/lookcloser/arm_h40_grid128_huber_delta018_s42_39680_39936_save64/renders_artifact_selection_step-000039808` |
| `huber_delta022_s42_39680_39936_save64` | 42 | 39680 | 0.22 | 39744 | 29.4924 | 0.6821 | 0.3968 | 0.697 | 0.657 | 1.075 | 210.2 | 780.2 | Reject; closest of bracket, but still worse than accepted delta0.2 leader | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta_bracket_seed42/lookcloser/arm_h40_grid128_huber_delta022_s42_39680_39936_save64/renders_artifact_selection_step-000039744` |

### Insight

The accepted seed42 Huber delta0.2 remains the best seed42 setting. The bracket did not find a better detail/artifact tradeoff, and all selected candidates were materially worse by micro ROI/stand score. Do not spend more budget on nearby Huber deltas unless another change shifts the trajectory.

## Seed43 earlier Charbonnier switch

### What Was Tested

Tested the hypothesis that the rejected seed43 Charbonnier switch was dirty because it switched exactly at the accepted MSE boundary (`23295`). This run switched earlier, from clean MSE step `23039`, then continued with Charbonnier to `23296` using ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, train-time eval disabled, and micro artifact-aware selection over all 3 eval views.

### Results

| Run | Seed | Source step | Loss | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `charb_s43_from_mse23039_23296_save32` | 43 | 23039 | Charbonnier | 23264 | 29.2669 | 0.6757 | 0.3909 | 0.628 | 0.571 | 1.553 | 150.1 | 1111.7 | Reject; switching earlier still creates stand/ROI artifacts and worsens LPIPS vs accepted MSE | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_charb_seed43_early_micro/lookcloser/arm_h40_grid128_charb_s43_from_mse23039_23296_save32/renders_artifact_selection_step-000023264` |

Per-candidate read:

| Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---|
| 23040 | 29.2804 | 0.6751 | 0.3905 | 1.304 | 3.445 | Dirty immediately after switch |
| 23168 | 29.3142 | 0.6734 | 0.3923 | 1.164 | 1.560 | Dirty |
| 23200 | 29.3229 | 0.6746 | 0.3916 | 1.121 | 1.550 | Dirty |
| 23232 | 29.2996 | 0.6751 | 0.3917 | 0.717 | 1.568 | Dirty |
| 23264 | 29.2669 | 0.6757 | 0.3909 | 0.628 | 1.553 | Least bad, still dirty |
| 23295 | 29.2565 | 0.6750 | 0.3909 | 0.628 | 1.556 | Dirty |

### Insight

Charbonnier is not a useful seed43 boundary fix even when switched before the known dirty boundary. It immediately introduces the same stand/ROI artifact class and does not beat the accepted seed43 MSE step `23295` (`29.2883`/`0.6758`/`0.3898`, artifact/stand `0.000`). Keep seed43 on MSE for now.

## Seed42 batch2048 continuation check

### What Was Tested

Tested whether reducing the train batch from `4096` to `2048` after the accepted seed42 Huber leader step `39936` can perturb the ARM trajectory enough to clean remaining micro artifacts. The run kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, `occupancy_warmup_steps=4096`, `occupancy_binary_warmup_steps=4096`, Huber delta `0.2`, train-time eval disabled, and micro artifact-aware selection over all 3 eval views.

### Results

| Run | Seed | Source step | Train rays/batch | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `huber_delta02_s42_batch2048_39936_40128_save16` | 42 | 39936 | 2048 | 39968 | 29.4982 | 0.6858 | 0.3965 | 0.680 | 0.598 | 0.255 | 180.1 | 1440.0 | Reject; no LPIPS/detail gain and no ROI/stand cleanup | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_batch2048_seed42_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_batch2048_39936_40128_save16/renders_artifact_selection_step-000039968` |

Per-candidate read:

| Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---|
| 39952 | 29.5050 | 0.6858 | 0.3965 | 0.691 | 0.255 | Close to leader, no improvement |
| 39968 | 29.4982 | 0.6858 | 0.3965 | 0.680 | 0.255 | Selected by artifact, still not better |
| 39984 | 29.4977 | 0.6859 | 0.3965 | 0.736 | 0.259 | Worse artifact |
| 40000 | 29.4956 | 0.6880 | 0.3965 | 0.735 | 0.259 | SSIM up, artifact worse |
| 40016 | 29.4944 | 0.6882 | 0.3965 | 0.735 | 0.259 | Worse artifact |
| 40032 | 29.4935 | 0.6882 | 0.3965 | 0.735 | 0.259 | Worse artifact |
| 40048 | 29.4922 | 0.6884 | 0.3965 | 0.735 | 0.259 | Worse artifact |
| 40064 | 29.4917 | 0.6885 | 0.3965 | 0.735 | 0.255 | Worse artifact |
| 40080 | 29.4881 | 0.6885 | 0.3965 | 0.782 | 0.255 | Dirty |
| 40096 | 29.4863 | 0.6886 | 0.3965 | 0.740 | 0.255 | Worse artifact |
| 40112 | 29.4871 | 0.6887 | 0.3965 | 0.740 | 0.259 | Worse artifact |
| 40127 | 29.4865 | 0.6887 | 0.3965 | 0.733 | 0.259 | Worse artifact |

### Insight

Smaller batch size is not a useful cleanup lever for the current seed42 Huber branch. It slightly changes SSIM and checkpoint ordering, but does not recover the old H40 LPIPS/detail and does not reduce the small-region ROI/stand issue. Keep `train_num_rays_per_batch=4096` as the accepted setting unless a future from-scratch sweep shows a stronger variance effect.

## Seed42 fpl4 capacity check

### What Was Tested

Tested whether increasing field capacity by changing `hash_features_per_level` from `2` to `4` helps the ARM-only artifact/detail problem. The run was from scratch with the accepted seed42 ARM recipe otherwise: ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, Huber delta `0.2`, `max_num_iterations=200000`, eval/save interval `8192`, early stop on eval loss, and micro artifact-aware selection over all 3 eval views.

### Results

| Run | Seed | Hash features/level | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | Max ROI micro | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `fpl4_huber_delta02_s42_200k_i8192` | 42 | 4 | 8192 | 14.0866 | 0.3651 | 0.8862 | 2.867 | 2.867 | 559.217 | 0.000 | 3365.5 | 3689.4 | Reject; eval collapsed and artifacts are massive | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_capacity_fpl4_seed42_micro/lookcloser/arm_h40_grid128_fpl4_huber_delta02_s42_200k_i8192/renders_artifact_selection_step-000008192` |

Per-candidate read:

| Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Max ROI micro | Read |
|---:|---:|---:|---:|---:|---:|---|
| 8192 | 14.0866 | 0.3651 | 0.8862 | 2.867 | 559.217 | Least bad by artifact, still collapsed |
| 16384 | 14.0911 | 0.3693 | 0.8751 | 5.718 | 584.760 | Eval loss best but artifact worse |
| 24576 | 14.0405 | 0.3723 | 0.8693 | 7.137 | 743.912 | Early stop after eval loss worsened |

### Insight

This is not a useful capacity direction in its tested form. Doubling hash features with Huber from scratch produced a severe eval collapse before `24576` and early-stopped far below the accepted recipe. This does not fully rule out a two-stage Charbonnier-to-Huber fpl4 schedule, but fpl4 is not a low-risk next step while the ARM-only artifact issue remains the priority.

## Seed42 fpl4 Charbonnier capacity check

### What Was Tested

Retested the same capacity idea with a safer from-scratch loss schedule: `hash_features_per_level=4`, Charbonnier reconstruction loss, ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, `max_num_iterations=200000`, eval/save interval `8192`, early stop on eval loss, and micro artifact-aware selection over all 3 eval views.

### Results

| Run | Seed | Hash features/level | Loss | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | ROI micro | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `fpl4_charb_s42_200k_i8192` | 42 | 4 | Charbonnier | 16384 | 29.5127 | 0.6774 | 0.4068 | 1.175 | 1.001 | 0.000 | 0.000 | 3486.3 | 3933.0 | Reject as replacement; selected artifact checkpoint is still dirty full-frame and worse LPIPS | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_capacity_fpl4_charb_seed42_micro/lookcloser/arm_h40_grid128_fpl4_charb_s42_200k_i8192/renders_artifact_selection_step-000016384` |

Metric-best candidate:

| Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | ROI micro | Stand connector | Renders | Read |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 32768 | 29.7857 | 0.6843 | 0.3912 | 1.723 | 0.831 | 0.000 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_capacity_fpl4_charb_seed42_micro/lookcloser/arm_h40_grid128_fpl4_charb_s42_200k_i8192/renders_artifact_selection_step-000032768` | Strong PSNR/LPIPS signal but dirty |

Per-candidate read:

| Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | ROI micro | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8192 | 28.5604 | 0.6499 | 0.4452 | 1.730 | 0.263 | 0.263 | Dirty and weak |
| 16384 | 29.5127 | 0.6774 | 0.4068 | 1.175 | 0.000 | 0.000 | Least dirty full-frame, still not clean |
| 24576 | 29.7945 | 0.6825 | 0.3941 | 1.828 | 0.000 | 0.000 | Better metrics, dirty |
| 32768 | 29.7857 | 0.6843 | 0.3912 | 1.723 | 0.831 | 0.000 | Best LPIPS, dirty |

### Insight

The fpl4 capacity direction is real for PSNR/LPIPS under Charbonnier, unlike the failed Huber-from-scratch run, but it is not artifact-clean. Later checkpoints recover much of the current leader's LPIPS gap (`0.3912` vs `0.3964`) and improve PSNR, but micro artifacts rise far above the current leader. Treat fpl4+Charbonnier as a metric/detail signal that needs a separate artifact stabilizer, not as a replacement recipe.

## Seed42 fpl4 Charbonnier distortion 0.02 check

### What Was Tested

Tested whether stronger geometry compactness regularization can stabilize the dirty fpl4+Charbonnier metric branch. The run used `hash_features_per_level=4`, Charbonnier loss, `distortion_loss_mult=0.02`, ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, `max_num_iterations=200000`, eval/save interval `8192`, early stopping, and micro artifact-aware selection over all 3 eval views.

### Results

| Run | Seed | Hash features/level | Distortion mult | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | ROI micro | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `fpl4_charb_dist02_s42_200k_i8192` | 42 | 4 | 0.02 | 32768 | 29.8571 | 0.6927 | 0.4025 | 0.597 | 0.432 | 0.758 | 0.000 | 5079.8 | 5862.8 | Reject as replacement; full-frame micro improves, but ROI and LPIPS are worse than current leader | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_capacity_fpl4_charb_dist02_seed42_micro/lookcloser/arm_h40_grid128_fpl4_charb_dist02_s42_200k_i8192/renders_artifact_selection_step-000032768` |

Metric-best candidate:

| Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | ROI micro | Stand connector | Renders | Read |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 49152 | 29.9633 | 0.6964 | 0.3970 | 0.780 | 0.889 | 0.000 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_capacity_fpl4_charb_dist02_seed42_micro/lookcloser/arm_h40_grid128_fpl4_charb_dist02_s42_200k_i8192/renders_artifact_selection_step-000049152` | Very strong PSNR/SSIM, LPIPS still worse than leader, dirty ROI |

Per-candidate read:

| Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | ROI micro | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8192 | 28.6208 | 0.6535 | 0.4576 | 1.116 | 0.752 | 0.306 | Dirty and weak |
| 16384 | 29.5665 | 0.6856 | 0.4162 | 0.692 | 0.523 | 0.000 | Similar full-frame micro to leader, ROI/LPIPS worse |
| 24576 | 29.8232 | 0.6946 | 0.4061 | 0.832 | 0.535 | 0.000 | Dirty |
| 32768 | 29.8571 | 0.6927 | 0.4025 | 0.597 | 0.758 | 0.000 | Selected by artifact, not a replacement |
| 40960 | 29.9406 | 0.6930 | 0.4006 | 1.072 | 0.817 | 0.000 | Dirty |
| 49152 | 29.9633 | 0.6964 | 0.3970 | 0.780 | 0.889 | 0.000 | Metric-best, dirty |

### Insight

Stronger distortion partially stabilizes fpl4 full-frame micro artifacts, but at a clear LPIPS/detail and ROI cost. Compared with fpl4+Charbonnier at default distortion, step `32768` changes from LPIPS `0.3912`, micro `1.723`, ROI `0.831` to LPIPS `0.4025`, micro `0.597`, ROI `0.758`. This suggests distortion is a real stabilizer for the capacity branch, but `0.02` is too strong to become the current recipe. A middle value such as `0.015` is the next reasonable bracket if continuing this direction.

Follow-up significant-preset rescoring showed that `0.02` also does not contain an official-clean checkpoint. Significant full-frame max scores by step are: `8192=2.055`, `16384=2.102`, `24576=2.124`, `32768=2.108`, `40960=2.116`, and `49152=2.133`; significant ROI remains `0.000` at every saved checkpoint. The dirty component is therefore a persistent full-frame eval0 issue, not only the micro ROI issue used for original selection.

## Seed42 fpl4 Charbonnier distortion 0.015 check

### What Was Tested

Bracketed the fpl4+Charbonnier distortion stabilizer between default `0.01` and too-strong `0.02`. The run used `hash_features_per_level=4`, Charbonnier loss, `distortion_loss_mult=0.015`, ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, `max_num_iterations=200000`, eval/save interval `8192`, early stopping, and micro artifact-aware selection over all 3 eval views.

### Results

| Run | Seed | Hash features/level | Distortion mult | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | ROI micro | Stand connector | Significant full-frame | Significant ROI | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `fpl4_charb_dist015_s42_200k_i8192` | 42 | 4 | 0.015 | 24576 | 29.9584 | 0.6865 | 0.3910 | 0.560 | 0.415 | 0.578 | 0.000 | 1.817 | 0.000 | 3425.9 | 3869.8 | Reject as final; strong metrics but official significant full-frame fails on eval0 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_capacity_fpl4_charb_dist015_seed42_micro/lookcloser/arm_h40_grid128_fpl4_charb_dist015_s42_200k_i8192/renders_artifact_selection_step-000024576` |

Metric-best candidate:

| Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | ROI micro | Stand connector | Significant full-frame | Significant ROI | Renders | Read |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 32768 | 30.0591 | 0.6902 | 0.3880 | 0.661 | 0.000 | 0.000 | 1.801 | 0.000 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_capacity_fpl4_charb_dist015_seed42_micro/lookcloser/arm_h40_grid128_fpl4_charb_dist015_s42_200k_i8192/renders_artifact_selection_step-000032768` | Best metrics so far in fpl4 bracket, but significant eval0 full-frame fails |

Per-candidate read:

| Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | ROI micro | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8192 | 28.8486 | 0.6521 | 0.4435 | 1.648 | 3.551 | 0.608 | Dirty and weak |
| 16384 | 29.7397 | 0.6790 | 0.4034 | 0.591 | 1.766 | 0.249 | Full micro lower, ROI dirty |
| 24576 | 29.9584 | 0.6865 | 0.3910 | 0.560 | 0.578 | 0.000 | Selected by micro, official significant dirty |
| 32768 | 30.0591 | 0.6902 | 0.3880 | 0.661 | 0.000 | 0.000 | Metric-best, official significant dirty |

### Significant Eval0 Failure

The significant preset fails only on eval0 for selected step `24576` and metric-best step `32768`; eval1/eval2 and significant ROI scoring are clean. A follow-up rescore of the earlier saved checkpoints showed the same full-frame eval0 issue already exists before the metric peak:

| Step | Significant max full-frame | Significant eval0 | Significant eval1 | Significant eval2 | Significant ROI |
|---:|---:|---:|---:|---:|---:|
| 8192 | 3.085 | 3.085 | 0.243 | 0.000 | 0.000 |
| 16384 | 1.804 | 1.804 | 0.000 | 0.000 | 0.000 |
| 24576 | 1.817 | 1.817 | 0.000 | 0.000 | 0.000 |
| 32768 | 1.801 | 1.801 | 0.000 | 0.000 | 0.000 |

Detected eval0 significant components for the selected and metric-best checkpoints are in the lower half of the full frame, outside the curated stand/cable ROI set:

| Step | Significant eval0 score | Major bboxes `(area, x0, y0, x1, y1)` |
|---:|---:|---|
| 24576 | 1.817 | `(1080, 244, 999, 290, 1037)`, `(596, 1193, 1044, 1234, 1066)`, `(582, 554, 1045, 594, 1074)`, `(569, 1150, 1033, 1182, 1061)` |
| 32768 | 1.801 | `(1071, 244, 999, 290, 1037)`, `(596, 1193, 1044, 1234, 1066)`, `(569, 1150, 1033, 1182, 1061)`, `(556, 555, 1044, 593, 1066)` |

### Insight

`distortion_loss_mult=0.015` is the best fpl4 balance so far by metrics and improves the capacity branch substantially, but it still fails the full-frame significant gate. The failures are not in the curated stand/cable ROI under the significant preset, which suggests the next step should either add a targeted eval0 lower-frame ROI to the visual gate or stabilize the same fpl4 branch specifically around those eval0 components. It is not acceptable as the final artifact-clean recipe.

## Seed42 fpl4 Charbonnier distortion 0.0175 check

### What Was Tested

Tested a midpoint between the strong-but-dirty `distortion_loss_mult=0.015` branch and the more stable-but-detail-costly `0.02` branch. The run used `hash_features_per_level=4`, Charbonnier loss, `distortion_loss_mult=0.0175`, ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, explicit `max_num_iterations=200000`, eval/save interval `8192`, early stopping enabled, and significant artifact-aware selection configured over all 3 eval views.

### Results

| Run | Seed | Hash features/level | Distortion mult | Last eval step | PSNR | SSIM | LPIPS | Eval loss | Train time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `fpl4_charb_dist0175_s42_200k_i8192` | 42 | 4 | 0.0175 | 16384 | 16.5803 | 0.6206 | 0.5156 | 0.121642 | ~1755 | Reject; full-eval collapse, stopped manually before artifact render stage | none |

Per-eval read:

| Step | PSNR | SSIM | LPIPS | Eval loss | Read |
|---:|---:|---:|---:|---:|---|
| 8192 | 16.8762 | 0.6122 | 0.5415 | 0.124119 | Already far below neighboring fpl4 runs |
| 16384 | 16.5803 | 0.6206 | 0.5156 | 0.121642 | Eval loss slightly improves, but quality remains collapsed |

### Insight

`distortion_loss_mult=0.0175` unexpectedly collapsed on full-eval quality and is not a useful bracket point in this exact trajectory. This is worse than both neighboring completed fpl4 runs: `0.015` reached PSNR `29.9584`/`30.0591` around steps `24576`/`32768`, and `0.02` reached PSNR `29.8571`/`29.9633` around steps `32768`/`49152`. Because the run was clearly rejected before offline artifact rendering, no selected render directory was created. The heavy checkpoints were deleted after recording the metrics.

## Seed43 fpl4 Charbonnier distortion 0.015 variance check

### What Was Tested

Tested whether the best fpl4 seed42 metric branch (`hash_features_per_level=4`, Charbonnier, `distortion_loss_mult=0.015`) is variance-safe. The run used the same ARM-only setup as seed42: ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, explicit `max_num_iterations=200000`, eval/save interval `8192`, and significant artifact selection configured over all 3 eval views.

### Results

| Run | Seed | Hash features/level | Distortion mult | Last eval step | PSNR | SSIM | LPIPS | Eval loss | Train time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `fpl4_charb_dist015_s43_200k_i8192` | 43 | 4 | 0.015 | 24576 | 26.7102 | 0.7039 | 0.4296 | 0.033812 | ~2566 | Reject; weak PSNR/LPIPS variance, stopped before artifact render stage | none |

Per-eval read:

| Step | PSNR | SSIM | LPIPS | Eval loss | Read |
|---:|---:|---:|---:|---:|---|
| 8192 | 26.1574 | 0.6767 | 0.4758 | 0.036314 | Undertrained versus seed42 fpl4 |
| 16384 | 26.6000 | 0.6993 | 0.4404 | 0.034773 | SSIM high, but PSNR/LPIPS still weak |
| 24576 | 26.7102 | 0.7039 | 0.4296 | 0.033812 | Improving loss, but still not competitive |

### Insight

The fpl4+Charbonnier `distortion_loss_mult=0.015` branch is not variance-safe as a next leader. Seed42 gave very strong metrics but persistent eval0 full-frame significant artifacts; seed43 stayed much weaker by PSNR/LPIPS through `24576` despite improving eval loss and high SSIM. This reduces confidence that fpl4 capacity is the next low-risk ARM-only fix without a separate trajectory/stability change.

## Seed42 leader dense micro checkpoint check

### What Was Tested

Tested whether the current seed42 leader only missed a cleaner nearby checkpoint because the accepted run saved every `16` steps. Continued the accepted leader checkpoint `39936` to `39968` with the same ARM recipe and save interval `2`, keeping ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Huber delta `0.2`, and micro artifact-aware selection over all 3 eval views.

### Results

| Run | Seed | Source step | Save interval | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `huber_delta02_s42_39936_39968_save2_micro` | 42 | 39936 | 2 | 39942 | 29.5084 | 0.6857 | 0.3964 | 0.690 | 0.650 | 0.255 | 240.2 | 1931.0 | Reject as replacement; essentially tied, no ROI/stand cleanup | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_seed42_leader_dense_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_39936_39968_save2_micro/renders_artifact_selection_step-000039942` |

Per-candidate read:

| Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---|
| 39938 | 29.5084 | 0.6857 | 0.3964 | 0.690 | 0.255 | Tied with leader |
| 39940 | 29.5083 | 0.6857 | 0.3964 | 0.691 | 0.255 | Tied |
| 39942 | 29.5084 | 0.6857 | 0.3964 | 0.690 | 0.255 | Selected, no real cleanup |
| 39944 | 29.5085 | 0.6857 | 0.3964 | 0.691 | 0.255 | Tied |
| 39946 | 29.5084 | 0.6857 | 0.3964 | 0.690 | 0.255 | Tied |
| 39948 | 29.5085 | 0.6858 | 0.3964 | 0.691 | 0.255 | Tied |
| 39950 | 29.5085 | 0.6858 | 0.3964 | 0.691 | 0.255 | Tied |
| 39952 | 29.5048 | 0.6858 | 0.3965 | 0.691 | 0.255 | Metrics worse |
| 39954 | 29.5050 | 0.6858 | 0.3965 | 0.730 | 0.255 | Worse artifact |
| 39956 | 29.5051 | 0.6858 | 0.3965 | 0.730 | 0.255 | Worse artifact |
| 39958 | 29.5051 | 0.6858 | 0.3965 | 0.730 | 0.255 | Worse artifact |
| 39960 | 29.5051 | 0.6858 | 0.3965 | 0.730 | 0.255 | Worse artifact |
| 39962 | 29.5054 | 0.6858 | 0.3965 | 0.730 | 0.255 | Worse artifact |
| 39964 | 29.5054 | 0.6858 | 0.3965 | 0.730 | 0.255 | Worse artifact |
| 39966 | 29.5056 | 0.6859 | 0.3965 | 0.712 | 0.255 | Still worse |
| 39967 | 29.5056 | 0.6859 | 0.3965 | 0.712 | 0.255 | Still worse |

### Insight

Dense checkpointing around the seed42 leader does not reveal a hidden cleaner checkpoint. Steps `39938`-`39950` form a near-identical plateau, then the later steps lose PSNR/LPIPS and worsen micro artifact score. The accepted current leader remains step `39936`; the remaining small-region issue needs a model/training change rather than denser checkpoint selection in this local window.

## Seed42 gradient-scaling continuation check

### What Was Tested

Tested whether enabling nerfstudio-style gradient scaling can stabilize the current seed42 Huber leader trajectory and reduce the residual micro small-region artifacts. The run continued the accepted clean leader checkpoint `39936` to `40128` with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, Huber delta `0.2`, `use_gradient_scaling=True`, train-time eval disabled, save interval `16`, and micro artifact-aware selection over all 3 eval views.

### Results

| Run | Seed | Source step | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | ROI micro | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `huber_delta02_s42_gradscale_39936_40128_save16` | 42 | 39936 | 39968 | 29.5036 | 0.6858 | 0.3965 | 0.690 | 0.598 | 0.255 | 0.255 | 180.2 | 1392.9 | Reject; no metric gain and no micro cleanup | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_gradscale_seed42_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_gradscale_39936_40128_save16/renders_artifact_selection_step-000039968` |

Per-candidate read:

| Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---|
| 39952 | 29.5045 | 0.6857 | 0.3965 | 0.690 | 0.255 | Tied artifact, weaker metrics than leader |
| 39968 | 29.5036 | 0.6858 | 0.3965 | 0.690 | 0.255 | Selected by artifact, still weaker |
| 39984 | 29.4977 | 0.6879 | 0.3965 | 0.736 | 0.259 | SSIM up, artifact worse |
| 40000 | 29.4966 | 0.6879 | 0.3965 | 0.729 | 0.255 | Worse artifact |
| 40016 | 29.4944 | 0.6879 | 0.3965 | 0.736 | 0.255 | Worse artifact |
| 40032 | 29.4929 | 0.6880 | 0.3965 | 0.735 | 0.255 | Worse artifact |
| 40048 | 29.4924 | 0.6881 | 0.3965 | 0.736 | 0.255 | Worse artifact |
| 40064 | 29.4906 | 0.6881 | 0.3965 | 0.735 | 0.255 | Worse artifact |
| 40080 | 29.4859 | 0.6881 | 0.3966 | 0.769 | 0.255 | Dirty micro |
| 40096 | 29.4842 | 0.6882 | 0.3966 | 0.768 | 0.255 | Dirty micro |
| 40112 | 29.4840 | 0.6882 | 0.3966 | 0.770 | 0.255 | Dirty micro |
| 40127 | 29.4837 | 0.6883 | 0.3966 | 0.770 | 0.255 | Dirty micro |

### Insight

Gradient scaling is not a useful stabilizer for the current seed42 Huber leader. It reproduces the same small-region micro score at the best early continuation checkpoints and then worsens micro artifact score while trading PSNR/LPIPS for slightly higher SSIM. Keep `use_gradient_scaling=False` for the accepted recipe unless a future from-scratch variance sweep gives contrary evidence.

## Seed42 log2 hashmap 24 capacity check

### What Was Tested

Tested whether reducing hash-grid collisions by increasing `log2_hashmap_size` from `23` to `24` improves ARM detail without the instability of increasing `hash_features_per_level` to `4`. The run was from scratch with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `hash_features_per_level=2`, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, Charbonnier loss, `distortion_loss_mult=0.01`, explicit `max_num_iterations=200000`, eval/save interval `8192`, and micro artifact selection configured over all 3 eval views.

### Results

| Run | Seed | log2 hashmap | Last eval step | PSNR | SSIM | LPIPS | Eval loss | Train time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `log2hash24_charb_s42_200k_i8192` | 42 | 24 | 16384 | 27.5576 | 0.6675 | 0.4294 | 0.030371 | ~1640 | Reject; weak metrics before artifact render stage | none |

Per-eval read:

| Step | PSNR | SSIM | LPIPS | Eval loss | Read |
|---:|---:|---:|---:|---:|---|
| 8192 | 27.1605 | 0.6451 | 0.4652 | 0.033109 | Not collapsed, but weak |
| 16384 | 27.5576 | 0.6675 | 0.4294 | 0.030371 | Improving, still far below current leader and fpl4 seed42 metric signal |

### Insight

Increasing hash table size alone is not a useful low-risk detail lever in this setup. It avoids the severe fpl4 Huber collapse, but by step `16384` remains far behind the current clean leader (`29.5082`/`0.6857`/`0.3964`) and does not justify continuing to later checkpoints. The more promising capacity midpoint, if capacity is revisited, is `hash_features_per_level=3` rather than larger hash table size alone.

## Seed42 batch8192 continuation from current leader

Tested whether increasing training rays per batch from `4096` to `8192` can smooth the residual small-region micro artifacts without restarting the whole run. The run continued the current seed42 Huber leader checkpoint `39936` to `40256` with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, Huber delta `0.2`, train-time eval disabled, save interval `64`, and micro artifact-aware selection over all 3 eval views.

| Run | Seed | Train rays | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | ROI micro | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `batch8192_39936_40256_save64` | 42 | 8192 | 40000 | 29.4978 | 0.6860 | 0.3965 | 0.735 | 0.650 | 0.255 | 0.255 | 80.1 | 455.5 | Reject; worse micro artifact and no metric gain over current leader | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_batch8192_seed42_micro2/lookcloser/arm_h40_grid128_huber_delta02_s42_batch8192_39936_40256_save64/renders_artifact_selection_step-000040000` |

Candidate table:

| Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | ROI micro | Stand connector | Selected renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 40000 | 29.4978 | 0.6860 | 0.3965 | 0.735 | 0.650 | 0.255 | 0.255 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_batch8192_seed42_micro2/lookcloser/arm_h40_grid128_huber_delta02_s42_batch8192_39936_40256_save64/renders_artifact_selection_step-000040000` |
| 40064 | 29.4898 | 0.6884 | 0.3965 | 0.740 | 0.651 | 0.255 | 0.255 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_batch8192_seed42_micro2/lookcloser/arm_h40_grid128_huber_delta02_s42_batch8192_39936_40256_save64/renders_artifact_selection_step-000040064` |
| 40128 | 29.4834 | 0.6887 | 0.3966 | 0.886 | 0.796 | 0.259 | 0.259 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_batch8192_seed42_micro2/lookcloser/arm_h40_grid128_huber_delta02_s42_batch8192_39936_40256_save64/renders_artifact_selection_step-000040128` |
| 40192 | 29.4946 | 0.6887 | 0.3965 | 0.848 | 0.759 | 0.259 | 0.259 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_batch8192_seed42_micro2/lookcloser/arm_h40_grid128_huber_delta02_s42_batch8192_39936_40256_save64/renders_artifact_selection_step-000040192` |
| 40255 | 29.5078 | 0.6886 | 0.3964 | 0.848 | 0.758 | 0.259 | 0.259 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_batch8192_seed42_micro2/lookcloser/arm_h40_grid128_huber_delta02_s42_batch8192_39936_40256_save64/renders_artifact_selection_step-000040255` |

### Insight

Batch8192 continuation is not a useful cleanup lever for the current seed42 leader. It slightly raises SSIM at later checkpoints, but PSNR/LPIPS do not improve and every tested checkpoint is worse than the current leader on micro full-frame artifact score (`0.735`-`0.886` vs `0.691` in the same micro detector). Keep the current batch4096 checkpoint as the leader.

## ARM occupancy-bypass sampling ratio check

Tested the hypothesis that small residual ARM artifacts may come from never sampling cells pruned by the occupancy grid. A temporary per-training-batch implementation was used: with probability `p`, the current ARM sampler temporarily treated occupancy binaries as fully occupied for that batch; eval/render still used the normal occupancy grid. Tested `p=0.10`, `0.30`, and `0.40` from scratch with the accepted ARM-only seed42 recipe: `grid_resolution=128`, batch `4096`, Huber delta `0.2`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, occupancy warmups `4096/4096`, FAS disabled, Feature Reweighting disabled, `max_num_iterations=200000`, and interval `8192`.

The implementation was intentionally removed after the test because all three variants collapsed by the first eval.

| Run | Bypass probability | Last train step before stop | Eval step | PSNR | SSIM | LPIPS | Artifact score | Train time (s) | Decision | Renders |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| `occmix_bypass10_s42_200k_i8192` | 0.10 | 10440 | 8192 | 13.9984 | 0.3691 | 0.9090 | not scored; rejected before artifact selection | ~2472 | Reject: severe underfit/collapse | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occmix_bypass_200k/lookcloser/arm_h40_grid128_occmix_bypass10_s42_200k_i8192/renders_stopped_step-000008192` |
| `occmix_bypass30_s42_200k_i8192` | 0.30 | 10410 | 8192 | 13.8849 | 0.3597 | 0.9041 | not scored; rejected before artifact selection | ~2478 | Reject: severe underfit/collapse | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occmix_bypass_200k/lookcloser/arm_h40_grid128_occmix_bypass30_s42_200k_i8192/renders_stopped_step-000008192` |
| `occmix_bypass40_s42_200k_i8192` | 0.40 | 10410 | 8192 | 13.5845 | 0.3543 | 0.9121 | not scored; rejected before artifact selection | ~2477 | Reject: severe underfit/collapse | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_occmix_bypass_200k/lookcloser/arm_h40_grid128_occmix_bypass40_s42_200k_i8192/renders_stopped_step-000008192` |

### Insight

Per-batch fully-occupied traversal is the wrong form of the fallback idea. Even `p=0.10` collapses the first full eval to PSNR `13.9984`, while the same family without this bypass previously reached PSNR `27.7624` at step `8192` and later became clean after longer training. This likely overexposes the model to broad low-value intervals and saturates ARM sample allocation instead of providing a small corrective signal for missed thin details. If revisiting this idea, use a much narrower safety mechanism such as a tiny number of uniform fallback samples per ray or a targeted low-frequency/low-weight auxiliary loss, not whole-batch fully-occupied occupancy traversal.

## Seed42 fpl3 capacity feasibility check

Tested the planned capacity midpoint `hash_features_per_level=3` with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Charbonnier loss, `distortion_loss_mult=0.0125`, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, occupancy warmups `4096/4096`, and explicit `max_num_iterations=200000`.

The run failed during model construction before training:

```text
RuntimeError: GridEncoding: n_features_per_level must be 1, 2, 4, or 8.
```

### Insight

The intended fpl3 midpoint is not available with the current tiny-cuda-nn hash-grid encoding. Testing this capacity midpoint would require changing the encoding implementation, so it is not a low-hanging experiment. The nearest valid capacity values remain `2` and `4`; `4` is already known to be metric-promising under Charbonnier but not artifact-clean or variance-safe.

## Seed42 tiny adaptive fixed-fallback check

Tested whether a much smaller version of fallback sampling could provide the "sample a little everywhere" effect without the catastrophic behavior of whole-batch occupancy bypass or the previously rejected larger fallback. Both runs continued the current seed42 Huber leader checkpoint `39936` to `40064`, with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, Huber delta `0.2`, train-time eval disabled, save interval `16`, and micro artifact-aware selection over all 3 eval views.

| Run | Fallback samples/ray | Source step | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | ROI micro | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `fallback4_39936_40064_save16` | 4 | 39936 | 40063 | 18.9848 | 0.5062 | 0.6282 | 49.554 | 49.427 | 29.195 | 15.603 | 90.1 | 859.0 | Reject; fallback intervals dominate rendering | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_tinyfallback_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_fallback4_39936_40064_save16/renders_artifact_selection_step-000040063` |
| `fallback8_39936_40064_save16` | 8 | 39936 | 40000 | 21.1140 | 0.6064 | 0.4966 | 6.211 | 6.036 | 13.963 | 0.641 | 90.1 | 879.6 | Reject; metrics/artifacts far worse than leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_tinyfallback_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_fallback8_39936_40064_save16/renders_artifact_selection_step-000040000` |

Candidate table:

| Run | Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | ROI micro | Stand connector |
|---|---:|---:|---:|---:|---:|---:|---:|
| fallback4 | 39952 | 17.6075 | 0.4842 | 0.6442 | 56.258 | 30.487 | 18.544 |
| fallback4 | 39968 | 17.8822 | 0.4889 | 0.6410 | 54.719 | 30.592 | 18.238 |
| fallback4 | 39984 | 18.1429 | 0.4931 | 0.6380 | 53.665 | 30.465 | 17.820 |
| fallback4 | 40000 | 18.3658 | 0.4967 | 0.6355 | 55.893 | 30.480 | 17.361 |
| fallback4 | 40016 | 18.5545 | 0.4997 | 0.6332 | 54.993 | 29.277 | 17.002 |
| fallback4 | 40032 | 18.7209 | 0.5022 | 0.6313 | 50.633 | 29.213 | 16.533 |
| fallback4 | 40048 | 18.8641 | 0.5044 | 0.6296 | 50.209 | 29.013 | 16.131 |
| fallback4 | 40063 | 18.9848 | 0.5062 | 0.6282 | 49.554 | 29.195 | 15.603 |
| fallback8 | 39952 | 20.4910 | 0.5971 | 0.5070 | 6.651 | 14.817 | 2.976 |
| fallback8 | 39968 | 20.7237 | 0.6007 | 0.5028 | 6.663 | 14.372 | 1.091 |
| fallback8 | 39984 | 20.9321 | 0.6038 | 0.4994 | 6.640 | 14.119 | 1.382 |
| fallback8 | 40000 | 21.1140 | 0.6064 | 0.4966 | 6.211 | 13.963 | 0.641 |
| fallback8 | 40016 | 21.2744 | 0.6087 | 0.4941 | 6.611 | 13.825 | 1.159 |
| fallback8 | 40032 | 21.4140 | 0.6105 | 0.4921 | 6.580 | 13.637 | 1.106 |
| fallback8 | 40048 | 21.5360 | 0.6122 | 0.4903 | 6.567 | 13.383 | 0.618 |
| fallback8 | 40063 | 21.6420 | 0.6136 | 0.4887 | 6.490 | 13.228 | 0.605 |

### Insight

The existing `adaptive_fixed_fallback_samples_per_ray` mechanism is not a valid low-risk safety sampler for this purpose. It appends uniformly spaced fallback intervals across the full ray; with only `4` or `8` intervals, each fallback interval is very large and strongly changes volume rendering instead of adding a tiny corrective supervision signal. Do not use this flag as an artifact fix in its current form. If revisiting fallback coverage, implement a separate low-weight/point-like auxiliary sampling path rather than mixing these large fallback intervals into the rendered sample set.

## Seed42 render-only dense ARM traversal check

Tested whether the current seed42 Huber leader already contains useful field density that is being missed by the default eval traversal. This was render/eval only from the selected clean checkpoint `39936`; no training was run. The eval config changed only traversal/render settings from the current leader: `adaptive_coarse_step_size=0.003125`, `adaptive_max_step_size=0.003125`, `max_steps_per_ray=4096`, and `eval_num_rays_per_chunk=128`. ARM and occupancy-grid sampling stayed enabled.

| Run | Source checkpoint | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | Significant artifact | ROI micro | Stand connector | Decision | Renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `dense003125_cap4096` | 39936 | 29.5920 | 0.6875 | 0.4006 | 520.837 | 520.798 | 1.450 | 0.000 | 0.000 | Reject; denser render traversal improves PSNR/SSIM but hurts LPIPS and creates full-frame artifacts | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_renderonly_dense_eval/lookcloser/seed42_huber39936_dense003125_cap4096/renders_dense_step-000039936` |

Per-view artifact read:

| Preset | eval0 | eval1 | eval2 | Max |
|---|---:|---:|---:|---:|
| micro | 3.542 | 0.339 | 520.837 | 520.837 |
| significant | 1.450 | 0.000 | 0.000 | 1.450 |

### Insight

The current leader is not failing merely because eval traversal is too sparse. Denser render-only traversal raises PSNR/SSIM, but LPIPS becomes worse and full-frame artifact scores explode, especially under the micro detector on eval2. This supports the current interpretation that the remaining visual issue is field/checkpoint trajectory rather than a simple render traversal miss. Do not use dense render override as a selection or final rendering fix.

## ARM maxfreq12 soft-ceiling check

Tested the explorer recommendation that lowering the ARM frequency ceiling from `adaptive_max_frequency_level=13` to `12` might reduce high-frequency boundary instability while keeping the current safe ARM recipe. Both runs used ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `max_steps_per_ray=2048`, occupancy warmups `4096/4096`, `max_num_iterations=200000`, eval/save interval `8192`, and micro artifact-aware selection over all 3 eval views.

| Run | Seed | Loss | Last train/eval step | Selected step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | ROI micro | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `maxfreq12_huber_delta02_s42_200k_i8192` | 42 | Huber delta0.2 | 8192 eval / stopped at 10530 | none | 14.1575 | 0.3663 | 0.9281 | not scored | not scored | not scored | not scored | interrupted after first eval | n/a | Reject; first eval collapsed | none |
| `maxfreq12_mse_s43_200k_i8192` | 43 | MSE | 32768 | 8192 | 28.3773 | 0.6481 | 0.4556 | 1.050 | 0.935 | 1.720 | 0.251 | 2944.7 | 3254.1 | Reject; artifact-selected checkpoint is weak and all candidates are micro-dirty | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_maxfreq12_soft_micro/lookcloser/arm_h40_grid128_maxfreq12_mse_s43_200k_i8192/renders_artifact_selection_step-000008192` |

Seed43 candidate table:

| Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | ROI micro | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8192 | 28.3773 | 0.6481 | 0.4556 | 1.050 | 0.935 | 1.720 | 0.251 | least dirty, but weak |
| 16384 | 29.0926 | 0.6786 | 0.4228 | 1.980 | 1.797 | 0.000 | 0.000 | metrics improve, full-frame dirty |
| 24576 | 29.3072 | 0.6822 | 0.4133 | 1.502 | 1.340 | 0.852 | 0.000 | best eval loss, dirty |
| 32768 | 29.3700 | 0.6886 | 0.4104 | 1.724 | 1.615 | 0.434 | 0.000 | best metrics, dirty and still worse LPIPS than current clean seed43 |

### Insight

Lowering the adaptive ceiling to `12` is not a useful stabilizer. Seed42 Huber collapses at the first eval, and seed43 MSE never reaches a clean or competitive LPIPS checkpoint. This rejects the softer-ceiling hypothesis for the current recipe; the useful minfreq4/maxfreq13 setting remains the safer baseline.

## Seed43 fpl4 MSE capacity-only check

Tested whether the fpl4 capacity signal could be separated from the earlier Charbonnier/distortion bundle. This run used `hash_features_per_level=4` with seed43 MSE, ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, default `distortion_loss_mult=0.01`, occupancy warmups `4096/4096`, and explicit `max_num_iterations=200000`. The run was stopped manually after eval step `24576` because LPIPS remained weak; saved checkpoints were manually evaluated and micro-scored.

| Run | Seed | Hash features/level | Loss | Last eval step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | ROI micro | Stand connector | Decision | Renders |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `fpl4_mse_s43_200k_i8192` | 43 | 4 | MSE | 24576 | 29.5842 | 0.6808 | 0.4143 | 516.833 | 516.833 | 8.068 | 0.000 | Reject; capacity-only MSE is dirty and not LPIPS-competitive | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_capacity_fpl4_mse_micro/lookcloser/arm_h40_grid128_fpl4_mse_s43_200k_i8192/renders_artifact_selection_step-000024576` |

Candidate table:

| Step | PSNR | SSIM | LPIPS | Micro full-frame artifact | Micro serious artifact | ROI micro | Stand connector | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8192 | 28.5994 | 0.6423 | 0.4566 | 503.440 | 503.440 | 3.831 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_capacity_fpl4_mse_micro/lookcloser/arm_h40_grid128_fpl4_mse_s43_200k_i8192/renders_artifact_selection_step-000008192` |
| 16384 | 29.3589 | 0.6711 | 0.4235 | 514.521 | 514.485 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_capacity_fpl4_mse_micro/lookcloser/arm_h40_grid128_fpl4_mse_s43_200k_i8192/renders_artifact_selection_step-000016384` |
| 24576 | 29.5842 | 0.6808 | 0.4143 | 516.833 | 516.833 | 8.068 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_capacity_fpl4_mse_micro/lookcloser/arm_h40_grid128_fpl4_mse_s43_200k_i8192/renders_artifact_selection_step-000024576` |

### Insight

Capacity-only fpl4 with MSE is not the missing stabilizer. It avoids the fpl4+Huber catastrophic PSNR collapse, but it is far dirtier than the fpl4+Charbonnier branch under the micro detector and does not approach the current clean seed43 LPIPS. The earlier fpl4 metric signal remains tied to Charbonnier-like training, but that direction still needs a separate artifact stabilizer before it can replace the current leader.

## Seed42 Huber long-continuation / occupancy-freeze check

Tested the user hypothesis that the current seed42 Huber leader was simply under-trained. Both runs continued the current clean leader checkpoint `39936` with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Huber delta `0.2`, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, occupancy warmups `4096/4096`, and explicit `max_num_iterations=200000`. The ordinary continuation was stopped manually after eval step `65536` because all rendered late checkpoints were dirty. The second run repeated the same continuation with `occupancy_update_interval=999999` to freeze the occupancy grid loaded from the clean leader checkpoint.

Ordinary long-continuation selected the least-bad dirty checkpoint `45056`, not an acceptable replacement:

| Step | PSNR | SSIM | LPIPS | Micro artifact | Serious artifact | Renders |
|---:|---:|---:|---:|---:|---:|---|
| 40960 | 29.4427 | 0.6939 | 0.3967 | 0.971 | 0.870 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_long39936_200k_save4096/renders_artifact_selection_step-000040960` |
| 45056 | 29.0787 | 0.6963 | 0.3958 | 0.918 | 0.699 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_long39936_200k_save4096/renders_artifact_selection_step-000045056` |
| 49152 | 28.8914 | 0.6951 | 0.3943 | 1.990 | 1.879 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_long39936_200k_save4096/renders_artifact_selection_step-000049152` |
| 53248 | 28.8802 | 0.6871 | 0.3914 | 2.980 | 2.854 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_long39936_200k_save4096/renders_artifact_selection_step-000053248` |
| 57344 | 28.9199 | 0.6878 | 0.3914 | 3.990 | 3.927 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_long39936_200k_save4096/renders_artifact_selection_step-000057344` |
| 61440 | 28.9239 | 0.6848 | 0.3917 | 3.590 | 3.534 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_long39936_200k_save4096/renders_artifact_selection_step-000061440` |
| 65536 | 28.9921 | 0.6846 | 0.3891 | 3.051 | 2.821 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_long39936_200k_save4096/renders_artifact_selection_step-000065536` |

Manual selection summary:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_long39936_200k_save4096/manual_artifact_selection_summary.json`

The ordinary continuation's selected dirty step `45056` has ROI max `3.453`, with `left_stand_connector_eval0=2.394`, `left_stand_eval0=1.656`, and `floor_crack_eval0=3.453`. Its artifact-to-occupancy debug flipped the diagnosis from the earlier field-only failures: `grid_miss_likely=true`, `field_issue_likely=false`, with summary at:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_long39936_200k_save4096/artifact_occ_debug_eval0_step45056/artifact_occupancy_debug.md`

This means the late continuation is not just under-training. Eval loss continued improving through `65536`, and LPIPS improved to `0.3891`, but late occupancy state/traversal started missing relevant surface/ray voxels and visible artifacts became worse.

The occupancy-freeze continuation partially reduced the first late artifact but did not clean the run:

| Run | Occupancy updates | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Serious artifact | ROI max | Stand connector | Train wall (s) | Decision | Renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| ordinary long continuation | normal | 45056 | 29.0787 | 0.6963 | 0.3958 | 0.918 | 0.699 | 3.453 | 2.394 | ≈1782 | Reject; dirty despite improving eval loss | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_long39936_200k_save4096/renders_artifact_selection_step-000045056` |
| occupancy-freeze continuation | frozen after load | 40960 | 29.4489 | 0.6911 | 0.3960 | 0.823 | 0.670 | 1.624 | 1.037 | ≈600 | Reject; artifact reduced but not zero | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occfreeze_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occfreeze39936_200k_save4096/renders_artifact_selection_step-000040960` |

### Insight

The latest evidence splits the artifact problem into two regimes. Up to the accepted current leader, artifact-aware checkpoint selection can find clean ARM states. Past that point, longer Huber training improves eval loss and sometimes SSIM/LPIPS, but micro artifacts appear immediately and then grow. The new occupancy debugger result on the late dirty step says the late failure has an occupancy-grid miss component. Freezing the loaded clean occupancy grid helps only slightly, so the next low-hanging direction is not "train longer" by itself. It should test a targeted late-occupancy policy, such as lower late occupancy threshold, delayed/EMA-smoothed updates after the clean checkpoint, or a conservative refresh schedule, while keeping ARM and occupancy-grid sampling enabled and Feature Reweighting disabled.

## Seed42 late occupancy-threshold bracket

Tested the direct follow-up from the `grid_miss_likely=true` late-continuation debug: continue the current clean seed42 Huber leader step `39936`, keep ARM and occupancy-grid sampling enabled, keep FAS/Feature Reweighting disabled, and lower only `occupancy_occ_thre` during the late continuation. This is not the old rejected from-scratch low-threshold test; it is a targeted late-stage check after the clean checkpoint.

Shared settings: Huber delta `0.2`, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, occupancy warmups `4096/4096`, explicit `max_num_iterations=200000`, save interval `4096`, and manual micro artifact selection over all three eval views.

| Run | `occupancy_occ_thre` | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Serious artifact | ROI max | Serious ROIs | Stand connector | Decision | Renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `occ001_39936_200k_save4096` | 0.001 | 45056 | 29.1693 | 0.6964 | 0.3948 | 0.427 | 0.360 | 1.384 | 0/10 | 0.000 | Reject; much cleaner than ordinary late continuation but full-frame micro still nonzero | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ001_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ001_39936_200k_save4096/renders_artifact_selection_step-000045056` |
| `occ0001_39936_200k_save4096` | 0.0001 | 40960 | 29.4964 | 0.6958 | 0.3945 | 0.316 | 0.292 | 0.000 | 0/10 | 0.000 | Reject as final; best threshold bracket point, ROI-clean, but full-frame micro still nonzero | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_39936_200k_save4096/renders_artifact_selection_step-000040960` |
| `occ00001_39936_200k_save4096` | 0.00001 | 40960 | 29.4936 | 0.6958 | 0.3945 | 0.315 | 0.292 | not rescored; same visual/full-frame regime as 0.0001 | n/a | n/a | Reject; lowering below `1e-4` plateaus | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ00001_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ00001_39936_200k_save4096/renders_artifact_selection_step-000040960` |

Per-view full-frame micro scores for the best `0.0001` point:

| View | Artifact | Serious artifact |
|---|---:|---:|
| eval_img_0000.png | 0.261 | 0.162 |
| eval_img_0001.png | 0.316 | 0.292 |
| eval_img_0002.png | 0.107 | 0.000 |

Manual summaries:

- `0.001`: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ001_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ001_39936_200k_save4096/manual_artifact_selection_summary.json`
- `0.0001`: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_39936_200k_save4096/manual_artifact_selection_summary.json`
- `0.00001`: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ00001_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ00001_39936_200k_save4096/manual_artifact_selection_summary.json`

### Insight

Lowering late `occupancy_occ_thre` is a real improvement direction for this late seed42 branch, unlike the earlier global/from-scratch low-threshold result. It removes the severe stand/floor ROI failures and improves full-frame micro artifact from ordinary late `0.918` to `0.316` while preserving strong metrics. But it plateaus around `1e-4`: `1e-5` does not improve the residual eval1 full-frame component. The next low-hanging variant should not lower the threshold further; it should combine the `1e-4` late threshold with a more local stabilizer, such as denser checkpointing around `40960`, a conservative clamp multiplier/EMA schedule, or significant-vs-micro component inspection on eval1 to decide whether the remaining component is a true hole or detector-sensitive texture.

## Seed42 occ_thre=1e-4 dense clean-window scan

Ran the local follow-up from the threshold bracket: continue the current seed42 Huber leader step `39936` with `occupancy_occ_thre=1e-4`, ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, save interval `128`, train-time eval disabled, and stop at `41216`. The goal was to find a nearby checkpoint with lower full-frame micro artifact while preserving the improved SSIM/LPIPS signal.

| Step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 40064 | 29.6110 | 0.6841 | 0.3948 | 0.265 | 0.265 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/renders_artifact_selection_step-000040064` |
| 40192 | 29.6003 | 0.6877 | 0.3947 | 0.276 | 0.260 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/renders_artifact_selection_step-000040192` |
| 40320 | 29.5762 | 0.6900 | 0.3947 | 0.287 | 0.258 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/renders_artifact_selection_step-000040320` |
| 40448 | 29.5552 | 0.6920 | 0.3946 | 0.260 | 0.258 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/renders_artifact_selection_step-000040448` |
| 40576 | 29.5363 | 0.6934 | 0.3946 | 0.256 | 0.256 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/renders_artifact_selection_step-000040576` |
| 40704 | 29.5202 | 0.6945 | 0.3946 | 0.276 | 0.254 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/renders_artifact_selection_step-000040704` |
| 40832 | 29.5032 | 0.6952 | 0.3946 | 0.316 | 0.293 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/renders_artifact_selection_step-000040832` |
| 40960 | 29.4948 | 0.6958 | 0.3946 | 0.315 | 0.292 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/renders_artifact_selection_step-000040960` |
| 41088 | 29.4855 | 0.6962 | 0.3945 | 0.315 | 0.292 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/renders_artifact_selection_step-000041088` |
| 41215 | 29.4747 | 0.6965 | 0.3945 | 0.314 | 0.292 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/renders_artifact_selection_step-000041215` |

Selected checkpoint: step `40576`, train time `180.2s`, total time `1012.8s`.

Significant-preset audit for the selected checkpoint is fully clean: full-frame artifact `0.000`, ROI artifact `0.000`, stand connector `0.000`. Diagnostic micro is still nonzero (`0.256`), but substantially lower than the previous current leader's micro score (`0.691`) and with ROI/stand fully clean (`0.000` vs previous micro ROI/stand `0.255`).

### Insight

Dense checkpointing around the late low-threshold branch does not find micro artifact `0.000`, but it finds a better practical leader. Step `40576` improves every headline metric versus the previous seed42 Huber leader (`29.5082/0.6857/0.3964 -> 29.5363/0.6934/0.3946`), keeps the official significant artifact gate at `0.000`, and cuts diagnostic micro artifact by more than half. This becomes the current single leader, with the caveat that the remaining micro detector components still need visual/user inspection before claiming the final artifact problem is solved.

## Seed42 occ_thre=1e-4 plus dilation check

Tested whether a one-voxel occupancy dilation can remove the remaining off-ROI micro components in the new dense low-threshold window. The run repeated the `occ_thre=1e-4` dense scan but added `occupancy_dilation_radius=1`.

| Run | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `occ0001_dilate1_dense39936_41216_save128` | 40576 | 29.5332 | 0.6935 | 0.3946 | 0.256 | 0.256 | 0.000 | 0.000 | 210.2 | 1125.1 | Reject as replacement; ties micro but slightly worsens metrics and costs more | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dilate1_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dilate1_dense39936_41216_save128/renders_artifact_selection_step-000040576` |

### Insight

Late dilation is not the missing stabilizer. It selects the same nominal checkpoint and the same micro score as the non-dilated leader, but with slightly worse PSNR/LPIPS and longer runtime. Keep `occupancy_dilation_radius=0` for the current leader.

## Seed42 occ_thre=1e-4 plus EMA 0.99 check

Tested whether smoothing late occupancy updates with `occupancy_ema_decay=0.99` can remove the remaining off-ROI micro components. The run repeated the `occ_thre=1e-4` dense scan from clean step `39936` to `41216` with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, save interval `128`, train-time eval disabled, and explicit diagnostic windowing. Future long/diagnostic variants should use the repository default and explicit cap `max_num_iterations=200000`; this EMA run was intentionally short to isolate the known boundary window.

| Run | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `occ0001_ema099_dense39936_41216_save128` | 40576 | 29.5358 | 0.6934 | 0.3946 | 0.256 | 0.256 | 0.000 | 0.000 | 180.2 | 1035.3 | Reject as replacement; ties micro but slightly worsens PSNR versus the non-EMA leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_ema099_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_ema099_dense39936_41216_save128/renders_artifact_selection_step-000040576` |

Candidate artifact scores were effectively the same trajectory as the non-EMA dense scan: `40064=0.265`, `40192=0.276`, `40320=0.287`, `40448=0.289`, `40576=0.256`, `40704=0.290`, `40832=0.315`, `40960=0.315`, `41088=0.314`, and `41215=0.314`.

### Insight

Late EMA smoothing does not explain or fix the residual micro artifacts. It reproduces the same selected checkpoint and micro score as the current leader, while slightly lowering PSNR. Keep the current leader on normal EMA behavior and `occupancy_dilation_radius=0`. The next useful ARM-only direction should target the residual components more directly, for example with a conservative update cadence/clamp policy or another occupancy-debug pass on the selected residual eval view, rather than repeating global EMA or dilation.

## Seed42 occ_thre=1e-4 conservative update cadence check

Tested whether reducing late occupancy-grid update frequency can prevent the dirty late trajectory while preserving the LPIPS gains. Both runs continued the clean seed42 Huber checkpoint `39936` with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `occupancy_occ_thre=1e-4`, Huber delta `0.2`, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, minfreq4/maxfreq13, cap2048, artifact-aware micro selection over all 3 eval views, and explicit `max_num_iterations=200000`. The only tested change was `occupancy_update_interval=64` or `128` versus the current default `16`. Both runs stopped by eval-loss plateau at step `61440`.

| Run | `occupancy_update_interval` | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `occ0001_upd64_39936_200k_save4096` | 64 | 40960 | 29.4970 | 0.6957 | 0.3946 | 0.356 | 0.253 | 0.000 | 0.000 | 3726.9 | 4364.4 | Reject; worse micro than current leader and later checkpoints dirty | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_updatecadence_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_upd64_39936_200k_save4096/renders_artifact_selection_step-000040960` |
| `occ0001_upd128_39936_200k_save4096` | 128 | 40960 | 29.4966 | 0.6958 | 0.3946 | 0.315 | 0.292 | 0.000 | 0.000 | 3727.0 | 4372.5 | Reject; matches coarse `occ_thre=1e-4` behavior but does not beat dense leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_updatecadence_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_upd128_39936_200k_save4096/renders_artifact_selection_step-000040960` |

Candidate artifact trajectory:

| Step | `upd64` PSNR / SSIM / LPIPS | `upd64` micro / ROI | `upd128` PSNR / SSIM / LPIPS | `upd128` micro / ROI |
|---:|---:|---:|---:|---:|
| 40960 | 29.4970 / 0.6957 / 0.3946 | 0.356 / 0.000 | 29.4966 / 0.6958 / 0.3946 | 0.315 / 0.000 |
| 45056 | 29.1793 / 0.7003 / 0.3948 | 0.420 / 1.308 | 29.1798 / 0.7006 / 0.3947 | 0.375 / 1.289 |
| 49152 | 28.9913 / 0.6905 / 0.3928 | 1.782 / 1.293 | 28.9984 / 0.6934 / 0.3928 | 1.792 / 1.293 |
| 53248 | 29.0007 / 0.6826 / 0.3895 | 2.529 / 0.499 | 29.0099 / 0.6821 / 0.3895 | 2.583 / 0.500 |
| 57344 | 29.0452 / 0.6813 / 0.3876 | 2.637 / 0.530 | 29.0533 / 0.6818 / 0.3876 | 2.554 / 0.522 |
| 61440 | 29.0749 / 0.6815 / 0.3875 | 2.846 / 0.488 | 29.0741 / 0.6819 / 0.3876 | 2.346 / 0.489 |

### Insight

Conservative occupancy update cadence is not the missing stabilizer. The `128` run reproduces the coarse low-threshold selected checkpoint (`40960`, micro `0.315`) but does not beat the dense-window leader (`40576`, micro `0.256`), and the `64` run is worse. More importantly, the metric-attractive late checkpoints reach LPIPS `0.3875`-`0.3876` but become strongly dirty (`micro >2.3`) with nonzero ROI scores. This reinforces the current tradeoff: the field can improve LPIPS late, but the ARM/occupancy trajectory creates visible artifacts before those metric gains are usable.

## Current leader residual artifact debug

Debugged the remaining diagnostic micro components in the current leader, using the correct side-by-side render interpretation for artifact logs (`--panels 2 --gt 0 --cand 1`). The largest residual micro components are outside the curated ROI/stand regions:

| View | Component bbox | Micro score context | Occupancy debug result | Debug output |
|---|---|---|---|---|
| eval1 | `[1725,437,1753,448]` | eval1 max component, run max micro `0.256` | `grid_miss_likely=false`, `field_issue_likely=true` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/artifact_occ_debug_eval1_micro_bbox1725_437_1753_448/artifact_occupancy_debug.md` |
| eval0 | `[1767,155,1781,177]` | eval0 micro `0.229` | `grid_miss_likely=false`, `field_issue_likely=true` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/artifact_occ_debug_eval0_micro_bbox1767_155_1781_177/artifact_occupancy_debug.md` |

### Insight

The current leader's remaining micro artifacts are not explained by occupancy-grid misses. This differs from the later dirty long-continuation step `45056`, where the debugger did report `grid_miss_likely=true`. For the accepted `40576` leader, additional global occupancy conservativeness is unlikely to remove the residual micro score; next fixes should target field/checkpoint trajectory or ARM training integration.

Extended the same debugger to report ARM sample counts for artifact pixels and reran both residual bboxes. Neither component is explained by local `max_steps_per_ray` truncation:

| View | Bbox | Mean samples/ray | Max samples/ray | Cap | Saturated rays | Zero-sample rays | Surface occupied rate | Debug output |
|---|---|---:|---:|---:|---:|---:|---:|---|
| eval1 | `[1725,437,1753,448]` | 465.3 | 506 | 2048 | 0.0% | 0.0% | 100.0% | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/artifact_occ_debug_eval1_micro_bbox1725_437_1753_448_samplecounts/artifact_occupancy_debug.md` |
| eval0 | `[1767,155,1781,177]` | 439.6 | 517 | 2048 | 0.0% | 0.0% | 97.7% | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/artifact_occ_debug_eval0_micro_bbox1767_155_1781_177_samplecounts/artifact_occupancy_debug.md` |

### Insight

The current leader residuals are not occupancy misses, not local cap truncation, and not zero-sample rays. They are field/checkpoint trajectory artifacts in occupied, adequately sampled voxels. More global occupancy conservativeness, denser render traversal, or higher sample cap should not be the next lever for this residual.

## Seed42 occ_thre=1e-4 ultra-dense scan

Ran a save-every-16 local scan around the current leader's selected neighborhood to check whether the save-every-128 dense scan missed a cleaner checkpoint. The run continued from the same seed42 low-threshold branch with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `occupancy_occ_thre=1e-4`, Huber delta `0.2`, and diagnostic micro artifact selection over all three eval views.

| Run | Source step | Scan window | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Renders |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `occ0001_micro40448_40704_save16` | 40448 | 40448-40704 | 40576 | 29.5374 | 0.6934 | 0.3946 | 0.257 | 0.257 | 0.000 | 0.000 | 120.1 | 1454.1 | Reject as replacement; does not beat current leader micro `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_ultradense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_micro40448_40704_save16/renders_artifact_selection_step-000040576` |

Candidate micro scores around the selected region: `40464=0.260`, `40480=0.280`, `40496=0.279`, `40512=0.279`, `40528=0.280`, `40544=0.280`, `40560=0.281`, `40576=0.257`, `40592=0.281`, `40608=0.281`, `40624=0.281`, `40640=0.281`, `40656=0.278`, `40672=0.262`, `40688=0.277`, `40703=0.276`.

### Insight

The current save-every-128 leader is not missing a nearby zero-micro checkpoint. The artifact floor in this window is about `0.256`-`0.257`.

## Packed distortion-spacing normalization check

Fixed a code inconsistency in the packed ARM/occupancy paths: `frustums.starts`, `frustums.ends`, and `deltas` remain in raw ray `t`, but `spacing_starts` and `spacing_ends` are now normalized per ray as `(t - near) / (far - near)` for distortion-loss consistency with the fixed/fallback dense paths. This is a code-correctness change for training loss only; it does not change render geometry directly.

| Run | Code path | Source step | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Renders |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `occ0001_distnorm39936_41216_save128` | normalized packed spacing | 39936 | 41215 | 29.6039 | 0.6882 | 0.3944 | 0.259 | 0.257 | 0.000 | 0.000 | 180.2 | 1018.2 | Keep as code correctness fix, but reject as leader; micro is worse than `0.256` and SSIM drops | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_distnorm/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_distnorm39936_41216_save128/renders_artifact_selection_step-000041215` |
| `occ0001_distnorm41088_41472_save16` | normalized packed spacing, ultra-dense late scan | 41088 | 41264 | 29.6012 | 0.6883 | 0.3944 | 0.259 | 0.257 | 0.000 | 0.000 | 180.1 | 2139.8 | Reject as leader; stable artifact plateau at `0.259` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_distnorm_ultradense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_distnorm41088_41472_save16/renders_artifact_selection_step-000041264` |

The late ultra-dense normalized-spacing scan produced a long plateau near micro `0.259` from steps `41136` through `41424`, then worsened to about `0.281` at `41440+`.

### Insight

The spacing normalization is still worth keeping because the old packed paths were inconsistent with the normalized distortion-loss convention, and the patched branch made later checkpoints less dirty than some pre-patch late checkpoints. It is not an accepted recipe improvement yet: it does not beat the current `40576` leader and does not remove the residual micro artifacts.

## Seed42 normalized-spacing long continuation

Ran the same low-threshold seed42 continuation as the current leader, but with the packed distortion-spacing normalization fix active and a normal long cap (`max_num_iterations=200000`). The run loaded the clean source checkpoint `39936`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Huber delta `0.2`, `occupancy_occ_thre=1e-4`, save/eval interval `4096`, and micro artifact selection over all three eval views.

| Step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 40960 | 29.6114 | 0.6898 | 0.3945 | 0.258 | 0.258 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_distnorm_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_distnorm39936_200k_save4096/renders_artifact_selection_step-000040960` |
| 45056 | 29.5498 | 0.6901 | 0.3942 | 0.309 | 0.309 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_distnorm_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_distnorm39936_200k_save4096/renders_artifact_selection_step-000045056` |
| 49152 | 29.4664 | 0.6896 | 0.3901 | 0.390 | 0.369 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_distnorm_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_distnorm39936_200k_save4096/renders_artifact_selection_step-000049152` |
| 53248 | 29.4302 | 0.6881 | 0.3868 | 0.800 | 0.706 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_distnorm_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_distnorm39936_200k_save4096/renders_artifact_selection_step-000053248` |
| 57344 | 29.4261 | 0.6875 | 0.3862 | 1.521 | 1.386 | 0.486 | 0.270 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_distnorm_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_distnorm39936_200k_save4096/renders_artifact_selection_step-000057344` |
| 61440 | 29.4427 | 0.6873 | 0.3870 | 1.466 | 1.368 | 0.280 | 0.280 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_distnorm_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_distnorm39936_200k_save4096/renders_artifact_selection_step-000061440` |

Selected checkpoint: step `40960`, train time `1863.6s`, total time `2340.8s`. It is rejected as a replacement because micro artifact `0.258` is slightly worse than the current leader's `0.256`, despite better PSNR.

The late LPIPS-friendly best-eval checkpoint `57344` was debugged on the dominant eval0 artifact. Output:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_distnorm_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_distnorm39936_200k_save4096/artifact_occ_debug_eval0_step57344/artifact_occupancy_debug.md`

Key debugger result: `grid_miss_likely=true`, `field_issue_likely=false`. The artifact bbox was `[1875,1014,1916,1076]`; only `173/421` surface-depth pixels (`41.1%`) landed in occupied surface voxels even with `occupancy_occ_thre=1e-4`, while all rays had at least one occupied voxel somewhere along the path.

### Insight

The normalized-spacing long run confirms the core tradeoff: longer training can push seed42 LPIPS into the `0.386` range, close to the useful detail regime, but the clean artifact window is lost before those metrics become usable. Unlike the current leader's tiny residual components (`field_issue_likely=true`), the late LPIPS-friendly failure is again an occupancy-grid miss. This makes late-stage occupancy policy a valid lever only for the metric-attractive dirty window, not for the current leader's small residual micro floor.

## Seed42 targeted late occ_thre=1e-5 continuation

Because step `57344` showed a real late grid miss under `occupancy_occ_thre=1e-4`, tested a targeted lower-threshold continuation from the LPIPS-friendly step `49152`. This run kept the same ARM/FAS/FR-off settings, lowered only `occupancy_occ_thre` to `1e-5`, used `max_num_iterations=200000`, save/eval interval `1024`, and selected by micro artifact score.

| Step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 50176 | 29.4436 | 0.6896 | 0.3910 | 0.392 | 0.371 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ00001_distnorm_late/lookcloser/arm_h40_grid128_huber_delta02_s42_occ00001_distnorm49152_200k_save1024/renders_artifact_selection_step-000050176` |
| 51200 | 29.4413 | 0.6898 | 0.3906 | 0.417 | 0.373 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ00001_distnorm_late/lookcloser/arm_h40_grid128_huber_delta02_s42_occ00001_distnorm49152_200k_save1024/renders_artifact_selection_step-000051200` |
| 52224 | 29.4383 | 0.6889 | 0.3918 | 0.362 | 0.286 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ00001_distnorm_late/lookcloser/arm_h40_grid128_huber_delta02_s42_occ00001_distnorm49152_200k_save1024/renders_artifact_selection_step-000052224` |
| 53248 | 29.4283 | 0.6889 | 0.3906 | 0.770 | 0.694 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ00001_distnorm_late/lookcloser/arm_h40_grid128_huber_delta02_s42_occ00001_distnorm49152_200k_save1024/renders_artifact_selection_step-000053248` |

Selected checkpoint: step `52224`, train time `481.7s`, total time `799.4s`.

### Insight

Lowering the late threshold from `1e-4` to `1e-5` does not clean the LPIPS-friendly dirty branch. It slightly reduces the worst step `53248` micro score (`0.800 -> 0.770`) and improves the selected micro score versus raw step `49152` (`0.390 -> 0.362`), but it remains worse than the current leader and does not keep the `0.386` LPIPS point. Do not lower the threshold further as the next standalone knob; the next targeted grid-miss test should be a different late occupancy policy, such as dilation or a refresh/freeze schedule, and it should be validated against the `57344` debug failure.

## Seed42 targeted late dilation continuation

Tested the next targeted grid-miss policy from the same LPIPS-friendly step `49152`: keep `occupancy_occ_thre=1e-4`, add `occupancy_dilation_radius=1`, keep ARM/FAS/FR-off settings unchanged, use `max_num_iterations=200000`, save/eval interval `1024`, and select by micro artifact score.

| Step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 50176 | 29.4382 | 0.6896 | 0.3910 | 0.392 | 0.371 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dilate_late/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dilate1_distnorm49152_200k_save1024/renders_artifact_selection_step-000050176` |
| 51200 | 29.4392 | 0.6898 | 0.3907 | 0.740 | 0.665 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dilate_late/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dilate1_distnorm49152_200k_save1024/renders_artifact_selection_step-000051200` |
| 52224 | 29.4392 | 0.6890 | 0.3919 | 0.778 | 0.703 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dilate_late/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dilate1_distnorm49152_200k_save1024/renders_artifact_selection_step-000052224` |
| 53248 | 29.4344 | 0.6890 | 0.3906 | 0.806 | 0.730 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dilate_late/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dilate1_distnorm49152_200k_save1024/renders_artifact_selection_step-000053248` |

Selected checkpoint: step `50176`, train time `541.7s`, total time `872.1s`.

### Insight

Late dilation is rejected. It does not preserve the `0.386` LPIPS point and it makes the branch dirtier than the `1e-5` threshold continuation. This suggests the late grid-miss failure is not a simple one-voxel binary-neighbor hole. The remaining promising direction is a policy that changes late occupancy update dynamics over time, not just the binary threshold shape at a loaded dirty branch. A reasonable next test is to start before the dirty LPIPS transition and switch occupancy policy only after a stable clean checkpoint, e.g. refresh/freeze or reset occupancy statistics around `40960-49152`, still with ARM enabled and Feature Reweighting disabled.

## Seed42 temporary full-occupancy late continuation

Tested the direct diagnostic for the late grid-miss regime: continue from the LPIPS-friendly dirty step `49152`, keep ARM enabled and Feature Reweighting/FAS disabled, but set `occupancy_binary_warmup_steps=60000` so saved/eval checkpoints before `60000` use a fully occupied binary grid. This keeps the occupancy-grid path active but removes binary pruning in the late window.

| Step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 50176 | 29.4380 | 0.6949 | 0.3911 | 0.390 | 0.328 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_fullocc_late/lookcloser/arm_h40_grid128_huber_delta02_s42_fullocc60000_distnorm49152_200k_save1024/renders_artifact_selection_step-000050176` |
| 51200 | 29.4394 | 0.6946 | 0.3907 | 0.416 | 0.372 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_fullocc_late/lookcloser/arm_h40_grid128_huber_delta02_s42_fullocc60000_distnorm49152_200k_save1024/renders_artifact_selection_step-000051200` |
| 52224 | 29.4387 | 0.6930 | 0.3919 | 0.798 | 0.703 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_fullocc_late/lookcloser/arm_h40_grid128_huber_delta02_s42_fullocc60000_distnorm49152_200k_save1024/renders_artifact_selection_step-000052224` |
| 53248 | 29.4351 | 0.6926 | 0.3905 | 0.790 | 0.694 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_fullocc_late/lookcloser/arm_h40_grid128_huber_delta02_s42_fullocc60000_distnorm49152_200k_save1024/renders_artifact_selection_step-000053248` |

Selected checkpoint: step `50176`, train time `751.8s`, total time `1144.2s`.

### Insight

Temporary fully occupied binaries do not clean the late branch. They improve SSIM but leave the first saved candidate at micro `0.390`, then later candidates jump to about `0.79`. This means the late dirty LPIPS branch is not fixed by simply removing occupancy pruning during render/training; at least part of the failure is already baked into the field/trajectory by step `49152`, while later grid misses make it worse.

## Seed42 frozen-occupancy continuation from step 40960

Tested whether freezing the earlier low-threshold occupancy state can prevent the later dirty transition. This run loaded step `40960` from the normalized-spacing long branch, kept ARM enabled and Feature Reweighting/FAS disabled, used `occupancy_occ_thre=1e-4`, and set `occupancy_update_interval=999999` so the loaded occupancy grid was not updated during the continuation.

| Step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 41984 | 29.5943 | 0.6926 | 0.3940 | 0.259 | 0.259 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occfreeze_distnorm/lookcloser/arm_h40_grid128_huber_delta02_s42_occfreeze40960_200k_save1024/renders_artifact_selection_step-000041984` |
| 43008 | 29.5815 | 0.6938 | 0.3937 | 0.298 | 0.298 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occfreeze_distnorm/lookcloser/arm_h40_grid128_huber_delta02_s42_occfreeze40960_200k_save1024/renders_artifact_selection_step-000043008` |
| 44032 | 29.5669 | 0.6945 | 0.3940 | 0.306 | 0.306 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occfreeze_distnorm/lookcloser/arm_h40_grid128_huber_delta02_s42_occfreeze40960_200k_save1024/renders_artifact_selection_step-000044032` |
| 45056 | 29.5453 | 0.6948 | 0.3941 | 0.352 | 0.352 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occfreeze_distnorm/lookcloser/arm_h40_grid128_huber_delta02_s42_occfreeze40960_200k_save1024/renders_artifact_selection_step-000045056` |

Selected checkpoint: step `41984`, train time `541.7s`, total time `872.8s`.

### Insight

Freezing the earlier occupancy grid is also rejected. It keeps ROI/stand clean and gives good SSIM, but the best micro score is `0.259`, just worse than the current leader's `0.256`, and later checkpoints become steadily dirtier. The update-dynamics checks now reject static lower threshold, dilation, fully occupied binaries, and freeze. The remaining issue looks like a coupled field/trajectory problem: the clean current leader is near the boundary, and once the model moves toward lower LPIPS, artifacts appear before simple occupancy policies can recover them.

## Seed42 frequency-grid boundary index + non-stratified ARM training check

Tested two temporary code-level instability hypotheses from the ARM audit: fix frequency-grid boundary indexing so max-side AABB positions can query voxel index `resolution-1`, and disable stratified coarse nerfacc traversal during ARM training with `--disable-adaptive-stratified-training`. The run loaded the clean seed42 Huber checkpoint `39936`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `occupancy_occ_thre=1e-4`, Huber delta `0.2`, `grid_resolution=128`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, train-time eval disabled, save interval `128`, and micro artifact-aware selection over all 3 eval views.

| Step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 40064 | 29.5795 | 0.6818 | 0.3951 | 0.267 | 0.267 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_nostrat_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx_nostrat39936_41216_save128/renders_artifact_selection_step-000040064` |
| 40192 | 29.6133 | 0.6837 | 0.3950 | 0.266 | 0.266 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_nostrat_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx_nostrat39936_41216_save128/renders_artifact_selection_step-000040192` |
| 40320 | 29.6181 | 0.6851 | 0.3948 | 0.262 | 0.262 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_nostrat_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx_nostrat39936_41216_save128/renders_artifact_selection_step-000040320` |
| 40448 | 29.6172 | 0.6862 | 0.3947 | 0.262 | 0.262 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_nostrat_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx_nostrat39936_41216_save128/renders_artifact_selection_step-000040448` |
| 40576 | 29.6154 | 0.6872 | 0.3946 | 0.282 | 0.261 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_nostrat_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx_nostrat39936_41216_save128/renders_artifact_selection_step-000040576` |
| 40704 | 29.6092 | 0.6884 | 0.3946 | 0.292 | 0.258 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_nostrat_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx_nostrat39936_41216_save128/renders_artifact_selection_step-000040704` |
| 40832 | 29.6074 | 0.6893 | 0.3946 | 0.346 | 0.258 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_nostrat_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx_nostrat39936_41216_save128/renders_artifact_selection_step-000040832` |
| 40960 | 29.6074 | 0.6900 | 0.3946 | 0.314 | 0.259 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_nostrat_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx_nostrat39936_41216_save128/renders_artifact_selection_step-000040960` |
| 41088 | 29.6034 | 0.6905 | 0.3945 | 0.260 | 0.256 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_nostrat_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx_nostrat39936_41216_save128/renders_artifact_selection_step-000041088` |
| 41215 | 29.6027 | 0.6909 | 0.3945 | 0.259 | 0.257 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_nostrat_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx_nostrat39936_41216_save128/renders_artifact_selection_step-000041215` |

Selected checkpoint: step `41215`, train time `180.2s`, total time `1017.7s`.

### Insight

This combined code-level test is rejected as a leader. It improves PSNR/SSIM versus the current leader but does not reduce diagnostic micro artifact below the current `0.256` floor; the selected full-frame micro is `0.259`. Deterministic ARM training is not a standalone artifact fix. Because the change did not improve artifacts, the temporary no-stratified flag was removed after the test.

## Seed42 frequency-grid boundary index-only check

Isolated the boundary-index change by rerunning the same seed42 continuation with normal stratified ARM training. The temporary patch allowed frequency-grid queries to reach max-side voxel index `resolution-1`, while training otherwise matched the prior recipe. Other settings stayed the same as the current low-threshold leader: load clean step `39936`, ARM enabled, occupancy-grid sampling enabled, FAS/Feature Reweighting disabled, `occupancy_occ_thre=1e-4`, Huber delta `0.2`, grid128, coarse00625, minfreq4/maxfreq13, cap2048, train-time eval disabled, save interval `128`, and micro artifact selection over all 3 eval views.

| Step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 40064 | 29.5789 | 0.6818 | 0.3951 | 0.267 | 0.267 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx39936_41216_save128/renders_artifact_selection_step-000040064` |
| 40192 | 29.6133 | 0.6837 | 0.3950 | 0.266 | 0.266 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx39936_41216_save128/renders_artifact_selection_step-000040192` |
| 40320 | 29.6185 | 0.6851 | 0.3947 | 0.262 | 0.262 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx39936_41216_save128/renders_artifact_selection_step-000040320` |
| 40448 | 29.6174 | 0.6862 | 0.3946 | 0.261 | 0.261 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx39936_41216_save128/renders_artifact_selection_step-000040448` |
| 40576 | 29.6152 | 0.6871 | 0.3946 | 0.282 | 0.261 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx39936_41216_save128/renders_artifact_selection_step-000040576` |
| 40704 | 29.6116 | 0.6880 | 0.3945 | 0.292 | 0.258 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx39936_41216_save128/renders_artifact_selection_step-000040704` |
| 40832 | 29.6071 | 0.6890 | 0.3946 | 0.363 | 0.258 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx39936_41216_save128/renders_artifact_selection_step-000040832` |
| 40960 | 29.6093 | 0.6898 | 0.3945 | 0.296 | 0.258 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx39936_41216_save128/renders_artifact_selection_step-000040960` |
| 41088 | 29.6054 | 0.6903 | 0.3945 | 0.278 | 0.256 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx39936_41216_save128/renders_artifact_selection_step-000041088` |
| 41215 | 29.6039 | 0.6907 | 0.3944 | 0.278 | 0.257 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_grididx_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_grididx39936_41216_save128/renders_artifact_selection_step-000041215` |

Selected checkpoint: step `40448`, train time `180.2s`, total time `1021.5s`.

### Insight

Boundary-index fix alone is rejected as a recipe improvement for this seed42 window. It gives higher PSNR than the current leader but the best full-frame micro artifact score is `0.261`, worse than `0.256`, and the current leader's step `40576` becomes dirtier (`0.282`). The no-stratified flag was not the cause of the combined run's rejection; the boundary-index change itself shifts the continuation trajectory. The temporary boundary-index patch was reverted after this negative result so future default runs remain comparable to the accepted leader. Keep this as an audit finding, but do not replace the current leader or use this branch as evidence of artifact progress.

## Seed42 field-parameter checkpoint averaging check

Tested a cheap field/checkpoint-trajectory stabilizer after occupancy/cap diagnostics pointed away from traversal. Built two eval-only averaged checkpoints from the existing save-every-16 neighborhood around the current leader. Only floating field parameter tensors were averaged (`encoding`, `mlp_geo`, `direction_encoding`, `mlp_color`); occupancy grid, frequency grid, and other buffers were kept from central step `40576`. No training was run.

| Variant | Averaged steps | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Decision | Renders |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `swa_field_avg3_40464_40576_40672` | 40464, 40576, 40672 | 29.5395 | 0.6933 | 0.3946 | 0.257 | 0.257 | 0.000 | 0.000 | Reject; tiny metric tie but artifact worse than `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_swa_eval/lookcloser/swa_field_avg3_40464_40576_40672/renders_swa_step-000040576` |
| `swa_field_avg5_40544_40560_40576_40592_40608` | 40544, 40560, 40576, 40592, 40608 | 29.5375 | 0.6934 | 0.3946 | 0.257 | 0.257 | 0.000 | 0.000 | Reject; artifact worse than `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_swa_eval/lookcloser/swa_field_avg5_40544_40560_40576_40592_40608/renders_swa_step-000040576` |

### Insight

Local field-parameter averaging almost preserves current-leader quality, but does not remove the residual micro components. This weakens the hypothesis that the `0.256` floor is just high-frequency checkpoint noise that can be averaged out locally. Current leader remains unchanged.

## Seed42 low-LR continuation from current leader

Tested whether the current `occupancy_occ_thre=1e-4` leader can be improved by continuing more gently instead of using the original optimizer trajectory. Loaded current leader step `40576`, reset the scheduler with `--no-load-scheduler`, set fields LR `1e-4 -> 1e-5`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Huber delta `0.2`, grid128, coarse00625, minfreq4/maxfreq13, cap2048, train-time eval disabled, save interval `1024`, and micro artifact selection over all 3 eval views. No occupancy, cap, or traversal changes were made.

| Step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 40960 | 29.5692 | 0.6930 | 0.3944 | 0.294 | 0.294 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000040960` |
| 41984 | 29.5817 | 0.6894 | 0.3941 | 0.297 | 0.297 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000041984` |
| 43008 | 29.5825 | 0.6897 | 0.3941 | 0.298 | 0.298 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000043008` |
| 44032 | 29.5850 | 0.6900 | 0.3943 | 0.306 | 0.306 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000044032` |
| 45056 | 29.5619 | 0.6901 | 0.3944 | 0.353 | 0.353 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000045056` |
| 46080 | 29.5387 | 0.6897 | 0.3941 | 0.330 | 0.310 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000046080` |
| 47104 | 29.5107 | 0.6893 | 0.3938 | 0.339 | 0.318 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000047104` |
| 48128 | 29.4974 | 0.6895 | 0.3930 | 0.337 | 0.316 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000048128` |
| 49151 | 29.4746 | 0.6896 | 0.3919 | 0.341 | 0.320 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000049151` |

Selected checkpoint: step `40960`, train time `720.8s`, total time `1440.6s`.

### Insight

Low LR from the current leader is rejected as a standalone fix. It improves the late dirty branch relative to ordinary continuation (`45056` micro `0.353` here versus about `0.918` before), and LPIPS slowly improves to `0.3919` by step `49151`, but every checkpoint is dirtier than the current leader's `0.256` micro floor. The result is useful because it shows trajectory speed matters, but simply lowering LR does not produce artifact-clean detail recovery.

## Seed42 Charbonnier low-LR continuation from current leader

Tested a loss-side continuation from the current seed42 leader step `40576`: switched to Charbonnier, reset the scheduler with `--no-load-scheduler`, used fields LR `1e-4 -> 1e-5`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `occupancy_occ_thre=1e-4`, grid128, coarse00625, minfreq4/maxfreq13, cap2048, train-time eval disabled, save interval `1024`, and micro artifact selection over all 3 eval views.

| Step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 40960 | 29.5573 | 0.6825 | 0.3946 | 0.271 | 0.271 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_charb_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_charb_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000040960` |
| 41984 | 29.5753 | 0.6823 | 0.3953 | 0.269 | 0.269 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_charb_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_charb_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000041984` |
| 43008 | 29.5799 | 0.6824 | 0.3954 | 0.273 | 0.273 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_charb_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_charb_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000043008` |
| 44032 | 29.5872 | 0.6821 | 0.3955 | 0.276 | 0.276 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_charb_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_charb_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000044032` |
| 45056 | 29.5771 | 0.6820 | 0.3946 | 0.281 | 0.281 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_charb_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_charb_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000045056` |
| 46080 | 29.5936 | 0.6820 | 0.3928 | 0.310 | 0.276 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_charb_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_charb_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000046080` |
| 47104 | 29.6039 | 0.6821 | 0.3913 | 0.316 | 0.316 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_charb_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_charb_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000047104` |
| 48128 | 29.5986 | 0.6823 | 0.3901 | 0.315 | 0.315 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_charb_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_charb_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000048128` |
| 49151 | 29.5983 | 0.6823 | 0.3889 | 0.315 | 0.315 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_charb_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_charb_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000049151` |

Selected checkpoint: step `41984`, train time `810.9s`, total time `1564.7s`.

### Insight

Charbonnier low-LR is rejected as a replacement. It eventually improves LPIPS to `0.3889`, but every checkpoint is dirtier than the current leader (`0.269` best micro versus `0.256`) and SSIM drops sharply to about `0.682`. This again shows the detail direction is reachable, but this loss switch does not stabilize the clean boundary.

## Seed42 per-ray full-occupancy ARM training mix

Tested a temporary training-only per-ray full-occupancy bypass. Each training batch was split by ray: the standard subset used normal ARM + occupancy-grid traversal, and the bypass subset kept ARM frequency subdivision but temporarily treated occupancy binaries as fully occupied. Eval/render still used the normal occupancy-grid path, so artifact scores measured the deployable renderer. This replaced the older rejected whole-batch bypass test, which was much more memory-spiky.

All runs loaded clean source checkpoint step `39936`, kept ARM enabled, occupancy-grid sampling enabled, FAS/Feature Reweighting disabled, Huber delta `0.2`, `occupancy_occ_thre=1e-4`, grid128, coarse00625, minfreq4/maxfreq13, cap2048, `max_num_iterations=200000`, save/eval batch interval `4096`, and micro artifact selection over all 3 eval views.

| Bypass fraction | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.10 | 40960 | 29.6073 | 0.6900 | 0.3946 | 0.296 | 0.258 | 0.000 | 0.000 | 6071.5 | 6800.6 | Reject; worse than current leader micro `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass010v2_39936_200k_save4096/renders_artifact_selection_step-000040960` |
| 0.30 | 40960 | 29.6090 | 0.6893 | 0.3945 | 0.296 | 0.258 | 0.000 | 0.000 | 6131.4 | 6873.0 | Reject; same artifact floor as 0.10 and worse than leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass030v2_39936_200k_save4096/renders_artifact_selection_step-000040960` |
| 0.40 | 40960 | 29.6089 | 0.6900 | 0.3946 | 0.296 | 0.258 | 0.000 | 0.000 | 6131.3 | 6770.6 | Reject; no artifact improvement | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass040v2_39936_200k_save4096/renders_artifact_selection_step-000040960` |

Candidate details:

| Bypass fraction | Step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.10 | 40960 | 29.6073 | 0.6900 | 0.3946 | 0.296 | 0.258 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass010v2_39936_200k_save4096/renders_artifact_selection_step-000040960` |
| 0.10 | 45056 | 29.5516 | 0.6901 | 0.3943 | 0.353 | 0.353 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass010v2_39936_200k_save4096/renders_artifact_selection_step-000045056` |
| 0.10 | 49152 | 29.4509 | 0.6896 | 0.3902 | 0.524 | 0.479 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass010v2_39936_200k_save4096/renders_artifact_selection_step-000049152` |
| 0.10 | 53248 | 29.4168 | 0.6885 | 0.3869 | 0.859 | 0.767 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass010v2_39936_200k_save4096/renders_artifact_selection_step-000053248` |
| 0.10 | 57344 | 29.4393 | 0.6877 | 0.3858 | 1.504 | 1.428 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass010v2_39936_200k_save4096/renders_artifact_selection_step-000057344` |
| 0.10 | 61440 | 29.4413 | 0.6872 | 0.3869 | 1.592 | 1.466 | 3.982 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass010v2_39936_200k_save4096/renders_artifact_selection_step-000061440` |
| 0.30 | 40960 | 29.6090 | 0.6893 | 0.3945 | 0.296 | 0.258 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass030v2_39936_200k_save4096/renders_artifact_selection_step-000040960` |
| 0.30 | 45056 | 29.5495 | 0.6902 | 0.3943 | 0.352 | 0.352 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass030v2_39936_200k_save4096/renders_artifact_selection_step-000045056` |
| 0.30 | 49152 | 29.4504 | 0.6898 | 0.3901 | 0.389 | 0.368 | 0.260 | 0.260 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass030v2_39936_200k_save4096/renders_artifact_selection_step-000049152` |
| 0.30 | 53248 | 29.4125 | 0.6884 | 0.3871 | 0.806 | 0.713 | 1.130 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass030v2_39936_200k_save4096/renders_artifact_selection_step-000053248` |
| 0.30 | 57344 | 29.4139 | 0.6874 | 0.3863 | 1.739 | 1.630 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass030v2_39936_200k_save4096/renders_artifact_selection_step-000057344` |
| 0.30 | 61440 | 29.4445 | 0.6872 | 0.3871 | 1.544 | 1.419 | 3.987 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass030v2_39936_200k_save4096/renders_artifact_selection_step-000061440` |
| 0.40 | 40960 | 29.6089 | 0.6900 | 0.3946 | 0.296 | 0.258 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass040v2_39936_200k_save4096/renders_artifact_selection_step-000040960` |
| 0.40 | 45056 | 29.5446 | 0.6903 | 0.3943 | 0.373 | 0.353 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass040v2_39936_200k_save4096/renders_artifact_selection_step-000045056` |
| 0.40 | 49152 | 29.4497 | 0.6898 | 0.3902 | 0.398 | 0.366 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass040v2_39936_200k_save4096/renders_artifact_selection_step-000049152` |
| 0.40 | 53248 | 29.4144 | 0.6884 | 0.3871 | 0.792 | 0.699 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass040v2_39936_200k_save4096/renders_artifact_selection_step-000053248` |
| 0.40 | 57344 | 29.4318 | 0.6876 | 0.3862 | 1.485 | 1.376 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass040v2_39936_200k_save4096/renders_artifact_selection_step-000057344` |
| 0.40 | 61440 | 29.4393 | 0.6877 | 0.3869 | 1.485 | 1.375 | 0.000 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass040v2_39936_200k_save4096/renders_artifact_selection_step-000061440` |

### Insight

Per-ray full-occupancy ARM training mix is rejected. It does not reduce the clean-window residual micro artifact score: all three ratios select step `40960` with micro `0.296`, worse than the current leader's `0.256`. The late detail branch still appears (`LPIPS ~0.386` at `57344`), but artifacts grow strongly (`1.485`-`1.739`), so sparse fully occupied training coverage does not fix the field/trajectory boundary. The temporary `occupancy_training_full_bypass_fraction` code and CLI flag were removed after this negative result.

## Appearance embedding ARM-only screen

Tested optional appearance embeddings as a field/data-fit capacity lever, not Feature Reweighting. Both runs were from scratch because appearance embedding changes the field checkpoint shape. Common settings: ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Huber delta `0.2`, `grid_resolution=128`, `occupancy_occ_thre=1e-4`, occupancy warmups `4096/4096`, coarse00625, minfreq4/maxfreq13, cap2048, `train_num_rays_per_batch=4096`, `max_num_iterations=200000`, eval/save interval `8192`, early stop on eval-loss no-improve, and micro artifact selection over all 3 eval views.

| Appearance dim | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 8 | 8192 | 12.8835 | 0.3613 | 0.9446 | 7.772 | 7.717 | 321.948 | 0.000 | 2915.3 | 3152.0 | Reject; severe quality collapse and dirty artifact score | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_appearance_seed42_micro/lookcloser/arm_h40_grid128_huber_s42_app8_200k_i8192/renders_artifact_selection_step-000008192` |
| 16 | 16384 | 14.2921 | 0.3492 | 0.8992 | 9.112 | 8.963 | 716.594 | 0.000 | 2915.3 | 3154.8 | Reject; later checkpoint slightly lowers artifact but remains unusable | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_appearance_seed42_micro/lookcloser/arm_h40_grid128_huber_s42_app16_200k_i8192/renders_artifact_selection_step-000016384` |

### Insight

Appearance embedding is rejected for the current ARM-only path. Both dimensions stopped at step `16384` because eval loss worsened after the first boundary, and selected renders are far below the current leader by every metric and artifact gate. The temporary eval-time average-appearance behavior tested for this screen was removed after the negative result, so the default `appearance_embedding_dim=0` behavior remains unchanged.

## Seed43 MSE low-threshold occupancy transfer

Tested the direct transfer of the current seed42 late low-threshold occupancy trick to seed43. The run loaded the accepted clean seed43 MSE checkpoint `23295`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, occupancy warmups `4096/4096`, coarse00625, minfreq4/maxfreq13, cap2048, and changed/kept the late occupancy threshold at `occupancy_occ_thre=1e-4`. Checkpoints were saved every `16` steps to `23424`, train-time eval was disabled, and selection used micro artifact scoring over all 3 eval views.

| Run | Source step | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_mse_s43_occ0001_from23295_23424_save16` | 23295 | 23423 | 29.4255 | 0.6742 | 0.3898 | 0.402 | 0.243 | 0.000 | 0.000 | 90.1 | 872.1 | Reject; low-threshold transfer improves PSNR slightly but fails micro artifact gate | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_seed43_dense/lookcloser/arm_h40_grid128_mse_s43_occ0001_from23295_23424_save16/renders_artifact_selection_step-000023423` |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Micro artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 23296 | 29.3830 | 0.6760 | 0.3899 | 0.403 | 0.000 | 0.000 | Dirty immediately after source |
| 23312 | 29.4166 | 0.6755 | 0.3899 | 0.404 | 0.000 | 0.000 | Dirty |
| 23328 | 29.4090 | 0.6755 | 0.3898 | 0.404 | 0.000 | 0.000 | Dirty despite small LPIPS improvement |
| 23344 | 29.4087 | 0.6747 | 0.3898 | 0.406 | 0.000 | 0.000 | Dirty |
| 23360 | 29.4088 | 0.6743 | 0.3898 | 0.404 | 0.000 | 0.000 | Dirty |
| 23376 | 29.4116 | 0.6744 | 0.3899 | 0.404 | 0.000 | 0.000 | Dirty |
| 23392 | 29.4189 | 0.6742 | 0.3898 | 0.403 | 0.000 | 0.000 | Dirty |
| 23408 | 29.4249 | 0.6742 | 0.3899 | 0.402 | 0.000 | 0.000 | Dirty |
| 23423 | 29.4255 | 0.6742 | 0.3898 | 0.402 | 0.000 | 0.000 | Least dirty, still rejected |

### Insight

The seed42 low-threshold occupancy trick does not transfer to seed43 as a standalone fix. It preserves ROI and stand cleanliness and slightly improves PSNR, but every post-source checkpoint is full-frame micro dirty at about `0.402`-`0.406`. This strengthens the read that seed43's post-boundary failure is a field/trajectory artifact rather than a simple occupancy-threshold miss. The accepted seed43 clean point remains `mse_s43_micro23232_23296_save16` step `23295`.

## Seed42 current-leader Charbonnier normal-scheduler scan

Tested the second direct loss-transfer candidate from the current leader. The run loaded current leader step `40576`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `occupancy_occ_thre=1e-4`, grid128, coarse00625, minfreq4/maxfreq13, cap2048, and switched only the reconstruction loss to Charbonnier while keeping the loaded scheduler. Checkpoints were saved every `128` steps to `40960`, train-time eval was disabled, and selection used micro artifact scoring over all 3 eval views.

| Run | Source step | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_charb_s42_from40576_40960_save128` | 40576 | 40832 | 29.5762 | 0.6823 | 0.3945 | 0.271 | 0.271 | 0.000 | 0.000 | 90.1 | 337.6 | Reject; no LPIPS gain and dirtier than current leader micro `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_charb_boundary_transfer/lookcloser/arm_h40_grid128_charb_s42_from40576_40960_save128/renders_artifact_selection_step-000040832` |

Candidate timeline:

| Step | PSNR | SSIM | LPIPS | Micro artifact | ROI artifact | Stand connector | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 40704 | 29.6168 | 0.6824 | 0.3948 | 0.271 | 0.000 | 0.000 | Dirty and worse LPIPS than leader |
| 40832 | 29.5762 | 0.6823 | 0.3945 | 0.271 | 0.000 | 0.000 | Selected, still dirtier than leader |
| 40959 | 29.5577 | 0.6825 | 0.3946 | 0.271 | 0.000 | 0.000 | Dirty |

### Insight

Charbonnier from the current leader with the normal scheduler is rejected. It does not reproduce the later low-LR Charbonnier LPIPS direction and immediately raises the micro artifact score from the current leader's `0.256` to `0.271`. The current seed42 leader remains unchanged.

## Seed44 loss-transfer from clean LPIPS boundary

Tested whether the clean seed44 Charbonnier detail point can be pushed upward in PSNR/SSIM without losing its old-H40-level LPIPS. Both runs loaded the clean seed44 step `14861`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `occupancy_occ_thre=1e-4`, occupancy warmups `4096/4096`, coarse00625, minfreq4/maxfreq13, cap2048, and used micro artifact-aware selection over all 3 eval views. These were deliberately short boundary-window scans from `14861` to `14912` with save interval `4`; future full non-smoke runs use the `max_num_iterations=200000` cap.

| Run | Loss | Source step | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_huber_s44_from14861_14912_save4` | Huber delta0.2 | 14861 | 14896 | 28.6150 | 0.6486 | 0.3648 | 0.504 | 0.403 | 5.540 | 0.568 | 120.1 | 1791.0 | Reject; immediately dirty and weak PSNR/SSIM | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_seed44_loss_transfer/lookcloser/arm_h40_grid128_huber_s44_from14861_14912_save4/renders_artifact_selection_step-000014896` |
| `arm_h40_grid128_mse_s44_from14861_14912_save4` | MSE | 14861 | 14896 | 28.9104 | 0.6522 | 0.3639 | 0.523 | 0.451 | 5.518 | 0.585 | 120.1 | 1779.9 | Reject; better PSNR than Huber but still dirty with severe ROI score | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_seed44_loss_transfer/lookcloser/arm_h40_grid128_mse_s44_from14861_14912_save4/renders_artifact_selection_step-000014896` |

### Insight

Seed44's clean LPIPS boundary is not stable under immediate Huber or MSE continuation. MSE improves LPIPS slightly to `0.3639` and PSNR to `28.9104`, but both runs become full-frame and ROI dirty right after the source checkpoint. This rejects simple loss transfer from the clean seed44 detail point. Keep seed44 step `14861` only as a clean LPIPS/detail comparison, not as the active leader.

## Seed42 Huber-delta bracket on current low-threshold leader branch

Tested whether reducing Huber delta below the current leader's `0.2` improves the ARM field trajectory or lowers the residual micro artifact score. Both runs loaded the clean seed42 source checkpoint `39936`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `occupancy_occ_thre=1e-4`, occupancy warmups `4096/4096`, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, save interval `1024`, artifact-aware micro selection over all 3 eval views, and explicit `max_num_iterations=200000`. Early stopping stopped both runs before the cap.

| Run | Huber delta | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_huber_delta01_s42_occ0001_39936_200k_save1024` | 0.10 | 40960 | 29.6061 | 0.6899 | 0.3945 | 0.296 | 0.258 | 0.000 | 0.000 | 3546.7 | 5848.4 | Reject; PSNR improves but micro artifact is worse than current leader `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta_bracket_200k/lookcloser/arm_h40_grid128_huber_delta01_s42_occ0001_39936_200k_save1024/renders_artifact_selection_step-000040960` |
| `arm_h40_grid128_huber_delta005_s42_occ0001_39936_200k_save1024` | 0.05 | 40960 | 29.5999 | 0.6903 | 0.3946 | 0.260 | 0.258 | 0.000 | 0.000 | 5201.0 | 8163.5 | Reject; nearly ties current leader's micro but does not beat it and SSIM is worse | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta_bracket_200k/lookcloser/arm_h40_grid128_huber_delta005_s42_occ0001_39936_200k_save1024/renders_artifact_selection_step-000040960` |

Notable late points:

| Huber delta | Late step | PSNR | SSIM | LPIPS | Micro artifact | ROI artifact | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.10 | 57344 | 29.4073 | 0.6872 | 0.3863 | about 1.5 | about 0.3-0.5 | LPIPS improves, but the branch is clearly dirty |
| 0.05 | 77824 | 29.4219 | 0.6875 | 0.3864 | 2.270 | 1.147 | Later LPIPS branch remains dirty and grows ROI score |

### Insight

Lowering Huber delta does not solve the ARM artifact boundary. Delta `0.05` almost matches the current leader's residual full-frame micro score (`0.260` vs `0.256`) but does not improve it, while delta `0.10` is clearly worse. Both runs still move toward lower LPIPS late in training, and both make that late region dirty. Keep the current Huber delta `0.2` leader unchanged.

## Seed42 current-leader distortion bracket

Tested whether stronger field/geometry regularization can stabilize the low-LPIPS late branch after the current seed42 leader. Both runs loaded current leader checkpoint `40576`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Huber delta `0.2`, `occupancy_occ_thre=1e-4`, grid128, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, save interval `1024`, artifact-aware micro selection over all 3 eval views, and explicit `max_num_iterations=200000`. Only `distortion_loss_mult` changed.

| Run | Distortion | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_huber_s42_from40576_dist015_200k_save1024` | 0.015 | 41984 | 29.5328 | 0.6910 | 0.3942 | 0.311 | 0.256 | 0.000 | 0.000 | 6283.9 | 9749.8 | Reject; selected checkpoint is dirtier than current leader micro `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_distortion_bracket_200k/lookcloser/arm_h40_grid128_huber_s42_from40576_dist015_200k_save1024/renders_artifact_selection_step-000041984` |
| `arm_h40_grid128_huber_s42_from40576_dist02_200k_save1024` | 0.020 | 41984 | 29.4821 | 0.6923 | 0.3942 | 0.310 | 0.255 | 0.000 | 0.000 | 5862.6 | 9166.2 | Reject; no artifact improvement and lower PSNR than `0.015` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_distortion_bracket_200k/lookcloser/arm_h40_grid128_huber_s42_from40576_dist02_200k_save1024/renders_artifact_selection_step-000041984` |

Late metric/detail points:

| Distortion | Late step | PSNR | SSIM | LPIPS | Micro artifact | ROI artifact | Selected renders |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.015 | 81920 | 29.4220 | 0.6860 | 0.3815 | 2.592 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_distortion_bracket_200k/lookcloser/arm_h40_grid128_huber_s42_from40576_dist015_200k_save1024/renders_artifact_selection_step-000081920` |
| 0.020 | 77824 | 29.2970 | 0.6833 | 0.3826 | 2.918 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_distortion_bracket_200k/lookcloser/arm_h40_grid128_huber_s42_from40576_dist02_200k_save1024/renders_artifact_selection_step-000077824` |

### Insight

Stronger distortion regularization is not the missing stabilizer for the current seed42 ARM branch. It lets validation loss and LPIPS improve for much longer under the `200000` cap, but the artifact-clean selector still falls back to an early checkpoint with micro `0.310`-`0.311`, worse than the current leader's `0.256`. The late low-LPIPS region remains strongly dirty despite ROI/stand scores staying `0.000`, so the current leader remains unchanged.

## Seed42 late occupancy threshold clamp bracket

Tested whether making the late low-threshold occupancy update more conservative can stabilize the dirty LPIPS-friendly branch. All three runs loaded clean seed42 source checkpoint `39936`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `grid_resolution=128`, `occupancy_occ_thre=1e-4`, occupancy warmups `4096/4096`, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, Huber delta `0.2`, save interval `1024`, artifact-aware micro selection over all 3 eval views, and explicit `max_num_iterations=200000`. Only `occupancy_thre_clamp_mult` changed.

| Run | Clamp mult | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_huber_s42_occ0001_clamp025_39936_200k_save1024_r3` | 0.25 | 40960 | 29.6072 | 0.6898 | 0.3945 | 0.258 | 0.258 | 0.000 | 0.000 | 4629.3 | 7189.9 | Reject; ties the same early checkpoint but misses current leader micro `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_clamp_seed42_200k/lookcloser/arm_h40_grid128_huber_s42_occ0001_clamp025_39936_200k_save1024_r3/renders_artifact_selection_step-000040960` |
| `arm_h40_grid128_huber_s42_occ0001_clamp050_39936_200k_save1024_r3` | 0.50 | 40960 | 29.6114 | 0.6898 | 0.3945 | 0.258 | 0.258 | 0.000 | 0.000 | 4569.1 | 7058.0 | Reject; slightly higher PSNR but no artifact improvement | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_clamp_seed42_200k/lookcloser/arm_h40_grid128_huber_s42_occ0001_clamp050_39936_200k_save1024_r3/renders_artifact_selection_step-000040960` |
| `arm_h40_grid128_huber_s42_occ0001_clamp075_39936_200k_save1024_r3` | 0.75 | 40960 | 29.6072 | 0.6898 | 0.3945 | 0.258 | 0.258 | 0.000 | 0.000 | 4629.7 | 7128.2 | Reject; no artifact improvement and later checkpoints show ROI/stand dirt | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_clamp_seed42_200k/lookcloser/arm_h40_grid128_huber_s42_occ0001_clamp075_39936_200k_save1024_r3/renders_artifact_selection_step-000040960` |

Late LPIPS-best checkpoints:

| Clamp mult | LPIPS-best step | PSNR | SSIM | LPIPS | Micro artifact | ROI artifact | Read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.25 | 54272 | 29.4088 | 0.6879 | 0.3861 | 0.794 | 0.000 | Better LPIPS, still dirty |
| 0.50 | 56320 | 29.4121 | 0.6876 | 0.3858 | 1.553 | 0.488 | Dirty and now ROI-positive |
| 0.75 | 56320 | 29.4139 | 0.6877 | 0.3861 | 1.515 | 0.000 | Dirty |

### Insight

`occupancy_thre_clamp_mult` is not the missing stabilizer for the seed42 low-threshold branch. The artifact-aware selector always falls back to step `40960` with micro `0.258`, which is slightly worse than the current leader's `0.256`, while the validation/LPIPS-favored late checkpoints remain clearly dirty. Keep the current leader unchanged and do not spend more budget on this clamp bracket unless paired with a different field-trajectory change.

## Seed42 current-leader loss-side scan

Tested whether changing only the reconstruction loss after the current leader can move the field trajectory below the residual micro artifact floor. All three runs loaded current leader checkpoint `40576`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `occupancy_occ_thre=1e-4`, `grid_resolution=128`, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, save interval `128`, artifact-aware micro selection over all 3 eval views, and `max_num_iterations=42112` for this short boundary scan. Only the loss changed.

| Run | Loss | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_mse_s42_from40576_42112_save128` | MSE | 42111 | 29.6752 | 0.6846 | 0.3955 | 0.266 | 0.266 | 0.000 | 0.000 | 360.4 | 1760.0 | Reject; strong PSNR but artifact and LPIPS worse than current leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_loss_from_leader_micro/lookcloser/arm_h40_grid128_mse_s42_from40576_42112_save128/renders_artifact_selection_step-000042111` |
| `arm_h40_grid128_huber025_s42_from40576_42112_save128` | Huber delta0.25 | 40704 | 29.5512 | 0.6933 | 0.3944 | 0.262 | 0.255 | 0.000 | 0.000 | 390.4 | 1739.0 | Reject; nearly ties serious score but full micro worse than current leader `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_loss_from_leader_micro/lookcloser/arm_h40_grid128_huber025_s42_from40576_42112_save128/renders_artifact_selection_step-000040704` |
| `arm_h40_grid128_huber030_s42_from40576_42112_save128` | Huber delta0.30 | 40704 | 29.5514 | 0.6933 | 0.3944 | 0.262 | 0.255 | 0.000 | 0.000 | 420.4 | 1758.0 | Reject; same artifact floor as delta0.25 with no metric advantage | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_loss_from_leader_micro/lookcloser/arm_h40_grid128_huber030_s42_from40576_42112_save128/renders_artifact_selection_step-000040704` |

Notable metric/detail points:

| Loss | Step | PSNR | SSIM | LPIPS | Micro artifact | ROI artifact | Read |
|---|---:|---:|---:|---:|---:|---:|---|
| MSE | 40704 | 29.6366 | 0.6884 | 0.3939 | 0.284 | 0.000 | Best LPIPS inside MSE scan, dirty |
| MSE | 42111 | 29.6752 | 0.6846 | 0.3955 | 0.266 | 0.000 | Selected by artifact, but still worse than leader |
| Huber 0.25 | 42111 | 29.5788 | 0.6894 | 0.3941 | 0.290 | 0.000 | Slight LPIPS gain, dirtier |
| Huber 0.30 | 42111 | 29.5791 | 0.6894 | 0.3941 | 0.296 | 0.000 | Slight LPIPS gain, dirtier |

### Insight

Changing the loss after the current leader does not remove the residual micro artifacts. Huber deltas above `0.2` almost preserve the current metrics but bottom out at micro `0.262`, and MSE trades a large PSNR increase for worse SSIM/LPIPS and micro `0.266`. Together with the earlier rejected Charbonnier scan, this closes the simple loss-switch path around the current leader. The active leader remains unchanged.

## Seed42 current-leader render-compositing check

Tested whether the residual micro artifacts in the current leader are mainly alpha/background compositing holes by re-evaluating checkpoint `40576` with `background_color: last_sample`. This required adding packed-ray support for `last_sample` in `RGBRenderer.combine_rgb`; default black-background rendering and training behavior are unchanged.

| Render mode | Step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | Train time (s) | Decision | Renders |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| black background, current leader | 40576 | 29.5363 | 0.6934 | 0.3946 | 0.256 | 0.256 | 180.2 | Keep leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/renders_artifact_selection_step-000040576` |
| `last_sample` background | 40576 | 22.1575 | 0.6118 | 0.4059 | 558.034 | 558.034 | n/a eval-only | Reject; severe render degradation | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/renders_bg_last_sample_step-000040576` |

Per-view `last_sample` micro scores were `487.526` on eval0, `511.193` on eval1, and `558.034` on eval2.

### Insight

`last_sample` does not mask or fix the residual artifacts; it catastrophically worsens both metrics and detector score. The current leader's remaining defects should still be treated as field/checkpoint trajectory issues rather than a simple black-background compositing artifact. Keep the packed `last_sample` renderer fix as code correctness for that exposed config option, but do not use `last_sample` for this experiment family.

## ARM occupancy-grid levels=2 screen

Tested whether cascade-like occupancy coverage can stabilize the ARM late grid-miss/detail problem. All three runs were from scratch because changing `occupancy_grid_levels` changes occupancy buffer shapes versus the current `levels=1` checkpoints. Common settings: ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Huber delta `0.2`, `grid_resolution=128`, `occupancy_grid_levels=2`, occupancy warmups `4096/4096`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, batch `4096`, `max_num_iterations=200000`, eval/save interval `8192`, early stopping, and micro artifact selection over all 3 eval views.

| Seed | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 42 | 16384 | 13.8492 | 0.3635 | 0.8795 | 4.106 | 4.004 | 733.144 | 0.000 | 4298.2 | 4611.8 | Reject; severe collapse | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occlevels2_200k_micro/lookcloser/arm_h40_grid128_occlevels2_huber_s42_200k_i8192/renders_artifact_selection_step-000016384` |
| 43 | 8192 | 14.0741 | 0.3630 | 0.9042 | 7.366 | 7.318 | 0.000 | 0.000 | 4298.6 | 4613.0 | Reject; severe collapse | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occlevels2_200k_micro/lookcloser/arm_h40_grid128_occlevels2_huber_s43_200k_i8192/renders_artifact_selection_step-000008192` |
| 44 | 8192 | 14.2081 | 0.3642 | 0.9122 | 8.006 | 7.923 | 633.542 | 0.000 | 4298.0 | 4616.1 | Reject; severe collapse | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occlevels2_200k_micro/lookcloser/arm_h40_grid128_occlevels2_huber_s44_200k_i8192/renders_artifact_selection_step-000008192` |

Second eval checkpoints stayed collapsed: seed42 step `16384` was `13.8492/0.3635/0.8795`, seed43 step `16384` was `13.9509/0.3591/0.8831`, and seed44 step `16384` was `14.1514/0.3675/0.8873`. All three runs stopped by eval-loss no-improve after the second boundary.

### Insight

`occupancy_grid_levels=2` is rejected for the current bounded ARM recipe. It does not act as a conservative coverage fix; it changes the occupancy/traversal behavior enough that all three seeds plateau around PSNR `14` with large full-frame/ROI artifact scores. Keep `occupancy_grid_levels=1`. The remaining artifact/detail problem should be attacked through field/capacity or a more targeted late occupancy policy, not multi-level occupancy cascades.

## ARM color MLP depth 3 screen

Tested a softer capacity lever than `hash_features_per_level=4`: increase only `color_num_layers` from `2` to `3`, leaving the hash-grid feature count and occupancy-grid recipe unchanged. All runs were from scratch with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Huber delta `0.2`, `grid_resolution=128`, `occupancy_grid_levels=1`, occupancy warmups `4096/4096`, `adaptive_coarse_step_size=0.00625`, `adaptive_min_frequency_level=4`, `adaptive_max_frequency_level=13`, `max_steps_per_ray=2048`, batch `4096`, `max_num_iterations=200000`, eval/save interval `8192`, early stopping, and micro artifact selection over all 3 eval views.

| Seed | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 42 | 16384 | 14.2234 | 0.3675 | 0.9015 | 8.068 | 8.044 | 712.285 | 0.000 | 4358.7 | 4657.8 | Reject; severe collapse | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_color3_200k_micro/lookcloser/arm_h40_grid128_color3_huber_s42_200k_i8192/renders_artifact_selection_step-000016384` |
| 43 | 16384 | 14.6583 | 0.3827 | 0.9024 | 9.323 | 9.265 | 331.008 | 0.000 | 4358.1 | 4669.2 | Reject; severe collapse | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_color3_200k_micro/lookcloser/arm_h40_grid128_color3_huber_s43_200k_i8192/renders_artifact_selection_step-000016384` |
| 44 | 8192 | 14.5486 | 0.3714 | 0.9355 | 6.455 | 6.428 | 361.816 | 0.000 | 4358.2 | 4663.3 | Reject; severe collapse | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_color3_200k_micro/lookcloser/arm_h40_grid128_color3_huber_s44_200k_i8192/renders_artifact_selection_step-000008192` |

All three stopped after the second eval boundary because eval loss did not improve. First evals were already collapsed (`PSNR 14.49`-`14.95`, LPIPS `0.922`-`0.936`), and second evals did not recover.

### Insight

`color_num_layers=3` is rejected for the current ARM Huber recipe. It behaves like a capacity/training mismatch rather than a useful detail lever: quality collapses early and artifact scores become very large. Keep `color_num_layers=2`. The next capacity attempt, if any, should not simply add MLP depth under the same Huber schedule; it would need a different initialization/loss schedule or should return to the already-known fpl4+Charbonnier metric branch with a targeted artifact stabilizer.

## fpl4 + Charbonnier dist015 reproduction with checkpoints

Retested the strongest dirty fpl4 metric branch with checkpoints preserved for debugging. Common settings: ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `hash_features_per_level=4`, Charbonnier loss, `distortion_loss_mult=0.015`, grid128, `occupancy_grid_levels=1`, occupancy warmups `4096/4096`, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, eval/save interval `8192`, `max_num_iterations=200000`, early stopping, and micro artifact-aware selection over all 3 eval views.

| Run | Seed | Occupancy threshold | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_fpl4_charb_dist015_s42_repro_200k_i8192` | 42 | 0.01 | 24576 | 29.9098 | 0.6835 | 0.3971 | 0.384 | 0.303 | 0.000 | 0.000 | 11515.0 | 12296.5 | Reject; strong PSNR but not artifact-clean and LPIPS worse than current leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_fpl4_charb_dist015_repro_debug/lookcloser/arm_h40_grid128_fpl4_charb_dist015_s42_repro_200k_i8192/renders_artifact_selection_step-000024576` |
| `arm_h40_grid128_fpl4_charb_dist015_s44_repro_200k_i8192` | 44 | 0.01 | 16384 | 29.3357 | 0.6799 | 0.4112 | 1.188 | 1.188 | 0.000 | 0.000 | 12597.2 | 13121.9 | Reject; variance is much dirtier than seed42 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_fpl4_charb_dist015_repro_debug/lookcloser/arm_h40_grid128_fpl4_charb_dist015_s44_repro_200k_i8192/renders_artifact_selection_step-000016384` |
| `arm_h40_grid128_fpl4_charb_dist015_occ0001_s42_200k_i8192` | 42 | 0.0001 | 16384 manual | 28.9753 | 0.6690 | 0.4196 | 503.243 | 503.172 | n/a | n/a | interrupted after second eval | n/a | Reject; denser threshold is slower and catastrophically dirty on eval0/eval2 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_fpl4_charb_dist015_occ0001_debug/lookcloser/arm_h40_grid128_fpl4_charb_dist015_occ0001_s42_200k_i8192/renders_manual_step-000016384` |

Notable metric checkpoints:

| Run | Step | PSNR | SSIM | LPIPS | Micro artifact | Read |
|---|---:|---:|---:|---:|---:|---|
| seed42, `occ_thre=0.01` | 32768 | 30.0103 | 0.6853 | 0.3950 | 0.520 | metric-best by eval loss, dirtier than selected step |
| seed42, `occ_thre=0.01` | 40960 | 29.9841 | 0.6848 | 0.3930 | 0.652 | lower LPIPS but dirtier |
| seed44, `occ_thre=0.01` | 40960 | 29.6783 | 0.6935 | 0.3945 | 1.494 | strong SSIM, dirty |
| seed44, `occ_thre=0.01` | 49152 | 29.7284 | 0.6939 | 0.3916 | 1.290 | lower LPIPS, still dirty |

Artifact-to-occupancy debugging on seed42 selected step `24576`, eval0, reported:

`grid_miss_likely=false`, `field_issue_likely=true`

Debug path:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_fpl4_charb_dist015_repro_debug/lookcloser/arm_h40_grid128_fpl4_charb_dist015_s42_repro_200k_i8192/artifact_occ_debug_eval0_step24576/artifact_occupancy_debug.md`

### Insight

The fpl4 + Charbonnier `dist015` branch is reproducibly a PSNR/SSIM capacity signal, but it is not the current cleanup path. With checkpoints preserved, the least-dirty seed42 checkpoint still has full-frame micro artifact `0.384`, and occupancy debug says the main eval0 artifacts already project through occupied voxels. Lowering `occupancy_occ_thre` to `1e-4` from scratch makes the branch much denser and much worse, so this artifact mode is not fixed by more conservative occupancy. Treat this branch as field/trajectory instability, not occupancy pruning. The current Huber leader remains unchanged.

## fpl4 + Charbonnier dist015 loss-transfer cleanup

Tested whether the dirty fpl4+Charbonnier seed42 branch can be cleaned by switching loss after its least-dirty checkpoint. Both runs loaded `arm_h40_grid128_fpl4_charb_dist015_s42_repro_200k_i8192` step `24576`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, `hash_features_per_level=4`, `occupancy_occ_thre=0.01`, `distortion_loss_mult=0.015`, grid128, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, save/eval every `1024`, `max_num_iterations=200000`, and micro artifact selection over all 3 eval views.

| Run | Loss | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_fpl4_from_charb24576_huber_s42_200k_i1024` | Huber delta0.2 | 30720 | 29.5865 | 0.6931 | 0.3911 | 0.482 | 0.472 | 6.986 | 0.000 | 1623.1 | 2163.7 | Reject; LPIPS improves but full-frame/ROI artifacts worsen | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_fpl4_transfer_from_charb24576_200k/lookcloser/arm_h40_grid128_fpl4_from_charb24576_huber_s42_200k_i1024/renders_artifact_selection_step-000030720` |
| `arm_h40_grid128_fpl4_from_charb24576_mse_s42_200k_i1024` | MSE | 28672 | 29.9386 | 0.6880 | 0.3924 | 0.467 | 0.467 | 1.598 | 1.598 | 1172.6 | 1696.6 | Reject; high PSNR but still dirtier than source fpl4 and current leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_fpl4_transfer_from_charb24576_200k/lookcloser/arm_h40_grid128_fpl4_from_charb24576_mse_s42_200k_i1024/renders_artifact_selection_step-000028672` |

Notable metric points:

| Run | Step | PSNR | SSIM | LPIPS | Micro artifact | Read |
|---|---:|---:|---:|---:|---:|---|
| Huber transfer | 29696 | 29.6247 | 0.6948 | 0.3932 | 0.767 | best eval-loss checkpoint, dirty |
| Huber transfer | 30720 | 29.5865 | 0.6931 | 0.3911 | 0.482 | best artifact checkpoint, still dirty |
| MSE transfer | 27648 | 29.9158 | 0.6861 | 0.3953 | 0.649 | best eval-loss checkpoint, dirty |
| MSE transfer | 28672 | 29.9386 | 0.6880 | 0.3924 | 0.467 | best artifact checkpoint, still dirty |

### Insight

Loss transfer does not clean fpl4+Charbonnier artifacts. Huber and MSE can push LPIPS toward `0.391`-`0.392`, but artifact scores stay above the source fpl4 selected checkpoint (`0.384`) and far above the current leader (`0.256` diagnostic micro, significant `0.000`). This further closes fpl4 capacity as the immediate low-hanging path unless a different field-regularization mechanism is introduced.

## Seed42 current-leader frequency-grid freeze

Tested whether residual artifacts after the current leader are caused by noisy runtime frequency-grid updates. The run loaded current leader checkpoint `40576`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Huber delta `0.2`, `occupancy_occ_thre=1e-4`, grid128, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, save/eval every `1024`, `max_num_iterations=200000`, and set `grid_update_interval=999999` so the loaded frequency grid stayed frozen.

| Run | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_huber_s42_from40576_freqfreeze_200k_save1024` | 40960 | 29.5680 | 0.6929 | 0.3943 | 0.294 | 0.294 | 0.000 | 0.000 | 241.5 | 403.6 | Reject; worse than current leader artifact `0.256` and no metric gain | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqfreeze_from_leader_200k/lookcloser/arm_h40_grid128_huber_s42_from40576_freqfreeze_200k_save1024/renders_artifact_selection_step-000040960` |

The second checkpoint `41984` reached PSNR `29.5764`, SSIM `0.6894`, LPIPS `0.3939`, but was slightly dirtier (`micro=0.297`) and eval loss did not improve.

### Insight

Freezing the loaded frequency grid does not remove the residual micro artifacts. This rejects the simple hypothesis that post-leader frequency-grid updates alone push the clean checkpoint dirty. The remaining frequency-grid code hypothesis is more specific: runtime updates may be projected from arbitrary pixels instead of patch centers during training, which requires a targeted patch-center update test rather than disabling updates entirely.

## Seed42 current-leader patch-center frequency update

Temporarily added an experimental `patch_center` runtime frequency-grid update mode, leaving the default code path unchanged, then tested it from the same current leader checkpoint `40576`. Common settings matched the freeze test: ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Huber delta `0.2`, `occupancy_occ_thre=1e-4`, grid128, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, save/eval every `1024`, `max_num_iterations=200000`, and micro artifact selection over all 3 eval views.

| Run | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_huber_s42_from40576_patchcenter_200k_save1024` | 40960 | 29.5710 | 0.6930 | 0.3944 | 0.294 | 0.294 | 0.000 | 0.000 | 241.4 | 404.0 | Reject; same artifact score as freeze and worse than current leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_patchcenter_from_leader_200k/lookcloser/arm_h40_grid128_huber_s42_from40576_patchcenter_200k_save1024/renders_artifact_selection_step-000040960` |

The experimental flag/code was removed after the run because it did not improve the accepted recipe.

### Insight

Patch-center runtime frequency updates do not fix the current-leader residual micro artifacts. Together with the freeze test, this makes runtime frequency-grid update projection unlikely to be the immediate low-hanging fix for the seed42 Huber leader. The remaining issue still looks like field/checkpoint trajectory: small full-frame artifacts persist even when occupancy and frequency-grid update policies are changed.

## Seed42 current-leader Huber low-LR continuation

Tested whether the current leader can be continued with a gentler field trajectory. The run loaded checkpoint `40576`, reset the scheduler with `--no-load-scheduler`, set fields LR `1e-4 -> 1e-5`, and otherwise kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Huber delta `0.2`, `occupancy_occ_thre=1e-4`, grid128, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, save/eval every `1024`, `max_num_iterations=200000`, and micro artifact selection over all 3 eval views.

| Run | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_huber_s42_from40576_lowlr_200k_save1024` | 40960 | 29.5695 | 0.6930 | 0.3944 | 0.294 | 0.294 | 0.000 | 0.000 | 241.4 | 404.1 | Reject; same artifact floor as freeze/patch-center and worse than current leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_lowlr_from_leader_200k/lookcloser/arm_h40_grid128_huber_s42_from40576_lowlr_200k_save1024/renders_artifact_selection_step-000040960` |

The second checkpoint `41984` reached PSNR `29.5812`, SSIM `0.6894`, LPIPS `0.3942`, but artifact score rose to `0.297` and eval loss did not improve.

### Insight

Gentler Huber continuation does not preserve the current leader's artifact floor. The repeated pattern across normal/freeze/patch-center/low-LR continuations is that the first post-leader checkpoint around `40960` lands at micro artifact about `0.294`, so the accepted checkpoint remains `40576`. The current issue is not fixed by simply lowering LR after the leader.

## ARM field hidden width 128 screen

Tested a capacity increase that does not change hash-grid feature count: `field_hidden_dim=128` instead of `64`. The run was from scratch with ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Huber delta `0.2`, grid128, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, eval/save interval `8192`, `max_num_iterations=200000`, early stopping, and micro artifact selection over all 3 eval views.

| Run | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_hidden128_huber_s42_200k_i8192` | 16384 | 13.8125 | 0.3715 | 0.8931 | 3.208 | 3.091 | 560.012 | 0.000 | 1563.2 | 1722.4 | Reject; severe quality collapse | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_hidden128_200k_micro/lookcloser/arm_h40_grid128_hidden128_huber_s42_200k_i8192/renders_artifact_selection_step-000016384` |

First eval step `8192` was already collapsed (`13.8662` / `0.3713` / `0.9066`, micro `8.679`), and eval loss did not improve at step `16384`.

### Insight

Increasing hidden width is rejected for the current ARM Huber recipe. It behaves like the earlier color-depth and appearance-capacity screens: training collapses early and artifact scores are unusable. Keep `field_hidden_dim=64`.

## Seed42 current-leader fresh-optimizer continuation

Tested whether the post-leader dirty branch is caused by loaded Adam optimizer moments rather than by the field state itself. All runs loaded current leader checkpoint `40576`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Huber delta `0.2`, `occupancy_occ_thre=1e-4`, grid128, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, reset the scheduler, and did not load optimizer/scaler state. The new `--no-load-optimizers` flag preserves default checkpoint loading unless explicitly set.

First screen used save/eval interval `1024`, `max_num_iterations=200000`, and micro artifact selection over all 3 eval views.

| Run | Fresh LR | Selected step | PSNR | SSIM | LPIPS | Significant artifact | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_huber_s42_from40576_freshopt_lr1e4_200k_save1024` | `1e-4 -> 1e-5` | 41984 | 29.4606 | 0.6899 | 0.3942 | 0.000 | 0.338 | 0.308 | 0.000 | 0.000 | 662.4 | 1023.6 | Reject; micro and metrics worse than leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freshopt_from_leader_200k/lookcloser/arm_h40_grid128_huber_s42_from40576_freshopt_lr1e4_200k_save1024/renders_artifact_selection_step-000041984` |
| `arm_h40_grid128_huber_s42_from40576_freshopt_lr5e5_200k_save1024` | `5e-5 -> 5e-6` | 40960 | 29.5540 | 0.6951 | 0.3933 | 0.000 | 0.300 | 0.300 | 0.000 | 0.000 | 662.4 | 1024.1 | Reject; better metrics but worse micro than leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freshopt_from_leader_200k/lookcloser/arm_h40_grid128_huber_s42_from40576_freshopt_lr5e5_200k_save1024/renders_artifact_selection_step-000040960` |
| `arm_h40_grid128_huber_s42_from40576_freshopt_lr2e5_200k_save1024` | `2e-5 -> 2e-6` | 40960 | 29.5750 | 0.6938 | 0.3937 | 0.000 | 0.297 | 0.297 | 0.000 | 0.000 | 602.2 | 935.5 | Reject; significant-clean but micro worse than leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freshopt_from_leader_200k/lookcloser/arm_h40_grid128_huber_s42_from40576_freshopt_lr2e5_200k_save1024/renders_artifact_selection_step-000040960` |

Because `lr5e-5` and `lr2e-5` improved global metrics while failing only the diagnostic micro score, a denser boundary scan was run with save/eval interval `128` to check whether a short clean window existed between step `40576` and `40960`.

| Run | Fresh LR | Selected step | PSNR | SSIM | LPIPS | Significant artifact | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_huber_s42_from40576_freshopt_lr5e5_200k_save128` | `5e-5 -> 5e-6` | 40704 | 29.5715 | 0.6934 | 0.3935 | 0.000 | 0.297 | 0.297 | 0.000 | 0.000 | 661.8 | 1171.9 | Reject; no clean micro window | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freshopt_dense_from_leader_200k/lookcloser/arm_h40_grid128_huber_s42_from40576_freshopt_lr5e5_200k_save128/renders_artifact_selection_step-000040704` |
| `arm_h40_grid128_huber_s42_from40576_freshopt_lr2e5_200k_save128` | `2e-5 -> 2e-6` | 40704 | 29.5724 | 0.6929 | 0.3940 | 0.000 | 0.288 | 0.256 | 0.000 | 0.000 | 451.7 | 829.0 | Reject; best fresh optimizer micro still above leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freshopt_dense_from_leader_200k/lookcloser/arm_h40_grid128_huber_s42_from40576_freshopt_lr2e5_200k_save128/renders_artifact_selection_step-000040704` |

Notable metric checkpoints from the dense `lr5e-5` scan:

| Step | PSNR | SSIM | LPIPS | Micro artifact | ROI artifact | Read |
|---:|---:|---:|---:|---:|---:|---|
| 40832 | 29.5780 | 0.6943 | 0.3931 | 0.320 | 0.000 | best LPIPS in dense scan, dirty |
| 41088 | 29.5431 | 0.6951 | 0.3931 | 0.338 | 1.130 | lower eval loss, dirty |

### Insight

Resetting Adam state is not the missing stabilizer. It reliably improves LPIPS/SSIM under the official significant gate (`0.000`), but the diagnostic micro components grow immediately and remain worse than the current leader's `0.256`. The dense boundary scan rules out a missed short clean window at 128-step cadence. Keep the current leader unchanged and treat the remaining problem as field/trajectory instability that appears even with fresh optimizer moments.

## Official significant gate sensitivity check

After the fresh-optimizer runs, several existing low-LPIPS late renders were rescored with the official significant preset over all 3 eval views. This was a no-training check to see whether `artifact_score=0.000` under the significant gate still separates visually suspicious micro artifacts at this stage.

| Candidate | Step | PSNR | SSIM | LPIPS | Known micro artifact | Significant artifact | Render path |
|---|---:|---:|---:|---:|---:|---:|---|
| current leader | 40576 | 29.5363 | 0.6934 | 0.3946 | 0.256 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/renders_artifact_selection_step-000040576` |
| fresh optimizer `lr5e-5` dense | 40832 | 29.5780 | 0.6943 | 0.3931 | 0.320 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freshopt_dense_from_leader_200k/lookcloser/arm_h40_grid128_huber_s42_from40576_freshopt_lr5e5_200k_save128/renders_artifact_selection_step-000040832` |
| low-threshold distnorm late | 57344 | 29.4261 | 0.6875 | 0.3862 | 1.521 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_distnorm_long200k/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_distnorm39936_200k_save4096/renders_artifact_selection_step-000057344` |
| Charbonnier low-LR late | 49151 | 29.5983 | 0.6823 | 0.3889 | 0.315 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_charb_low_lr_from_leader_micro/lookcloser/arm_h40_grid128_charb_s42_occ0001_lowLR40576_49152_save1024/renders_artifact_selection_step-000049151` |
| per-ray bypass late | 57344 | 29.4318 | 0.6876 | 0.3862 | 1.485 | 0.000 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_perray_bypass_ratio/lookcloser/arm_h40_grid128_huber_s42_perrayBypass040v2_39936_200k_save4096/renders_artifact_selection_step-000057344` |

### Insight

The official significant preset is now too coarse for the remaining problem. It reports `0.000` even on late renders that the diagnostic micro detector scores around `1.5` and that are likely to contain the small holes/obstructions the visual gate cares about. Keep reporting significant scores for continuity, but do not promote a new leader unless it also improves the diagnostic micro score and passes visual inspection. Under this stricter criterion, the current leader remains unchanged.

## Field checkpoint interpolation toward metric-improved fresh optimizer

Tested whether the metric gains from the dirty fresh-optimizer checkpoint can be partially blended into the current clean leader without crossing the micro-artifact boundary. This was eval-only, no new training. The base checkpoint was current leader step `40576`; the target checkpoint was fresh-optimizer `lr5e-5` dense step `40832`, which has better PSNR/SSIM/LPIPS but micro artifact `0.320`. Only trainable field parameters were interpolated (`encoding`, `mlp_geo`, `mlp_color`); frequency grid, occupancy grid, and sampler state stayed from the current leader. Fake eval runs use `max_num_iterations=200000` in config for consistency, but train time is `0.0s` because no optimizer steps were run.

| Alpha | Step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Eval time (s) | Artifact time (s) | Decision | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.10 | 40576 | 29.5464 | 0.6937 | 0.3944 | 0.261 | 0.254 | 0.000 | 0.000 | 0.0 | 110.6 | 49.4 | Reject; slightly better metrics, but micro still above leader `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_checkpoint_interp/lookcloser/interp_field_leader_to_freshopt40832_a010/renders_artifact_selection_step-000040576` |
| 0.15 | 40576 | 29.5509 | 0.6938 | 0.3943 | 0.261 | 0.255 | 0.000 | 0.000 | 0.0 | 111.5 | 51.7 | Reject; same artifact floor as alpha `0.10` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_checkpoint_interp/lookcloser/interp_field_leader_to_freshopt40832_a015/renders_artifact_selection_step-000040576` |
| 0.20 | 40576 | 29.5549 | 0.6939 | 0.3942 | 0.261 | 0.255 | 0.000 | 0.000 | 0.0 | 109.2 | 41.6 | Reject; metrics improve but artifact score does not match leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_checkpoint_interp/lookcloser/interp_field_leader_to_freshopt40832_a020/renders_artifact_selection_step-000040576` |
| 0.25 | 40576 | 29.5593 | 0.6940 | 0.3941 | 0.261 | 0.255 | 0.000 | 0.000 | 0.0 | 109.3 | 42.7 | Reject; metrics improve but micro is above leader `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_checkpoint_interp/lookcloser/interp_field_leader_to_freshopt40832_a025/renders_artifact_selection_step-000040576` |
| 0.50 | 40576 | 29.5753 | 0.6942 | 0.3937 | 0.295 | 0.295 | 0.000 | 0.000 | 0.0 | 107.5 | 50.0 | Reject; artifact growth outweighs metric gain | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_checkpoint_interp/lookcloser/interp_field_leader_to_freshopt40832_a050/renders_artifact_selection_step-000040576` |
| 0.75 | 40576 | 29.5805 | 0.6944 | 0.3933 | 0.296 | 0.296 | 0.000 | 0.000 | 0.0 | 104.9 | 51.8 | Reject; best metrics but clearly worse micro | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_checkpoint_interp/lookcloser/interp_field_leader_to_freshopt40832_a075/renders_artifact_selection_step-000040576` |

### Insight

Local interpolation confirms the tradeoff is continuous: moving toward the metric-improved checkpoint improves PSNR/SSIM/LPIPS, but even a small `0.10` blend raises diagnostic micro from `0.256` to `0.261`. The `0.10` through `0.25` blends all hit the same `0.261` micro floor, then artifacts grow at `0.50+`. The remaining artifact floor is not just a discrete checkpoint-selection miss that can be solved by field averaging/interpolation. Current leader remains unchanged.

## Seed42 occ0001 switch trajectory boundary screen

Tested three short boundary continuations from the clean seed42 source checkpoint `39936` to isolate the known transition into the current late `occupancy_occ_thre=1e-4` window. ARM and occupancy-grid sampling stayed enabled; FAS and Feature Reweighting stayed disabled. All runs used Huber delta `0.2`, grid128, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, occupancy warmups `4096/4096`, occupancy update interval `16`, train-time eval disabled, save/eval interval `128`, and micro artifact selection over all 3 eval views.

These scans intentionally used `max_num_iterations=41216` because they target the narrow `39936 -> 41216` boundary around a known clean/dirty transition. This is a diagnostic exception. The default and future full ARM experiment cap is `max_num_iterations=200000`.

| Run | Change | Selected step | PSNR | SSIM | LPIPS | Significant artifact | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_huber_s42_from39936_occ0001_freshopt_lr2e5_39936_41216_save128` | fresh optimizer, fields LR `2e-5 -> 2e-6`, scheduler reset | 40448 | 29.5838 | 0.6835 | 0.3952 | 0.000 | 0.257 | 0.224 | 0.000 | 0.000 | 420.4 | 1709.2 | Reject; nearly ties but does not beat current leader micro `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_switch_boundary/lookcloser/arm_h40_grid128_huber_s42_from39936_occ0001_freshopt_lr2e5_39936_41216_save128/renders_artifact_selection_step-000040448` |
| `arm_h40_grid128_huber_s42_from39936_occ0001_freshopt_lr5e5_39936_41216_save128` | fresh optimizer, fields LR `5e-5 -> 5e-6`, scheduler reset | 40064 | 29.5597 | 0.6826 | 0.3955 | 0.000 | 0.260 | 0.226 | 0.000 | 0.000 | 390.4 | 1701.2 | Reject; metric trajectory is good, but micro artifact is worse than leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_switch_boundary/lookcloser/arm_h40_grid128_huber_s42_from39936_occ0001_freshopt_lr5e5_39936_41216_save128/renders_artifact_selection_step-000040064` |
| `arm_h40_grid128_huber_s42_from39936_occ0001_matchedmax_39936_41216_save128` | set `adaptive_max_step_size=adaptive_coarse_step_size=0.00625` | 40064 | 29.5658 | 0.6895 | 0.3934 | 0.000 | 0.299 | 0.299 | 0.000 | 0.000 | 420.4 | 1754.6 | Reject; LPIPS improves, but artifact score is clearly worse | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_switch_boundary/lookcloser/arm_h40_grid128_huber_s42_from39936_occ0001_matchedmax_39936_41216_save128/renders_artifact_selection_step-000040064` |

### Insight

Fresh optimizer from the clean source checkpoint can improve PSNR/SSIM/LPIPS locally, but it does not reduce the diagnostic micro artifact below the current leader. The `2e-5` fresh run is the closest miss (`0.257` versus `0.256`), which confirms the boundary is very narrow but not a missed checkpoint from stale Adam state. Matching the ARM max step to the coarse step improves LPIPS but makes the micro artifact substantially worse. Keep the current leader unchanged.

## Runtime frequency-grid validity filtering

Tested whether runtime frequency-grid updates are poisoning the ARM trajectory by projecting invalid/low-confidence rendered depths into the max-only 3D frequency grid. A temporary reversible patch filtered `_update_frequency_grid()` updates by projected AABB membership and, in two variants, rendered accumulation. The patch was removed after the test because it did not improve artifacts.

All runs loaded the clean seed42 source checkpoint `39936`, kept ARM and occupancy-grid sampling enabled, kept FAS and Feature Reweighting disabled, used Huber delta `0.2`, grid128, `occupancy_occ_thre=1e-4`, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, `max_num_iterations=200000`, eval-loss early stop, save interval `1024`, and micro artifact selection over all 3 eval views. Training stopped at step `57344` for all variants after eval loss stopped improving.

| Run | Frequency-grid update filter | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Selected renders |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_huber_s42_freqvalid_aabb_39936_200k_save1024` | AABB only | 40960 | 29.4823 | 0.6901 | 0.3935 | 0.338 | 0.317 | 0.000 | 0.000 | 3606.2 | 5634.1 | Reject; worse than current leader micro `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqgrid_validity_200k/lookcloser/arm_h40_grid128_huber_s42_freqvalid_aabb_39936_200k_save1024/renders_artifact_selection_step-000040960` |
| `arm_h40_grid128_huber_s42_freqvalid_acc005_aabb_39936_200k_save1024` | AABB + accumulation >= `0.05` | 40960 | 29.4826 | 0.6901 | 0.3935 | 0.335 | 0.313 | 0.000 | 0.000 | 3546.0 | 5533.5 | Reject; best of this bracket, still worse than leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqgrid_validity_200k/lookcloser/arm_h40_grid128_huber_s42_freqvalid_acc005_aabb_39936_200k_save1024/renders_artifact_selection_step-000040960` |
| `arm_h40_grid128_huber_s42_freqvalid_acc020_aabb_39936_200k_save1024` | AABB + accumulation >= `0.20` | 40960 | 29.4828 | 0.6901 | 0.3935 | 0.361 | 0.317 | 0.000 | 0.000 | 3606.3 | 5559.7 | Reject; stricter accumulation filter worsens artifact | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqgrid_validity_200k/lookcloser/arm_h40_grid128_huber_s42_freqvalid_acc020_aabb_39936_200k_save1024/renders_artifact_selection_step-000040960` |

Notable dirty detail checkpoints:

| Run | Step | PSNR | SSIM | LPIPS | Micro artifact | ROI artifact | Read |
|---|---:|---:|---:|---:|---:|---:|---|
| AABB only | 50176 | 29.2364 | 0.6855 | 0.3605 | 1.705 | 0.000 | Beats old H40 LPIPS but dirty |
| `acc005` + AABB | 56320 | 29.4309 | 0.6949 | 0.3552 | 2.762 | 0.637 | Excellent metrics, unusable artifacts |
| `acc020` + AABB | 56320 | 29.4220 | 0.6922 | 0.3545 | 2.484 | 1.249 | Best LPIPS in this bracket, dirty |

### Insight

Runtime frequency-grid update validity filtering is rejected. It does not clean the current seed42 late branch; the best selected checkpoint is still dirtier than the current leader, and the LPIPS-friendly late checkpoints remain strongly artifacted. This weakens the hypothesis that random invalid frequency-grid updates are the main cause of the remaining ARM artifacts. The temporary code was removed. The useful signal is that the field can still reach LPIPS `0.354`-`0.361` with strong PSNR/SSIM, but the artifact boundary is not controlled by this frequency-grid filter.

Occupancy debug on the strongest late dirty detail point, `acc020` step `56320`, eval0, used the explicit checkpoint and render path:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqgrid_validity_200k/lookcloser/arm_h40_grid128_huber_s42_freqvalid_acc020_aabb_39936_200k_save1024/artifact_occ_debug_eval0_step56320_explicit/artifact_occupancy_debug.md`

Result: `grid_miss_likely=true`, `field_issue_likely=false`. Only `35.0%` of selected artifact surface pixels landed in occupied surface voxels, while ray samples were not capped (`max=192`, `max_steps_per_ray=2048`, saturation `0.000`). This means the late low-LPIPS regime is still an occupancy/traversal coverage failure, not a sample-cap failure and not fixed by filtering invalid frequency-grid updates.

## Sticky binary occupancy retention

Tested whether late low-LPIPS ARM artifacts come from cells being switched off after they were once occupied. A temporary `sticky` binary occupancy retention patch was used, then removed after rejection. It captured loaded checkpoint binaries before the first occupancy update and OR-preserved them during later binary postprocessing. All runs kept ARM and occupancy-grid sampling enabled, FAS and Feature Reweighting disabled, Huber delta `0.2`, grid128, `occupancy_occ_thre=1e-4`, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, occupancy warmups `4096/4096`, and `max_num_iterations=200000`.

The main run loaded the current leader checkpoint `40576` and preserved optimizer/scheduler state. It reached excellent metrics but the metric-improved checkpoints were strongly dirty under the micro artifact detector:

| Step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Train time (s) | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 40960 | 29.5214 | 0.6955 | 0.3930 | 0.347 | 0.306 | 1.136 | ~7772 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occsticky_200k/lookcloser/arm_h40_grid128_huber_s42_occsticky_from40576_keepopt_40576_200k_save1024/renders_artifact_selection_step-000040960` |
| 49152 | 29.2182 | 0.6846 | 0.3622 | 1.703 | 1.487 | 0.741 | ~7772 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occsticky_200k/lookcloser/arm_h40_grid128_huber_s42_occsticky_from40576_keepopt_40576_200k_save1024/renders_manual_micro_step-000049152` |
| 53248 | 29.4093 | 0.6894 | 0.3580 | 1.688 | 1.561 | 0.603 | ~7772 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occsticky_200k/lookcloser/arm_h40_grid128_huber_s42_occsticky_from40576_keepopt_40576_200k_save1024/renders_manual_micro_step-000053248` |
| 57344 | 29.5860 | 0.6975 | 0.3551 | 1.808 | 1.690 | 0.378 | ~7772 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occsticky_200k/lookcloser/arm_h40_grid128_huber_s42_occsticky_from40576_keepopt_40576_200k_save1024/renders_manual_micro_step-000057344` |
| 61440 | 29.7116 | 0.7035 | 0.3503 | 1.921 | 1.804 | 0.000 | ~7772 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occsticky_200k/lookcloser/arm_h40_grid128_huber_s42_occsticky_from40576_keepopt_40576_200k_save1024/renders_manual_micro_step-000061440` |
| 65536 | 29.7640 | 0.7036 | 0.3476 | not scored | not scored | not scored | ~7772 | not rendered manually |
| 69632 | 29.8266 | 0.7087 | 0.3449 | not scored | not scored | not scored | ~7772 | not rendered manually |

Control runs:

| Run | Read | Best/last useful metrics | Artifact evidence | Train time (s) |
|---|---|---|---|---:|
| `arm_h40_grid128_huber_s42_occsticky_from39936_keepopt_39936_200k_save1024` | Rejected. Starting sticky retention from source step `39936` still dirties early. | Step `57344`: 29.5363 / 0.6920 / 0.3565 | Scoring was stopped after dirty candidates: step `40960` micro `0.317`, step `45056` micro `1.590`, ROI `0.812`, stand `0.320`. | ~5186 |
| `arm_h40_grid128_huber_s42_occsticky_from40576_fresh2e5_40576_200k_save1024` | Rejected. Fresh optimizer with LR `2e-5 -> 2e-6` stayed weak on LPIPS. | Step `69632`: 29.4689 / 0.6929 / 0.3941 | Not artifact-scored because metrics never approached the target; run was stopped as a weak control. | ~7848 |

Follow-up diagnostics on the dirty high-metric main checkpoint:

- Occupancy debugger on step `61440`, eval0, exact render/checkpoint: `grid_miss_likely=false`, `field_issue_likely=true`.
- Debug report: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occsticky_200k/lookcloser/arm_h40_grid128_huber_s42_occsticky_from40576_keepopt_40576_200k_save1024/artifact_occ_debug_eval0_step61440_manual_micro/artifact_occupancy_debug.md`
- Eval-only `adaptive_interval_level_mode=max3` on the same checkpoint did not clean the artifact: PSNR `29.7296`, SSIM `0.7040`, LPIPS `0.3506`, micro artifact `1.943`, ROI `0.000`.
- Max3 renders: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occsticky_200k/lookcloser/arm_h40_grid128_huber_s42_occsticky_from40576_keepopt_40576_200k_save1024/renders_manual_max3_step-000061440`

### Insight

Sticky binary occupancy retention is rejected and the temporary code was removed. It proves the field can reach the desired metric regime after the current leader (`LPIPS 0.3449`, PSNR `29.8266`, SSIM `0.7087`), but preserving loaded binary occupancy cells does not keep that regime artifact-clean. Dirty micro scores appear as soon as LPIPS improves below the old H40 value (`0.3653`). The exact dirty step is mostly occupied and remains dirty under eval-only `max3`, so the next cleanup path should target the field/geometry trajectory or training distribution, not binary occupancy retention or render-time interval query alone.

## Current-leader ultra-micro boundary rescan

Tested whether the current leader's residual diagnostic micro artifact is just a missed checkpoint between the `128`-step saves around `40576`. Both rescans loaded the previous saved checkpoint `40448`, kept ARM and occupancy-grid sampling enabled, kept FAS and Feature Reweighting disabled, and otherwise matched the current leader recipe: Huber delta `0.2`, `occupancy_occ_thre=1e-4`, grid128, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, warmups `4096/4096`, and micro artifact selection over all three eval views. This is a deliberate short boundary scan, not a full `200000` run.

| Run | Train-time eval cadence | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Train time (s) | Total time (s) | Decision | Renders |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_huber_s42_occ0001_from40448_40640_save8` | eval/save every `8` | 40472 | 29.5440 | 0.6940 | 0.3939 | 0.311 | 0.256 | 0.000 | 271.5 | 692.6 | Reject; denser eval cadence did not beat leader micro `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_ultramicro_boundary/lookcloser/arm_h40_grid128_huber_s42_occ0001_from40448_40640_save8/renders_artifact_selection_step-000040472` |
| `arm_h40_grid128_huber_s42_occ0001_from40448_40576_save8_noeval` | save every `8`, no train-time eval in window | 40528 | 29.5593 | 0.6939 | 0.3932 | 0.291 | 0.260 | 0.000 | 120.1 | 1464.0 | Reject; best rescan checkpoint improves LPIPS but remains dirtier than leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_ultramicro_boundary/lookcloser/arm_h40_grid128_huber_s42_occ0001_from40448_40576_save8_noeval/renders_artifact_selection_step-000040528` |

No-train-eval scan detail:

| Step | PSNR | SSIM | LPIPS | Micro artifact |
|---:|---:|---:|---:|---:|
| 40456 | 29.5338 | 0.6938 | 0.3944 | 0.318 |
| 40472 | 29.5430 | 0.6940 | 0.3939 | 0.310 |
| 40504 | 29.5538 | 0.6938 | 0.3934 | 0.298 |
| 40528 | 29.5593 | 0.6939 | 0.3932 | 0.291 |
| 40575 | 29.5498 | 0.6943 | 0.3930 | 0.299 |

### Insight

The current leader is not simply missing a nearby cleaner checkpoint. Replaying from step `40448` with denser saves finds slightly better LPIPS (`0.3930`-`0.3932`) but never recovers the original leader's lower micro artifact floor (`0.256`). The best rescan is `0.291`. This reinforces the trajectory/cadence sensitivity: the accepted `40576` checkpoint remains the best artifact-balanced point, and future work should change the field trajectory before the boundary rather than only saving more densely around it.

## Fresh-optimizer ultra-dense boundary follow-up

Tested a save-every-16 fresh-optimizer continuation from the clean seed42 Huber source checkpoint `39936` to `40576`: fresh optimizer/scheduler, fields LR `2e-5 -> 2e-6`, ARM and occupancy-grid sampling enabled, FAS and Feature Reweighting disabled, Huber delta `0.2`, grid128, `occupancy_occ_thre=1e-4`, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, no train-time eval, and micro artifact selection over all three eval views.

| Run | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arm_h40_grid128_huber_s42_from39936_fresh2e5_39936_40576_save16` | 40512 | 29.5875 | 0.6839 | 0.3952 | 0.256 | 0.223 | 0.000 | 0.000 | 270.3 | 4843.2 | Reject as leader; ties current leader micro but loses SSIM/LPIPS and took expensive artifact selection | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freshopt_ultradense_from39936/lookcloser/arm_h40_grid128_huber_s42_from39936_fresh2e5_39936_40576_save16/renders_artifact_selection_step-000040512` |

### Insight

Fresh optimizer plus ultra-dense checkpointing can match the current leader's diagnostic micro score, but it does not beat it. The best selected checkpoint improves PSNR but regresses SSIM and LPIPS against the current leader (`29.5363/0.6934/0.3946`, micro `0.256`). The remaining artifact floor is not solved by denser checkpointing or stale-optimizer reset.

## ARM full-occupancy training mix

Hypothesis: the remaining small holes may come from training distribution gaps caused by binary occupancy pruning during ARM training. The next test keeps ARM enabled and still uses normal occupancy-grid sampling for evaluation, but during training it randomly runs some batches with the same ARM sampler after temporarily treating occupancy binaries as fully occupied. This is a reversible experimental code path controlled by `adaptive_full_occupancy_train_ratio`; default is `0.0`.

Planned ratios are `0.10`, `0.30`, and `0.40` full-occupancy train batches. This interprets the requested `10/90`, `30/70`, and `40/50` variants as the fully-occupied bypass fraction versus the normal ARM fraction. All runs should stay on the current trusted seed42 Huber source setup: load clean step `39936`, ARM on, occupancy-grid sampling on, FAS off, Feature Reweighting off, Huber delta `0.2`, grid128, `occupancy_occ_thre=1e-4`, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, warmups `4096/4096`, micro artifact selection over all three eval views, and `max_num_iterations=200000`.

Status: completed and rejected. Ratios `0.10` and `0.30` ran in parallel. Ratio `0.40` was first started with the other two, then stopped before the first eval because three training jobs plus the active ultra-dense artifact-selection job pushed GPU memory to about `44.4/46.1 GB`; it was rerun sequentially as `arm_h40_grid128_huber_s42_fulloccmix040_from39936_200k_save1024_retry`. The temporary code path was removed after rejection.

| Ratio | Run | Selected step | PSNR | SSIM | LPIPS | Micro artifact | ROI artifact | Train time (s) | Total time (s) | Decision | Renders |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.10 | `arm_h40_grid128_huber_s42_fulloccmix010_from39936_200k_save1024` | 41984 | 29.4548 | 0.6900 | 0.3980 | 0.367 | 0.000 | 962.5 | 1350.9 | Reject; worse than current leader artifact and metrics | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_fullocc_mix_200k/lookcloser/arm_h40_grid128_huber_s42_fulloccmix010_from39936_200k_save1024/renders_artifact_selection_step-000041984` |
| 0.30 | `arm_h40_grid128_huber_s42_fulloccmix030_from39936_200k_save1024` | 40960 | 29.5113 | 0.6897 | 0.3965 | 0.319 | 0.000 | 1172.9 | 1558.1 | Reject; lower artifact than 0.10 but still worse than current leader `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_fullocc_mix_200k/lookcloser/arm_h40_grid128_huber_s42_fulloccmix030_from39936_200k_save1024/renders_artifact_selection_step-000040960` |
| 0.40 | `arm_h40_grid128_huber_s42_fulloccmix040_from39936_200k_save1024_retry` | 40960 | 29.5094 | 0.6897 | 0.3967 | 0.340 | 0.000 | 872.3 | 1409.4 | Reject; still worse than current leader artifact `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_fullocc_mix_200k/lookcloser/arm_h40_grid128_huber_s42_fulloccmix040_from39936_200k_save1024_retry/renders_artifact_selection_step-000040960` |

### Insight

Batch-level full-occupancy ARM mixing does not fix the residual micro holes. All three ratios kept ROI/stand scores at `0.000`, but full-frame micro artifact stayed above the current leader: `0.367` for ratio `0.10`, `0.319` for `0.30`, and `0.340` for `0.40`, versus leader `0.256`. Metrics also did not improve. This rejects the hypothesis that a simple fraction of unpruned ARM batches is enough to cover the remaining small misses. Do not keep the temporary sampling-mix code.

## ARM stochastic replay from step 39936 with 200k cap

Tested whether the remaining artifact floor is mostly trajectory variance after the clean seed42 Huber source checkpoint. All runs loaded step `39936`, kept ARM enabled, occupancy-grid sampling enabled, FAS disabled, Feature Reweighting disabled, Huber delta `0.2`, grid128, `occupancy_occ_thre=1e-4`, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, warmups `4096/4096`, save interval `512`, micro artifact selection over all three eval views, and `max_num_iterations=200000`. Early stopping on eval-loss plateau stopped seed47 at `53248` and seeds45/46 at `61440`.

Artifact-selected checkpoints:

| Seed | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Stand connector | Train time (s) | Total time (s) | Decision | Renders |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 45 | 41472 | 29.4827 | 0.6895 | 0.3973 | 0.323 | 0.323 | 0.000 | 0.000 | 4599.3 | 9025.5 | Reject; best artifact checkpoint is worse than current leader micro `0.256` and worse LPIPS | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s45_occ0001_rng_from39936_200k_save512/renders_artifact_selection_step-000041472` |
| 46 | 40960 | 29.4982 | 0.6898 | 0.3966 | 0.314 | 0.314 | 0.000 | 0.000 | 4599.4 | 9106.8 | Reject; best artifact checkpoint is still worse than current leader micro `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s46_occ0001_rng_from39936_200k_save512/renders_artifact_selection_step-000040960` |
| 47 | 40448 | 29.5390 | 0.6908 | 0.3951 | 0.306 | 0.306 | 0.000 | 0.000 | 3156.2 | 6692.3 | Reject; closest replay, but still dirtier than current leader and no LPIPS gain | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s47_occ0001_rng_from39936_200k_save512/renders_artifact_selection_step-000040448` |

Metric-best dirty checkpoints:

| Seed | Step | PSNR | SSIM | LPIPS | Micro artifact | ROI artifact | Note | Renders |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 45 | 61440 | 29.6613 | 0.7014 | 0.3576 | 1.500 | 0.000 | Better than old H40 LPIPS, but strongly dirty | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s45_occ0001_rng_from39936_200k_save512/renders_artifact_selection_step-000061440` |
| 46 | 60928 | 29.5576 | 0.6949 | 0.3588 | 2.116 | 0.000 | Better than old H40 LPIPS, but strongly dirty | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s46_occ0001_rng_from39936_200k_save512/renders_artifact_selection_step-000060928` |
| 47 | 53248 | 29.3976 | 0.6882 | 0.3724 | 2.056 | 1.588 | LPIPS improves, but full-frame and ROI artifacts are large | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s47_occ0001_rng_from39936_200k_save512/renders_artifact_selection_step-000053248` |

### Insight

Stochastic replay with a high `200000` cap confirms that the current ARM recipe can reach and exceed the old H40 LPIPS target, but not cleanly. The clean-ish early selected checkpoints are all worse than the current leader by diagnostic micro artifact (`0.306`-`0.323` versus `0.256`) and do not improve LPIPS. As training continues, PSNR/SSIM/LPIPS improve strongly, but micro artifacts jump to `~1.5`-`2.1` and ROI/stand scores sometimes become nonzero. This is evidence for a field/geometry trajectory tradeoff rather than an occupancy warmup, RNG, or missing-long-training issue. Current leader remains unchanged.

## Loss-side high-frequency RGB weighting

Tested a reversible loss-side high-frequency RGB weighting idea without Feature Reweighting and without FAS. The sampler stayed uniform because `enable_fas=False`; the temporary code only attached frequency-map levels to sampled rays and upweighted RGB loss for higher-frequency pixels. All runs loaded clean seed42 step `39936`, kept ARM enabled, occupancy-grid sampling enabled, Huber delta `0.2`, grid128, `occupancy_occ_thre=1e-4`, coarse00625, minfreq4/maxfreq13, cap2048, batch4096, warmups `4096/4096`, save interval `512`, micro artifact selection over all three eval views, and `max_num_iterations=200000`. The temporary code path was removed after rejection.

Artifact-selected checkpoints:

| Weighting | Selected step | PSNR | SSIM | LPIPS | Micro artifact | Micro serious | ROI artifact | Train time (s) | Total time (s) | Decision | Renders |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| strength `0.25`, min level `10`, max weight `1.25` | 40448 | 29.5246 | 0.6908 | 0.3926 | 0.328 | 0.305 | 0.000 | 4118.1 | 7970.7 | Reject; worse than current leader micro `0.256` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss025_l10_w125_from39936_200k_save512/renders_artifact_selection_step-000040448` |
| strength `0.50`, min level `10`, max weight `1.50` | 40960 | 29.4844 | 0.6903 | 0.3935 | 0.316 | 0.316 | 0.000 | 4087.9 | 7908.2 | Reject; best of this sweep, but still dirtier than current leader | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss050_l10_w150_from39936_200k_save512/renders_artifact_selection_step-000040960` |
| strength `1.00`, min level `12`, max weight `1.50` | 40448 | 29.5269 | 0.6907 | 0.3926 | 0.330 | 0.307 | 0.000 | 4118.4 | 7948.0 | Reject; no artifact improvement | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss100_l12_w150_from39936_200k_save512/renders_artifact_selection_step-000040448` |

LPIPS-best dirty checkpoints:

| Weighting | Step | PSNR | SSIM | LPIPS | Micro artifact | ROI artifact | Note | Renders |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `0.25/l10/w1.25` | 56320 | 29.4381 | 0.6909 | 0.3544 | 2.567 | 1.229 | Excellent LPIPS, strongly dirty | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss025_l10_w125_from39936_200k_save512/renders_artifact_selection_step-000056320` |
| `0.50/l10/w1.50` | 56320 | 29.4174 | 0.6918 | 0.3559 | 2.779 | 1.205 | Excellent LPIPS, strongly dirty | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss050_l10_w150_from39936_200k_save512/renders_artifact_selection_step-000056320` |
| `1.00/l12/w1.50` | 56320 | 29.4561 | 0.6970 | 0.3544 | 1.668 | 1.244 | Best dirty metric point, still artifact-heavy | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss100_l12_w150_from39936_200k_save512/renders_artifact_selection_step-000056320` |

### Insight

Loss-side high-frequency RGB weighting repeats the known tradeoff: it can push LPIPS below the old H40 target (`0.3653`) and as low as about `0.354`, but only in artifact-heavy states. Artifact-aware selection falls back to early checkpoints with worse LPIPS and still worse micro artifacts than the current leader. This rejects simple frequency-weighted RGB loss as the cleanup mechanism and strengthens the conclusion that the late detail/metric regime needs a different field or occupancy/traversal stability fix rather than more high-frequency supervision.

## Current leader

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense/lookcloser/arm_h40_grid128_huber_delta02_s42_occ0001_dense39936_41216_save128/renders_artifact_selection_step-000040576`

This is the single current leader because it improves PSNR, SSIM, LPIPS, official significant artifact score, ROI/stand artifact score, and diagnostic micro artifact score versus the previous seed42 Huber leader. It keeps ARM enabled, occupancy-grid sampling enabled, FAS disabled, and Feature Reweighting disabled. It uses late `occupancy_occ_thre=1e-4` after loading the clean seed42 Huber checkpoint.

| Seed | Selected step | PSNR | SSIM | LPIPS | Significant artifact | Significant ROI | Diagnostic micro | Micro ROI | Stand connector | Train time (s) | Total time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 40576 | 29.5363 | 0.6934 | 0.3946 | 0.000 | 0.000 | 0.256 | 0.000 | 0.000 | 180.2 | 1012.8 |

Previous leader for comparison:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_seed42_micro/lookcloser/arm_h40_grid128_transfix_coarse00625_minfreq4_huber_delta02_s42_micro39935_40000_save16/renders_artifact_selection_step-000039936`

Previous leader metrics: PSNR `29.5082`, SSIM `0.6857`, LPIPS `0.3964`, significant artifact `0.000`, diagnostic micro `0.691`, micro ROI/stand `0.255`.

Historical comparison render path for the old pre-fix H40 metric leader:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_continue36864_r4096/renders_full_step-000034816`

Old H40 metrics: PSNR `28.8982`, SSIM `0.6659`, LPIPS `0.3653`.

## Runs With Artifact ROI And Runtime Metrics

| Timestamp | Selection | Train Seconds | Eval Seconds | Artifact Seconds | Total Seconds | Artifact Score | Serious Artifact Score | ROI Artifact Score | ROI Serious Score | ROI Serious Count | Stand Connector | Params | Checkpoint | PSNR | SSIM | LPIPS | Eval JSON | Renders |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|
| arm_h40_grid128_huber_s42_from39936_occ0001_freshopt_lr5e5_39936_41216_save128 | best_artifact_checkpoint_step_40064 | 390.364 | 81.286531 | 45.826063 | 1701.235 | 0.260000 | 0.226000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `{"adaptive_coarse_step_size": 0.00625, "adaptive_fixed_fallback_samples_per_ray": 0, "adaptive_interval_level_mode": "midpoint", "adaptive_max_frequency_level": 13.0, "adaptive_max_step_size": 0.1, "adaptive_min_frequency_level": 4.0, "adaptive_min_step_size": 0.0001, "adaptive_warmup_steps": 0, "alpha_thre": 0.0, "appearance_embedding_dim": 0, "artifact_crop_bottom": 0, "artifact_crop_left": 0, "artifact_crop_right": 0, "artifact_crop_top": 0, "artifact_detector_preset": "micro", "artifact_render_names": ["eval_img_0000.png", "eval_img_0001.png", "eval_img_0002.png"], "artifact_roi_crop_names": ["left_stand_connector_eval0", "left_stand_eval0", "left_hand_background_eval0", "left_hand_outlet_stand_eval0", "floor_crack_eval0", "fingers_right_tight_eval1", "stand_label_eval2", "tangled_cable_eval2", "fingers_center_eval2"], "artifact_roi_drop_border_components": 0, "artifact_roi_score": true, "background_color": "black", "center_method": "focus", "color_num_layers": 2, "cone_angle": 0.0, "enable_adaptive_ray_marching": true, "enable_fas": false, "enable_feature_reweighting": false, "enable_frequency_grid": true, "eval_num_rays_per_batch": 4096, "eval_num_rays_per_chunk": 2048, "fallback_frequency_level": 0.0, "far_plane": 1000.0, "fas_decay_start_steps": -1, "fas_decay_steps": 0, "fas_level_count_alpha": 0.0, "fas_max_sampling_level": -1, "fas_patch_group_size": 1, "fas_ramp_steps": 0, "fas_strength": 1.0, "fas_warmup_steps": 0, "feature_reweighting_strength": 1.0, "fields_lr": 5e-05, "fields_lr_final": 5e-06, "fixed_num_samples_per_ray": 256, "frequency_map_dir": "lookcloser_frequencies", "geo_num_layers": 1, "grid_resolution": 128, "grid_update_batch_size": 2048, "grid_update_interval": 1024, "huber_delta": 0.2, "load_optimizers": false, "load_scheduler": false, "max_num_iterations": 41216, "max_res": 8192.0, "max_res_base": 2048.0, "max_steps_per_ray": 2048, "min_res": 16.0, "near_plane": 0.01, "num_frequency_levels": 16, "occupancy_binary_warmup_steps": 4096, "occupancy_dilation_radius": 0, "occupancy_ema_decay": 0.95, "occupancy_fixed_fallback_samples_per_ray": 0, "occupancy_grid_levels": 1, "occupancy_occ_thre": 0.0001, "occupancy_thre_clamp_mult": 1.0, "occupancy_update_interval": 16, "occupancy_update_step_size": null, "occupancy_warmup_steps": 4096, "orientation_method": "up", "ray_sampling_mode": "adaptive", "reconstruction_loss_type": "huber", "render_step_size": null, "render_step_size_mult": 1.0, "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.0, "scene_scale": 1.5, "seed": 143, "step_interval": 128, "train_num_rays_per_batch": 4096, "transmittance_threshold": 0.0, "use_gradient_scaling": false}` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_switch_boundary/lookcloser/arm_h40_grid128_huber_s42_from39936_occ0001_freshopt_lr5e5_39936_41216_save128/nerfstudio_models/step-000040064.ckpt` | 29.559740 | 0.682562 | 0.395536 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_switch_boundary/lookcloser/arm_h40_grid128_huber_s42_from39936_occ0001_freshopt_lr5e5_39936_41216_save128/eval_artifact_selection_step-000040064.json` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_switch_boundary/lookcloser/arm_h40_grid128_huber_s42_from39936_occ0001_freshopt_lr5e5_39936_41216_save128/renders_artifact_selection_step-000040064` |
| arm_h40_grid128_huber_s42_from39936_occ0001_freshopt_lr2e5_39936_41216_save128 | best_artifact_checkpoint_step_40448 | 420.442 | 87.180324 | 52.389800 | 1709.232 | 0.257000 | 0.224000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `{"adaptive_coarse_step_size": 0.00625, "adaptive_fixed_fallback_samples_per_ray": 0, "adaptive_interval_level_mode": "midpoint", "adaptive_max_frequency_level": 13.0, "adaptive_max_step_size": 0.1, "adaptive_min_frequency_level": 4.0, "adaptive_min_step_size": 0.0001, "adaptive_warmup_steps": 0, "alpha_thre": 0.0, "appearance_embedding_dim": 0, "artifact_crop_bottom": 0, "artifact_crop_left": 0, "artifact_crop_right": 0, "artifact_crop_top": 0, "artifact_detector_preset": "micro", "artifact_render_names": ["eval_img_0000.png", "eval_img_0001.png", "eval_img_0002.png"], "artifact_roi_crop_names": ["left_stand_connector_eval0", "left_stand_eval0", "left_hand_background_eval0", "left_hand_outlet_stand_eval0", "floor_crack_eval0", "fingers_right_tight_eval1", "stand_label_eval2", "tangled_cable_eval2", "fingers_center_eval2"], "artifact_roi_drop_border_components": 0, "artifact_roi_score": true, "background_color": "black", "center_method": "focus", "color_num_layers": 2, "cone_angle": 0.0, "enable_adaptive_ray_marching": true, "enable_fas": false, "enable_feature_reweighting": false, "enable_frequency_grid": true, "eval_num_rays_per_batch": 4096, "eval_num_rays_per_chunk": 2048, "fallback_frequency_level": 0.0, "far_plane": 1000.0, "fas_decay_start_steps": -1, "fas_decay_steps": 0, "fas_level_count_alpha": 0.0, "fas_max_sampling_level": -1, "fas_patch_group_size": 1, "fas_ramp_steps": 0, "fas_strength": 1.0, "fas_warmup_steps": 0, "feature_reweighting_strength": 1.0, "fields_lr": 2e-05, "fields_lr_final": 2e-06, "fixed_num_samples_per_ray": 256, "frequency_map_dir": "lookcloser_frequencies", "geo_num_layers": 1, "grid_resolution": 128, "grid_update_batch_size": 2048, "grid_update_interval": 1024, "huber_delta": 0.2, "load_optimizers": false, "load_scheduler": false, "max_num_iterations": 41216, "max_res": 8192.0, "max_res_base": 2048.0, "max_steps_per_ray": 2048, "min_res": 16.0, "near_plane": 0.01, "num_frequency_levels": 16, "occupancy_binary_warmup_steps": 4096, "occupancy_dilation_radius": 0, "occupancy_ema_decay": 0.95, "occupancy_fixed_fallback_samples_per_ray": 0, "occupancy_grid_levels": 1, "occupancy_occ_thre": 0.0001, "occupancy_thre_clamp_mult": 1.0, "occupancy_update_interval": 16, "occupancy_update_step_size": null, "occupancy_warmup_steps": 4096, "orientation_method": "up", "ray_sampling_mode": "adaptive", "reconstruction_loss_type": "huber", "render_step_size": null, "render_step_size_mult": 1.0, "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.0, "scene_scale": 1.5, "seed": 142, "step_interval": 128, "train_num_rays_per_batch": 4096, "transmittance_threshold": 0.0, "use_gradient_scaling": false}` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_switch_boundary/lookcloser/arm_h40_grid128_huber_s42_from39936_occ0001_freshopt_lr2e5_39936_41216_save128/nerfstudio_models/step-000040448.ckpt` | 29.583782 | 0.683489 | 0.395225 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_switch_boundary/lookcloser/arm_h40_grid128_huber_s42_from39936_occ0001_freshopt_lr2e5_39936_41216_save128/eval_artifact_selection_step-000040448.json` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_switch_boundary/lookcloser/arm_h40_grid128_huber_s42_from39936_occ0001_freshopt_lr2e5_39936_41216_save128/renders_artifact_selection_step-000040448` |
| arm_h40_grid128_huber_s42_from39936_occ0001_matchedmax_39936_41216_save128 | best_artifact_checkpoint_step_40064 | 420.449 | 103.230407 | 50.142943 | 1754.584 | 0.299000 | 0.299000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `{"adaptive_coarse_step_size": 0.00625, "adaptive_fixed_fallback_samples_per_ray": 0, "adaptive_interval_level_mode": "midpoint", "adaptive_max_frequency_level": 13.0, "adaptive_max_step_size": 0.00625, "adaptive_min_frequency_level": 4.0, "adaptive_min_step_size": 0.0001, "adaptive_warmup_steps": 0, "alpha_thre": 0.0, "appearance_embedding_dim": 0, "artifact_crop_bottom": 0, "artifact_crop_left": 0, "artifact_crop_right": 0, "artifact_crop_top": 0, "artifact_detector_preset": "micro", "artifact_render_names": ["eval_img_0000.png", "eval_img_0001.png", "eval_img_0002.png"], "artifact_roi_crop_names": ["left_stand_connector_eval0", "left_stand_eval0", "left_hand_background_eval0", "left_hand_outlet_stand_eval0", "floor_crack_eval0", "fingers_right_tight_eval1", "stand_label_eval2", "tangled_cable_eval2", "fingers_center_eval2"], "artifact_roi_drop_border_components": 0, "artifact_roi_score": true, "background_color": "black", "center_method": "focus", "color_num_layers": 2, "cone_angle": 0.0, "enable_adaptive_ray_marching": true, "enable_fas": false, "enable_feature_reweighting": false, "enable_frequency_grid": true, "eval_num_rays_per_batch": 4096, "eval_num_rays_per_chunk": 2048, "fallback_frequency_level": 0.0, "far_plane": 1000.0, "fas_decay_start_steps": -1, "fas_decay_steps": 0, "fas_level_count_alpha": 0.0, "fas_max_sampling_level": -1, "fas_patch_group_size": 1, "fas_ramp_steps": 0, "fas_strength": 1.0, "fas_warmup_steps": 0, "feature_reweighting_strength": 1.0, "fields_lr": null, "fields_lr_final": null, "fixed_num_samples_per_ray": 256, "frequency_map_dir": "lookcloser_frequencies", "geo_num_layers": 1, "grid_resolution": 128, "grid_update_batch_size": 2048, "grid_update_interval": 1024, "huber_delta": 0.2, "load_optimizers": true, "load_scheduler": true, "max_num_iterations": 41216, "max_res": 8192.0, "max_res_base": 2048.0, "max_steps_per_ray": 2048, "min_res": 16.0, "near_plane": 0.01, "num_frequency_levels": 16, "occupancy_binary_warmup_steps": 4096, "occupancy_dilation_radius": 0, "occupancy_ema_decay": 0.95, "occupancy_fixed_fallback_samples_per_ray": 0, "occupancy_grid_levels": 1, "occupancy_occ_thre": 0.0001, "occupancy_thre_clamp_mult": 1.0, "occupancy_update_interval": 16, "occupancy_update_step_size": null, "occupancy_warmup_steps": 4096, "orientation_method": "up", "ray_sampling_mode": "adaptive", "reconstruction_loss_type": "huber", "render_step_size": null, "render_step_size_mult": 1.0, "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.0, "scene_scale": 1.5, "seed": 42, "step_interval": 128, "train_num_rays_per_batch": 4096, "transmittance_threshold": 0.0, "use_gradient_scaling": false}` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_switch_boundary/lookcloser/arm_h40_grid128_huber_s42_from39936_occ0001_matchedmax_39936_41216_save128/nerfstudio_models/step-000040064.ckpt` | 29.565767 | 0.689498 | 0.393393 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_switch_boundary/lookcloser/arm_h40_grid128_huber_s42_from39936_occ0001_matchedmax_39936_41216_save128/eval_artifact_selection_step-000040064.json` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_switch_boundary/lookcloser/arm_h40_grid128_huber_s42_from39936_occ0001_matchedmax_39936_41216_save128/renders_artifact_selection_step-000040064` |
| arm_h40_grid128_huber_s47_occ0001_rng_from39936_200k_save512 | best_artifact_checkpoint_step_40448 | 3156.185 | 150.336901 | 48.955713 | 6692.347 | 0.306000 | 0.306000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `{"adaptive_coarse_step_size": 0.00625, "adaptive_fixed_fallback_samples_per_ray": 0, "adaptive_interval_level_mode": "midpoint", "adaptive_max_frequency_level": 13.0, "adaptive_max_step_size": 0.1, "adaptive_min_frequency_level": 4.0, "adaptive_min_step_size": 0.0001, "adaptive_warmup_steps": 0, "alpha_thre": 0.0, "appearance_embedding_dim": 0, "artifact_crop_bottom": 0, "artifact_crop_left": 0, "artifact_crop_right": 0, "artifact_crop_top": 0, "artifact_detector_preset": "micro", "artifact_render_names": ["eval_img_0000.png", "eval_img_0001.png", "eval_img_0002.png"], "artifact_roi_crop_names": ["left_stand_connector_eval0", "left_stand_eval0", "left_hand_background_eval0", "left_hand_outlet_stand_eval0", "floor_crack_eval0", "fingers_right_tight_eval1", "stand_label_eval2", "tangled_cable_eval2", "fingers_center_eval2"], "artifact_roi_drop_border_components": 0, "artifact_roi_score": true, "background_color": "black", "center_method": "focus", "color_num_layers": 2, "cone_angle": 0.0, "enable_adaptive_ray_marching": true, "enable_fas": false, "enable_feature_reweighting": false, "enable_frequency_grid": true, "eval_num_rays_per_batch": 4096, "eval_num_rays_per_chunk": 2048, "fallback_frequency_level": 0.0, "far_plane": 1000.0, "fas_decay_start_steps": -1, "fas_decay_steps": 0, "fas_level_count_alpha": 0.0, "fas_max_sampling_level": -1, "fas_patch_group_size": 1, "fas_ramp_steps": 0, "fas_strength": 1.0, "fas_warmup_steps": 0, "feature_reweighting_strength": 1.0, "fields_lr": null, "fields_lr_final": null, "fixed_num_samples_per_ray": 256, "frequency_map_dir": "lookcloser_frequencies", "geo_num_layers": 1, "grid_resolution": 128, "grid_update_batch_size": 2048, "grid_update_interval": 1024, "huber_delta": 0.2, "load_optimizers": true, "load_scheduler": true, "max_num_iterations": 200000, "max_res": 8192.0, "max_res_base": 2048.0, "max_steps_per_ray": 2048, "min_res": 16.0, "near_plane": 0.01, "num_frequency_levels": 16, "occupancy_binary_warmup_steps": 4096, "occupancy_dilation_radius": 0, "occupancy_ema_decay": 0.95, "occupancy_fixed_fallback_samples_per_ray": 0, "occupancy_grid_levels": 1, "occupancy_occ_thre": 0.0001, "occupancy_thre_clamp_mult": 1.0, "occupancy_update_interval": 16, "occupancy_update_step_size": null, "occupancy_warmup_steps": 4096, "orientation_method": "up", "ray_sampling_mode": "adaptive", "reconstruction_loss_type": "huber", "render_step_size": null, "render_step_size_mult": 1.0, "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.0, "scene_scale": 1.5, "seed": 47, "step_interval": 4096, "train_num_rays_per_batch": 4096, "transmittance_threshold": 0.0, "use_gradient_scaling": false}` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s47_occ0001_rng_from39936_200k_save512/nerfstudio_models/step-000040448.ckpt` | 29.538988 | 0.690759 | 0.395061 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s47_occ0001_rng_from39936_200k_save512/eval_artifact_selection_step-000040448.json` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s47_occ0001_rng_from39936_200k_save512/renders_artifact_selection_step-000040448` |
| arm_h40_grid128_huber_s45_occ0001_rng_from39936_200k_save512 | best_artifact_checkpoint_step_41472 | 4599.348 | 75.114780 | 49.553445 | 9025.493 | 0.323000 | 0.323000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `{"adaptive_coarse_step_size": 0.00625, "adaptive_fixed_fallback_samples_per_ray": 0, "adaptive_interval_level_mode": "midpoint", "adaptive_max_frequency_level": 13.0, "adaptive_max_step_size": 0.1, "adaptive_min_frequency_level": 4.0, "adaptive_min_step_size": 0.0001, "adaptive_warmup_steps": 0, "alpha_thre": 0.0, "appearance_embedding_dim": 0, "artifact_crop_bottom": 0, "artifact_crop_left": 0, "artifact_crop_right": 0, "artifact_crop_top": 0, "artifact_detector_preset": "micro", "artifact_render_names": ["eval_img_0000.png", "eval_img_0001.png", "eval_img_0002.png"], "artifact_roi_crop_names": ["left_stand_connector_eval0", "left_stand_eval0", "left_hand_background_eval0", "left_hand_outlet_stand_eval0", "floor_crack_eval0", "fingers_right_tight_eval1", "stand_label_eval2", "tangled_cable_eval2", "fingers_center_eval2"], "artifact_roi_drop_border_components": 0, "artifact_roi_score": true, "background_color": "black", "center_method": "focus", "color_num_layers": 2, "cone_angle": 0.0, "enable_adaptive_ray_marching": true, "enable_fas": false, "enable_feature_reweighting": false, "enable_frequency_grid": true, "eval_num_rays_per_batch": 4096, "eval_num_rays_per_chunk": 2048, "fallback_frequency_level": 0.0, "far_plane": 1000.0, "fas_decay_start_steps": -1, "fas_decay_steps": 0, "fas_level_count_alpha": 0.0, "fas_max_sampling_level": -1, "fas_patch_group_size": 1, "fas_ramp_steps": 0, "fas_strength": 1.0, "fas_warmup_steps": 0, "feature_reweighting_strength": 1.0, "fields_lr": null, "fields_lr_final": null, "fixed_num_samples_per_ray": 256, "frequency_map_dir": "lookcloser_frequencies", "geo_num_layers": 1, "grid_resolution": 128, "grid_update_batch_size": 2048, "grid_update_interval": 1024, "huber_delta": 0.2, "load_optimizers": true, "load_scheduler": true, "max_num_iterations": 200000, "max_res": 8192.0, "max_res_base": 2048.0, "max_steps_per_ray": 2048, "min_res": 16.0, "near_plane": 0.01, "num_frequency_levels": 16, "occupancy_binary_warmup_steps": 4096, "occupancy_dilation_radius": 0, "occupancy_ema_decay": 0.95, "occupancy_fixed_fallback_samples_per_ray": 0, "occupancy_grid_levels": 1, "occupancy_occ_thre": 0.0001, "occupancy_thre_clamp_mult": 1.0, "occupancy_update_interval": 16, "occupancy_update_step_size": null, "occupancy_warmup_steps": 4096, "orientation_method": "up", "ray_sampling_mode": "adaptive", "reconstruction_loss_type": "huber", "render_step_size": null, "render_step_size_mult": 1.0, "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.0, "scene_scale": 1.5, "seed": 45, "step_interval": 4096, "train_num_rays_per_batch": 4096, "transmittance_threshold": 0.0, "use_gradient_scaling": false}` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s45_occ0001_rng_from39936_200k_save512/nerfstudio_models/step-000041472.ckpt` | 29.482666 | 0.689474 | 0.397312 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s45_occ0001_rng_from39936_200k_save512/eval_artifact_selection_step-000041472.json` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s45_occ0001_rng_from39936_200k_save512/renders_artifact_selection_step-000041472` |
| arm_h40_grid128_huber_s46_occ0001_rng_from39936_200k_save512 | best_artifact_checkpoint_step_40960 | 4599.421 | 76.246392 | 50.770794 | 9106.836 | 0.314000 | 0.314000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `{"adaptive_coarse_step_size": 0.00625, "adaptive_fixed_fallback_samples_per_ray": 0, "adaptive_interval_level_mode": "midpoint", "adaptive_max_frequency_level": 13.0, "adaptive_max_step_size": 0.1, "adaptive_min_frequency_level": 4.0, "adaptive_min_step_size": 0.0001, "adaptive_warmup_steps": 0, "alpha_thre": 0.0, "appearance_embedding_dim": 0, "artifact_crop_bottom": 0, "artifact_crop_left": 0, "artifact_crop_right": 0, "artifact_crop_top": 0, "artifact_detector_preset": "micro", "artifact_render_names": ["eval_img_0000.png", "eval_img_0001.png", "eval_img_0002.png"], "artifact_roi_crop_names": ["left_stand_connector_eval0", "left_stand_eval0", "left_hand_background_eval0", "left_hand_outlet_stand_eval0", "floor_crack_eval0", "fingers_right_tight_eval1", "stand_label_eval2", "tangled_cable_eval2", "fingers_center_eval2"], "artifact_roi_drop_border_components": 0, "artifact_roi_score": true, "background_color": "black", "center_method": "focus", "color_num_layers": 2, "cone_angle": 0.0, "enable_adaptive_ray_marching": true, "enable_fas": false, "enable_feature_reweighting": false, "enable_frequency_grid": true, "eval_num_rays_per_batch": 4096, "eval_num_rays_per_chunk": 2048, "fallback_frequency_level": 0.0, "far_plane": 1000.0, "fas_decay_start_steps": -1, "fas_decay_steps": 0, "fas_level_count_alpha": 0.0, "fas_max_sampling_level": -1, "fas_patch_group_size": 1, "fas_ramp_steps": 0, "fas_strength": 1.0, "fas_warmup_steps": 0, "feature_reweighting_strength": 1.0, "fields_lr": null, "fields_lr_final": null, "fixed_num_samples_per_ray": 256, "frequency_map_dir": "lookcloser_frequencies", "geo_num_layers": 1, "grid_resolution": 128, "grid_update_batch_size": 2048, "grid_update_interval": 1024, "huber_delta": 0.2, "load_optimizers": true, "load_scheduler": true, "max_num_iterations": 200000, "max_res": 8192.0, "max_res_base": 2048.0, "max_steps_per_ray": 2048, "min_res": 16.0, "near_plane": 0.01, "num_frequency_levels": 16, "occupancy_binary_warmup_steps": 4096, "occupancy_dilation_radius": 0, "occupancy_ema_decay": 0.95, "occupancy_fixed_fallback_samples_per_ray": 0, "occupancy_grid_levels": 1, "occupancy_occ_thre": 0.0001, "occupancy_thre_clamp_mult": 1.0, "occupancy_update_interval": 16, "occupancy_update_step_size": null, "occupancy_warmup_steps": 4096, "orientation_method": "up", "ray_sampling_mode": "adaptive", "reconstruction_loss_type": "huber", "render_step_size": null, "render_step_size_mult": 1.0, "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.0, "scene_scale": 1.5, "seed": 46, "step_interval": 4096, "train_num_rays_per_batch": 4096, "transmittance_threshold": 0.0, "use_gradient_scaling": false}` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s46_occ0001_rng_from39936_200k_save512/nerfstudio_models/step-000040960.ckpt` | 29.498154 | 0.689820 | 0.396643 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s46_occ0001_rng_from39936_200k_save512/eval_artifact_selection_step-000040960.json` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_occ0001_rng_replay_200k/lookcloser/arm_h40_grid128_huber_s46_occ0001_rng_from39936_200k_save512/renders_artifact_selection_step-000040960` |
| arm_h40_grid128_huber_s42_freqloss050_l10_w150_from39936_200k_save512 | best_artifact_checkpoint_step_40960 | 4087.886 | 65.351514 | 44.725948 | 7908.198 | 0.316000 | 0.316000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `{"adaptive_coarse_step_size": 0.00625, "adaptive_fixed_fallback_samples_per_ray": 0, "adaptive_interval_level_mode": "midpoint", "adaptive_max_frequency_level": 13.0, "adaptive_max_step_size": 0.1, "adaptive_min_frequency_level": 4.0, "adaptive_min_step_size": 0.0001, "adaptive_warmup_steps": 0, "alpha_thre": 0.0, "appearance_embedding_dim": 0, "artifact_crop_bottom": 0, "artifact_crop_left": 0, "artifact_crop_right": 0, "artifact_crop_top": 0, "artifact_detector_preset": "micro", "artifact_render_names": ["eval_img_0000.png", "eval_img_0001.png", "eval_img_0002.png"], "artifact_roi_crop_names": ["left_stand_connector_eval0", "left_stand_eval0", "left_hand_background_eval0", "left_hand_outlet_stand_eval0", "floor_crack_eval0", "fingers_right_tight_eval1", "stand_label_eval2", "tangled_cable_eval2", "fingers_center_eval2"], "artifact_roi_drop_border_components": 0, "artifact_roi_score": true, "background_color": "black", "center_method": "focus", "color_num_layers": 2, "cone_angle": 0.0, "enable_adaptive_ray_marching": true, "enable_fas": false, "enable_feature_reweighting": false, "enable_frequency_grid": true, "eval_num_rays_per_batch": 4096, "eval_num_rays_per_chunk": 2048, "fallback_frequency_level": 0.0, "far_plane": 1000.0, "fas_decay_start_steps": -1, "fas_decay_steps": 0, "fas_level_count_alpha": 0.0, "fas_max_sampling_level": -1, "fas_patch_group_size": 1, "fas_ramp_steps": 0, "fas_strength": 1.0, "fas_warmup_steps": 0, "feature_reweighting_strength": 1.0, "fields_lr": null, "fields_lr_final": null, "fixed_num_samples_per_ray": 256, "frequency_loss_max_weight": 1.5, "frequency_loss_min_level": 10.0, "frequency_loss_weight_strength": 0.5, "frequency_map_dir": "lookcloser_frequencies", "geo_num_layers": 1, "grid_resolution": 128, "grid_update_batch_size": 2048, "grid_update_interval": 1024, "huber_delta": 0.2, "load_optimizers": true, "load_scheduler": true, "max_num_iterations": 200000, "max_res": 8192.0, "max_res_base": 2048.0, "max_steps_per_ray": 2048, "min_res": 16.0, "near_plane": 0.01, "num_frequency_levels": 16, "occupancy_binary_warmup_steps": 4096, "occupancy_dilation_radius": 0, "occupancy_ema_decay": 0.95, "occupancy_fixed_fallback_samples_per_ray": 0, "occupancy_grid_levels": 1, "occupancy_occ_thre": 0.0001, "occupancy_thre_clamp_mult": 1.0, "occupancy_update_interval": 16, "occupancy_update_step_size": null, "occupancy_warmup_steps": 4096, "orientation_method": "up", "ray_sampling_mode": "adaptive", "reconstruction_loss_type": "huber", "render_step_size": null, "render_step_size_mult": 1.0, "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.0, "scene_scale": 1.5, "seed": 42, "step_interval": 4096, "train_num_rays_per_batch": 4096, "transmittance_threshold": 0.0, "use_gradient_scaling": false}` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss050_l10_w150_from39936_200k_save512/nerfstudio_models/step-000040960.ckpt` | 29.484396 | 0.690296 | 0.393510 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss050_l10_w150_from39936_200k_save512/eval_artifact_selection_step-000040960.json` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss050_l10_w150_from39936_200k_save512/renders_artifact_selection_step-000040960` |
| arm_h40_grid128_huber_s42_freqloss100_l12_w150_from39936_200k_save512 | best_artifact_checkpoint_step_40448 | 4118.392 | 92.983139 | 50.332060 | 7948.048 | 0.330000 | 0.307000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `{"adaptive_coarse_step_size": 0.00625, "adaptive_fixed_fallback_samples_per_ray": 0, "adaptive_interval_level_mode": "midpoint", "adaptive_max_frequency_level": 13.0, "adaptive_max_step_size": 0.1, "adaptive_min_frequency_level": 4.0, "adaptive_min_step_size": 0.0001, "adaptive_warmup_steps": 0, "alpha_thre": 0.0, "appearance_embedding_dim": 0, "artifact_crop_bottom": 0, "artifact_crop_left": 0, "artifact_crop_right": 0, "artifact_crop_top": 0, "artifact_detector_preset": "micro", "artifact_render_names": ["eval_img_0000.png", "eval_img_0001.png", "eval_img_0002.png"], "artifact_roi_crop_names": ["left_stand_connector_eval0", "left_stand_eval0", "left_hand_background_eval0", "left_hand_outlet_stand_eval0", "floor_crack_eval0", "fingers_right_tight_eval1", "stand_label_eval2", "tangled_cable_eval2", "fingers_center_eval2"], "artifact_roi_drop_border_components": 0, "artifact_roi_score": true, "background_color": "black", "center_method": "focus", "color_num_layers": 2, "cone_angle": 0.0, "enable_adaptive_ray_marching": true, "enable_fas": false, "enable_feature_reweighting": false, "enable_frequency_grid": true, "eval_num_rays_per_batch": 4096, "eval_num_rays_per_chunk": 2048, "fallback_frequency_level": 0.0, "far_plane": 1000.0, "fas_decay_start_steps": -1, "fas_decay_steps": 0, "fas_level_count_alpha": 0.0, "fas_max_sampling_level": -1, "fas_patch_group_size": 1, "fas_ramp_steps": 0, "fas_strength": 1.0, "fas_warmup_steps": 0, "feature_reweighting_strength": 1.0, "fields_lr": null, "fields_lr_final": null, "fixed_num_samples_per_ray": 256, "frequency_loss_max_weight": 1.5, "frequency_loss_min_level": 12.0, "frequency_loss_weight_strength": 1.0, "frequency_map_dir": "lookcloser_frequencies", "geo_num_layers": 1, "grid_resolution": 128, "grid_update_batch_size": 2048, "grid_update_interval": 1024, "huber_delta": 0.2, "load_optimizers": true, "load_scheduler": true, "max_num_iterations": 200000, "max_res": 8192.0, "max_res_base": 2048.0, "max_steps_per_ray": 2048, "min_res": 16.0, "near_plane": 0.01, "num_frequency_levels": 16, "occupancy_binary_warmup_steps": 4096, "occupancy_dilation_radius": 0, "occupancy_ema_decay": 0.95, "occupancy_fixed_fallback_samples_per_ray": 0, "occupancy_grid_levels": 1, "occupancy_occ_thre": 0.0001, "occupancy_thre_clamp_mult": 1.0, "occupancy_update_interval": 16, "occupancy_update_step_size": null, "occupancy_warmup_steps": 4096, "orientation_method": "up", "ray_sampling_mode": "adaptive", "reconstruction_loss_type": "huber", "render_step_size": null, "render_step_size_mult": 1.0, "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.0, "scene_scale": 1.5, "seed": 42, "step_interval": 4096, "train_num_rays_per_batch": 4096, "transmittance_threshold": 0.0, "use_gradient_scaling": false}` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss100_l12_w150_from39936_200k_save512/nerfstudio_models/step-000040448.ckpt` | 29.526915 | 0.690720 | 0.392647 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss100_l12_w150_from39936_200k_save512/eval_artifact_selection_step-000040448.json` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss100_l12_w150_from39936_200k_save512/renders_artifact_selection_step-000040448` |
| arm_h40_grid128_huber_s42_freqloss025_l10_w125_from39936_200k_save512 | best_artifact_checkpoint_step_40448 | 4118.063 | 96.200791 | 49.004131 | 7970.686 | 0.328000 | 0.305000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `{"adaptive_coarse_step_size": 0.00625, "adaptive_fixed_fallback_samples_per_ray": 0, "adaptive_interval_level_mode": "midpoint", "adaptive_max_frequency_level": 13.0, "adaptive_max_step_size": 0.1, "adaptive_min_frequency_level": 4.0, "adaptive_min_step_size": 0.0001, "adaptive_warmup_steps": 0, "alpha_thre": 0.0, "appearance_embedding_dim": 0, "artifact_crop_bottom": 0, "artifact_crop_left": 0, "artifact_crop_right": 0, "artifact_crop_top": 0, "artifact_detector_preset": "micro", "artifact_render_names": ["eval_img_0000.png", "eval_img_0001.png", "eval_img_0002.png"], "artifact_roi_crop_names": ["left_stand_connector_eval0", "left_stand_eval0", "left_hand_background_eval0", "left_hand_outlet_stand_eval0", "floor_crack_eval0", "fingers_right_tight_eval1", "stand_label_eval2", "tangled_cable_eval2", "fingers_center_eval2"], "artifact_roi_drop_border_components": 0, "artifact_roi_score": true, "background_color": "black", "center_method": "focus", "color_num_layers": 2, "cone_angle": 0.0, "enable_adaptive_ray_marching": true, "enable_fas": false, "enable_feature_reweighting": false, "enable_frequency_grid": true, "eval_num_rays_per_batch": 4096, "eval_num_rays_per_chunk": 2048, "fallback_frequency_level": 0.0, "far_plane": 1000.0, "fas_decay_start_steps": -1, "fas_decay_steps": 0, "fas_level_count_alpha": 0.0, "fas_max_sampling_level": -1, "fas_patch_group_size": 1, "fas_ramp_steps": 0, "fas_strength": 1.0, "fas_warmup_steps": 0, "feature_reweighting_strength": 1.0, "fields_lr": null, "fields_lr_final": null, "fixed_num_samples_per_ray": 256, "frequency_loss_max_weight": 1.25, "frequency_loss_min_level": 10.0, "frequency_loss_weight_strength": 0.25, "frequency_map_dir": "lookcloser_frequencies", "geo_num_layers": 1, "grid_resolution": 128, "grid_update_batch_size": 2048, "grid_update_interval": 1024, "huber_delta": 0.2, "load_optimizers": true, "load_scheduler": true, "max_num_iterations": 200000, "max_res": 8192.0, "max_res_base": 2048.0, "max_steps_per_ray": 2048, "min_res": 16.0, "near_plane": 0.01, "num_frequency_levels": 16, "occupancy_binary_warmup_steps": 4096, "occupancy_dilation_radius": 0, "occupancy_ema_decay": 0.95, "occupancy_fixed_fallback_samples_per_ray": 0, "occupancy_grid_levels": 1, "occupancy_occ_thre": 0.0001, "occupancy_thre_clamp_mult": 1.0, "occupancy_update_interval": 16, "occupancy_update_step_size": null, "occupancy_warmup_steps": 4096, "orientation_method": "up", "ray_sampling_mode": "adaptive", "reconstruction_loss_type": "huber", "render_step_size": null, "render_step_size_mult": 1.0, "sampling_ramp_end": 3.0, "sampling_ramp_start": 1.0, "scale_factor": 1.0, "scene_scale": 1.5, "seed": 42, "step_interval": 4096, "train_num_rays_per_batch": 4096, "transmittance_threshold": 0.0, "use_gradient_scaling": false}` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss025_l10_w125_from39936_200k_save512/nerfstudio_models/step-000040448.ckpt` | 29.524557 | 0.690791 | 0.392610 | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss025_l10_w125_from39936_200k_save512/eval_artifact_selection_step-000040448.json` | `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_freqloss_200k/lookcloser/arm_h40_grid128_huber_s42_freqloss025_l10_w125_from39936_200k_save512/renders_artifact_selection_step-000040448` |
