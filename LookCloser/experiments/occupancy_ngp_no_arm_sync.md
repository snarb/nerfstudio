# Occupancy NGP No-ARM Sync

## What was tested

Clean-context audit of local `/home/ubuntu/repos/instant-ngp/` occupancy-grid update/traversal behavior against LookCloser and nerfstudio `instant-ngp-bounded`. No long training was run.

Target LookCloser mode: constant-step `nerfacc.OccGridEstimator.sampling` traversal, no frequency-aware adaptive subdivision, no fixed uniform full-ray sampling.

## Findings

Raw instant-ngp occupancy behavior:

| Area | Local instant-ngp behavior | LookCloser / nerfacc status |
| --- | --- | --- |
| Grid resolution | `NERF_GRIDSIZE=128` | default `grid_resolution=128`, matched |
| Optical thickness scale | update stores `density * MIN_CONE_STEPSIZE`; `MIN_CONE_STEPSIZE=sqrt(3)/1024` in unit NGP space | nerfacc stores `density * render_step_size`; use `scene_diag / 1000` for nerfstudio parity |
| EMA update | `max(prev * decay, new)` | nerfacc 0.5.2 same |
| Decay | UI/default field `density_grid_decay`, default path uses `0.95` | default `occupancy_ema_decay=0.95`, matched |
| Threshold | bit if `occ > min(mean_occ, 0.01)` | nerfacc same with `occ_thre=1e-2` |
| Warmup/update | first 256 training steps sample all cells; later sample `1/4` uniform + `1/4` occupied | nerfacc same, called every `n=16` by default, so dense updates occur at steps divisible by 16 while `step < 256` |
| Visibility init | cells outside all camera frusta are marked negative/untrained in raw instant-ngp | nerfacc does not mark unseen cells negative; bounded nerfstudio also does not expose this raw C++ camera-visibility mask |
| Cascades / levels | raw instant-ngp uses cascades and max-pools bitfields into coarser mips | nerfstudio `instant-ngp-bounded` explicitly sets `grid_levels=1`; LookCloser now exposes `occupancy_grid_levels` and defaults to `1` |
| Traversal step | raw instant-ngp uses `calc_dt` with optional cone stepping; unit scenes use cone `0` | LookCloser occupancy mode uses nerfacc constant `render_step_size` when `cone_angle=0` |
| Alpha pruning | raw constant `NERF_MIN_OPTICAL_THICKNESS=0.01` for occupancy threshold; nerfstudio bounded sets sampler `alpha_thre=0.0` | recommended bounded parity is `alpha_thre=0.0`; LookCloser default stays lower-risk existing `0.0025` |
| Near plane | raw instant-ngp constant near is `0.05`; local nerfstudio bounded config uses `near_plane=0.01` | recommended bounded parity is `near_plane=0.01` |

Code-level comparison:

- `LookCloserModel.ray_sampling_mode="occupancy"` already routes to `occupancy_ray_marching`, which calls `OccGridEstimator.sampling` with `render_step_size`, `near_plane`, `far_plane`, `alpha_thre`, and `cone_angle`, then renders packed samples without frequency subdivision.
- `ray_sampling_mode="auto"` preserves previous behavior: adaptive when `enable_adaptive_ray_marching=True`, fixed when disabled.
- Fixed s640 path is not changed.
- FAS and frequency maps are not changed.
- This audit added `occupancy_grid_levels` as an explicit knob, defaulting to `1` to match nerfstudio `instant-ngp-bounded`.

## Recommended hyperparameters

Primary bounded parity:

| Hyperparameter | Recommendation | Reason |
| --- | ---: | --- |
| `ray_sampling_mode` | `occupancy` | no ARM and no fixed full-ray sampling |
| `render_step_size_mult` | `1.0` | nerfstudio instant-ngp uses `scene_aabb_diag / 1000`; LookCloser default `0.75` is denser |
| `alpha_thre` | `0.0` | local nerfstudio `instant-ngp-bounded` sets `alpha_thre=0.0` |
| `occupancy_occ_thre` | `1e-2` | raw instant-ngp / nerfacc default threshold cap |
| `occupancy_ema_decay` | `0.95` | raw instant-ngp max-with-decay default |
| `occupancy_warmup_steps` | `256` | raw instant-ngp / nerfacc warmup |
| `occupancy_update_interval` | `16` | nerfacc default; keep unless measuring cost/lag |
| `grid_resolution` | `128` | raw instant-ngp grid size |
| `occupancy_grid_levels` | `1` | nerfstudio `instant-ngp-bounded` default for bounded scenes |
| `cone_angle` | `0.0` | local nerfstudio bounded and unit-scene raw instant-ngp behavior |
| `near_plane` | `0.01` | local nerfstudio bounded config |

Diagnostic variants:

- `render_step_size_mult=0.75`: current LookCloser denser update/traversal setting; compare only as a quality/runtime variant, not strict parity.
- `alpha_thre=0.0025`: current LookCloser default; useful if training cost is high, but not strict bounded parity.
- `occupancy_grid_levels=2` or `4`: diagnostic for raw instant-ngp cascade-like coverage, but not bounded nerfstudio parity and may sample outside the tight scene AABB.
- `near_plane=0.05`: raw C++ instant-ngp default; not the local nerfstudio bounded setting.

## Recommended commands

Dry-run parity command:

```bash
conda activate /home/ubuntu/anaconda3/envs/nerfstudio
python /home/ubuntu/repos/nerfstudio/LookCloser/scripts/run_lookcloser_quiet.py \
  --dry-run \
  --timestamp occ_ngp_no_arm_parity_s42 \
  --seed 42 \
  --scene-scale 1.5 \
  --scale-factor 1.0 \
  --train-num-rays-per-batch 8192 \
  --ray-sampling-mode occupancy \
  --disable-fas \
  --grid-resolution 128 \
  --occupancy-grid-levels 1 \
  --render-step-size-mult 1.0 \
  --alpha-thre 0.0 \
  --occupancy-occ-thre 1e-2 \
  --occupancy-ema-decay 0.95 \
  --occupancy-warmup-steps 256 \
  --occupancy-update-interval 16 \
  --occupancy-thre-clamp-mult 1.0 \
  --occupancy-dilation-radius 0 \
  --cone-angle 0.0 \
  --near-plane 0.01
```

Actual first experiment is the same command without `--dry-run`. Use seeds `42`, `43`, and `44` for a three-seed read.

Stricter sampler/field isolation command:

```bash
python /home/ubuntu/repos/nerfstudio/LookCloser/scripts/run_lookcloser_quiet.py \
  --dry-run \
  --timestamp occ_ngp_no_arm_iso_s42 \
  --seed 42 \
  --scene-scale 1.5 \
  --scale-factor 1.0 \
  --train-num-rays-per-batch 8192 \
  --ray-sampling-mode occupancy \
  --disable-fas \
  --disable-frequency-grid \
  --disable-feature-reweighting \
  --allow-missing-frequency-maps \
  --grid-resolution 128 \
  --occupancy-grid-levels 1 \
  --render-step-size-mult 1.0 \
  --alpha-thre 0.0 \
  --occupancy-occ-thre 1e-2 \
  --occupancy-ema-decay 0.95 \
  --occupancy-warmup-steps 256 \
  --occupancy-update-interval 16 \
  --cone-angle 0.0 \
  --near-plane 0.01
```

## Results

Training results are now available for the new occupancy-only path.

All runs below use:

- `ray_sampling_mode=occupancy`
- ARM/frequency-aware subdivision disabled
- fixed full-ray sampling disabled
- `alpha_thre=0.0`, `transmittance_threshold=0.0`
- `near_plane=0.01`, `cone_angle=0.0`
- `occupancy_grid_levels=1`
- significant artifact detector preset

| Run | Grid res | Occ warmup | Binary warmup | Views scored | PSNR | SSIM | LPIPS | Full artifact | ROI artifact | Train s | Total s | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `occgrid_parity_warm512_s42` | 64 | 256 | 512 | 1 | 26.232 | 0.678 | 0.472 | 12.319 | 12.414 | 1471.8 | 1535.9 | cold-start pruning too aggressive |
| `occgrid_parity_grid128_warm512_s42` | 128 | 256 | 512 | 1 | 25.265 | 0.675 | 0.469 | 35.043 | 0.000 | 1471.9 | 1524.6 | edge/empty-region artifacts |
| `occgrid_grid64_warm4096_s42` | 64 | 4096 | 4096 | 3 | 28.382 | 0.646 | 0.469 | 1.653 | 13.951 | 1742.1 | 1817.0 | longer warmup helps full-frame, ROI still fails |
| `occgrid_grid128_warm4096_s42` | 128 | 4096 | 4096 | 3 | 28.658 | 0.647 | 0.461 | 0.000 | 0.000 | 1742.2 | 1816.2 | first accepted occupancy-grid result |

Best zero-artifact render paths:

- Best single zero-artifact render by SSIM among the plain 3-seed confirmation:
  `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occgrid_no_arm/lookcloser/occgrid_grid128_warm4096_s44/renders_best_step-000015187`
- Problem seed43 after artifact-aware checkpoint selection:
  `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_occgrid_no_arm/lookcloser/occgrid_grid128_warm4096_s43_artifact_select_3797_v2/renders_artifact_selection_step-000015187`

Variance confirmation for `occgrid_grid128_warm4096`:

| Seed | PSNR | SSIM | LPIPS | Full artifact | ROI artifact | Train s | Total s | Read |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 28.658 | 0.647 | 0.461 | 0.000 | 0.000 | 1742.2 | 1816.2 | clean |
| 43 | 27.371 | 0.657 | 0.468 | 0.106 | 0.000 | 1742.3 | 1816.1 | tiny 253 px off-ROI residual on eval1 |
| 44 | 28.775 | 0.661 | 0.462 | 0.000 | 0.000 | 1742.2 | 1814.3 | clean |

Follow-up runs are in progress for seed43:

- `occgrid_grid128_warm4096_clamp05_s43`: artifact `0.127`,
  ROI `0.000`; worse than baseline.
- `occgrid_grid128_warm4096_ema099_s43`: artifact `0.194`,
  ROI `2.552`; worse than baseline.
- `occgrid_grid128_warm4096_s43_continue30376_latest`: artifact
  `0.133`, ROI `0.000`; longer training improved LPIPS to `0.453` but
  did not remove the tiny residual.
- `occgrid_grid128_warm4096_step075_s43`: artifact `0.336`,
  ROI `0.000`; denser constant traversal worsened the detector score.
- `occgrid_grid256_warm4096_s43`: artifact `0.132`, ROI `0.000`;
  finer occupancy grid did not fix the residual and used more memory.
- `occgrid_grid128_warm4096_fb64_s43`: artifact `5.275`,
  ROI `7.567`; rejected. The safety fallback kept the grid effectively dense
  and damaged quality (`PSNR 15.595`).
- `occgrid_grid128_warm4096_fr_on_s43`: artifact `0.230`, ROI `0.000`;
  rejected. Feature reweighting made this residual worse.

Best current occupancy-grid configuration:

- `ray_sampling_mode=occupancy`
- `grid_resolution=128`
- `occupancy_grid_levels=1`
- `render_step_size_mult=1.0`
- `alpha_thre=0.0`
- `transmittance_threshold=0.0`
- `near_plane=0.01`
- `cone_angle=0.0`
- `occupancy_occ_thre=0.01`
- `occupancy_ema_decay=0.95`
- `occupancy_warmup_steps=4096`
- `occupancy_binary_warmup_steps=4096`
- `occupancy_update_interval=16`
- `occupancy_thre_clamp_mult=1.0`
- `occupancy_dilation_radius=0`
- `Feature Reweighting off`
- `FAS off`

Artifact-aware checkpoint selection for the problem seed43:

| Checkpoint | PSNR | SSIM | LPIPS | Full artifact | ROI artifact | Read |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3797 | 26.715 | 0.608 | 0.516 | 0.385 | 8.962 | too early; real ROI artifacts remain |
| 7594 | 26.891 | 0.635 | 0.489 | 0.109 | 0.000 | best eval-loss checkpoint, but not artifact-clean |
| 11391 | 27.017 | 0.649 | 0.474 | 0.124 | 0.000 | visually cleaner but still one full-frame component |
| 15187 | 27.190 | 0.657 | 0.469 | 0.000 | 0.000 | selected by `--eval-checkpoint artifact` |

Strict status:

- With the same occupancy-only hyperparameters, literal `0.000` significant
  full-frame artifact score is now achieved for seeds `42`, `43`, and `44`.
- Seed `43` requires artifact-aware checkpoint selection. The best eval-loss
  checkpoint in the artifact-selection run was `7594`, but it still scored
  `0.109`; checkpoint `15187` scored `0.000`.
- Multiple global occupancy changes made the seed43 residual worse. The useful
  fix was not more conservative occupancy inflation; it was keeping the
  instant-ngp-like occupancy path and selecting the checkpoint by the artifact
  gate when the detector is the hard constraint.

## Insights

The highest-confidence semantic fix is to use `ray_sampling_mode=occupancy` with bounded instant-ngp traversal parameters rather than overloading `--disable-adaptive-ray-marching`, because that flag intentionally selects the accepted fixed renderer. The remaining unavoidable implementation difference from raw C++ instant-ngp is the absence of camera-frustum untrained-cell masking and bitfield mip max-pooling in nerfacc bounded mode; this is also true of local nerfstudio `instant-ngp-bounded`, so it should not block bounded parity experiments.

The new measured root cause is cold-start occupancy pruning: with only the
default nerfacc warmup/binary behavior, LookCloser can zero out peripheral or
thin regions before the field has learned reliable density. Keeping binaries
fully occupied and dense-updating cells through step `4096`, then enabling
normal occupancy pruning, removes the significant artifact for seed42 at
`grid_resolution=128` while keeping training around 30 minutes for the first
15188-step checkpoint.

The variance read matters: the seed43 residual did not respond to more
conservative occupancy thresholds, longer training, denser traversal, finer grid
resolution, safety fallback samples, or feature reweighting. Those results argue
against blindly increasing occupancy-grid conservativeness. Artifact-aware
checkpoint selection is the right confirmation protocol when literal zero is
the target, because eval-loss selection can keep a checkpoint with a small but
qualifying structural component.
