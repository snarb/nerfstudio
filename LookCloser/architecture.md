## Key files:

- `nerfstudio/scripts/lookcloser_preprocess.py` — 2D patch frequency preprocessing.
- `nerfstudio/scripts/lookcloser_debug_preprocess.py` — focused standalone preprocessing debug checks.
- `nerfstudio/model_components/lookcloser_grid.py` — 3D frequency grid.
- `nerfstudio/model_components/lookcloser_samplers.py` — packed frequency-aware adaptive sampler.
- `nerfstudio/fields/lookcloser_field.py` — frequency-aware field.
- `nerfstudio/lookcloser_pixel_sampler.py` — frequency-aware sampler.
- `nerfstudio/models/lookcloser.py` — LookCloser model and adaptive ray marching.
- `nerfstudio/pipelines/lookcloser_pipeline.py` — training pipeline and grid updates.
- `nerfstudio/configs/method_configs.py` — `lookcloser` method config.

## Training monitoring additions

Baseline runs can enable `--logging.csv-writer.enable True` to write compact `metrics_compact.csv` rows for train/eval trends, `best_eval_*`, plateau and overfit status, which is useful because recent 3k baselines plateau early and best checkpoint metrics are more informative than final-step metrics.

LookCloser frequency-grid experiments use `scripts/run_lookcloser_preprocess_quiet.py` for dataset frequency-map generation, `scripts/run_lookcloser_quiet.py` for one quiet run, and `scripts/run_lookcloser_sweep.py` for 3-seed candidate sweeps. These mirror the bounded Instant-NGP protocol but add training-time capture, best-per-metric reporting, frequency-grid/FAS flags, and a hard preflight check for dataset `lookcloser_frequencies` unless explicitly bypassed for smoke tests. The preprocessing wrapper keeps the tuned visual settings from `experiments/preprocess_heatmap_hyperparameter_tuning_6k.md`: `patch_size=8`, `ssim_window_size=7`, and `high_frequency_level=13`. Full-HD dataset maps use `train_steps_per_level=1000` and `ssim_threshold=0.95`; single full-image checks found `0.97` over-assigned max level and `0.93` likely under-labeled high frequencies.

Experiment decisions consider SSIM, LPIPS, and PSNR together. Sweep reports rank by mean SSIM first, then mean LPIPS, then mean PSNR, with eval loss and training time used as supporting signals; no runs are rerun solely to backfill this policy.

Manual guarded seed loops can be summarized with `scripts/summarize_lookcloser_runs.py <experiment_dir>` once all three `run_summary.json` files exist. The helper reads the selected best-eval-loss checkpoint summaries plus `metrics_compact.csv`, then emits per-run metrics, 3-seed means, and best single results for SSIM, LPIPS, PSNR, eval loss, and training time.

Use `scripts/inspect_lookcloser_frequency_maps.py` after preprocessing to record map count, metadata count, level histograms, and max-level fraction before starting training sweeps.

Adaptive Ray Marching now uses a packed nerfacc-compatible path instead of the original Python per-step field-query loop. `FrequencyAwareVolumetricSampler` first traverses the occupancy grid at the adaptive max interval, then vectorizes paper-style interval subdivision from the frequency grid. The paper interval `1 / (2 * N_l)` is treated as normalized AABB/hash-grid distance and converted to ray `t` units with `dt = dt_norm / ||ray_dir / aabb_size||`. A scalar longest-side diagnostic produced indistinguishable crops from this per-axis AABB conversion and was rolled back. A packed linear-time distortion loss replaces padded max-step history for adaptive outputs. Isolated smoke with Frequency Grid on, Adaptive RM on, Feature Re-weighting off, and FAS off completed 16 iterations at 64 rays/chunk with no full-image eval; after startup, iteration time dropped to about `0.11s`, train samples averaged about `33` per ray, and saturation was `0`.

The current isolated Interval Adjustment metric leader is `adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_continue36864_r4096`, checkpoint step `34816`: PSNR `28.8982`, SSIM `0.6659`, LPIPS `0.3653`, with renders in `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_continue36864_r4096/renders_full_step-000034816`. The best visual-balance candidate is `adaptive_fg_arm_iso_h41_batch8192_coarse00125_continue34816_r8192`, checkpoint step `34815`: PSNR `28.8879`, SSIM `0.6660`, LPIPS `0.3664`. Both keep Frequency Grid and Adaptive RM enabled with Feature Re-weighting and FAS disabled; the next module test should enable FAS alone before testing Feature Re-weighting.

Follow-up Interval Adjustment tuning found that adaptive-from-scratch was the core failure mode: empty/noisy frequency grids and weak early density caused underfit or sample-cap saturation. `LookCloserModelConfig.adaptive_warmup_steps` now allows fixed ray marching during initial training before switching to adaptive marching, and `adaptive_min_frequency_level` / `adaptive_max_frequency_level` bound interval sizing without changing stored frequency-grid values. The first stable isolated candidate used `grid_resolution=64`, `adaptive_warmup_steps=2048`, and `adaptive_max_frequency_level=12`, reaching compact eval-batch loss `0.0294421` in `adaptive_fg_arm_iso_h20_grid64_warmup2048_maxfreq12_continue30376_r4096`. A crop-gate bug was later fixed: `scripts/render_lookcloser_crop_gate.py` now passes nerfstudio camera coordinates as `(row/y, col/x)` and defaults to the current bounded Instant-NGP render directory from `experiments/bounded_ngp_param_sweep.md`. Under the corrected gate, reducing only nerfacc coarse traversal improved results: `adaptive_coarse_step_size=0.025` reached compact loss `0.0284642` and improved stand-label/center-finger crops in `adaptive_fg_arm_iso_h27_coarse0025_continue32768_r4096`; `0.0125` reached compact loss `0.0282583` and gave the best cable/center-finger SSIM so far in `adaptive_fg_arm_iso_h28_coarse00125_continue33792_r4096`, with a slight stand-label/right-hand tradeoff. A `0.00625` smoke was mostly flat and rejected. Later H40/H41 runs superseded the H27/H28 bracket for metric and visual-balance selection, though stand-label and tangled-cable crops still remain below the matched Instant-NGP baseline. A 12288-ray continuation from H20 was rejected because it worsened compact eval loss and did not improve the crop gate overall.

A density-activation parity diagnostic temporarily tested Instant-NGP-style `trunc_exp` in `LookCloserField`; it regressed early eval loss and high-frequency crops under the isolated adaptive setup, so it was rolled back. The carried LookCloser density activation remains `softplus(h + 1)`.

Additional corrected-gate diagnostics rejected `geo_num_layers=2` and the older `grid_resolution=128` / `adaptive_max_frequency_level=14` branch. Geo depth 2 gave one center-hand crop gain at an early warmup checkpoint but regressed label, cable, right-hand, and floor crops; maxfreq14/grid128 was worse than H20 latest on every target crop SSIM.

`LookCloserField` also exposes optional `appearance_embedding_dim` for baseline-capacity experiments. It defaults to `0`; a 32D appearance-embedding test with the isolated adaptive setup regressed early eval loss and was rejected.

Because `/fsx` reached 100% usage during the first baseline sweep attempt, `scripts/run_lookcloser_sweep.py` defaults its output directory to local `LookCloser/repro_runs/lookcloser_runs`. The dataset and frequency maps still live on `/fsx`.

Runtime grid update sampling in `nerfstudio/pipelines/lookcloser_pipeline.py` now handles per-ray tensor image bounds using `torch.minimum(..., H/W - 1)` instead of passing tensor bounds to `torch.clamp`, and keeps camera-index tensors on CPU for `cameras.fx/fy` indexing before moving generated rays to CUDA. These fixes were verified by a 1030-step frequency-grid smoke run that crossed the first grid update at step 1024.

Implementation-doubt follow-up: a paper-aligned runtime update that sampled frequency-map patch cells directly, rendered the center pixel of each sampled patch, and updated from that patch's scalar 2D frequency was tested in `lookcloser_fix_runtime_patch_centers`. The 3-seed experiment regressed mean SSIM/LPIPS/PSNR, so the runtime update sampler was restored to the previous arbitrary-pixel sampling plus pixel-to-frequency-map lookup.

Implementation-doubt follow-up: FAS bucket sampling was tested with probabilities renormalized over only non-empty frequency buckets in `lookcloser_fix_fas_nonempty_buckets`. It improved mean LPIPS, PSNR, and eval loss but regressed the SSIM-first decision metric, so the sampler was restored to the previous all-level probability normalization.

Implementation-doubt follow-up: sparse SfM frequency-grid initialization was tested in `lookcloser_fix_sparse_init` by loading COLMAP point observations, mapping `colmap_im_id` values to train images with frequency maps, and transforming COLMAP points through the nerfstudio dataparser transform/scale. The smoke touched 2,272 unique voxels, but the 3-seed experiment regressed mean SSIM, LPIPS, and PSNR versus the carried no-sparse reference, so the sparse-init hook was reverted.

Implementation-doubt follow-up: sparse depth supervision was tested with explicit COLMAP sparse-depth `.npy` maps, a sparse-depth-only dataset path that refused Zoe/pseudo-depth fallback, `depth_loss_steps` gating, and paper-style Charbonnier depth loss. The generated experiment dataset had 66 train sparse-depth maps, 3 zero eval placeholders for dataparser compatibility, and 222,001 supervised train pixels. The smoke verified finite depth loss before the gate and no depth term after it. The 3-seed quality run regressed mean SSIM, LPIPS, PSNR, and eval loss versus the carried reference, so all transient sparse-depth code was reverted and only the report/artifacts remain in `experiments/lookcloser_fix_sparse_depth_supervision.md`.

`nerfstudio/models/lookcloser.py` reports PSNR, SSIM, and LPIPS for image evals, matching the metric surface expected by the frequency-grid sweeps. `scripts/run_lookcloser_sweep.py` can recover `train_seconds` from the quiet wrapper log when a run completed but crashed before writing `run_summary.json`, which allowed the completed seed-42 baseline to be reused after LPIPS was added and its eval JSON was refreshed.

`scripts/run_lookcloser_quiet.py` prunes non-selected `step-*.ckpt` files after writing `run_summary.json` unless `--keep-all-checkpoints` is passed. The selected best checkpoint, final eval JSON, and rendered outputs are preserved. This keeps local sweep storage viable while `/fsx` is full.

`scripts/run_lookcloser_quiet.py` also supports optional `--eval-batch-interval`, `--eval-image-interval`, `--eval-all-interval`, and `--save-interval` overrides for smoke tests that need eval-batch health checks without accidentally launching full-image/full-dataset evaluation.

FAS tuning added partial and scheduled FAS controls to `LookCloserPixelSampler`: `fas_strength` mixes FAS-selected pixels with uniform pixels in each batch, while `fas_warmup_steps` and `fas_ramp_steps` delay and linearly ramp that mixture over training steps. Defaults preserve the previous full-FAS behavior when FAS is enabled: `fas_strength=1.0`, `fas_warmup_steps=0`, and `fas_ramp_steps=0`. The quiet runner and sweep runner expose these as `--fas-strength`, `--fas-warmup-steps`, and `--fas-ramp-steps`.

The first isolated FAS tuning pass kept Feature Re-weighting disabled and tested FAS alone on the carried Frequency Grid + Adaptive RM metric leader. Full paper-style FAS improved LPIPS but regressed mean PSNR/SSIM. Mixed FAS with `fas_strength=0.35`, `fas_warmup_steps=2048`, and `fas_ramp_steps=4096` became the numeric PSNR/SSIM leader, with mean PSNR `29.051932` and SSIM `0.673978`, but it failed the high-frequency visual crop gate and worsened mean LPIPS to `0.378441`. Aggressive `sampling_ramp_start=0`, `sampling_ramp_end=3` was rejected early. See `experiments/lookcloser_fas_tuning.md`.

An artifact follow-up added `left_stand_eval0` to `scripts/render_lookcloser_crop_gate.py` after the seed-44 mixed-FAS render showed a broken vertical stand on `eval_img_0000.png`. `scripts/find_render_artifact_outliers.py` scans sliding local windows across eval renders and reports both worst candidate-vs-GT patches and worst candidate-vs-baseline regressions; this caught the same class of thin stand/wire failures. Delayed mixed FAS (`fas_strength=0.35`, `fas_warmup_steps=8192`, `fas_ramp_steps=8192`) and lower-strength mixed FAS (`fas_strength=0.20`, `fas_warmup_steps=2048`, `fas_ramp_steps=4096`) were both rejected early because eval loss regressed after the FAS ramp. Among the completed `0.35/2048/4096` runs, seed 43 is now the recommended visual FAS candidate: PSNR `29.135916`, SSIM `0.681484`, LPIPS `0.367407`, with a better eval0 left-stand crop than seed 44.

Frequency-map scalar-resolution metadata is intentionally checked against the model frequency schedule. The current HD maps were generated for `min_res=16`, `max_res=8192`, and `n_levels=16` (`max_res_base=2048` with this scene scale). Sweeping `max_res_base`, explicit `max_res`, or `num_frequency_levels` requires regenerating frequency maps for that schedule; otherwise FAS and runtime grid updates would interpret the same scalar values under a different level ladder.

The first valid Frequency Grid hyperparameter sweep selected `grid_resolution=64` by mean SSIM. `grid_resolution=256` had the best mean LPIPS, so it remains a useful counter-signal for visual comparisons, but the carried config for subsequent update/fallback sweeps is `grid_resolution=64`. `scripts/run_lookcloser_sweep.py` supports base overrides such as `--base-grid-resolution 64` so later stages can start from the carried setting without changing historical defaults.

The update-parameter sweep selected `grid_update_interval=512` and `grid_update_batch_size=4096` with `grid_resolution=64`. This candidate produced the best 3-seed mean SSIM, LPIPS, and PSNR among update settings: SSIM `0.555427`, LPIPS `0.425247`, PSNR `25.729128`, eval loss `0.03694857`, and training time `2635.440s`. Use this as the fixed hyperparameter baseline for implementation-doubt and improvement experiments unless a later 3-seed sweep beats it.

After the fixed-sample improvement, `scripts/run_lookcloser_sweep.py` carries `fixed_num_samples_per_ray=512` in its base config for LookCloser-internal metric sweeps. This is not an accepted quality improvement over the bounded Instant-NGP baseline after visual audit. Future baseline comparisons must match the Instant-NGP dataparser scale (`scene_scale=1.5`, `scale_factor=1.0`) and pass a visual crop gate on small writing, cables, fingers, and floor cracks.

A non-paper fixed-renderer improvement tested higher `fixed_num_samples_per_ray` values against the carried 256-sample fixed renderer using the same Frequency Grid settings. `384` improved the 3-seed mean SSIM from `0.555427` to `0.585817`, LPIPS from `0.425247` to `0.402468`, PSNR from `25.729128` to `26.616046`, and eval loss from `0.03694857` to `0.03442633`, with mean training time increasing from `2635.440s` to `2935.752s`. A follow-up `512` run further improved mean SSIM to `0.595257`, LPIPS to `0.390554`, PSNR to `27.009452`, and eval loss to `0.03358400`, with mean training time `3686.840s`. A later visual audit found that these runs used `scene_scale=2.0` and `scale_factor=1.15` while the bounded Instant-NGP baseline used `scene_scale=1.5` and `scale_factor=1.0`; it also showed worse tiny-detail crops despite higher PSNR/LPIPS. Treat these fixed-sample results as internal diagnostics until an apples-to-apples rerun confirms them.

## Scene bounds / AABB

LookCloser now replaces the default `NearFarCollider(near=2, far=6)` with `AABBBoxCollider(scene_box)` when `pipeline.model.enable_collider=True`. This is important for fixed-step ablations because the fixed marcher should sample only the nerfstudio scene box instead of a hand-picked near/far slab.

For the 3k `007740` split use `nerfstudio-data --scene-scale 2.5` for current 3k LookCloser runs unless a later full validation contradicts it.

## Shared hash-grid defaults

With `scene_scale=2.5`, a short 3k sweep over LookCloser `max_res_base` found `2048` to be the best early eval-PSNR setting among `1024`, `2048`, and `4096`. Keep `pipeline.model.max_res_base=2048` as the current quality-first default; `1024` is close and slightly faster, so it is useful for fast debugging runs. 

## Nerfstudio instant-ngp comparison hooks

For raw instant-ngp parity experiments, `nerfstudio.models.instant_ngp.InstantNGPModelConfig` exposes the underlying `NerfactoField` hash-grid and MLP shape: `base_res`, `num_levels`, `features_per_level`, `num_layers`, `hidden_dim`, `num_layers_color`, and `hidden_dim_color`. This allows testing raw-like settings such as 8 hash levels with 4 features per level without changing the default nerfstudio `instant-ngp` behavior.

The same comparison path also exposes `rgb_output_activation`, `loss_type`, and `raw_no_appearance_embedding`. These are for ablations against raw instant-ngp only: raw-like Huber loss and removing the appearance embedding were tested separately from the default `instant-ngp-big` baseline because they changed optimization behavior substantially.

`nerfstudio.data.dataparsers.instant_ngp_dataparser.InstantNGP` now reads `fl_y` directly when it is present in `transforms.json`. This avoids silently falling back to `fl_x` for non-square intrinsics in instant-ngp formatted transform files.

## Configurable LookCloser modules

The paper-level modules can be ablated independently through config flags.

- Frequency Grid: `pipeline.model.enable_frequency_grid` controls grid queries in the model; `pipeline.enable_frequency_grid` controls loading 2D maps and periodic grid updates. When disabled, the grid returns `fallback_frequency_level` and update steps are skipped.
- Current processed 3k data does not include `lookcloser_frequencies`, so Frequency Grid update experiments log a missing-map warning until the preprocessing path is restored.
- Feature Re-weighting: `pipeline.model.enable_feature_reweighting` controls Eq. 6 weighting in `LookCloserField`. When disabled, raw hash-grid features are passed to the MLP.
- FAS: `pipeline.datamanager.pixel_sampler.enable_fas` controls frequency-averaged pixel sampling. When disabled, `LookCloserPixelSampler` falls back to uniform `PixelSampler` behavior.
- Adaptive RM: `pipeline.model.enable_adaptive_ray_marching` controls adaptive ray marching. When disabled, `LookCloserModel` uses a fixed-step renderer with `fixed_num_samples_per_ray`.

Preprocessing now prefers `train_steps_per_level` over the legacy `steps_per_image`, so every frequency level receives enough optimization before SSIM assignment. The quiet wrapper currently calls `python -m nerfstudio.scripts.lookcloser_preprocess` directly so it does not depend on an installed console entrypoint.
