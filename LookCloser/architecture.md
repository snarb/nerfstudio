## Key files:

- `nerfstudio/scripts/lookcloser_preprocess.py` — 2D patch frequency preprocessing.
- `nerfstudio/model_components/lookcloser_grid.py` — 3D frequency grid.
- `nerfstudio/fields/lookcloser_field.py` — frequency-aware field.
- `nerfstudio/lookcloser_pixel_sampler.py` — frequency-aware sampler.
- `nerfstudio/models/lookcloser.py` — LookCloser model and adaptive ray marching.
- `nerfstudio/pipelines/lookcloser_pipeline.py` — training pipeline and grid updates.
- `nerfstudio/configs/method_configs.py` — `lookcloser` method config.

## Preprocessing debug/test additions

Recent changes are scoped to validating 2D frequency-map preprocessing, not the full LookCloser model.

- `lookcloser_preprocess.py` now supports direct image runs via `--image-path`, so HD/6K crops can be tested without a Nerfstudio dataparser.
- Added `debug-overfit` artifacts for 2D HashGrid crop overfit: `gt.png`, `recon_full.png`, `diff.png`, `stats.json`.
- Added progressive level visualization for fixed debug levels, defaulting to `0,2,4,8,12,15`, to verify that lower levels are blurry and higher levels add detail.
- Added UV/patch audit output with fixed patches: GT patch, max-level prediction, diff, and labeled `(x, y)` patch coordinates.
- Added frequency-map diagnostics: `freq_heatmap.png`, `freq_overlay.png`, `freq_histogram.png`, and `freq_stats.json` / `stats.json`.
- Added patch audit mosaics: `low_freq_patches.png`, `high_freq_patches.png`, and `random_freq_patches.png`.
- Patch audit entries include GT, assigned-level reconstruction, max-level reconstruction, assigned level/resolution, assigned SSIM, and max SSIM.
- Added `sweep` mode for the minimal hyperparameter sweep over steps, SSIM threshold, patch size, and max resolution. Results are summarized in `sweep_summary.csv`.
- Frequency maps still store scalar resolution values, but preprocessing now writes sidecar JSON metadata containing `patch_size`, `stride`, `min_res`, `max_res`, `n_levels`, and the level-resolution schedule.
- `lookcloser_pixel_sampler.py` and `lookcloser_pipeline.py` now read this metadata when available, avoiding hidden `patch_size=32` assumptions for new maps.
- Progressive 2D preprocessing trains and evaluates each HashGrid prefix with the same `render_masked(..., level)` path, and casts tiny-cuda-nn half outputs safely for training loss and SSIM/debug artifacts.
- Baseline HashGrid defaults follow the paper setup: 16 levels (`0..15`), 2 features per level, `min_res=16`, `max_res=2048 * scene_size`, and `log2_hashmap_size=23`. The LookCloser model infers `scene_size` from the longest AABB side; preprocessing infers it from the dataparser scene box for dataset runs and accepts `--scene-size` for direct image debugging.
- Frequency-averaged sampling buckets scalar frequency maps using the per-map metadata `min_res/max_res/n_levels` when present, so maps generated with scene-size-scaled `max_res` are not decoded with stale fallback constants.
- Debug frequency maps now include level-based diagnostics: `level_heatmap.png`, `level_overlay.png`, `level_heatmap_legend.png`, and `high_frequency_mask_L{threshold}_plus.png`. Compatibility files `freq_heatmap.png` and `freq_overlay.png` use the level-based visualization; scalar-resolution heatmaps are saved separately.
- Heatmap debugging now also writes quantile-stretched diagnostic views, `level_heatmap_quantile.png` and `level_overlay_quantile.png`, so narrow assigned-level ranges are easier to inspect without changing `level_heatmap.png` semantics. The high-frequency diagnostic threshold is configurable with `--high-frequency-level`, and a readable red overlay is saved as `high_frequency_overlay_L{threshold}_plus.png`.
- 6K heatmap tuning on the same tiny-detail crop selected `patch_size=8`, `ssim_window_size=7`, `ssim_threshold=0.97`, and diagnostic `high_frequency_level=13` as the default preprocessing/debug configuration. The final 64x64 patch grid populated 13 levels, with L12+ fraction `0.402`, L13+ fraction `0.125`, and max-level fraction `0.026`; it produced a clearer overlay on rods/cables/hard edges than patch sizes 12 or 16. Results are documented in `experiments/preprocess_heatmap_hyperparameter_tuning_6k.md`.
- Kneedle selector comparison was tested on the 6K tiny-detail crop and then removed from the runtime preprocessing script. Plain Kneedle selected levels about 2.5-3.0 levels lower than the existing selection logic and visually under-selected rods/cables, so the codebase keeps the tuned SSIM-threshold assignment path instead of adding a Kneedle selector.

## Configurable LookCloser modules

The paper-level modules can be ablated independently through config flags.

- Frequency Grid: `pipeline.model.enable_frequency_grid` controls grid queries in the model; `pipeline.enable_frequency_grid` controls loading 2D maps and periodic grid updates. When disabled, the grid returns `fallback_frequency_level` and update steps are skipped.
- Feature Re-weighting: `pipeline.model.enable_feature_reweighting` controls Eq. 6 weighting in `LookCloserField`. When disabled, raw hash-grid features are passed to the MLP.
- FAS: `pipeline.datamanager.pixel_sampler.enable_fas` controls frequency-averaged pixel sampling. When disabled, `LookCloserPixelSampler` falls back to uniform `PixelSampler` behavior.
- Adaptive RM: `pipeline.model.enable_adaptive_ray_marching` controls adaptive ray marching. When disabled, `LookCloserModel` uses a fixed-step renderer with `fixed_num_samples_per_ray`.

Preprocessing now prefers `train_steps_per_level` over the legacy `steps_per_image`, so every frequency level receives enough optimization before SSIM assignment. The CLI entrypoint is `ns-process-lookcloser-freqs`.
