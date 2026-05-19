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
