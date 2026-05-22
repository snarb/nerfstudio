## Key files:

- `nerfstudio/scripts/lookcloser_preprocess.py` — 2D patch frequency preprocessing.
- `nerfstudio/scripts/lookcloser_debug_preprocess.py` — focused standalone preprocessing debug checks.
- `nerfstudio/model_components/lookcloser_grid.py` — 3D frequency grid.
- `nerfstudio/fields/lookcloser_field.py` — frequency-aware field.
- `nerfstudio/lookcloser_pixel_sampler.py` — frequency-aware sampler.
- `nerfstudio/models/lookcloser.py` — LookCloser model and adaptive ray marching.
- `nerfstudio/pipelines/lookcloser_pipeline.py` — training pipeline and grid updates.
- `nerfstudio/configs/method_configs.py` — `lookcloser` method config.

## Training monitoring additions

Baseline runs can enable `--logging.csv-writer.enable True` to write compact `metrics_compact.csv` rows for train/eval trends, `best_eval_*`, plateau and overfit status, which is useful because recent 3k baselines plateau early and best checkpoint metrics are more informative than final-step metrics.

## Configurable LookCloser modules

The paper-level modules can be ablated independently through config flags.

- Frequency Grid: `pipeline.model.enable_frequency_grid` controls grid queries in the model; `pipeline.enable_frequency_grid` controls loading 2D maps and periodic grid updates. When disabled, the grid returns `fallback_frequency_level` and update steps are skipped.
- Feature Re-weighting: `pipeline.model.enable_feature_reweighting` controls Eq. 6 weighting in `LookCloserField`. When disabled, raw hash-grid features are passed to the MLP.
- FAS: `pipeline.datamanager.pixel_sampler.enable_fas` controls frequency-averaged pixel sampling. When disabled, `LookCloserPixelSampler` falls back to uniform `PixelSampler` behavior.
- Adaptive RM: `pipeline.model.enable_adaptive_ray_marching` controls adaptive ray marching. When disabled, `LookCloserModel` uses a fixed-step renderer with `fixed_num_samples_per_ray`.

Preprocessing now prefers `train_steps_per_level` over the legacy `steps_per_image`, so every frequency level receives enough optimization before SSIM assignment. The CLI entrypoint is `ns-process-lookcloser-freqs`.
