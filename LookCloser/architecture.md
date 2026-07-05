## Key files:

- `nerfstudio/scripts/lookcloser_preprocess.py` — 2D patch frequency preprocessing.
- `nerfstudio/scripts/lookcloser_debug_preprocess.py` — focused standalone preprocessing debug checks.
- `nerfstudio/model_components/lookcloser_grid.py` — 3D frequency grid.
- `nerfstudio/fields/lookcloser_field.py` — frequency-aware field.
- `nerfstudio/lookcloser_pixel_sampler.py` — frequency-aware sampler.
- `nerfstudio/models/lookcloser.py` — LookCloser model and adaptive ray marching.
- `nerfstudio/pipelines/lookcloser_pipeline.py` — training pipeline and grid updates.
- `nerfstudio/configs/method_configs.py` — `lookcloser` method config.

## Dataset format

A LookCloser dataset is a standard nerfstudio dataset directory (`images/` + `transforms.json`)
plus a `lookcloser_frequencies/` subfolder holding one precomputed frequency map per train image:

```
<dataset_root>/
  images/
  transforms.json
  lookcloser_frequencies/
    frame_train_00001.pt     ← float32 tensor, scalar resolution per patch
    frame_train_00001.json   ← sidecar metadata (patch_size, stride, level schedule, etc.)
    ...
```

Maps are produced by `nerfstudio/scripts/lookcloser_preprocess.py` (see Key files).

### Static (single-frame) dataset
```
fsx: /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns/
```

### Temporal (per-frame) dataset

45 stride-7 frames, each its own nerfstudio dataset with the layout above, so any single frame
can be trained on its own like the static dataset:

```
fsx:           /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/temporal_perframe_stride7_45f/<frame_id>/
clever-shadow: /home/brans/temporal_perframe_stride7_45f/<frame_id>/
```

`<frame_id>` ranges from `007740` to `008048`. When working on `clever-shadow`, use its local
copy above — it does not have `/fsx` mounted.

## Training monitoring additions

Baseline runs can enable `--logging.csv-writer.enable True` to write compact `metrics_compact.csv` rows for train/eval trends, `best_eval_*`, plateau and overfit status, which is useful because recent 3k baselines plateau early and best checkpoint metrics are more informative than final-step metrics.

Use `scripts/detect_structural_artifacts.py` as the automatic detector for serious structural artifacts in rendered crops or triptychs, especially broken/dislocated thin structures, holes, and floaters. Include its `artifact_score` alongside the current evaluation signals (SSIM, PSNR, LPIPS, and eval loss) when comparing candidate checkpoints; lower `artifact_score` is better and `0.0` means no qualifying severe local-SSIM artifact blobs. When needed, the script also saves bbox overlays, heatmaps, and suspicion maps for visual analysis of problematic artifacts.

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

Preprocessing now prefers `train_steps_per_level` over the legacy `steps_per_image`, so every frequency level receives enough optimization before SSIM assignment. The CLI entrypoint is `ns-process-lookcloser-freqs`.
