# LookCloser Progressive 2D Frequency Validation

## What was tested

Validated `nerfstudio/scripts/lookcloser_preprocess.py` on the real preprocessing debug images:

- HD preview: `/home/ubuntu/repos/look-closer/E004_D014_HD.jpg`, crop `[x=384, y=64, w=512, h=512]`.
- 6K source: `/home/ubuntu/repos/look-closer/E004_D014_graded.png`, crop `[x=1843, y=754, w=512, h=512]`.

Both runs used progressive 2D HashGrid training with `render_masked(..., max_active_level=L)` during training and evaluation, `patch_size=32`, `ssim_threshold=0.95`, `train_steps_per_level=120`, and debug levels `[0,2,4,8,12,15]`.

HashGrid parameters for these historical validation runs were `n_levels=16`, `n_features_per_level=2`, `min_resolution=16`, `max_resolution=2048`, and `log2_hashmap_size=19`. After this validation, the LookCloser baseline default was changed to `max_resolution=2048 * scene_size` and `log2_hashmap_size=23`.

Commands:

```bash
conda run -p /home/ubuntu/anaconda3/envs/nerfstudio python ../nerfstudio/scripts/lookcloser_preprocess.py --run-mode debug-overfit --image-path /home/ubuntu/repos/look-closer/E004_D014_HD.jpg --output-root experiments/preprocess_progressive_real_data_artifacts/hd_crop --debug-crop-size 512 --crop-x 384 --crop-y 64 --train-steps-per-level 120 --train-batch-size 8192 --patch-size 32 --debug-levels 0,2,4,8,12,15 --debug-patch-count 24 --audit-patch-count 12 --uv-audit-patch-count 8 --ssim-threshold 0.95

conda run -p /home/ubuntu/anaconda3/envs/nerfstudio python ../nerfstudio/scripts/lookcloser_preprocess.py --run-mode debug-overfit --image-path /home/ubuntu/repos/look-closer/E004_D014_graded.png --output-root experiments/preprocess_progressive_real_data_artifacts/6k_crop --debug-crop-size 512 --crop-x 1843 --crop-y 754 --train-steps-per-level 120 --train-batch-size 8192 --patch-size 32 --debug-levels 0,2,4,8,12,15 --debug-patch-count 24 --audit-patch-count 12 --uv-audit-patch-count 8 --ssim-threshold 0.95
```

## Results

| Crop | L0 | L2 | L4 | L8 | L12 | L15 | Non-empty levels | Min frac | Max frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HD preview | 0.402 | 0.508 | 0.650 | 0.883 | 0.973 | 0.984 | 10 | 0.008 | 0.000 |
| 6K source | 0.684 | 0.729 | 0.760 | 0.907 | 0.981 | 0.985 | 4 | 0.000 | 0.000 |

Artifacts:

- HD patch levels: [patch_mosaic.png](preprocess_progressive_real_data_artifacts/hd_crop/overfit_hd/patch_mosaic.png)
- HD frequency overlay: [freq_overlay.png](preprocess_progressive_real_data_artifacts/hd_crop/overfit_hd/freq_overlay.png)
- HD patch audits: [low](preprocess_progressive_real_data_artifacts/hd_crop/overfit_hd/patch_audit/low_freq_patches.png), [high](preprocess_progressive_real_data_artifacts/hd_crop/overfit_hd/patch_audit/high_freq_patches.png), [random](preprocess_progressive_real_data_artifacts/hd_crop/overfit_hd/patch_audit/random_freq_patches.png)
- 6K patch levels: [patch_mosaic.png](preprocess_progressive_real_data_artifacts/6k_crop/overfit_hd/patch_mosaic.png)
- 6K frequency overlay: [freq_overlay.png](preprocess_progressive_real_data_artifacts/6k_crop/overfit_hd/freq_overlay.png)
- 6K patch audits: [low](preprocess_progressive_real_data_artifacts/6k_crop/overfit_hd/patch_audit/low_freq_patches.png), [high](preprocess_progressive_real_data_artifacts/6k_crop/overfit_hd/patch_audit/high_freq_patches.png), [random](preprocess_progressive_real_data_artifacts/6k_crop/overfit_hd/patch_audit/random_freq_patches.png)
- Raw stats: [HD stats.json](preprocess_progressive_real_data_artifacts/hd_crop/overfit_hd/stats.json), [6K stats.json](preprocess_progressive_real_data_artifacts/6k_crop/overfit_hd/stats.json)

## Insights

The progressive logic now runs on the real images. Mean SSIM increases with active level for both crops, and the mosaics visually progress from low-frequency reconstructions at L0/L2 to detailed reconstructions at L12/L15.

The assignment path uses unresolved-only patches, so a patch keeps the first level that crosses the SSIM threshold instead of being overwritten by later levels. `render_masked` is used in training and evaluation, avoiding the invalid full-level-training plus inference-only-mask test.

P0 preprocessing hyperparameters from `lookcloser_hyperparameter_todo.md` are the relevant knobs for this task. These short runs support the current defaults structurally, but `train_steps_per_level`, `ssim_threshold`, and `patch_size` still need the planned sweep before selecting final dataset-wide values.
