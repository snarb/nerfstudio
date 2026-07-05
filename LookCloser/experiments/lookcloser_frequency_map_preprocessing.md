# LookCloser Frequency Map Preprocessing

## What was tested

Generated HD dataset frequency maps for:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns`

Starting point came from `experiments/preprocess_heatmap_hyperparameter_tuning_6k.md`: `patch_size=8`, `ssim_window_size=7`, and high-frequency debug threshold `L13+`.

Single-image full-HD checks showed:

| Setting | Non-empty levels | Max-level frac | Mean scalar res | Median scalar res | Decision |
|---|---:|---:|---:|---:|---|
| `train_steps_per_level=160`, `ssim_threshold=0.97` | 16 | 0.957 | 7921 | 8192 | Rejected; full-image maps collapsed to max. |
| `train_steps_per_level=1000`, `ssim_threshold=0.97` | 16 | 0.368 | n/a | n/a | Rejected; still over-assigns max level. |
| `train_steps_per_level=1000`, `ssim_threshold=0.95` | 16 | 0.0968 | 3284 | 2353 | Selected. |
| `train_steps_per_level=1000`, `ssim_threshold=0.93` | 16 | 0.0154 | 2245 | 2353 | Rejected; likely under-labels high frequencies. |

## Results

Full preprocessing command:

```bash
python scripts/run_lookcloser_preprocess_quiet.py --force-recompute --debug-save --debug-max-images 2
```

Completed outputs:

| Item | Value |
|---|---:|
| Frequency maps | 66 |
| Metadata files | 66 |
| Runtime seconds | 12132.235 |
| Mean non-empty levels | 16 |
| Mean min-level fraction | 0.033056 |
| Mean max-level fraction | 0.118817 |
| Worst max-level fraction | 0.188735 |

Artifacts:

- Inspection JSON: `experiments/lookcloser_frequency_map_inspection.json`
- Preprocess log: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/lookcloser_preprocess_stdout.log`
- Debug artifacts: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns/lookcloser_debug`

## Insights

The 6K crop-tuned `0.97` threshold was useful as a visual starting point but was too strict for full-HD image-wide preprocessing. The selected `0.95` threshold preserves all 16 levels while avoiding max-level collapse.

Use these maps for the first frequency-grid baseline and hyperparameter sweeps. Treat preprocessing threshold as a later improvement axis only if frequency-grid metrics or crop inspection suggest over/under-labeling.
