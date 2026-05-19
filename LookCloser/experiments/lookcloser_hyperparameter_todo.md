# LookCloser Hyperparameter Validation ToDo

## What was tested

Planned validation experiments for LookCloser preprocessing and the four paper modules. No experiments in this file have been run yet.

| Priority | Uncertainty | Module | Hyperparameter / Ablation | Candidate values | Validation method | Expected signal |
|---|---|---|---|---|---|---|
| P0 | High | Preprocess | `train_steps_per_level` | 250, 500, 1000, 2000 | Direct crop debug-overfit, patch mosaics, SSIM histogram | Max-level overuse should drop while high-detail patches remain high frequency |
| P0 | High | Preprocess | `ssim_threshold` | 0.90, 0.93, 0.95, 0.97 | Sweep on fixed HD/6K crops, inspect low/high/random patch audits | Frequency map should avoid collapse to min or max levels |
| P0 | High | Preprocess | `patch_size` / stride | 16, 32, 64 | Compare heatmaps and patch audits on detail-heavy crops | Smaller patches localize tiny details without noisy speckle |
| P1 | Medium | Preprocess / Field | `max_res` | 2048, 4096 | Overfit quality and downstream PSNR/LPIPS on short training runs | Higher max resolution helps tiny details without unstable maps |
| P1 | Medium | Field | `log2_hashmap_size` | 19, 21, 23 | Short training runs and memory/runtime logging | Larger tables reduce collisions on high-res scenes |
| P1 | Medium | FAS | `sampling_ramp_start/end` | 1:2, 1:3, 1:4 | Training curves plus rendered crops in high-frequency regions | Higher ramp improves details without hurting global structure |
| P1 | Medium | Frequency Grid | `grid_update_interval` | 512, 1024, 2048, disabled | Short training runs, grid histograms, rendered detail crops | Updates should improve consistency after geometry stabilizes |
| P1 | High | Frequency Grid | `fallback_frequency_level` | 0, 8, 15 | Ablation with grid disabled and feature re-weighting/adaptive RM enabled | Establish fair fallback for controlled no-grid baselines |
| P1 | Medium | Adaptive RM | `adaptive_min_step_size` / `adaptive_max_step_size` | min 1e-4/5e-4, max 0.05/0.1/0.2 | Runtime, samples per ray, detail crops, floaters | Smaller steps improve detail but should not explode runtime |
| P1 | Medium | Adaptive RM Off | `fixed_num_samples_per_ray` | 128, 256, 512 | Compare fixed-step baseline to adaptive RM | Fixed baseline should be stable enough for ablation |
| P2 | Medium | Feature Re-weighting | `enable_feature_reweighting` | on, off | Module ablation with same preprocessing maps | On should preserve detail with less capacity waste |
| P2 | Medium | Frequency Grid | `enable_frequency_grid` | on, off | Module ablation with fallback levels | Quantify value of 3D frequency projection/update |
| P2 | Medium | FAS | `enable_fas` | on, off | Module ablation using same train seed | On should improve high-frequency patch learning |
| P2 | Medium | Adaptive RM | `enable_adaptive_ray_marching` | on, off | Module ablation against fixed-step renderer | On should improve detail/runtime tradeoff |

## Results

- Progressive 2D frequency logic was checked on real HD and 6K crops in [preprocess_progressive_real_data.md](preprocess_progressive_real_data.md). The relevant P0 knobs remain `train_steps_per_level`, `ssim_threshold`, and `patch_size`; the short validation confirms the progressive mechanism, not final dataset-wide defaults.

## Insights

Start with P0 preprocessing experiments before full model ablations. Bad 2D maps will make Frequency Grid, FAS, and Adaptive RM conclusions unreliable.
