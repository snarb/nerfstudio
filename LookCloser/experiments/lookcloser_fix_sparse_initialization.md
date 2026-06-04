# LookCloser Sparse Frequency Grid Initialization Fix

## What was tested

Hypothesis: the current LookCloser pipeline missed the paper's sparse SfM initialization for the 3D frequency grid. A narrow implementation loaded `colmap/sparse/0/points3D.bin` and `images.bin`, mapped COLMAP image ids to train dataset indices through `transforms.json` `colmap_im_id`, transformed COLMAP point coordinates through the train dataparser transform and scale, sampled the existing per-image frequency maps at the observed 2D tracks, and called `FrequencyGridManager.initialize_from_sparse(...)` before training.

Smoke verification used a 1-step run before the 3-seed experiment. It loaded 66 frequency maps, prepared 34,598 COLMAP points and 252,410 train observations, and initialized 2,272 unique frequency-grid voxels (`nonzero_voxels=0->2272`, `max_level=15.000`) without crashing.

Experiment settings:

- Dataset: `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns`
- Experiment name: `007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_sparse_init`
- Seeds: `42, 43, 44`
- Checkpoint protocol: final eval on the checkpoint with the lowest in-training eval loss.
- Carried hyperparameters: `grid_resolution=64`, `max_res_base=2048`, `num_frequency_levels=16`, `grid_update_interval=512`, `grid_update_batch_size=4096`, fixed ray marching, FAS enabled, feature reweighting enabled.
- Decision rule: mean SSIM first, then mean LPIPS, mean PSNR, mean eval loss, mean training time.

## Results

Carried no-sparse reference from `experiments/lookcloser_frequency_grid_update_sweep.md`: SSIM `0.555427`, LPIPS `0.425247`, PSNR `25.729128`, eval loss `0.03694857`, training time `2635.440s`.

Sparse-init means: SSIM `0.554959`, LPIPS `0.434221`, PSNR `25.685141`, eval loss `0.03690490`, training time `2364.775s`.

Delta versus carried reference: SSIM `-0.000468`, LPIPS `+0.008974` worse, PSNR `-0.043987`, eval loss `-0.00004367` better, training time `-270.665s`.

| Seed | Selected step | Eval loss | PSNR | SSIM | LPIPS | Train s | Eval JSON | Renders |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 42 | 30376 | 0.037334 | 25.656107 | 0.553920 | 0.430769 | 2675.353 | `repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_sparse_init/lookcloser/control_current_baseline_seed42/eval_best_step-000030376.json` | `repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_sparse_init/lookcloser/control_current_baseline_seed42/renders_best_step-000030376` |
| 43 | 30376 | 0.036319 | 25.819952 | 0.556411 | 0.421423 | 2645.458 | `repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_sparse_init/lookcloser/control_current_baseline_seed43/eval_best_step-000030376.json` | `repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_sparse_init/lookcloser/control_current_baseline_seed43/renders_best_step-000030376` |
| 44 | 15188 | 0.037062 | 25.579365 | 0.554545 | 0.450469 | 1773.514 | `repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_sparse_init/lookcloser/control_current_baseline_seed44/eval_best_step-000015188.json` | `repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_sparse_init/lookcloser/control_current_baseline_seed44/renders_best_step-000015188` |
| **Mean** |  | **0.03690490** | **25.685141** | **0.554959** | **0.434221** | **2364.775** |  |  |

Best single results:

| Metric | Best value | Seed | Step |
|---|---:|---:|---:|
| SSIM | 0.556411 | 43 | 30376 |
| LPIPS | 0.421423 | 43 | 30376 |
| PSNR | 25.819952 | 43 | 30376 |
| Eval loss | 0.036319 | 43 | 30376 |
| Training time | 1773.514s | 44 | 15188 |

## Insights

The sparse initialization was technically valid and touched real grid voxels, but it regressed mean SSIM, LPIPS, and PSNR versus the carried no-sparse reference. Under the SSIM-first rule, reject this fix. The sparse-initialization code was reverted after the experiment; run artifacts and this report are retained.

The likely issue is not observation loading itself. The paper initializes a frequency grid from sparse points, but in this bounded Instant-NGP setup the fixed ray-marching implementation and runtime grid updates may already recover enough useful frequency state, while the sparse point initialization biases early training toward a sparse COLMAP surface distribution that does not improve held-out image metrics.
