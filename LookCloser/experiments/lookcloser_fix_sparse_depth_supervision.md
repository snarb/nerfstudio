# LookCloser Sparse COLMAP Depth Supervision Fix

## What was tested

Hypothesis: the current LookCloser implementation misses the paper's early sparse point-cloud depth supervision because the HD dataset has no `depth_file_path` entries and the `lookcloser` method uses the plain image dataset. A narrow implementation generated explicit sparse COLMAP depth maps, added them to an experiment dataset copy, enabled a sparse-depth-only dataset path with no Zoe/pseudo-depth fallback, enforced `depth_loss_steps`, and used the paper-style Charbonnier depth loss.

The quality experiment used the carried frequency-grid settings: `grid_resolution=64`, `max_res_base=2048`, `num_frequency_levels=16`, `grid_update_interval=512`, `grid_update_batch_size=4096`, fixed ray marching, FAS enabled, and feature reweighting enabled. The 3-seed carried reference is SSIM `0.555427`, LPIPS `0.425247`, PSNR `25.729128`, eval loss `0.03694857`, and train time `2635.440s`.

The transient sparse-depth code path was reverted after the experiment because the 3-seed mean regressed under the SSIM-first rule. Artifacts and reports are retained.

## Sparse-depth sanity checks

Generated dataset root: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/sparse_depth_datasets/007740_hd_aabb4_multicamera_eval3_ns_sparse_depth`

Preview: `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/sparse_depth_datasets/007740_hd_aabb4_multicamera_eval3_ns_sparse_depth/frame_train_00001_sparse_depth_preview.png`

| Check | Result |
|---|---:|
| COLMAP images | 69 |
| COLMAP points | 34,598 |
| Real train sparse-depth maps | 66 |
| Zero eval placeholder maps | 3 |
| Total nonzero sparse pixels | 222,001 |
| Nonzero pixels per train image | min 1,490 / mean 3,363.65 / median 3,529.5 / max 4,310 |
| Saved-depth range before dataparser scale | min 1.687087 / median-of-frame-medians 10.953273 / max 63.371727 |
| Experiment dataset size | 547M |

The zero eval placeholders were only used to satisfy the nerfstudio dataparser's all-or-none `depth_file_path` assertion. Depth supervision was gated to train loss and nonzero sparse pixels.

## Smoke result

A 16-step smoke with `depth_loss_steps=8` and fixed ray marching completed without OOM or non-finite losses. A direct pipeline loss check showed the gate worked:

| Step | Loss keys | Depth loss |
|---:|---|---:|
| 0 | `depth_loss`, `distortion_loss`, `rgb_loss` | 0.000941 |
| 8 | `distortion_loss`, `rgb_loss` | absent |

Smoke eval at step 8: eval loss `0.135944`, PSNR `15.5023`, SSIM `0.423445`, LPIPS `0.963024`, train time `45.045s`.

## Results

Decision: reject and keep the current code. Mean SSIM regressed from `0.555427` to `0.554464`. LPIPS, PSNR, and eval loss also moved the wrong way; only training time improved.

| Candidate | Mean SSIM | Delta SSIM | Mean LPIPS | Delta LPIPS | Mean PSNR | Delta PSNR | Mean Eval Loss | Delta Loss | Mean Train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| carried reference | 0.555427 | - | 0.425247 | - | 25.729128 | - | 0.03694857 | - | 2635.440 |
| sparse depth supervision | 0.554464 | -0.000963 | 0.429050 | +0.003803 | 25.675806 | -0.053322 | 0.03707527 | +0.00012670 | 2344.882 |

Best single sparse-depth results:

| Metric | Best value | Seed | Selected step |
|---|---:|---:|---:|
| SSIM | 0.555888 | 43 | 30376 |
| LPIPS | 0.416517 | 43 | 30376 |
| PSNR | 25.875462 | 43 | 30376 |
| Eval loss | 0.036025 | 43 | 30376 |
| Training time | 1773.652s | 44 | 15188 |

Per-run results:

| Seed | Selected step | Eval Loss | PSNR | SSIM | LPIPS | Train s | Eval JSON | Renders |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 42 | 30376 | 0.037538 | 25.583965 | 0.553400 | 0.423722 | 2645.427 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_sparse_depth/lookcloser/control_current_baseline_seed42/eval_best_step-000030376.json` | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_sparse_depth/lookcloser/control_current_baseline_seed42/renders_best_step-000030376` |
| 43 | 30376 | 0.036025 | 25.875462 | 0.555888 | 0.416517 | 2615.567 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_sparse_depth/lookcloser/control_current_baseline_seed43/eval_best_step-000030376.json` | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_sparse_depth/lookcloser/control_current_baseline_seed43/renders_best_step-000030376` |
| 44 | 15188 | 0.037663 | 25.567991 | 0.554102 | 0.446910 | 1773.652 | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_sparse_depth/lookcloser/control_current_baseline_seed44/eval_best_step-000015188.json` | `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_sparse_depth/lookcloser/control_current_baseline_seed44/renders_best_step-000015188` |

## Insights

The paper component is implementable for this dataset using sparse COLMAP observations, but the measured reconstruction quality does not justify keeping the code path. The likely reason is that the sparse signal is extremely thin, around `0.16%` of train pixels, and may over-constrain early geometry at sparse SfM locations that do not align with this current HD split's best image-metric optimum.

Next step: do not keep sparse depth supervision in the baseline. If revisited later, treat it as a separate improvement with a tuned depth weight/window rather than a correctness fix.
