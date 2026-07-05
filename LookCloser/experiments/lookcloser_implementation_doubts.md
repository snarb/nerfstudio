# LookCloser Implementation Doubts

## What was tested

The first isolated implementation check focused on whether runtime Frequency Grid updates can execute with the generated HD frequency maps.

Preprocessing settings came from the 6K tuning note as the visual starting point:

```bash
--patch-size 8 --ssim-window-size 7 --high-frequency-level 13
```

For the full-HD dataset, `experiments/lookcloser_frequency_map_preprocessing.md` records the selected adaptation:

```bash
--train-steps-per-level 1000 --ssim-threshold 0.95
```

## Results

| Check | Result |
|---|---|
| Initial 3-seed baseline attempt | Failed at first grid update with tensor-bound `torch.clamp` error. |
| Clamp fix smoke | Exposed CPU/GPU indexing mismatch for `cameras.fx/fy`. |
| CPU camera-index fix smoke | Passed through step 1024 grid update and wrote eval metrics. |
| First completed baseline seed | Training and rendering succeeded, but final reporting exposed that LookCloser eval did not emit LPIPS. |
| LPIPS metric fix | Added LookCloser LPIPS image metric and refreshed seed 42 eval JSON. |

Successful smoke command:

```bash
python scripts/run_lookcloser_quiet.py \
  --output-dir /home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs \
  --timestamp smoke_grid_update_fixed_2 \
  --max-num-iterations 1030 \
  --step-interval 1024 \
  --disable-adaptive-ray-marching \
  --no-render-final \
  --poll-seconds 10 \
  --train-num-rays-per-batch 64 \
  --eval-num-rays-per-batch 64 \
  --grid-update-batch-size 64
```

Smoke metrics at step 1024:

| eval loss | PSNR | SSIM | Training seconds |
|---:|---:|---:|---:|
| 0.0615459 | 19.917 | 0.444252 | 90.093 |

Completed seed-42 baseline metrics after LPIPS refresh:

| eval loss | PSNR | SSIM | LPIPS | Training seconds | Best step |
|---:|---:|---:|---:|---:|---:|
| 0.0365152 | 25.731192 | 0.554881 | 0.416669 | 3456.511 | 45564 |

## Insights

The runtime Frequency Grid path is now runnable with fixed ray marching. Adaptive ray marching remains disabled for the initial hyperparameter sweep because the frequency-enabled adaptive smoke OOMed even at 64 rays.

Initial Frequency Grid hyperparameter optimization is complete enough to fix the baseline for implementation-doubt experiments. Use `grid_resolution=64`, `grid_update_interval=512`, `grid_update_batch_size=4096`, `max_res_base=2048`, `num_frequency_levels=16`, fixed ray marching, and enabled Frequency Grid/FAS/Feature Re-weighting. The carried 3-seed reference from `experiments/lookcloser_frequency_grid_update_sweep.md` is SSIM `0.555427`, LPIPS `0.425247`, PSNR `25.729128`, eval loss `0.03694857`, and train time `2635.440s`.

Queued implementation doubts:

- Runtime update patch sampling: tested a paper-aligned patch-cell sampler that rendered patch centers. It regressed the 3-seed mean (`SSIM 0.555012`, `LPIPS 0.431763`, `PSNR 25.700442`) versus the carried reference (`SSIM 0.555427`, `LPIPS 0.425247`, `PSNR 25.729128`), so the code was reverted and the report is kept in `experiments/lookcloser_fix_runtime_patch_centers.md`.
- FAS empty-bucket handling: tested probability renormalization over non-empty frequency buckets. It improved mean LPIPS (`0.423117`) and PSNR (`25.745732`) but regressed mean SSIM (`0.555089`), so the code was reverted under the SSIM-first decision rule and the report is kept in `experiments/lookcloser_fix_fas_nonempty_buckets.md`.
- Adaptive ray marching: read-only audit found two blockers before quality experiments. The current Python loop preallocates `(n_rays, max_steps_per_ray, 1)` history tensors and keeps the full autograd graph, then the standard distortion loss broadcasts to a quadratic `(num_samples, num_samples)` tensor. The step size also appears to use normalized-scene frequency as a world-space interval. A smoke-first transient fix corrected world-scale `dt`, stacked only executed steps, and used an exact linear-time distortion loss. The formula matched the quadratic loss on a synthetic check and a tiny 8-ray eval-batch smoke completed, but it still took `35.027s` for two iterations and full-image eval did not reach step 1 in a practical window. Reject and revert; full report is in `experiments/lookcloser_fix_adaptive_ray_marching.md`.
- Feature re-weighting equation: audit found no concrete mismatch worth a 3-seed run. The main equation is ambiguous, but the supplementary text describes using the quantified frequency as a threshold and applying a singular down-weighting factor to higher components, which matches the current implementation.
- Sparse grid initialization / unknown voxels: tested COLMAP sparse initialization using `points3D.bin`, `images.bin`, `transforms.json` `colmap_im_id`, train dataparser transform/scale, and existing train frequency maps. Smoke touched 2,272 unique voxels (`nonzero_voxels=0->2272`, `max_level=15.000`). The 3-seed run regressed mean SSIM (`0.554959` vs carried `0.555427`), LPIPS (`0.434221` vs `0.425247`), and PSNR (`25.685141` vs `25.729128`), despite a slightly better eval loss and shorter training time. Reject and revert under the SSIM-first rule; full report is in `experiments/lookcloser_fix_sparse_initialization.md`.
- Sparse depth supervision: tested an explicit sparse-COLMAP-depth path with generated per-frame `.npy` maps, no Zoe/pseudo-depth fallback, `depth_loss_steps` gating, and paper-style Charbonnier depth loss. Sanity checks generated 66 train maps with 222,001 total nonzero pixels, 1,490-4,310 sparse pixels per train image, and a saved-depth range of 1.687-63.372 before dataparser scale. Smoke showed finite `depth_loss` at step 0 and no depth term after the gate. The 3-seed run regressed mean SSIM (`0.554464` vs carried `0.555427`), LPIPS (`0.429050` vs `0.425247`), PSNR (`25.675806` vs `25.729128`), and eval loss (`0.03707527` vs `0.03694857`), so the code was reverted under the SSIM-first rule; full report is in `experiments/lookcloser_fix_sparse_depth_supervision.md`.
