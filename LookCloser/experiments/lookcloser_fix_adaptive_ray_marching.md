# LookCloser Adaptive Ray Marching Fix

## What was tested

Hypothesis: adaptive ray marching is blocked by implementation issues rather than only hyperparameters. The tested transient fix changed the adaptive renderer to:

- scale the Nyquist interval back to world ray units using the AABB extent;
- stack only executed adaptive steps instead of preallocating the configured max-step history;
- replace the quadratic Nerfstudio distortion loss path with an exact linear-time sorted-interval equivalent;
- log adaptive step counts and cap saturation during smoke runs.

The fixed-code smoke used the carried frequency-grid settings: `grid_resolution=64`, `max_res_base=2048`, `num_frequency_levels=16`, `grid_update_interval=512`, `grid_update_batch_size=4096`, FAS enabled, and feature reweighting enabled.

## Results

The transient distortion-loss replacement matched the existing quadratic loss exactly on a synthetic sorted-interval check (`max_abs_diff=0.0`).

Smoke runs:

| Smoke | Rays | Max steps | Eval type | Result |
|---|---:|---:|---|---|
| `20260601_adaptive_smoke_256` | 128 | 256 | train plus full-image/full-all eval interval | Interrupted after only the step-0 row; it did not reach step 1 or eval before the runtime became unreasonable. |
| `20260601_adaptive_smoke_64_r8` | 8 | 64 | train plus full-image/full-all eval interval | Interrupted after only the step-0 row; full-image eval with adaptive marching was still too slow. |
| `20260601_adaptive_smoke_batch_eval` | 8 | 64 | train plus eval-batch only | Completed 2 iterations in `35.027s`; eval step 1 loss `0.233820`, eval batch PSNR `10.7319`. |

Batch-eval smoke adaptive counters from the transient logger:

| Step | Split | Mean steps | Max steps | Saturation rate |
|---:|---|---:|---:|---:|
| 0 | train | 35.500 | 40 | 0.000 |
| 1 | eval batch | 35.625 | 43 | 0.000 |

No 3-seed quality run was launched. Even the smallest valid batch-eval smoke took about `17.5s/iteration` at 8 rays, while the 128-ray smoke failed to reach step 1 in a practical window. Full-image SSIM/LPIPS/PSNR evaluation was not run because the full-image adaptive eval path was the slow part of the smoke, and the user explicitly requested no final render.

## Insights

The smoke proved the memory-safe direction can avoid immediate OOM at tiny batch sizes, and low-level empty-grid rays did not saturate a 64-step cap. However, the adaptive path remains far too slow for normal training or a 3-seed metric experiment. The main bottleneck is still the Python per-step field-query loop; replacing history storage and distortion loss is not sufficient.

Recommendation: reject this implementation-fix attempt and keep adaptive ray marching disabled for quality experiments. The transient adaptive model and writer changes were reverted. The only retained helper is the quiet-runner interval override, which is harmless and lets future smoke tests run eval-batch without accidentally triggering full-image eval.

The generated smoke checkpoint was removed after rejection to avoid keeping a 2 GB artifact; logs, configs, metrics CSVs, and the run summary remain under `repro_runs/lookcloser_runs/007740_hd_aabb4_multicamera_eval3_ns_lookcloser_fix_adaptive_rm_smoke`.

Next plausible adaptive direction, if revisited, should be a vectorized or occupancy-grid-compatible sampler rather than another small patch to the current Python loop.
