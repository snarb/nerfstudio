# Insight: Eliminating Visually Significant Artifacts

## What worked

**Fixed-step sampling + occupancy grid + feature reweighting (seed 43 leader resume)** removed visually significant artifacts (ROI artifact score = 0.0).

Key combination:
- `--ray-sampling-mode fixed --fixed-num-samples-per-ray 2048` — no ARM, stable gradients
- `--enable-feature-reweighting` — better high-frequency detail recovery without ARM interactions
- Resumed from seed 43 leader checkpoint (previously best artifact-clean checkpoint)
- Occupancy grid active (`grid-resolution 128`, `occ-thre 0.01`) for spatial sparsity

## Metrics (run C, step 45564)
| Metric | Value |
|--------|-------|
| PSNR   | 29.565 |
| SSIM   | 0.683 |
| LPIPS  | 0.365 |
| ROI artifact score | **0.0** |

## Why artifacts were suppressed

ARM (adaptive ray marching) was the root cause of stand/foreground artifacts — it under-sampled transitions near dense objects. Fixed-step mode eliminates this entirely. Feature reweighting was previously avoided due to interaction with ARM; without ARM it is safe and helps detail.

## Checkpoint selection fix

Noisy `eval_loss` (2048 px / 1 image) selected step 30376 (LPIPS 0.397) over step 45564 (LPIPS 0.365). Fix: use `eval_all_psnr` (100% of all 3 eval images) + LPIPS tie-breaker (< 0.07 dB threshold). Implemented in `run_lookcloser_quiet.py`.
