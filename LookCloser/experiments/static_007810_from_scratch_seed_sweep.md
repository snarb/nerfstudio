# Static LookCloser from scratch on 007810: three-seed sweep

## What was tested

Three independent LookCloser campaigns start from random initialization on
`/home/brans/temporal_perframe_stride7_45f/007810`. They run concurrently on GPU 0 with seeds
42, 43 and 44. No checkpoint from 007740, another frame, or another seed is loaded.

Each campaign uses the frozen static-leader weight recipe:

- 66 filename-split train views and 3 eval views;
- standard `lookcloser_frequencies` maps;
- B4096, hash23, max-res8192;
- adaptive ARM with 4096-update warmup, cap1024 and coarse step0.00625;
- FAS1.0 and frequency grid from update zero;
- Charbonnier RGB, distortion0.01 and early depth0.001;
- Adam `0.01→0.0001` over 200000 scheduler steps;
- Stage A with FR1.0 through step75940;
- same-campaign full-state continuation with FR0.3;
- checkpoint and full three-view eval every15188 updates.

After the fixed leader stages, each seed is extended one interval at a time until its last two
full-eval intervals both satisfy the numeric plateau thresholds: PSNR gain below0.03 dB, SSIM gain
below0.001 and LPIPS improvement below0.003. The final cross-seed selector maximizes full-eval
PSNR, then minimizes LPIPS within the inclusive0.07-dB window. Loss is not used in reporting.

Controller worktree:
`/home/brans/repos/nerfstudio_007810_scratch3`, branch
`scratch-007810-seed-sweep`.

Campaign root:
`/mnt/data/lookcloser_007810_from_scratch_seed_sweep`.

## Results

Training in progress.

## Insights

Pending final numeric and visual review.
