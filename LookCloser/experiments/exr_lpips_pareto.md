# EXR perceptual-loss and PQ-MSE continuation

## What was tested

Starting from the geometry-guard EXR leader at step 106316, the screen tested a PQ-domain AlexNet
LPIPS term on true contiguous 32×32 FAS patches, then short PQ-MSE and historical EAG-PQ-DSSIM
continuations. The implementation keeps scene-linear RGB prediction/compositing; PQ is used only
for the reconstruction objective and reported perceptual metrics.

## Results

All values are the mean over the three held-out EXR views in the calibrated PQ evaluation domain.

| Candidate | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Cable gaps |
|---|---:|---:|---:|---:|
| Geometry-guard EXR leader, step 106316 | 34.136342 | **0.899861** | 0.211813 | 0 |
| PQ-L1+LPIPS 0.02, 32×32 patches, step 110113 | 33.990125 | 0.895389 | **0.194845** | not promoted |
| PQ-L1+LPIPS then PQ-MSE, step 111982 | **34.307040** | 0.896696 | 0.197152 | **0** |
| PQ-MSE directly from leader, step 108186 | 34.414862 | 0.899213 | 0.211416 | not promoted |
| PQ-L1+LPIPS then EAG-DSSIM 0.3, step 111982 | 34.107605 | 0.899472 | 0.204612 | not promoted |

The current LPIPS-target leader is the PQ-L1+LPIPS → PQ-MSE checkpoint:

`/mnt/data/lookcloser_lpips_campaign/pq_lpips_v1/lookcloser/pqlpips_w002_p32_recover_pqmse_1869_s42/nerfstudio_models/step-000111982.ckpt`

Its native EXR renders, fixed-exposure review sheets, metrics, config, and cable crops are retained
in the checkpoint's run directory. The targeted cable detector reports zero gap pixels and zero
longest-gap length in all three eval views; the saved review crops were also visually checked.

## Insights

True spatial LPIPS patches cross the requested LPIPS 0.2 threshold, while a short PQ-MSE
continuation recovers PSNR above the previous leader without reopening the cable. It is a useful
Pareto candidate, but it is not yet the global promoted EXR leader: SSIM is 0.003165 below the
geometry-guard reference. PQ-MSE alone improves PSNR but does not materially improve LPIPS, while
EAG-DSSIM recovers SSIM at the cost of crossing back above LPIPS 0.2.
