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
| 64×64 PQ-L1+LPIPS then PQ-MSE, step 107008 | 34.345299 | 0.898023 | **0.195992** | **0** |
| Same step, dense 4× corrected adaptive rendering | **34.369545** | **0.899050** | 0.199267 | **0** |
| PQ-MSE directly from leader, step 108186 | 34.414862 | 0.899213 | 0.211416 | not promoted |
| PQ-L1+LPIPS then EAG-DSSIM 0.3, step 111982 | 34.107605 | 0.899472 | 0.204612 | not promoted |

This is the promoted field-only EXR leader and the perceptual source used by the later residual
renderer. The current end-to-end quality leader is documented in
`experiments/exr_residual_pareto_leader.md`. The field-only checkpoint is:

`/mnt/data/lookcloser_lpips_campaign/p64_staged_recovery_v1/lookcloser/p64_step106496_recover_pqmse_2048_s42/nerfstudio_models/step-000107008.ckpt`

The exact renderer config, native EXR renders, metrics, and cable review crops are retained under
`sampling_ablation_step107008/adaptive_dense4x_corrected` in the same run directory. The checkpoint
SHA-256 is `0aa2d92f6d83421ca5e2792e17d5c98c7df6623e120624345dd10fc835261266`.
The targeted cable detector reports zero gap pixels and zero longest-gap length in all three eval
views; the saved review crops were also visually checked.

## Insights

True spatial LPIPS patches cross the requested LPIPS 0.2 threshold, while a short PQ-MSE
continuation recovers structure without reopening the cable. Relative to the geometry-guard
reference, the promoted configuration changes PSNR/SSIM/LPIPS by `+0.233204 / -0.000811 / -0.012546`.
Dense 4× adaptive rendering is the selected quality setting; it is about 2× slower than the normal
adaptive renderer (`0.0512` versus `0.1038` FPS). PQ-MSE alone improves PSNR but does not materially
improve LPIPS, while EAG-DSSIM recovers SSIM at the cost of crossing back above LPIPS 0.2.
