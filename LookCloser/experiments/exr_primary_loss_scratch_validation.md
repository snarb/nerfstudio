# EXR primary-loss scratch validation

## What was tested

The missing scratch-vs-scratch control for the EXR primary loss. Seeds 43 and 44 used the same
linear-softplus RGB head, native linear compositing, knee frequency maps, geometry guard, corrected
ARM, optimizer and target cumulative exposure (`2.419e11` rendered point samples). Only the RGB loss
changed:

- PQ-L1: L1 between prediction and GT after both are transformed to PQ (`linear_pq` in code).
- PQ-MSE: MSE in the same PQ domain.
- EAG: `0.7 * PQ-L1 + 0.3 * DSSIM` in PQ.

Every completed arm was selected by eval PSNR with LPIPS tie-break inside 0.07 dB and evaluated with
dense4 corrected rendering. PQ-MSE was allowed to stop only after two consecutive catastrophic evals.

## Results

Final dense4 corrected metrics, mean of seeds 43/44:

| Primary loss | PSNR | SSIM | LPIPS | Cable gap pixels | Mean train time |
|---|---:|---:|---:|---:|---:|
| **EAG PQ-L1+DSSIM** | **34.763405** | **0.901025** | **0.212974** | **0** | 98.53 min |
| Pure PQ-L1 | 34.555962 | 0.899941 | 0.222590 | 12 | 98.59 min |
| Pure PQ-MSE | rejected | rejected | rejected | not rendered | 14.68 min to reject |

Pure PQ-L1 minus EAG: `-0.207443 dB` PSNR, `-0.001084` SSIM, `+0.009616` LPIPS, and no meaningful
speed benefit (`+3.2 s`, or `+0.05%`).

Per-seed dense4 results:

| Loss | Seed | PSNR | SSIM | LPIPS | Cable gap pixels | Train time |
|---|---:|---:|---:|---:|---:|---:|
| EAG | 43 | 34.993431 | 0.901516 | 0.209626 | 0 | 98.97 min |
| PQ-L1 | 43 | 34.697708 | 0.900315 | 0.222914 | 12 | 98.77 min |
| EAG | 44 | 34.533379 | 0.900534 | 0.216323 | 0 | 98.10 min |
| PQ-L1 | 44 | 34.414215 | 0.899567 | 0.222267 | 0 | 98.41 min |

PQ-MSE early-screen evidence:

| Seed | Eval | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|---:|
| 43 | 1 | 21.2647 | 0.699174 | 0.816511 |
| 43 | 2 | 21.1185 | 0.672797 | 0.810446 |
| 44 | 1 | 21.4305 | 0.700866 | 0.809786 |
| 44 | 2 | 21.0316 | 0.681118 | 0.795193 |

## Visual review

All six GT/EAG/PQ-L1 full-frame sheets were inspected. No perceptual advantage for pure PQ-L1 was
found. The cable detector found a 12px break in seed43 eval1; inspection of
`left_black_cable_eval1_review.png` confirms that it is a real break in the thin black cable rather
than a mask-tracking error. EAG has zero cable gaps on all six matched renders.

Artifacts:

- `/mnt/data/lookcloser_primary_loss_validation/visual_review`
- `/mnt/data/lookcloser_primary_loss_validation/exr_primary_loss_scratch_two_seed_v1/lookcloser/s43_pql1_scratch/evaluation_dense4/adaptive_dense4x_corrected/target_cable_gaps/left_black_cable_eval1_review.png`

## Insights

EAG PQ-L1+DSSIM is now validated as the better primary loss for this EXR scene: it wins PSNR, SSIM,
LPIPS and cable continuity at effectively identical training time. Pure PQ-MSE is not usable from
scratch with the current unscaled loss and optimizer. A small early relative regression is not a
valid reason to reject PQ-L1; only clearly catastrophic failures should use the generic early-stop
gate.
