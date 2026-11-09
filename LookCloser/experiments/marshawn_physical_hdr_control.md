# Marshawn physical HDR occupancy control

## What was tested

Frame1660, the same calibrated 63 train and 3 held-out cameras, and the sealed
gain4 positive scene-linear Rec709 EXR master. Exposure is baked into the input;
one linear scene unit is 203 nits. The existing LookCloser linear-softplus head
composites physical linear RGB, then the EAG PQ L1+DSSIM objective operates in
PQ space (`eag_dssim_weight=0.3`, patch11). PQ code values are not composited as
physical RGB. Display uses shared Reinhard+sRGB without an additional gain.

The control disables frequency grid/regularization/adaptive sampling, distortion
and gradient scaling. Native nerfacc occupancy uses cube±4, grid128, max-res8192,
hash21, fixed render and occupancy-update step0.003, alpha threshold0, cone0,
near0.01, transmittance threshold1e-4, native warmup256 and binary warmup0.
Batch3993 rays consists of 33 patches11×11. The explicit native pixel sampler
uses `ignore_mask=True`; all66 source masks were verified entirely valid.
This avoids upstream patch-size constructor rounding and mask layout bugs.

Training attempted exactly22000 iterations, global0 through21999. Exact full-state
resumes preserved the ranges0–2000,2001–8000,8001–21999. The pinned pre-eval8k
checkpoint restored Adam/scheduler/scaler/RNG; interrupted eval RNG draws were
discarded and future single-image evaluation disabled, as recorded in receipts.
Final Adam successful-update count is21988 because AMP skipped12 updates.
No auxiliary free-space objective, background prior or new model was installed.

## Results

All metrics below average the three full-HD held-out views in the configured
PQ203 metric space. These are not directly comparable with sRGB-space PSNR.

| Global step | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| 2000 | 26.8058 | 0.815803 | 0.576567 |
| 4000 | 27.8195 | 0.836515 | 0.478994 |
| 6000 | 28.0074 | 0.847803 | 0.410887 |
| 8000 | 28.5610 | 0.856977 | 0.373782 |
| 12000 | 29.0327 | 0.867386 | 0.332218 |
| 16000 | 28.8512 | 0.869654 | 0.302435 |
| 20000 | 29.6675 | 0.878099 | 0.281261 |
| **21999** | **29.8738** | **0.878277** | **0.275192** |

The formal selection rule chooses21999: maximum all-eval PSNR, LPIPS tie-breaker
within0.07dB. All eligible checkpoints remain available. Full-resolution linear
EXR, common-display PNG and depth/opacity outputs are saved under
[/staged_hdr_occupancy4/final21999_hd](/fsx/tmp/marshawn_001660_hd/staged_hdr_occupancy4/final21999_hd).

![All three held-out views, GT left and prediction right](/fsx/tmp/marshawn_001660_hd/staged_hdr_occupancy4/final21999_hd/manual_review/all3_gt_pred_640.jpg)

All three full-frame pairs, native-resolution central-body crops and twelve
marked train-witness crops were manually inspected. Central people have coherent
silhouettes, limbs, hair and clothing folds. Faces and small details remain soft;
the green upper garment trends gray/brown. White/gray fog remains around some
floor and mat regions, and some background people/rails remain blurred.

The same128 camera-balanced visible subject observations were traced at8k and
21999, with unchanged field-state hashes. LOS ends before the witness by
`max(0.01,0.05*depth)`; the surface band is `±max(0.005,0.01*depth)`.

| Matched native traversal measure | 8k | 21999 |
| --- | ---: | ---: |
| Median pre-surface termination mass | 0.03847 | 0.08782 |
| Rays with pre-surface mass above10% | 47/128 | 58/128 |
| Median surface-band mass | 0.77009 | 0.79206 |
| Rays with surface-band mass at least50% | 108/128 | 112/128 |
| Median expected-depth relative error | −2.31% | −8.01% |
| Median weighted-median-depth relative error | +0.237% | +0.229% |
| Subject witness point-cell misses | 3/495 | 1/495 |

Independent uniform field queries improve median pre-surface mass from0.15662
to0.13647 and median surface-band mass from0.68316 to0.75076. Native traversal
admits more of the remaining low-density prefix tail. The remaining skipped
subject point6876 has sigma260.47 at voxel[65,68,62], a strong density-sampling
miss candidate. Default occupancy therefore remains unqualified as an ideal grid.

## Insights

The physical HDR pipeline produces recognizable geometry with cube±4, whereas
the corresponding±32 control remained cloudy. Better image metrics coexist with
residual near-surface density tails and a high-density occupancy miss. A one-cell
guard comparison needs the separately tested eval-hook repair; this frozen
training control and its metric renders retain the original radius-zero source.

Sparse train-witness rays qualify individual visible intervals, not whole voxels
or hidden volume. The subject ROI includes a cord witness and is not a semantic
body mask. No volume was carved or declared empty from missing SfM points.

Receipts and detailed evidence under `/fsx/tmp/marshawn_001660_hd/staged_hdr_occupancy4`:
`training_completed_receipt.json`, `final_selection.json`, `body8k_vs_best.json`,
`best_body128/audit.json`, and `final21999_hd/manual_review/manual_review.json`.
Checkpoint SHA256: `ee4d804679a3422f8819abd698aa4b2239db36f78eab015aadc6b04f909dd612`.
No training or evaluation worker owned by this control remains active.
