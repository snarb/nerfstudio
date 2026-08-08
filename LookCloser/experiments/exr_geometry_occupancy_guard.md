# EXR geometry-aware occupancy guard

## What was tested

The target failure is the long thin black cable at the left of all three eval views. The frozen
field still contains cable density, but adaptive traversal skips false-negative occupancy voxels.
The target gate is therefore zero missing runs from `score_thin_cable_gaps.py`; aggregate image
metrics cannot override a cable failure.

The implemented training path is scene-generic and uses no cable coordinates or manual voxel edits:

1. `build_geometry_support_maps.py` reads the native linear EXRs and forms patch maps from Scharr
   edges plus multi-scale dark morphological ridges in PQ BT.709 luminance.
2. Each map is tie-aware rank-normalized, so its quantile is relative to each new image/scene.
3. Periodically sampled structural pixels are projected with a fixed uniform density probe which
   deliberately bypasses occupancy. Only rays with sufficient opacity and peak weight update a
   decayed 128-cubed 3D confidence grid.
4. Thresholded support is unioned into the binary traversal mask. It never edits density or
   occupancy EMA values. Cube, cross, and zero-halo controls isolate the required safety volume.

Map generation for this scene is reproducible with:

```bash
python scripts/build_geometry_support_maps.py \
  --images-dir /mnt/data/temporal_perframe_stride7_45f_exr_1920x1080/007740/images \
  --frequency-root /mnt/data/temporal_perframe_stride7_45f_exr_1920x1080/007740/lookcloser_frequencies_exr_auto \
  --out /mnt/data/temporal_perframe_stride7_45f_exr_1920x1080/007740/lookcloser_geometry_support_v2
```

All branches use the same seed-42 step91128 checkpoint and exactly 7594 additional updates, knee
frequency maps, corrected ARM, linear-softplus output and EAG PQ-DSSIM loss unless noted. Evaluation
uses adaptive traversal with no eval-only dilation.

## Results

| Candidate at step98722 | PQ PSNR | PQ SSIM | PQ LPIPS | Gap px | Longest gap |
|---|---:|---:|---:|---:|---:|
| Prior adaptive checkpoint | 34.049675 | 0.899265 | 0.213361 | 246 | 67 |
| Prior eval-only frequency-q75 repair | 34.034149 | 0.899406 | **0.212446** | **0** | **0** |
| Train-time frequency-q75 dilation + eval repair | 34.035337 | 0.899628 | 0.213128 | 0 | 0 |
| Geometry q80, cube radius1 | **34.180695** | **0.899974** | 0.213640 | **0** | **0** |
| Geometry q90, cube radius1 | 34.115214 | 0.899911 | 0.213109 | **0** | **0** |
| Geometry q95, cube radius1 | 34.092311 | 0.899819 | 0.213262 | **0** | **0** |
| Geometry q90, cube radius1, DSSIM0.4 | 34.128269 | 0.899902 | 0.213355 | **0** | **0** |
| Geometry q90, radius0 | 34.062265 | 0.899473 | 0.212896 | 161 | 38 |
| Geometry q90, cross radius1 | 34.105020 | 0.899760 | 0.212678 | **0** | **0** |
| Geometry q80, cross radius1 | 34.158597 | 0.899812 | 0.212850 | **0** | **0** |
| Geometry q80, cross radius1, edge loss0.1 | 34.144356 | 0.899825 | 0.212952 | **0** | **0** |
| q80+cross continuation, step106316 | 34.136342 | 0.899861 | **0.211813** | **0** | **0** |
| q80+cross continuation, step113910 | 34.045750 | 0.899568 | 0.211696 | **0** | **0** |

The q90 support grid contains 11,861 confidence voxels at threshold0.2, only 0.57% of the grid.
Cube dilation raises final binary occupancy from a raw 51.52% to 52.54%. Removing the halo improves
LPIPS slightly but immediately restores large cable gaps. A seven-cell cross is the smallest tested
neighborhood that retains zero gaps and improves LPIPS over the 27-cell cube. q80+cross is the
short-stage selection: it is within 0.07 dB of the maximum q80+cube PSNR and wins that tie on LPIPS.

Visual cable reviews for the current conservative winner are in:

`/mnt/data/lookcloser_geometry_campaign/geometry_guard_v1/lookcloser/geom_v2_q80_cross_s42/target_cable_gaps/`

## Insights

The cable was not absent from the learned field: occupancy-independent fixed probing finds its
surface. The defect arises because both training and rendering follow the same false-negative binary
occupancy mask. A geometry side channel is therefore useful during training; unlike eval dilation,
it also exposes the protected surface to subsequent optimization.

The guard must include neighboring voxels because a single projected surface cell does not cover
quantization and view-direction uncertainty. Full cube dilation is unnecessary: the axis cross
retains continuity with less metric disturbance. q90 is more stable than q95, while q80 maximizes
PSNR; the final quantile/continuation selection follows the 0.07-dB PSNR window with LPIPS as the
tie-breaker after the zero-gap veto.

The earlier PQ finite-difference edge term does not compose positively with the geometry guard: at
weight0.1 it keeps the cable but regresses both PSNR and LPIPS, so the selected loss remains plain
EAG PQ-DSSIM at DSSIM weight0.3.

Step106316 is the final selection. One more interval improves LPIPS by only0.000117 while losing
0.0906 dB PSNR, which moves step113910 outside the frozen 0.07-dB window. Final per-view metrics are:

| View | PQ PSNR | PQ SSIM | PQ LPIPS |
|---:|---:|---:|---:|
| eval0 | 34.030342 | 0.900119 | 0.250294 |
| eval1 | 34.726357 | 0.906554 | 0.218797 |
| eval2 | 33.652328 | 0.892909 | 0.166348 |
| **Mean** | **34.136342** | **0.899861** | **0.211813** |

All prediction EXRs have zero non-finite pixels, negative channels and over-peak channels. The
significant full-frame detector reports `artifact_score=0`, `serious=false` on all three eval views.
Native-scale visual inspection confirms the cable is continuous in all three saved review crops.

Final artifacts:

- checkpoint: `/mnt/data/lookcloser_geometry_campaign/geometry_guard_v1/lookcloser/geom_v2_q80_cross_cont1_s42/nerfstudio_models/step-000106316.ckpt`
  (`sha256:3a915e38b39bd3c376e72a5a2ce72206eed629e9da30fec05ab3fc8ae5bff5c2`);
- native EXR/PNG renders: `/mnt/data/lookcloser_geometry_campaign/geometry_guard_v1/lookcloser/geom_v2_q80_cross_cont1_s42/renders_latest_step-000106316/`;
- HDR review: `/mnt/data/lookcloser_geometry_campaign/geometry_guard_v1/lookcloser/geom_v2_q80_cross_cont1_s42/hdr_review_renders_latest_step-000106316/`;
- cable masks/crops: `/mnt/data/lookcloser_geometry_campaign/geometry_guard_v1/lookcloser/geom_v2_q80_cross_cont1_s42/target_cable_gaps/`.
