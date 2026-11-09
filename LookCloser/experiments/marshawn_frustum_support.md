# Marshawn calibration-only frustum support proposal

## What was tested

A conservative camera-frustum/voxel-AABB test on the same LOOSE128³ world grid for selected R0 step12000 MIN35 and selected R1 step20000 stored radius1. This audit reads calibration metadata, known fitted points/observations and checkpoint-bound masks. It reads no RGB pixels, queries no field/density and changes no worker, EMA, parameters or live mask.

For each camera, a voxel is outside only when **all8 corners are strictly outside one common frustum plane**. The implementation evaluates exact corner maxima using `plane(center)+abs(normal)·halfextent`. Cells touching or crossing a boundary stay admitted. The union over cameras is a conservative overestimate of possible frustum intersection; separately passing each plane does not certify exact intersection, visibility or occupied geometry.

Prepared OpenGL cameras use right/up/back, forward−Z; `u=fx*x/depth+cx`, `v=cy−fy*y/depth`, raster boundaries0..W/0..H. Cached observation UVs already use the half-pixel coordinate convention. Native near.052 is Euclidean ray distance, so the axial near-plane bound is conservatively `.052*min_cos(rastercorners)` (.040517–.040650 across cameras). This qualification changes0 union cells relative to standard axialnear.052 in this overlapping rig. Far is unbounded before clipping to the fixed world AABB.

## Results

TRAIN63 and ALL66 support unions are bitwise identical on this grid: eval-camera cones add no cells beyond the neighboring training cones. Raw support has1,584,144 cells; its exactlyONE voxel halo has1,619,117 cells.

| Actual source mask | Active cells | Outside raw support | Outside support+ONEhalo | Remaining with support+ONEhalo |
|---|---:|---:|---:|---:|
| R0 selected12k MIN35 | 1,363,103 | 342,332 (25.11%) | **319,726 (23.46%)** | 1,043,377 |
| R1 selected20k stored+1 | 1,887,128 | 511,593 (27.11%) | **477,356 (25.30%)** | 1,409,772 |

All9022 fitted TRAIN points,495 unique BODY points and52 MVS candidates remain covered by raw and support-halo candidate intersections for both models. Even everyONEhalo neighborhood around all9022 witness seed cells lies inside **raw** support; no extra point protection is necessary. R1 is intersected as stored: its mask is never dilated again.

All44,938 observations are inside their own declared TRAIN-camera frusta and reproduce the cached fitted UVs exactly. This is consistency with the fitted geometry, not an independent calibration measurement. Minimum observed axial depth is.401549 world units; minimum raster-edge margin is3.42163pixels.

The actual frozen `Nerfstudio.Cameras` CPU API generated all44,938 rays. Maximum direction discrepancy versus the OpenGL formula is1.90e−7. Sixteen points over each ray's **full AABB interval** (719,008 points total, including beyond the witness) all lie in retained support. All66 cameras also pass exact-raster-corner/half-pixel-center tests at Euclidean near.052. Sixteen distributed excluded cells have stored per-camera rejection certificates: every camera has one plane whose maximum signed distance across all8 corners is negative.

Eight CPU tests passed: OpenGL signs/half-pixel projection, near and raster boundaries, straddling retention, parallel/touching cases, explicit8-corner equivalence, rigid rotation, union monotonicity and the off-axis Euclidean-near qualification.

## Insights and next steps

The outside fraction is large enough to justify a **candidate-only** support guard review. The removed-region label is **UNOBSERVED/PHYSICALLY UNKNOWN**, never certified empty. The guard does not establish an ideal grid, hidden geometry truth or an image-quality improvement. Since calibrated input rays remain inside support, this should not be expected to fix their floor/mat color mottling; it limits unsupported traversal away from the input cones. Novel viewpoints may reveal real geometry outside those cones, so culling is not automatically appropriate for unrestricted viewpoints.

No intersection mask was installed or rendered. Both raw-support and support+ONEhalo candidate arrays are prepared for root review only. Source masks/checkpoints remain unchanged.

- [Audit and counts](/fsx/tmp/marshawn_001660_hd/frustum_support_native_near_ONLY_PRECHECK/audit.json)
- [Native camera/ray and16-cell rejection proof](/fsx/tmp/marshawn_001660_hd/frustum_support_native_near_ONLY_PRECHECK/camera_and_ray_geometry_proof.json)
- [Candidate-only NPZ](/fsx/tmp/marshawn_001660_hd/frustum_support_native_near_ONLY_PRECHECK/calibration_support_ONLY_NOT_APPLIED.npz)
- [Small3D geometry plot](/fsx/tmp/marshawn_001660_hd/frustum_support_native_near_ONLY_PRECHECK/unobserved_admission_3D.png)
- [Actual mask midplanes: retainedblue/unsupportedorange](/fsx/tmp/marshawn_001660_hd/frustum_support_native_near_ONLY_PRECHECK/candidate_midplanes_BLUEkept_ORANGEunobserved.png)
