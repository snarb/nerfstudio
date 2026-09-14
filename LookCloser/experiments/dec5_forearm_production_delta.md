# DEC5 forearm completion in the production renderer

## What was tested

Three real times, **001029 / 001033 / 001037**, transfer the earlier
[v3 plane prior](dec5_forearm_plane_transfer_v3.md) into the exact renderer of
the [phase-shifted dynamic movie](dec5_temporal_camera_phase.md). This is a
local geometry experiment, not another full-video publication.

`append_verified_mesh_delta.py` imports only added triangles, preserving the
production mesh's exact vertex/triangle prefix. It does not restore source faces
already removed by production carving, nor replace the existing head repairs.
`study_forearm_production_delta.py` then checks added visible surfaces against
**all 62 native measured depth maps**, on integer and half-pixel ray grids.
A foreground contradiction needs observed depth behind the candidate by >0.003
normalized units, corroborated by at least three other depth observations
through the existing measured-depth guard. Original triangles cannot be removed.
This is stricter than the old four-camera prior check. It is not a proof of
anatomical correctness or complete surface coverage.

Each variant renders the same moving camera and real train H004_A005_1210M6,
with both production source-mask and angular-prior wrappers installed. Fixed
camera profiles/exposure, source times and hard single-source RGB are unchanged.
There is no RGB averaging or held-out input. Baseline replay is byte-identical
between variants. Geometry variants use the same three frozen train-skin masks.

`collect_forearm_depth_anchors.py` also collects measured, multiview-supported
points within 32 reference pixels of the patch and 0.012 normalized depth of
the old plane. It tests an inverse-depth plane versus quadratic surface using
even/odd physical-camera partitions, then fits all cameras. The partition is
**not an independent held-out evaluation**: corroborating observations overlap.
`curve_forearm_delta.py` applies the quadratic along reference rays, with a
0.01 displacement bound, and repeats semantic and measured-depth checks.

Roots:

- Plane: `/mnt/data/dec5_forearm_production_delta`.
- Anchors: `/mnt/data/dec5_forearm_multiview_anchors`.
- Quadratic: `/mnt/data/dec5_forearm_production_curved_referenced`.
- Audit/review: `/mnt/data/dec5_forearm_production_review`.
- Earlier failed quadratic attempt, retained: `/mnt/data/dec5_forearm_production_curved`.

The first quadratic 001037 attempt correctly stopped on an excessive proposed
displacement, but this came exclusively from unused appended vertices. Applying
the same referenced-vertex rule to all three times fixed that implementation
error without relaxing the displacement bound. Earlier scripts are archived in
their root's `config/`; frozen requests are not rewritten to permit changed-code
resume. Replay into a new root with the desired version. These are opt-in study
helpers and do not change any model or existing runner default.

## Results

**Both completions fail the visual production gate.** The plane partially fills
holes at all three times; the quadratic is worse than the plane in all three
reported metrics. Neither has been merged into the movie.

The scorer uses the full frozen manual **train forearm-skin ROI**, including its
missing geometry, with display PSNR / SSIM / AlexNet LPIPS. These are train
reprojection diagnostics, not held-out face metrics. No full-frame metric, loss,
or modification of the campaign face CSV is involved.

| Time | Production baseline PSNR / SSIM / LPIPS | Plane | Quadratic |
|---|---|---|---|
| 001029 | 21.507 / .8184 / .2722 | 25.596 / .8468 / .2450 | 24.999 / .8384 / .2529 |
| 001033 | 13.622 / .5407 / .5550 | 17.726 / .6920 / .4390 | 17.408 / .6739 / .4585 |
| 001037 | 12.498 / .2324 / .7207 | 15.928 / .4780 / .6220 | 15.071 / .4124 / .6727 |

| Time | Fixed skin RGB holes: baseline / plane / quadratic | Final added triangles: plane / quadratic | Components: baseline / plane / quadratic |
|---|---|---|---|
| 001029 | 1110 / 228 / 304 | 1998 / 1910 | 60 / 76 / 197 |
| 001033 | 7673 / 2541 / 2756 | 11990 / 11296 | 50 / 119 / 194 |
| 001037 | 10229 / 3876 / 4958 | 12733 / 12077 | 52 / 146 / 233 |

All six final geometry variants pass 124 final measured-ray checks with zero
trusted contradictions under this guard. Exact production prefixes survive.
All 24 RGB receipts pass; their source lists contain 62 train cameras, no target
RGB and no averaging. Upper 1000 output rows are byte-unchanged by the local
edits (locality diagnostic, not a quality score). Added moving-view geometry
without RGB is only 7/61/126 pixels for the plane versus 21/121/174 for quadratic.
The much larger remaining black areas are primarily missing geometry, not just
failure to paint newly added surfaces.

The main agent inspected all six native moving RGB pairs, six clay comparisons,
six GT/baseline/candidate triplets and three anchor panels. All six explicit
verdicts are `fail`: holes and rough patch boundaries remain; coarse hands,
cuff tears and existing texture mosaics are not repaired. There is no uncertain
verdict silently converted into pass.

![001033 plane: train reference, baseline, completion](/mnt/data/dec5_forearm_production_delta/001033/train_reference_native.png)
![001037 quadratic: train reference, baseline, completion](/mnt/data/dec5_forearm_production_curved_referenced/001037/train_reference_native.png)

The quadratic decreases camera-partition p90 depth residual from
0.001366/0.001629/0.003056 to 0.000777/0.000813/0.001430. Nevertheless, its
observed anchors inside the old hole occupy only **375/1136, 1001/7585 and
562/8861** unique reference pixels. Most interior geometry remains inferred;
raw point counts of 845/2634/2947 overcount overlapping camera observations.
The anchors cluster near the boundary, so better residuals there do not validate
the full extrapolated surface.

For 001037, a separate read-only comparison of dense COLMAP pinhole calibration
with staged transforms found exactly zero intrinsics difference and zero stored
distortion. A distorted/undistorted intrinsics mix does not explain this canary;
this check was not generalized to every time.

## Insights

1. Preserving production geometry during composition works, but final all-camera
   confidence pruning leaves substantial holes. Passing a free-space guard is a
   safety condition, not evidence that the missing surface was recovered.
2. A better fit to observed boundary points is insufficient. This quadratic
   implementation also moves the appended one-pixel old-depth ring and uses a
   Euclidean edge bound where the older builder used per-axis extent. Those are
   confounds: the comparison does not isolate curvature alone. The next bounded
   test should preserve old-depth boundary anchors and use matched triangulation
   gates before attributing failure to curved priors generally.
3. Source selection also needs a separate diagnostic: H004_A supplies only
   276/11703 fixed-skin pixels in the 001037 plane render even at its own pose.
   Other sources may be required by visibility/support; this histogram alone
   does not prove a selector bug. Do not fix geometry holes by silently changing
   textures or weakening depth consistency.
4. The existing phase-shifted camera workaround remains the best published
   dynamic movie. It reduces visible cheek holes without reducing camera travel,
   but still has major forearm defects. No artifact-free completion is claimed.

Reproduce a fresh plane canary with `study_forearm_production_delta.py prepare
--root NEW_ROOT --frame 001029`, then `render` with the same arguments. To test
the quadratic, add `--curved-anchor-root /mnt/data/dec5_forearm_multiview_anchors`
to preparation. Repeat all three times uniformly; do not overwrite frozen roots.
`score_forearm_production_delta.py --root ROOT` and
`audit_forearm_production_delta.py` keep metrics, receipts and explicit visual
review separate. Focused transfer, depth-footprint and prior tests cover the
append-only safety and referenced-vertex correction.
