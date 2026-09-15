# Measured-source visibility and conditional depth peeling

## What was tested

Following [the lipstick-fin diagnosis](dec5_lipstick_fin_measured_free_space.md),
three fixed views of time `000995` were tested: the actual wide-left-arc movie
pose, `H004_C005_1210SZ`, and `K004_B005_1210DS`. The blue region came from real
train-camera shirt RGB projected onto an incorrect first mesh surface. This
study tests visibility, not exposure correction or cross-camera averaging.

Three opt-in controls were evaluated without changing production defaults:

1. Strict source admission on production and conservatively pruned geometry:
   require positive native measured COLMAP depth within 0.0015 normalized units.
   Unknown depth cannot supply RGB. Six full-resolution renders.
2. On the three strict-production renders, inspect actual deeper intersections
   of the same mesh where the first hit has no valid source. Require at least
   two coherent train observations and measured-depth-valid bilinear RGB taps.
   Select one source and the nearest admissible deeper intersection; no image
   inpainting, synthesized geometry, or RGB averaging.
3. Apply the strict/peeled result only where production's first hit is
   confidently contradicted by measurements: its selected source sees farther
   depth, there are zero near observations, at least six stable farther 5×5
   footprints, and at least six farther observations corroborated by three
   other cameras each. Preserve every other production RGB pixel exactly.

The 62 raw geometric depth maps, calibration, camera poses, fixed exposure and
color profiles are hash-bound. Held-out RGB, hand-drawn diagnostic ROIs and
segmentation masks are not used for these admission decisions. Geometry is
unchanged: **this is a render experiment, not an improved or repaired mesh**.

## Results

| View | Strict invalid-source rays | Admitted deeper receivers | Conditional changed pixels | Conditional deeper / background |
|---|---:|---:|---:|---:|
| Moving | 2858 | 1097 | 333 | 333 / 0 |
| H/C | 3210 | 553 | 377 | 0 / 377 |
| K/B | 6993 | 2149 | 793 | 24 / 769 |

Depth peeling took 2.66–2.91 seconds per view after the strict renders existed.
Conditional selection took 1.58–2.32 seconds. These are incremental stages,
not full reconstruction/render timings. Counts above are diagnostics, not
PSNR/SSIM/LPIPS or full-frame quality metrics.

The main agent inspected all six strict lipstick and head panels. Global
strict admission fails on both meshes: it removes blue shirt contamination
but produces a black lipstick ring, finger/jaw speckles, and crown cracks.
All three conditional lipstick, head, and actor comparison panels were also
inspected, including the intermediate peeled outputs. Peeling reveals actual
neck geometry behind part of the false fin. Conditional application avoids
the global hair damage and changes only 333 pixels in the moving view, with
no newly black pixels there. However, the side views retain incorrect brown
or blue strips near the tube. Existing crown defects remain.

**Verdict: partial local render improvement; not promoted.** This does not
establish a clean held-out result, improved mesh, or temporal-video pass.

- [Moving comparison](/mnt/data/dec5_confidence_gated_peeling/000995/moving/review/lipstick_native.png)
- [H/C comparison](/mnt/data/dec5_confidence_gated_peeling/000995/H004_C005_1210SZ/review/lipstick_native.png)
- [K/B comparison](/mnt/data/dec5_confidence_gated_peeling/000995/K004_B005_1210DS/review/lipstick_native.png)
- [Strict review](/mnt/data/dec5_measured_source_visibility/000995/visual_review.json)
- [Conditional review](/mnt/data/dec5_confidence_gated_peeling/000995/visual_review.json)
- [Artifact manifest](/mnt/data/dec5_confidence_gated_peeling/000995/artifact_manifest.json)

All three replay audits pass: actual triangle/barycentric points, nearest
admitted intersections, coherent depth votes, valid source footprints,
conditional free-space evidence, and unchanged RGB outside the admitted set.
An independent NumPy one-source RGB replay differs by at most 0/1/1 uint8
levels for moving/H/C/K/B. Evidence helpers are reused for depth-vote replay;
this is not a wholly independent reconstruction implementation.

The initial H/C and K/B audits rejected an arbitrary 2e-6 world-space ray
tolerance. Float32 grazing intersections reached 2.97e-6 and 3.77e-6 error.
No renders were changed. The audit now uses a units-based bound of 1% of the
actual 0.0005 TSDF voxel (5e-6), plus at most 0.05 native pixel reprojection
error for selected receivers. Measured maxima are 0.00234/0.01913/0.01424 px.
Failed first-attempt logs are retained. This numerical tolerance correction
does not waive geometric support or visual review.

The final seal rechecked **384 SHA-256 bindings** across requests, retained
outputs, original geometry, raw depths, RGB inputs and implementation files.
Five focused source-gate/intersection-selection tests pass. The source
installer is single-use in a fresh process; its test checks unexpected template
layout, not re-entrant installation. Production and published videos are unchanged.

## Insights

Visibility derived from the same faulty mesh can incorrectly validate a train
source. Measured source depth detects this circular consistency, but applying
strict support everywhere replaces ordinary coverage gaps with black holes.
Conditional peeling can recover an already existing deeper receiver without
inventing RGB or geometry; it cannot supply a receiver that does not exist,
and it cannot remove all false geometry that survives the confidence gate.
Further work needs a bounded geometry hypothesis, not broader unconditional
texture rejection. Do not treat these render-only outputs as a repaired mesh.
