# DEC5: boundary-preserving curved jaw caps

## What was tested

One-time geometry canary, 001193, following the
[measured-mask control](dec5_jaw_measured_mask.md). Three alternatives preserve
every production vertex and triangle, the original cap perimeter, fixed cameras,
exposure, hard source RGB, masks and native measured-depth vetoes.

`subdivide_boundary_caps.py` inserts one centroid per proposed triangle and
splits it into three. The curved arm fits a quadratic height function to the
cap boundary and two original-mesh adjacency rings, weighting boundary vertices
four times more strongly. Only centroids move; the maximum allowed displacement
is .0005 normalized units. Rank-deficient, poorly fitted or over-bound patches
stay planar. This is an inferred surface continuation, not measured anatomy.

The third arm doubles proposal extent/gap/plane-RMSE bounds to .006/.003/.0004
and permits 36 boundary vertices, while preserving the same .0005 displacement
limit and all admission/depth guards. It tests wider coverage, not a relaxed
free-space veto. No target RGB/camera or held-out view constructs the geometry.

All three candidates receive four fresh matched renders: production/repaired
at the actual phase+30 moving pose and physical train F004_E005_1210FP. The
earlier flat-repair renders are separately hash-verified. This is 12 new RGB
renders, not a full-video rerun or a new held-out evaluation.

## Results

| 001193 / frozen F/E skin diagnostics | Interior depth misses | Edge-inclusive depth misses | Edge-inclusive black RGB |
|---|---:|---:|---:|
| Production | 30 | 45 | 58 |
| Previous flat repair | 17 | 22 | 31 |
| Subdivided flat | 19 | 24 | 33 |
| Small curved | **14** | **16** | **26** |
| Larger curved | 19 | 21 | 30 |

These are the two previously frozen train skin polygons (3568/5712 pixels),
not face PSNR/SSIM/LPIPS or full-frame quality metrics. Smaller missing-pixel
counts are not proof of correct anatomy. Subdivision can alter sampling and
renderer triangle labels even without bending; its separate flat control matters.

The small curved proposal moves centroids by at most .000453313 and retains
149 of 1395 proposed triangles. The larger arm moves at most .000461492 and
retains 167 of 2169. The flat arm retains 156 of 1395. All three final meshes
have zero nonmanifold edges and preserve exact original geometry arrays.

The small/large curved moving renders are byte-identical. Compared with the
previous flat repair, they change only 22 moving-view RGB pixels. In F/E, the
small curved repair changes 68 RGB pixels. Thus the gain is real but local,
not a substantial whole-head/video improvement.

![Native real-view jaw comparison](/mnt/data/dec5_subdivided_jaw_caps/review_large/F004_E005_1210FP_detail.png)
![Moving-view comparison](/mnt/data/dec5_subdivided_jaw_caps/review_large/moving_detail.png)

Main-agent visual inspection covered all eight saved head/detail comparison
panels in `review/` and `review_large/`. The small spot and open under-chin edge
remain visibly broken in F/E. Existing hair/crown geometry and skin seams remain.
All candidates **fail artifact-free approval**; the small curved candidate is
retained only as a partial local improvement, not promoted into the movie.

An independent audit replays proposal geometry and bounded fitting, verifies
producer ancestry (initial versions archived before opt-in expansion), and runs
372 fresh native ray checks: both pixel offsets on all 62 cameras for each arm.
There are zero qualified measured free-space violations. Initial sample evidence
is hash-checked but not independently recomputed by this audit. Synthetic tests
verify exact original/perimeter preservation and a bounded nonzero curved fit.

The first large comparison started before the last render receipt existed and
correctly refused an incomplete frame. Its partial panel directory is retained
as `review_large_incomplete_render_race`; after the renderer exited normally,
comparison alone was rerun. No RGB was regenerated for this event.

Roots: `/mnt/data/dec5_subdivided_jaw_caps` and
`/mnt/data/dec5_large_curved_jaw_caps`. Source data, published movies and existing
model/runner defaults remain unchanged.

## Insights

Local curvature helps more than finer flat triangles, but simply enlarging this
boundary-arc family does not solve the residual neck opening. This is a small
measured mesh improvement, **not completion of the requested substantial repair**.
The next geometric hypothesis should change how a continuous missing surface
is proposed, rather than repeatedly expanding these caps or relaxing confidence.
The dynamic artifact-free video objective remains open.
