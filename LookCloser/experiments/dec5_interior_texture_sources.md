# DEC5: native hair-rim attribution and interior-source preference

## What was tested

2026-09-15. The wide-spiral `free` movie retains a tan/brown, ragged hair
outline. A read-only diagnostic traced six manually selected pixels at time
`001083` through the exact saved target raycast and source IDs to native train
RGB. These points locate witnesses only: they do not define a geometry edit,
target mask, or the subsequent algorithm's domain.

All 62 cameras were inspected numerically for foreground-mask distance. A
second diagnostic replayed the existing source visibility rule: masked mesh
depth, 0.0015 relative interpolated-depth tolerance, 0.003 relative four-tap
tolerance, the original image border, and pixel-center snapping. All six
original choices passed. Native witness sheets show the selected source plus
alternative **visible** sources farther inside the existing foreground mask.
Mask inclusion and depth agreement do not establish optical opacity.

The subsequent opt-in control uses the same meshes, camera poses/lenses,
calibration, fixed response/exposure, original foreground masks, and HD RGB at
`001083` and `001123`. At every mesh-triangle centroid and every target-surface
sample, it prefers geometrically admissible sources whose projected mask
distance is at least **16 native HD pixels**. If no such source is available,
the original admissible set is retained. The angular/incidence quality, graph,
visibility, and hard single-source RGB sampling remain in use. No camera RGB
averaging, alpha matting, target segmentation, per-frame manual exception,
geometry repair, or output crop is introduced.

This is a silhouette-interior heuristic, **not** an opacity estimator. The
16-pixel threshold is an initial test, not an optimized universal setting.
It applies to the whole surface, including neck and shoulders; that breadth
is important to the adverse observation below.

Code: `diagnose_cinematic_hair_rim.py`,
`diagnose_cinematic_hair_visibility.py`, `study_interior_texture_sources.py`,
and `seal_interior_texture_study.py`. Existing renderer/model defaults and
the concurrently rendering native-6K movie are unchanged.

## Results

| Diagnostic target x,y | Chosen-source mask distance | Admissible source count |
|---|---:|---:|
| 720,110 | 5.00 px | 18 |
| 744,157 | 8.06 px | 54 |
| 501,193 | 14.00 px | 32 |
| 289,359 | 7.81 px | 35 |
| 178,744 | 3.00 px | 27 |
| 710,240, interior control | 52.63 px | 60 |

Selected boundary-source patches already contain beige room color between
sparse strands. Alternative visible source patches show denser hair at the
same projected surface point. This is direct evidence that cross-camera
averaging is not required to produce the brown rim. It does not prove that
all contributing surface points are geometrically correct or that hair
correspondence is exact across viewpoints.

| Time | Changed source pixels | Changed RGB pixels | Newly black pixels | Depth / missing-source mask |
|---|---:|---:|---:|---|
| 001083 | 127,596 | 127,551 | 0 | Exactly equal |
| 001123 | 83,034 | 83,005 | 0 | Exactly equal |

These counts are diagnostics, **not PSNR/SSIM/LPIPS or face-quality scores**.
There is no exact RGB GT at either interpolated camera pose; none is fabricated.

The main LLM inspected six initial native-source panels, six visibility-qualified
panels, and six paired crown/face/body render crops. At both times, the tan/brown
hair band is reduced substantially. Existing jagged geometry, stretched fine
texture, and the crown opening remain. No conspicuous central-face degradation
was identified in these crops. The right neck/shoulder contour develops a more
conspicuous dark transition: globally forcing interior-source preference is
therefore **not approved for the full movie**. Temporal flicker is untested.

- [001083 crown A/B](/mnt/data/dec5_interior_texture_sources/review/001083_crown.png)
- [001123 crown A/B](/mnt/data/dec5_interior_texture_sources/review/001123_crown.png)
- [001123 face and neck A/B](/mnt/data/dec5_interior_texture_sources/review/001123_face.png)
- [Native visibility-qualified boundary witnesses](/mnt/data/dec5_cinematic_hair_rim_witnesses/001083/visibility/point_03.png)
- [Main visual verdict](/mnt/data/dec5_interior_texture_sources/visual_review.json)
- [Artifact inventory](/mnt/data/dec5_interior_texture_sources/artifact_manifest.json)

Three tests pass: preserve the last admissible source, never create visibility,
unchanged behavior for uniformly interior observations, inclusive threshold,
and rejection of malformed/non-finite evidence. The study verifier checks
retained hashes, source witnesses, exact target-depth equality, and exact
missing-source masks. No production mesh or video is replaced.

## Insights

1. Foreground segmentation is not foreground-color separation. An admitted
   pixel can contain both fine hair and room color; hard reprojection then
   paints that mixture onto an opaque triangle. Native 6K sampling sharpens
   detail but does not, by itself, remove this contamination.
2. This measured control establishes a useful texture improvement without
   changing geometry. It does **not** repair the holes under the cheek or the
   topological hair fringe, and cannot complete the artifact-free mesh goal.
3. Silhouette-interior preference has a real tradeoff on skin boundaries:
   changing source direction can reveal a darker shading/source transition.
   The next justified test is a train-derived hair/opacity-aware policy, with
   matched skin controls and a contiguous temporal sequence. Do not promote
   the global heuristic merely because the two hair crops improved.
4. Native-camera detail, geometric visibility, foreground opacity, and
   consistent surface position are separate requirements. A single binary
   mask or a count of agreeing depth cameras does not certify all four.
