# DEC5 dynamic camera phase: same path, different actor/view synchronization

## What was tested

The user authorized a shot-level workaround for the cheek hole while mesh
research continues. A uniform height offset hid late holes but exposed an early
hand/chin contact notch ([height control](dec5_residual_jaw_camera_height.md)).
This control instead cyclically shifts the **same** smooth elevated camera path
relative to the **unchanged** 150 real actor times. There is no reduced camera
excursion, fixed-camera substitution, crop, stabilization, mesh edit or new color
correction. Per-frame normalization is inverted and reapplied when transferring
a camera from another path index; normalized translations are never copied as
if they were common calibration coordinates.

`probe_temporal_camera_phase.py` produced 45 CPU clay views: offsets -30, -15,
0, +15 and +30 path samples at nine real times (000899, 000973, 001029, 001083,
001123, 001191, 001193, 001195, 001197). Zero phase checks the original pose.
Native +30 head crops were inspected on the initial and late problematic times;
overview comparisons screened the middle times. These are visibility tests,
not reconstructed-depth improvements or GT image metrics.

`study_camera_phase_rgb.py` freezes +30 (one fifth of the loop) and renders nine
initial RGB canaries with exact published source-mask/angular-prior wrappers,
unchanged meshes, camera profiles, exposure and hard source RGB. The immutable
request contains the full 150-frame candidate, gated before the remaining runs.

## Results

CPU probe root: `/mnt/data/dec5_temporal_camera_phase_probe`.
RGB/video candidate: `/mnt/data/dec5_phase30_dynamic_150`.
The earlier published movie remains untouched.

All nine initial GT-free native head comparisons were actually inspected and
hash-bound in `initial_review/visual_review.json`. The conspicuous isolated late
jaw spots are hidden or reduced to tiny edge speckles, without the uniform-height
control's stronger early hand/chin notch. Existing coarse hair/fringe, crown
holes (notably 001123), color seams and lipstick rear defects remain. Forearm
and lower-boundary geometry are not repaired by this trajectory change.

The initial gate permits **full-sequence evaluation with known limitations**;
it does not certify artifact-free images. No individual geometry reruns are
triggered merely by minor contour/color issues, following the user's instruction.

![Late jaw comparison](/mnt/data/dec5_phase30_dynamic_150/initial_review/001193_head.png)
![Initial lipstick/hand comparison](/mnt/data/dec5_phase30_dynamic_150/initial_review/000899_head.png)

`audit_temporal_camera_phase.py` verifies the chronological actor inventory,
unchanged mesh identity, a cyclic permutation of all 150 calibrated camera poses,
fixed intrinsics, 150 distinct nonstationary positions, train-only source lists,
and completed render hashes. A unit test independently exercises positive,
negative and wrapped phases under different per-frame scales/translations.

All 150 frames are rendered and encoded: `video.mp4`, 1080 x 1920, 24 fps,
6.25 seconds, no slow version. Four disjoint workers on clever-shadow finished
the remaining 141 frames in 1,022 seconds (17.0 minutes), reusing the nine
audited canaries. `checks.jsonl` contains 30-second supervision through normal
worker termination, without CUDA/OOM failures. Approximately 8.3 new frames/minute
is operational throughput, not an isolated speedup benchmark.

The complete phase/provenance audit passes. Independent publication checks find
150 unique actor times, meshes and renders; all RGB is exactly native rotation
without crop/stabilization. Six independent raycasts match saved depth. Camera
view span is 31.513 degrees, maximum/minimum loop step ratio 1.01238; fixed scene
landmark travel is 384.0 x 153.6 pixels and actual rendered foreground-centroid
travel is 301.4 x 150.4 pixels. The path is exactly the earlier elevated loop,
phase-shifted, not a newly narrowed excursion. Original-triangle preservation
and bounded-head locality pass independently across all 150 meshes.

All 15 ten-frame overviews and all 15 sheets decoded from the actual MP4 were
directly inspected; four distributed full native images were also reviewed.
`jaw_review/` contains 25 additional sheets covering **all 150 native jaw crops**
with explicit hash-bound notes and a completed review. Exposed cheek/chin regions
avoid conspicuous broad black cavities; early hand occlusion limits what can be
seen, and tiny late black flecks remain, notably 001195. This is partial shot-level
improvement, not zero missing pixels or geometry recovery.

Known failures remain: forearm/hand holes around 001029–001043 (overview groups
060–079 explicitly fail); rear lipstick skin/blue fins, especially around
000995–001007; crown/right-hair notches at 001123; changing skin/neck source seams
and open lower torso. None is removed from the output inventory. The decoded
sequence shows continuous viewpoint and actor changes; this inspection does not
pretend to be a real-time playback judgment. The camera is periodic, but actor
motion is not: looping the movie resets the actor pose.

Seven focused camera-path tests pass. `frames.zip` retains the 150 ordered PNGs;
publication checks bind archive contents, renders, requests, reviews and reports.
The delivered status is **camera and actor dynamic, artifacts remain**, not
artifact-free success. No full-frame quality metrics were computed.

## Insights

Camera-path phase is a useful independent control: it changes which view sees
each dynamic pose without sacrificing camera travel or smoothness. Fewer visible
holes must not be described as better mesh geometry. Remaining source seams and
hair/lipstick reconstruction problems require their own evidence and fixes.

Parallel geometry studies are intentionally separate from this immutable movie:
[12/24/36 source-count ablation](dec5_patchmatch_source_count_ablation.md) did
not establish a reliable crown-hole fix; 36 sources degraded all three face
metrics at both tested times and cost about 34% more runtime. Keep 12 for now.
[Confidence-gated local forearm completion](dec5_forearm_plane_transfer_v3.md)
gave a partial three-time improvement while preserving original triangles.
Its newly diagnosed image-border semantic veto was fixed consistently; the
001037 artificial horizontal split fell from 248 missing pixels to zero.
This remains inferred planar geometry with wrist/cuff defects, tested through
the raw local renderer, not accepted production geometry or learned anatomy.
Neither experiment changes the meshes or texture settings in this video.
