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

Full rendering is active with four disjoint frame workers on clever-shadow.
`checks.jsonl` records process/GPU/free-space status every 30 seconds. Initial
operational throughput is approximately eight frames/minute; this is not an
isolated performance benchmark. Full inventory, native/encoded temporal review,
encoding and publication audits remain required before delivering the candidate.

## Insights

Camera-path phase is a useful independent control: it changes which view sees
each dynamic pose without sacrificing camera travel or smoothness. Fewer visible
holes must not be described as better mesh geometry. Remaining source seams and
hair/lipstick reconstruction problems require their own evidence and fixes.
