# DEC5 camera flight: insufficient travel diagnosis and 3x3 / 4x4 controls

## What was tested

The user correctly rejected the apparent near-static camera in the previous
150-time video. We inspected the **actual retained poses**, their mesh-normalization
round trip, and the ray generator. `camera_depth` uses the requested camera pose
in its inverse extrinsic; the target renderer does not substitute an eval/train
camera or reuse an input RGB image as the final target.

The actual error was trajectory design and acceptance: the old controller used
a contracted four-anchor curve (G/I and B/D), then kept only 150 of its 360 loop
samples. The radius was a dimensionless blend amplitude, **not a number of camera
rows**. Its exact anchor-grid spans were only **1.22838 horizontal intervals and
0.64998 vertical intervals**. The first/last orientations differed by 8.572 degrees.
Smooth nonzero motion and convex containment were tested, but a minimum multi-row
travel extent and a camera-only visual control were missing.

Fixed look-at framing kept the actor centered, and simultaneous actor motion made
camera movement less obvious. Look-at is not itself a projection bug: the new
controls retain it. The missing range test, not removal of look-at, is the fix.

`diagnose_camera_grid_flight.py` now compares the old path against two **static
000973 camera-only** controls. All use the same existing hard-texture-v2 atlas,
same lighting encoded in that texture, and same fixed intrinsics. There is no
new training, diffusion, RGB blending or temporal actor animation. Resolution is
540x960 portrait, deliberately a quick camera-control preview rather than the
previous native 150-time deliverable.

New path: periodic cubic spline in two rig-grid coordinates, bilinear placement
inside four calibrated train corners, constant arc-length sampling, fixed look-at.
The 3x3 pilot spans columns G..I and rows B..D; the 4x4 pilot uses F..I and A..D
so none of its four anchors is held-out. Three camera levels mean two intervals;
four levels mean three intervals. A 2% inset at each end retains a safety margin.
Because the rig has only five vertical levels, 4x4 necessarily approaches an outer
level; it cannot also keep a two-row margin from both vertical edges.

## Results

| Control | Horizontal / vertical span, grid intervals | Duration | Frames | Angular speed min/median/max, degrees/s |
|---|---:|---:|---:|---:|
| Old path, frozen actor | 1.228 / 0.650 | 5 s | 150 | historical open path |
| 3x3 spline | 1.920 / 1.920 | 20 s | 480 | 2.241 / 2.376 / 2.657 |
| 4x4 spline | 2.880 / 2.880 | 30 s | 720 | 2.309 / 2.463 / 2.560 |

All new spline weights are nonnegative. Actual speed max/min is 1.00002235 for
3x3 and 1.00001223 for 4x4, including the closing interval. An initial control
omitted the final dense arc segment and had a 2.4–3.4% seam-speed discrepancy;
it is preserved at `/mnt/data/dec5_camera_grid_diagnosis`. Corrected final controls
are at **`/mnt/data/dec5_camera_grid_diagnosis_v2`**. Tests cover full two-axis span,
convex containment, closure and uniform speed. Existing model/renderer defaults
and prior immutable outputs are untouched.

[Actual camera-center travel](/mnt/data/dec5_camera_grid_diagnosis_v2/camera_travel.png),
[3x3 decoded details](/mnt/data/dec5_camera_grid_diagnosis_v2/3x3/encoded_detail.png),
[4x4 decoded details](/mnt/data/dec5_camera_grid_diagnosis_v2/4x4/encoded_detail.png).

Actual rendered and decoded quarter-cycle comparisons were inspected: both new
paths change top/bottom viewpoint and left/right parallax with the actor frozen;
4x4 makes this more visible. Known hair fringe, skin/shoulder texture seams and
small chin defects remain. These controls validate **camera travel**, not a new
artifact-free mesh or all-frame visual quality. No new image-quality metrics.
`integrity_audit.json` verifies all 1,350 PNG hashes, MP4 hashes, encoded frame
counts, resolutions and durations. Two focused regression tests passed.

Videos:

- `3x3/video.mp4`: `bfa77b00cd3c16aff167f0d695d9b66d7feea902e6d0f27c974480030d7245fa`
- `4x4/video.mp4`: `8998c29383f323f4c14671752104be1ba2a4ef8b111addcd7ab5d3d1bf46f345`

## Insights

“Inside the central cameras” and “smooth” do not establish that a flight traverses
several rows. A minimum **achieved span in both rig axes** is now an explicit gate.
The static-object control isolates camera motion from the girl's movement.

Do not compress a 20–30 second, multi-row slow path back into five seconds or use
only its first five seconds: the former restores excessive speed; the latter
recreates the insufficient-span mistake. Applying this path to moving footage
requires matching the source time interval and output duration explicitly. These
two pilots contain one source instant, not 480/720 independent temporal meshes.
