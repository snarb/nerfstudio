# DEC5: limits of camera phase and partial skin annotation

## What was tested

Two separate hypotheses, with the published movie unchanged:

1. Move the unchanged smooth camera loop relative to the actual actor times to
   avoid the bad hand/forearm interval. `probe_forearm_camera_phase.py` tests
   relative offsets 0/30/60/90/120 on 001029/001033/001037/001041. Each candidate
   retains all 150 calibrated poses, the same excursion and periodic spacing.
   Cameras are transferred through calibration coordinates, not copied between
   incompatible per-time normalizations. These are 20 CPU clay renders of the
   **published production meshes**, not the newer local forearm candidates.
2. Treat pixels outside the three inset positive skin annotations as unknown,
   rather than negative skin evidence. The new opt-in
   `--positive-only-annotations` still requires two positive cameras and the
   protected-production composition. It does not dilate masks, change geometry
   bounds, alter exposure or weaken measured free-space checks. All 62 native
   depth cameras still check newly added faces. This is an experimental evidence
   policy, not a claim that every unlabelled pixel really is skin.

The geometry canary uses 001037, with the same fitted quadric, 49 feasible depth
samples, native pins, renderer and fixed train H/A scoring ROI as the preceding
[protected-production control](dec5_protected_forearm_surface.md).

## Results

### Camera phase: rejected as a complete workaround

Root: `/mnt/data/dec5_forearm_camera_phase_probe`.
All four overview sheets were inspected, plus native 001037 arm crops at
offsets 0 and 120. Damage persists in the usable views. Offsets 30/60 place part
of the hand outside the image at the bad times: disappearing off-screen is not
accepted as repair. The phase control therefore does not justify a new full RGB
movie. This result does not negate the earlier useful phase+30 cheek workaround.

![Four-time phase probe, 001037](/mnt/data/dec5_forearm_camera_phase_probe/001037/overview.png)

### Partial annotation: negative RGB canary

Root: `/mnt/data/dec5_partial_skin_annotation_roundoff_control`.
Review: `/mnt/data/dec5_partial_skin_annotation_review`.

| 001037 / fixed train forearm ROI | PSNR | SSIM | LPIPS | Black RGB pixels | Depth misses |
|---|---:|---:|---:|---:|---:|
| Protected-production baseline | 21.90890 | 0.758793 | 0.355989 | 966 | 910 |
| Partial positive annotations | 20.94952 | 0.761375 | 0.365393 | 1167 | 1071 |

These are **train reprojection metrics, not held-out face scores**. Neither the
main face CSV nor any full-frame quality metric is changed/computed. Both new
RGB views are fresh renders using the identical camera, hard-source texture
policy, exposure and camera profiles as the matched baseline.

Domain pixels increase only 11435 → 11484; native pins 187 → 208; active
production-hole pixels 10023 → 10048. The changed constrained solution proposes
19752 faces and retains 18426 after four native-guard passes (previously 18745
retained after five). Allowing more points does not imply monotonic final
coverage: pins/solve and subsequent carving interact.

Independent replay reproduces the samples, solve, assembly and 124 native ray
checks. All 139053 production triangles remain intact, no new misses occur over
previous production faces, and nonmanifold edge count is zero. Index components
increase to 277 after carving, not an artifact-free/topologically welded mesh.

Both native RGB comparison panels were inspected. The lateral dark gap is
larger; the wrist hole and broken hand remain. The canary is **rejected** and is
not transferred to the other times or substituted into the movie.

![Moving view, protected / partial annotations](/mnt/data/dec5_partial_skin_annotation_review/001037/moving_comparison.png)
![Train GT and matched control](/mnt/data/dec5_partial_skin_annotation_review/001037/H004_A005_1210M6_comparison.png)

### Numerical guard correction

The first preparation stopped before mesh publication because subtracting model
depth from solved depth produced `0.01200000000000001` from an exact selected
residual `0.012`. Independent solver replay reproduced the excess
`1.0408340855860843e-17`. `displacement_within_bound` now permits only eight
float64 roundoff units scaled by operand magnitude, not a geometric margin.
The test rejects a real `1e-9` excess and nonfinite input. The failed request,
sample evidence and exact changed producer sources remain in
`/mnt/data/dec5_partial_skin_annotation_control`; the rerun has a separate root.

Twenty-four focused tests pass, including preservation of the measured
free-space veto under partial annotations, the two-positive-view requirement,
production protection, bound roundoff and camera phase transfer.

## Insights

A partial ROI should not casually be treated as an exhaustive silhouette, but
relaxing that interpretation is not the missing large repair in this bounded
forearm model. It adds only 49 eligible domain pixels and worsens the final
render after the coupled solve and carving. Keep the stronger previous local
candidate; do not promote this negative test on its slight SSIM improvement.

The remaining hand/wrist defects require a surface model and observations for
those anatomical regions. The existing inset forearm domain does not cover the
whole wrist or articulated hand. Camera phase alone cannot hide these defects
while keeping the hand in frame. The goal remains unfinished: there is no new
artifact-free video, and the published 150-time movie is unchanged.
