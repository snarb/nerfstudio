# Artifact-aware dynamic camera options

## What was tested

Three user-requested smooth camera alternatives reuse the published production
geometry and incidence-2, fixed-profile, fixed-exposure, hard single-source,
zero-registration renderer. Each uses all 150 distinct actual times
`000899..001197` with step 2, at normal **24 fps / 6.25 seconds**. Camera and actor
both move. No slowed version, repeated source time, RGB average, moving crop,
image stabilization or geometry repair is introduced.

The paths are **open shots**. They preserve the published initial camera position
and smoothly approach the real `G004_C005_121037` camera center at the end. This
avoids the rejected periodic workaround's late correction wrapping into the
initial hand pose. The final camera position settles while camera orientation
continues to follow the shot composition and actor time continues normally.

The middle paths interpolate actual train-camera centers in central G..J and
C..B. A 30-frame smooth transition preserves the production start. The elevated
path stays below row B; the rig has only five physical rows. The fixed virtual
lens and the production 384 × 154 pixel landmark travel remain; different
apparent actor sizes are caused by actual camera distance.

| Option | Camera behavior | Viewing-angle extent | Landmark travel |
|---|---|---:|---:|
| Sweep | Gradual lateral sweep, shallow elevation | 18.73° | 384 × 154 px |
| High arc | Rises and pulls farther back before approaching the ending view | 19.31° | 384 × 154 px |
| S curve | Crosses left, returns right, then reaches the ending view | 18.43° | 384 × 154 px |
| Closer portrait (additional candidate) | Same physical S curve, fixed 1.7× closer lens and analytic composition | 18.85° | 180 × 40 px |

These angles are smaller than the published 31.51° route by design. Screen-space
composition and physical camera translation remain visible. Constant camera
speed and a seamless video loop are not claimed.

## Results

The original three paths completed all **450 renders** with six workers exiting
normally. All three independent 150-frame integrity audits pass. The three
normal-speed MP4s are encoded and verified; the additional closer-portrait
full render remains in progress.

| Ready video | Format | PNG archive |
|---|---|---|
| [Sweep](/mnt/data/dec5_artifact_aware_variants/sweep/video.mp4) | 1080 × 1920, 24 fps, 6.25 s | [150 frames](/mnt/data/dec5_artifact_aware_variants/sweep/frames.zip) |
| [High arc](/mnt/data/dec5_artifact_aware_variants/high_arc/video.mp4) | 1080 × 1920, 24 fps, 6.25 s | [150 frames](/mnt/data/dec5_artifact_aware_variants/high_arc/frames.zip) |
| [S curve](/mnt/data/dec5_artifact_aware_variants/s_curve/video.mp4) | 1080 × 1920, 24 fps, 6.25 s | [150 frames](/mnt/data/dec5_artifact_aware_variants/s_curve/frames.zip) |

All 150 decoded frames in each MP4 were checked against the matching native PNG,
and the ZIP inventories were verified. Every source frame was visually inspected
in chronological thumbnail contact sheets, plus the native critical canary
comparisons and 15 decoded MP4 samples per clip. This is explicit frame/contact
inspection, not a claim of certified real-time playback review. Each option's
`publication.json` binds its request, input/code provenance, audit, video/ZIP,
review notes and the actual inspected images. The result is a reviewed choice
with known residuals, not a production replacement.

| Original path | Actual rendered foreground-centroid travel | Fresh saved-depth checks | Distinct times / meshes / renders |
|---|---:|---:|---:|
| Sweep | 401 × 192 px | 7 pass | 150 / 150 / 150 |
| High arc | 401 × 182 px | 7 pass | 150 / 150 / 150 |
| S curve | 368 × 190 px | 7 pass | 150 / 150 / 150 |

All requested camera centers are inside the train-camera-center convex hull.
Four additional casts of the same frozen actor mesh from separated movie poses
verify actual camera movement independently of actor motion. Every frame's
saved depth is finite and every output RGB is exactly the native render rotated
to portrait; there is no post-render crop, translation or stabilization.

The first three paths do **not** meet complete hole avoidance: full-sequence
overview inspection shows conspicuous inherited forearm/hand holes during
`001029..001045`. They remain useful camera choices but are not clean results.
An additional authorized closer-portrait candidate uses the same physical
S curve with a fixed 1.7× lens and fixed analytic composition. Its actual camera
field of view excludes the lower-forearm defect at `001029/001037` while retaining
the whole crown and painting hand at `000899/000995`. At `001037` the lowered
hand naturally leaves the frame. This is not a per-frame tracked or post-render
crop. Existing crown roughness/opening is more visible at the larger portrait
size, so the closer option is not artifact-free either.

- [Closer portrait keeps the painting hand](/mnt/data/dec5_artifact_aware_variants/closer_portrait/review/000899_comparison.png)
- [Lower forearm leaves the actual camera field](/mnt/data/dec5_artifact_aware_variants/closer_portrait/review/001029_comparison.png)
- [Later dropping hand exits naturally](/mnt/data/dec5_artifact_aware_variants/closer_portrait/review/001037_comparison.png)

All 27 native RGB canaries (9 times × 3 paths) completed normally. The agent
inspected all three geometry contact sheets and native comparisons for initial
`000899`, early `000929`, lipstick `000995/001005`, hand `001029`, middle `001059`,
crown `001123`, and late jaw `001193/001197`.

| Region | Actual canary observation |
|---|---|
| Initial chin, `000899` | Preserves the published appearance; no repeat of the earlier periodic correction's large initial cutout |
| Late jaw, `001193/001197` | Isolated fleck hidden from the common G/C ending view; thin jagged neck/chin boundary remains |
| Crown/cheek, `001123` | Broad appearance improved from this view; narrow crown opening and rough hair edges remain |
| Lipstick, `000995/001005` | Known fin remains, especially exposed in sweep/high arc; S curve stays closer to frontal here |
| Hand/forearm, `001029` | Known lower-forearm gap remains; these are camera changes only |
| High-arc framing | Actor is smaller during the early/middle pullback; real distance change with unchanged intrinsics |

- [Initial native jaw comparison](/mnt/data/dec5_artifact_aware_variants/review/000899_jaw.png)
- [Crown comparison](/mnt/data/dec5_artifact_aware_variants/review/001123_head.png)
- [Late native jaw comparison](/mnt/data/dec5_artifact_aware_variants/review/001193_jaw.png)
- [Lipstick comparison](/mnt/data/dec5_artifact_aware_variants/review/001005_head.png)
- [Explicit canary verdict](/mnt/data/dec5_artifact_aware_variants/canary_visual_review.json)

The outputs live under `/mnt/data/dec5_artifact_aware_variants/` in
`sweep`, `high_arc`, and `s_curve`. The source production output remains
`/mnt/data/dec5_incidence2_unwarped_dynamic_150`.

Copy the ready movies from a machine with SSH access to `clever-shadow`:

```bash
scp clever-shadow:/mnt/data/dec5_artifact_aware_variants/sweep/video.mp4 ./dec5_sweep.mp4
scp clever-shadow:/mnt/data/dec5_artifact_aware_variants/high_arc/video.mp4 ./dec5_high_arc.mp4
scp clever-shadow:/mnt/data/dec5_artifact_aware_variants/s_curve/video.mp4 ./dec5_s_curve.mp4
```

## Insights

Constraining the first and last views independently avoids the known periodic
regression and hides the late isolated jaw fleck. The different middle routes
trade viewing angle, size and defect visibility. These are choices for the user,
not an artifact-free reconstruction or a production replacement.

Camera views differ, so pixelwise PSNR/SSIM/LPIPS comparisons between these
movies would not be valid. No image-quality scores or loss are reported for
this path comparison.

Replay helpers:

```bash
../.venv/bin/python scripts/artifact_aware_video_variants.py init
../.venv/bin/python scripts/artifact_aware_video_variants.py screen
../.venv/bin/python scripts/artifact_aware_video_variants.py canary
../.venv/bin/python scripts/artifact_aware_video_variants.py panels
# After actual canary inspection:
../.venv/bin/python scripts/artifact_aware_video_variants.py supervise
../.venv/bin/python scripts/finalize_artifact_aware_video_variants.py sheets
../.venv/bin/python scripts/finalize_artifact_aware_video_variants.py audit
../.venv/bin/python scripts/finalize_artifact_aware_video_variants.py encode
```

The supervisor keeps exactly six workers total and writes process/GPU/free-space
checks every 30 seconds. The final audit verifies all native PNG receipts,
unchanged production meshes and source masks, source-time provenance, fixed
radiometry, finite saved depths, fresh raycasts at seven actual poses, and a
separate frozen-actor camera-motion control. Encoding requires the audit and
decodes all 150 MP4 frames to verify their order and content against source PNGs.
