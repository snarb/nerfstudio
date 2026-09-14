# DEC5: moving actor, moving grid camera, and background contamination

## What was tested

The user correctly rejected the preceding static 000973 camera pilots as a
replacement for a dynamic video. This experiment explicitly binds each of 150
chronological instants (000899–001197) to that instant's own mesh and train RGB.
It does not reuse one actor mesh, repeat temporal frames, or morph images.

`dynamic_grid_flythrough.py` replaces the truncated circle by a smooth open
cubic S curve. Translation is sampled at constant arc length, orientation looks
at the fixed reference optical target, and intrinsics remain fixed. The selected
4×4 train hull is F..I / A..D; all four anchors are train, never held-out views.
The smaller 3×3 option is also implemented and tested. The 4×4 curve covers
2.88 camera intervals on **each** grid axis. This extent is an acceptance gate,
not just a requested parameter. No loop restart is included.

Playback is explicitly **24 fps, 150 unique source times, 6.25 seconds**. Compared
with the former 30 fps playback, actor motion is slowed by 20%; no artificial
intermediate temporal frames are invented. An open traversal avoids forcing a
20–30-second full circle into this short capture. Peak camera angular speed is
5.0474 degrees/s; median is 4.9002. Consecutive translation direction changes
are at most 0.5266 degrees. Linear speed max/min is 1.00000355. Path-generation
translation values are in the reference mesh gauge; the final independent audit
recomputes camera translation in raw calibration coordinates.

The exact last pose of the rejected static 4×4 video was also replayed on both
the old atlas and the already repaired static asset. The old atlas geometry SHA
is `ab4436f42a2c047c1208a1912844743f8316f5b0b2acab897b624f89b32d23a9`:
it is the full-block TSDF mesh, but **not** the later locally carved/cylindrical
repair. That asset-selection regression reintroduced its known rear slab. The
matched comparison is under
`/mnt/data/dec5_dynamic_background_diagnosis/known_repair_comparison.png`.
This diagnostic remains static and is not a dynamic deliverable.

`train_foreground_guard.py` is a separate opt-in experiment. For every real time
it constructs 62 real-train silhouettes from cached DeepLab person probabilities
and deterministic real-RGB GrabCut refinement. An unrecognized held object is
not forced to background just because its person probability is low near the
hand. Interior mask holes are filled; the accepted candidate uses a four-native-
pixel silhouette margin and twelve outside-camera votes at **every** triangle
vertex. Source EXRs, fixed profiles, fixed exposure and original meshes stay
immutable. No eval RGB, generated RGB, per-time exposure, or RGB average enters.

Silhouettes are fallible evidence, not independent depth. If existing independent
depth evidence matches the exact input mesh hash and triangle inventory, two
near observations protect that triangle from semantic removal. A monotone
target-view safeguard restores proposals that would reveal deeper backing
geometry or create a new enclosed geometry hole. These safeguards do not certify
the entire remaining mesh. Real-source depth rasters are additionally masked,
so the existing native footprint/registration checks reject background RGB taps.

Output roots:

- Matched dynamic control: `/mnt/data/dec5_dynamic_grid_150_control`.
- Rejected aggressive silhouette experiment: `/mnt/data/dec5_dynamic_grid_150_guard_v1`.
- Conservative dynamic candidate: `/mnt/data/dec5_dynamic_grid_150_guard_v2`.
- Native before/after diagnostics: `/mnt/data/dec5_dynamic_background_diagnosis`.
- Unpromoted material diagnostic: `/mnt/data/dec5_dynamic_metal_diagnosis`.

## Results

Completed on 2026-09-14: **150/150 changing-time renders and an integrity-valid
dynamic video; artifact correction is only partially successful.** All 150
native face, ear/hair and lips/hand sheets were inspected in groups of four (two
in the final group), with hash-bound individual verdicts. There are **114
`accepted_known_artifacts`, 36 `fail`, zero artifact-free passes, and zero pending
or uncertain final visual verdicts**. Acceptance here follows the user's
allowance for small seams/rough edges; it does not mean clean geometry.

| Final check | Result |
|---|---|
| Distinct chronological source times / meshes / renders | 150 / 150 / 150 |
| Source interval | 000899–001197, stride 2 |
| Decoded MP4 | 1080×1920, 150 frames, 24 fps, 6.250000 s |
| Actual grid travel | 2.88 × 2.88 camera intervals inside the central 4×4 hull |
| Angular speed min / median / max | 4.2707 / 4.9002 / 5.0474 degrees/s |
| Linear speed max/min ratio | 1.00000355 |
| Foreground triangle removal fraction min / median / max | 4.734% / 7.404% / 10.087%; this is not an image-quality metric |
| Final visual verdicts | 114 accepted with known defects; 36 failures |
| Tests | 35 passed; original model/runner defaults unchanged |

The actual encoded overview and four **consecutive, native-resolution MP4**
patch groups (indices 24–27, 40–43, 52–55, 146–149) were inspected separately.
They confirm actor expression/hand/head changes rather than a frozen actor, and
also confirm temporal variation in the remaining neck holes and tube-side
contamination. Smooth camera matrices do **not** imply flicker-free geometry.
The failed interval is 000947–001017 inclusive at stride 2 (36 frames); these
frames remain in the output, not silently omitted or replaced.

Representative worst cases: **000949/000951/000957/000959** for black neck
openings; **000975/000983/000995/001003** for lipstick-side clothing/edge
contamination. The final 001191–001197 views retain a much smaller torn patch
under the chin. In the matched interior-neck rectangle `[335,1160,410,1245]` of
000951, both control and guard contain 815 black pixels, with zero new black
pixels: the guard did not create that opening, but also did not repair it.
See `/mnt/data/dec5_dynamic_background_diagnosis/000951_neck_comparison.json`.

Review and delivery:

- [Camera path](/mnt/data/dec5_dynamic_grid_150_guard_v2/camera_path.png).
- [Actual encoded overview](/mnt/data/dec5_dynamic_grid_150_guard_v2/encoded_overview.png).
- [Encoded neck failures](/mnt/data/dec5_dynamic_grid_150_guard_v2/encoded_review/024_027.png).
- [Encoded lipstick failures](/mnt/data/dec5_dynamic_grid_150_guard_v2/encoded_review/040_043.png).
- [Exact source-pixel trace](/mnt/data/dec5_dynamic_grid_150_guard_v2/pixel_traces/000983/source_trace.png).
- Portable output: `/mnt/data/dec5_dynamic_grid_150`, with `video.mp4`,
  `frames.zip`, 150 ordered `frames/FRAME.png`, contact sheets, verdicts,
  request, audit, report and a checksum publication manifest.
- Meshes and source-support caches remain under the conservative candidate root.
  These are extracted surface meshes, not serialized raw TSDF volumes.

Video SHA-256:
`51694ffe71a21c7e05b8b475d8a6aa4d26b58630fa93ff1e9982aa944e36bf4f`.

The control used four workers and finished in 1865.5 s. The guarded run was
resumed from 32 checksum-valid frames with eight workers on clever-shadow's
96GB GPU; the remaining 118 completed in 1654.4 s. That is not an isolated 4×/8×
speed benchmark: the guard adds segmentation/GrabCut work and the control ran
concurrently for part of the interval. All eight final workers exited zero.
The first publication attempt hit shared-mount directory-metadata restrictions;
the atomic destination was not published. A byte-only recursive copier and a
regression test replace `copytree` metadata operations; the partial staging
directory is retained separately and never treated as a completed output.
Supervisor PID, stage/frame, GPU memory and disk checks were recorded every
30 seconds. The original four-worker supervisor's nonzero exit was a documented
owned-process restart, not an unexplained CUDA/OOM failure. No new PatchMatch
or training jobs were launched; rendering/review overlapped.

| Hypothesis / control | Observed result | Decision |
|---|---|---|
| One static mesh can stand in for the requested dynamic flight | Actor is visibly frozen | Reject deliverable; require 150 unique chronological geometry/RGB inputs |
| Latest static pilot used the repaired lipstick | Hash/provenance and matched-pose replay show the older rear slab | Confirm asset-selection regression |
| Six outside votes and a two-pixel margin can safely trim contours | 000973 develops upper-hair holes; 001197 exceeds the 15% carving gate | Reject v1; preserve workspace and script snapshots |
| Twelve votes, four-pixel margin and conservative restoration reduce the fringe | Native comparisons at 000899, 000973, 000975, 001051, 001125, 001197 show substantially less hair/shoulder background without a new large hole | Included in dynamic delivery after full review; not artifact-free |
| A person silhouette completely fixes lipstick contamination | Thin brown/blue tube-side samples remain at 000973/000975 | Reject this stronger claim |
| Neutral color alone reliably identifies the metallic object | Multi-view clusters include tube, eyes, highlights and clothing boundaries | Diagnostic only; no automatic cylinder replacement promoted |
| Object-specific material silhouettes can safely remove the remaining side wall | 000973 removes 34 triangles without convincing benefit; 000975 removes 135 but damages the upper finger/contact | Reject local object-silhouette control, do not promote |

Known visible defects are recorded rather than omitted: 000899 retains its
preexisting square neck opening and hand/face source seams; 000973/000975 retain
thin tube-side contamination; later views retain small chin/neck or shoulder
edges and skin color seams. The static asset's local cylinder prior was **not**
silently copied to other moving times.

Exact pixel attribution is saved under the conservative candidate's
`pixel_traces/000983/`. Target portrait pixel `(393, 1249)`, RGB `(56,100,139)`,
comes from **shirt fabric** in real train camera `K004_A005_1210EF` at native
coordinates `(688.3384, 242.9707)`. Reprojecting and applying the frozen response
reconstructs `(55.9664,100.4411,138.9777)`, matching the rendered integer RGB.
It is preferred graph source 45, **not a fallback and not an RGB average**.
The retained surface projects beside the actual metal onto the clothing behind
it. A person silhouette correctly retains that clothing, so it cannot fix this
internal object-occlusion error. Calibration distortion coefficients k1/k2/p1/p2
are all zero; an omitted nonzero distortion term does not explain this trace.

`test_temporal_object_silhouette.py` tested an explicitly disclosed lower-face /
hand search rectangle in real H/C RGB, a strong elongated neutral-material
cluster, real-camera metal masks with a two-pixel margin, at least six visible
outside-mask votes at every proposed vertex, endpoint avoidance, and strong
neutral-core protection. Matched renders are under
`/mnt/data/dec5_dynamic_object_canary/frames/000973` and `000975`; their actual
visual verdict is **rejected**, saved in `visual_review.json`. The masks include
bright nail/skin pixels and material-space evidence does not reliably protect
the finger in times lacking independent depth evidence. This negative control
must not be presented as a successful lipstick repair or copied into the video.

No target-ground-truth novel views exist along this flight. No full-frame,
room, candidate-surface, or surrogate LPIPS/PSNR/SSIM is reported. These video
diagnostics do not replace the separate held-out face-only metric protocol.

## Insights

1. Mesh self-visibility is circular evidence: an erroneous protrusion can pass
   its own visibility test and acquire real background, skin or shirt pixels.
   Taking one source rather than averaging two sources does not prevent that.
2. Global foreground silhouettes address **room versus actor**, not internal
   **tube versus hand/face** occlusion. The latter requires object-specific,
   temporally verified support, not a stronger person-mask threshold.
3. More aggressive trimming is not automatically safer. A silhouette boundary
   error can cut hair or fingers; independent depth protection and explicit
   before/after review matter. Remaining defects must not be called solved.
4. Camera speed, achieved two-axis extent, and source-time diversity are separate
   acceptance conditions. All three are checked; a smooth static-actor pilot is
   useful only as a diagnostic and cannot substitute for the requested video.

Replay (existing model and single-frame runner defaults are unchanged):

```bash
python LookCloser/scripts/dynamic_grid_flythrough.py init \
  --output /mnt/data/dec5_dynamic_grid_150_guard_v2 --grid 4 --fps 24 --guard
python LookCloser/scripts/run_dynamic_grid_workers.py supervise \
  --output /mnt/data/dec5_dynamic_grid_150_guard_v2 --workers 8
python LookCloser/scripts/review_encode_smooth_temporal_video.py sheets \
  --output /mnt/data/dec5_dynamic_grid_150_guard_v2
python LookCloser/scripts/finalize_dynamic_grid_video.py encode \
  --output /mnt/data/dec5_dynamic_grid_150_guard_v2
python LookCloser/scripts/finalize_dynamic_grid_video.py encoded-crops \
  --output /mnt/data/dec5_dynamic_grid_150_guard_v2
python LookCloser/scripts/finalize_dynamic_grid_video.py audit \
  --output /mnt/data/dec5_dynamic_grid_150_guard_v2 --require-reviews
```

The final audit requires distinct ordered source times, meshes and renders;
checks all frame and guard receipts; revalidates actual camera positions against
convex rig weights; and separates artifact integrity from actual visual verdicts.
The encoder uses the request's FPS rather than the older hard-coded 30 fps.
Eight workers require at least 80 GB reported GPU memory; use four on smaller
hosts. Re-encoding deliberately resets encoded-review status to pending; actual
MP4 inspection and its explicit verdict are required before publication.
