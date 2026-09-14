# DEC5: independently verified wide dynamic camera traversal

## What was tested

The prior dynamic video was rejected by the user as visually stationary. It is
not treated as successful just because its pose metadata changed. This follow-up
tests the renderer itself on fixed geometry, and then combines the verified wide
camera path with **150 different chronological scene times**, 000899–001197.

The rig contains fourteen horizontal columns A..N, but only five vertical rows
A..E. The new horizontal sweep is **H−4 to H+4 (D..L)**. The vertical sweep uses
the entire available height **C−2 to C+2 (A..E)**. Literal ±4 vertical physical
rows do not exist in this calibration; no cameras outside the rig are invented.
This limitation was disclosed before the new rendering job.

`wide_dynamic_camera_flight.py` constructs a closed periodic cubic trajectory:
left → right → top → bottom → left. The entire spline is normalized to its
extrema, then sampled by 3D arc length; samples are not clipped. Convex weights
inside four real train anchors map the path into calibrated 3D positions.
Native camera X, not Y, fixes portrait-up for the sideways source sensors.
Intrinsics and optical target are fixed. Actual perspective is rendered;
there is no 2D pan/zoom effect and no generated intermediate temporal image.

One verified per-time mesh is used for each real time, with the same frozen
camera RGB profiles, display exposure, hard-source RGB and train-only source
masks as the prior dynamic run. A view-dependent silhouette safeguard cannot
simply be reused for a new camera: `prepare_wide_dynamic_geometry.py` restores
old proposed deletions where the new target would see a deeper backing surface
or a newly enclosed hole. Restoration is monotonic: no new triangles are deleted,
and no geometry absent from the original reconstruction is invented. The old
source images, original meshes, masks and published outputs remain unchanged.

## Results

Superseded by the user's subsequent selection of the earlier static 4×4 path.
The wide run rendered 150/150 frames in 722 seconds, but is **not** the chosen
video delivery and was not fully encoded/reviewed/published. Four native extreme
views were inspected. See `dec5_replayed_4x4_dynamic.md` for the active request.

The fixed-time diagnostic replays the **actual old and new poses** on the same
000973 mesh/atlas. The four old poses do produce different images: perspective,
hand/face occlusion and silhouette change. Independent fresh raycasts at old
frames 000899, 001049 and 001197 match their saved depth rasters. Thus the prior
failure was not a renderer silently replacing every pose by one camera. This
does not overrule the user's rejection of the visible result.

The old view-direction range was only −13.88°..+0.04° horizontally, with a single
diagonal traversal rather than a horizontal sweep, vertical sweep and return.
Vertical view directions spanned −8.63°..+17.35°. A fixed focus, missing room
parallax and simultaneous head movement make camera motion difficult to separate
from actor motion in that presentation. The last explanation is an inference;
the independent depth checks and fixed-time images are direct evidence.

The new direction ranges are −29.74°..+25.26° horizontally and
−19.76°..+20.60° vertically. Maximum pairwise view angle is **55.058°**.
The achieved grid spans are approximately **7.995 × 4.000 intervals**. The
right, top and bottom extrema occur at output indices 49, 78 and 118. The loop's
maximum/minimum per-frame translation ratio, including its closing step, is
1.0173. Four initial native dynamic renders at times 000899, 000997, 001055 and
001135 were inspected: opposite lateral perspectives and top/bottom perspectives
are plainly visible while hand/expression/head state changes with source time.

The wide views expose known geometry defects, including neck openings,
lipstick-side clothing contamination and hair/skin source seams. New-view
restoration addresses holes caused by *old silhouette deletions*, not holes
already missing from the original reconstruction. This is a camera-control
correction, not a claim that all reconstruction artifacts have been solved.

Artifacts:

- Diagnostic only (one actor time):
  `/mnt/data/dec5_wide_dynamic_flight_150/camera_probe/`.
- Initial dynamic canaries, without new-pose restoration:
  `/mnt/data/dec5_wide_dynamic_flight_150/frames/`.
- Final dynamic run with new-pose restoration:
  `/mnt/data/dec5_wide_dynamic_flight_150_v2`.

No novel-view ground truth exists, so no full-frame or substitute PSNR/SSIM/LPIPS
is computed. The original held-out face-only campaign remains a separate task.

## Insights

1. Pose metadata alone is insufficient. Check actual saved depths against an
   independent raycast, compare identical geometry under extreme poses, and then
   verify the *dynamic* encoded video. A static diagnostic is never the delivery.
2. Rig intervals, horizontal/vertical extent, traversal order, and a closed
   camera return need separate gates; a small smooth diagonal is not equivalent.
3. Portrait-up must match the delivered image orientation, not a sideways
   sensor's conventional Y-up assumption.
4. 150 frames cannot cover a wide multi-part path both slowly and at normal
   temporal speed without more scene samples. The normal version uses 24 fps
   (6.25 s; camera speed 21.27–32.70°/s). A separate **12 fps / 12.5 s** inspection
   version slows both actor and camera by two, with lower cadence. It does not
   fabricate smooth intermediate frames. Camera closure does not make the
   nonperiodic actor clip seamlessly loopable.

Replay from the repository root, using the same pinned local environment:

```bash
python LookCloser/scripts/wide_dynamic_camera_flight.py init
python LookCloser/scripts/wide_dynamic_camera_flight.py probe
python LookCloser/scripts/prepare_wide_dynamic_geometry.py
python LookCloser/scripts/run_wide_dynamic_workers.py supervise \
  --output /mnt/data/dec5_wide_dynamic_flight_150_v2 --workers 8
python LookCloser/scripts/finalize_wide_dynamic_flight.py sheets
python LookCloser/scripts/finalize_wide_dynamic_flight.py encode
# Inspect sheets and decoded MP4 samples; record explicit findings.
python LookCloser/scripts/finalize_wide_dynamic_flight.py audit --require-reviews
python LookCloser/scripts/finalize_wide_dynamic_flight.py publish
```
