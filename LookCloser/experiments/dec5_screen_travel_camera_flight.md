# DEC5: visible screen travel, not a permanently centered orbit

## What was tested

The user rejected the replayed 4×4 dynamic movie: the girl remained near image
center, so camera motion did not feel like a fly-through. Previous acceptance
checked changing rays, rig coverage and a moving actor, but did not require
visible object displacement inside the image. That was the wrong acceptance
criterion for the requested shot.

The old generator explicitly computed `z = camera_position - fixed_target` for
every pose. Its saved 720 optical axes converge on one scene point, whose image
coordinates remain `(960, 540)` with less than `4e-12` pixels variation. The old
renderer does **not** crop or stabilize the movie: it writes the native raycast
RGB and applies only a 90-degree rotation for portrait delivery. The encoder
reads those full images. Independent pixel-equality checks verify this on five
retained old frames. Thus the centering is in the **camera orientation**, not
postprocessing; it was not evidence that camera centers failed to move.

`screen_travel_camera_flight.py` is a separate opt-in route:

- Bilinear real camera centers expand from F..I to **D..K**, two columns farther
  on each side. Vertical A..D and the original loop phase are retained.
- A controlled optical-axis offset lets a fixed scene landmark travel across
  the portrait image, rather than always pointing the lens at it. This changes
  actual perspective rays, not rendered-image positions.
- The virtual focal lengths are fixed at 70% of the source reference lens for
  the entire movie, giving room for lateral composition. Principal point and
  image dimensions remain fixed. No animated zoom, crop, image shift or target
  RGB is used. Original physical calibration and source lenses are unchanged.
- 150 distinct chronological meshes and their own train RGB remain bound to
  times 000899–001197. Existing new-view conservative geometry restoration and
  hard texture selection are reused without new training or model-default changes.

`diagnose_screen_travel.py` renders the identical static 000973 atlas under old
and new poses to isolate screen displacement from the girl's own movements.
This comparison is diagnostic only, never presented as the dynamic movie.

## Results

Completed: 150 chronological dynamic renders and both encoded movies. All 150
frames were inspected on 15 original-render sheets and 15 sheets decoded from
the actual main MP4, in addition to four native-resolution phase checks.
Four native dynamic phases were inspected before the full run: the actor is
visibly left, central, right and raised in the image at different phases, while
her hand/head state also changes. All four heads remain in frame.

| Check | Result |
|---|---:|
| Fixed scene landmark screen span | 384.00 horizontal / 153.57 vertical pixels |
| Same-static-mesh silhouette centroid, old horizontal / vertical span | 71.81 / 31.55 pixels |
| Same-static-mesh silhouette centroid, new horizontal / vertical span | 286.51 / 177.79 pixels |
| Focused camera/normalization/temporal/audit tests | 44 passed |
| Horizontal expansion | 3 to 7 available rig intervals, F..I to D..K |
| Actual dynamic RGB silhouette-centroid span | 263.08 horizontal / 196.59 vertical pixels |
| Unique source times / meshes / renders | 150 / 150 / 150 |
| Native RGB equals portrait PNG after rotation only | 150 / 150 |
| Independent fresh raycast depth comparisons | 5 / 5 matched |
| Remaining full-run rendering, eight workers (four canaries reused) | 631.6 seconds |

The actual eight-image old/new static comparison was inspected: the new subject
visibly moves within the image, with roughly four times the old horizontal
silhouette-centroid travel. This isolates camera effects from temporal actor motion.

The wider fixed lens exposes more of the existing ragged lower-torso mesh
boundary. Hair-crown holes, small neck holes and source seams remain. This is
not an artifact-free reconstruction claim, and no novel-view PSNR/SSIM/LPIPS is
substituted for unavailable ground truth.

Camera motion and temporal identity pass independently of reconstruction quality.
All 150 visual verdicts retain `fail` for the unfinished lower-body boundary;
none is silently relabelled artifact-free. Especially visible failures include
the fragmented descending hand at 001039–001045, dark under-chin slit at 001083,
and crown openings around 001099–001149. The decoded sequence visibly traverses
left to right and returns along the upper arc while the hand lowers and head
turns. These are spatial frame-by-frame observations, not a claim of real-time
human playback evaluation or constant camera speed.

Evidence:

- [Identical-mesh old/new framing comparison](/mnt/data/dec5_screen_travel_dynamic_150_v2/framing_diagnosis/comparison.png)
- [Main MP4](/mnt/data/dec5_screen_travel_dynamic_150_v2/video.mp4)
- [Half-speed MP4](/mnt/data/dec5_screen_travel_dynamic_150_v2/video_slow_12fps.mp4)
- [Decoded middle-phase sheet](/mnt/data/dec5_screen_travel_dynamic_150_v2/decoded/sheet_070.png)
- [Decoded late-phase sheet](/mnt/data/dec5_screen_travel_dynamic_150_v2/decoded/sheet_120.png)
- [Integrity audit](/mnt/data/dec5_screen_travel_dynamic_150_v2/integrity_audit.json)
- [Encoded review and per-frame hashes](/mnt/data/dec5_screen_travel_dynamic_150_v2/video_manifest.json)

Video SHA-256:

```text
video.mp4            f1e907344bcebfbcbabc90638e4873e824693ef4211741c2bcce6805dce89f64
video_slow_12fps.mp4 e0d56d9fd849a9bc074c6c826ee82b67e7b260146127f86afcad5e661d053299
```

Output: `/mnt/data/dec5_screen_travel_dynamic_150_v2`.
The audit now requires both expected 3D poses and visible screen displacement,
checks all output PNGs against the rotated native render, and independently
raycasts five saved camera poses. Widening only the horizontal path axis makes
speed vary smoothly around the ellipse; local step continuity is checked rather
than falsely claiming constant speed. The camera loop is not an actor-time loop.
Normal playback is 24 fps / 6.25 s; the optional 12 fps / 12.5 s copy slows both
actor and camera and has lower cadence. No temporal frames are fabricated.

## Insights

- A centered look-at orbit can have substantial translation and parallax but
  still fail the desired visual composition. Verify the requested screen motion,
  not just nonzero camera coordinates.
- Constant lens orientation over this narrow-FOV rig would send the subject
  outside frame; instead, allow controlled residual screen travel while smoothly
  steering the optical axis. Disclose the fixed wider virtual lens.
- Wider framing reveals unfinished geometry formerly outside the image. Do not
  hide those defects behind a camera-motion success claim.

Replay with the existing pinned environment:

```bash
python LookCloser/scripts/screen_travel_camera_flight.py
python LookCloser/scripts/prepare_wide_dynamic_geometry.py \
  --parent /mnt/data/dec5_screen_travel_dynamic_150 \
  --output /mnt/data/dec5_screen_travel_dynamic_150_v2
python LookCloser/scripts/run_wide_dynamic_workers.py supervise \
  --output /mnt/data/dec5_screen_travel_dynamic_150_v2 --workers 8
```
