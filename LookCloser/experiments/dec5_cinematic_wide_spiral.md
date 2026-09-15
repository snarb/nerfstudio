# DEC5: a visibly wider first spiral turn

## What was tested

The user liked the previous cinematic render quality but could not see a
substantial spiral. Inspection found that the v4 `free_arc` was a single
Bezier arc, not a complete turn. This experiment changes camera motion only:
production geometry, hard RGB sampling, fixed radiometry and150 actual source
times000899..001197 remain unchanged. No retraining or new masks are involved.

Opt-in entrypoint: `scripts/cinematic_wide_spiral_centered.py`.
Output: `/mnt/data/dec5_cinematic_wide_spiral_v3`.
Two choices: `wide_spiral_lookat` and `wide_spiral_free`. Both use the same
camera centers; the second adds a controlled orientation/composition gesture.

The geometric path is a1.2-turn contracting ellipse across the FRONT rig,
not a360-degree orbit behind the actor. Nominal radius is2.7 columns and0.9
rows. Convex mixtures of C/B, K/B, K/D, C/D and endpoint H/C stay inside the
calibrated train hull. Actual row coordinate remains strictly inside B..D;
no outer A/E rows or A/B/M/N columns are used. Physical arc-length resampling
gives a short smooth speed ramp, cruise and long deceleration to index118.

The previous requested ending is preserved:118 pure3D frames,8 display-domain
dissolve frames, then24 actual dynamic RGB frames from H004_C005_1210SZ.
Camera pose and lens are fixed from index118. The room background appears in
the dissolve. The last second is explicitly NOT a reconstruction or mesh repair.
Virtual focal changes0.85x to1.9x and principal-x shift400px create the beauty
framing; these are distinct from camera-center translation. The physical path
is not a monotonically approaching radial dolly: reference target distance
starts0.6730 and ends0.8178. The close-up is therefore partly optical framing.

## Results

| Quantity | Previous v4 free arc | Selected centered spiral |
| --- | ---: | ---: |
| Physical center-ray angular extent |8.87 degrees|36.34 degrees|
| Full first-turn completion |No full turn|index88 /3.67s|
| Radius factor at full-turn completion |n/a|0.794|
| Actual rig horizontal coordinate span |n/a|−2.407..+2.693 columns|
| Actual rig vertical coordinate span |n/a|−0.864..+0.767 rows|
| First95-frame vertical physical extent ratio |1.0|1.313|
| Playback |24fps|24fps,150 moving times,6.25s|

Do not interpret the20.17x horizontal extent ratio against the previous free
arc as a20x viewing-angle improvement: the old path was nearly vertical.
The36.34-degree angular extent and saved-center plot are better comparisons.

Controls retained under `dec5_cinematic_wide_spiral_v1` and `_v2`:

- v1 nominal3.6-column ellipse, far-left start:50.63 degrees, but a late
  physical speed spike and a large black under-jaw notch in early RGB frames.
- v2 exact same geometric curve with physical arc-length timing: speed issue
  removed, but the same initial C-side surface gap. Neither was fully rendered.
- v3 recentered ellipse and higher/right initial phase:36.34 degrees, full
  broad turn retained, no analogous large starting notch in the24 RGB canaries.

Visual control links (absolute server paths):

- [Actual-center path and physical speed](/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_free/motion_plot.png).
- [Same actor mesh and fixed lens at eight actual saved poses](/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_free/motion_probe/contact.png).
- [Free-composition RGB canaries](/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_free/canary_review/contact.png).
- [Look-at RGB canaries](/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_lookat/canary_review/contact.png).

Initial main-LLM review covered both12-image contacts and native free-view
head crops000899,000995,001019. Rough brown hair/crown fringe, small
chin/lipstick membranes, thin facial seams and torso cutoff remain. Free
composition approaches the side border; final beauty framing clips crown.
This is not an artifact-free geometry result. Canary acceptance and reasons
for rejecting the widest control are saved in `canary_visual_review.json`.

Both full presentations completed on2026-09-15. The10 bounded renderer
shards and both postprocessing workers exited0; compact PID/GPU/disk/progress
records are in `full_fresh_workers/checks.jsonl` and `postprocess/checks.jsonl`.
No production worker was left running. Six fresh RGB processes ran at most;
each presentation began processing as soon as its126 raw renders were ready.

The main LLM inspected all30 chronological ten-frame sheets (300 presented
images across both choices), both decoded MP4 overview sheets, the24 RGB
canaries and the native crops noted above. No analogous large initial C-side
jaw notch or newly catastrophic geometry was seen on the selected path.
The known residuals remain; neither choice is labeled artifact-free. Free
composition partially clips hair/ear at the left border mid-turn. During the
short ending dissolve, a slight chin/silhouette mismatch remains visible.

Each raw audit checked126 distinct render/mesh hashes, the unchanged150-time
inventory, camera matrices, finite depths and seven independent fresh depth
raycasts. Presentation replay checked all150 images pixel-for-pixel against
the118+8+24 provenance. Both MP4s were decoded in full:150 unique1080x1920
frames,24fps,6.25s. Delivery packaging rechecks both MP4 hashes and all300
PNG hashes inside the frame archives, plus all64 prepared ending images
against the independently float64-replayed v4 actual-train ending.

Orientation was also checked from actual saved matrices: maximum per-step
angular speed27.88deg/s (look-at) and28.62deg/s (free); the final moving step
is0.000589deg/s, followed by exact zero. Physical translation cruise speed
is approximately0.288 reference-units/s, without the rejected v1 late surge.
Sixteen targeted trajectory, composition and audit tests passed.

Deliverables under the output root:

- `wide_spiral_choices.zip`: two MP4s, explanation, manifest, report and
  actual-center/speed plot. Suggested first watch: `02_wide_spiral_free.mp4`.
- Each variant: `presentation/video.mp4`, `presentation/frames.zip`,
  `publication.json`, `manual_visual_review.json`, camera/raw/presentation
  audits, source/hash manifests, chronological sheets and frozen script copies.
- `delivery_audit.json` and `bundle.json`: final rechecked delivery receipts.

Reproduce with `cinematic_wide_spiral_centered.py init`, `motion`, `canary`,
then actual canary review and `render`. `finish_cinematic_wide_spiral.py`
handles composition, raw/presentation audits, encoding and sheets. Record
only images actually inspected using `record`; then `publish` and
`bundle_cinematic_wide_spiral.py --base /mnt/data/dec5_cinematic_wide_spiral_v3`.
The v1/v2 producer scripts remain frozen so their rejected controls stay
auditable; the selected v3 producer is separate and hash-pinned.

## Insights

Prove motion from actual saved camera matrices and a fixed-actor/fixed-lens
control. Actor animation and focal changes alone cannot establish a flythrough.
A complete turn with enough radius retained is more legible than a small
single arc. Arc-length timing is essential: smooth parameter easing alone
does not prevent a speed surge where a spiral rapidly contracts.

Moving a view outside reliable surface coverage exposes existing holes; it
does not create proof that the source calibration or mesh changed. The
center/phase adjustment is a presentation workaround, not geometry repair.
Novel-view GT does not exist, so no PSNR/SSIM/LPIPS is invented for these movies.
