# Marshawn WS07/WS08: initial three-second AABB

## What was tested

User scope: **all fight performers**, including thrown/fallen participants, but
not crew, operators or safety staff. This supersedes the old handoff's two-actor
scope. No training is authorized past the current geometry review gate.

Branch `scene/marshawn1660-native63-20261109`, handoff `57d35d7e`.
Read local paper, handoff, geometry receipt and existing hull/overlay helpers.
Working resolution: Full HD bounding canvas, aspect preserved (1920 pixels wide,
height approximately 990); actual rounded X/Y scaling applied to intrinsics.
Colour previews decode PQ/AP1 to Rec709 after PQ-space area reduction; they are
geometry/debugging assets, **not accepted photometric training masters**.

Sources: workstation07 frames 908–1111 and workstation08 frames 1112–1314.
Seven samples: 908, 968, 1028, 1088, 1148, 1208, 1268. Their interpretation as
0–3 seconds in 0.5-second steps **assumes the previous take's 120 fps**; local
metadata examined so far does not independently establish FPS.

## Results

SSH access as nvuser verified on both workstations. Initial pavlo read-access
block resolved through read/traverse ACLs on data and required Python runtime
paths. No mask workers stopped. No model training launched.

Prepared 84 source/calibration records: seven times × twelve cameras
DC/HC/KA/NC/JE/AB/HB/HD/DB/DD/KB/LA. Their poses and scaled intrinsics remain
identical per camera over all seven samples (checked to 1e-10 absolute tolerance).
Source PNGs and masks are on the same undistorted native canvas; COLMAP poses
are converted to OpenGL camera-to-world. Source text hashes are in receipts.

Full-HD production-mask contact review shows the four performers, including
lying actors, selected while the spotter at the mat and camera crew are excluded
from foreground. SAM person instances outside production foreground are UNKNOWN
for carving, protecting potentially occluded actors. Four masks missing during
initial preparation at frame 1088 became available and were loaded before fitting.

Six widely spaced views gave unbounded silhouette aliases and no three-view
SIFT anchors; rejected. Added adjacent cameras. Initial permissive 12-view
hypotheses also retained remote false components; they are diagnostics, not boxes
accepted for training. Current candidate qualification requires at least eight
observed foreground views at leaf level, with at most two clear-background
contradictions. Adaptive carving before qualification uses support >=3, full-cell
corner footprints, 64-native-pixel guard, leaf upper bound .04 rig units. Keep
anchor-bearing components and fragments within .06 rig units. Anchor tracks use
three or more cameras and <1-pixel reprojection error. These thresholds are a
new empirical fit, not universal constants from the previous scene recipe.

Union qualified per-frame extents and independently triangulated actor anchors;
apply .05 rig-unit minimum margin, round outward on .01 lattice. Coordinate frame
is original COLMAP world rigidly reordered X,Z,-Y (right-handed). It is **not** the
old prepared-world transform or an accepted floor fit. An exploratory floor fit
from the sparse rig cloud was weak and is not used.

Artifacts under `lookcloser_debug_outputs/marshawn_ws07_aabb/`:
source/production-mask/instance previews, source receipts, retained cells,
three-view anchors, component qualification, bounds and review images.

Google Drive source previews:
- [Single frame 908 DC](https://drive.google.com/file/d/1Fnn0w9NbpyC_CtQwrE9hOOqxukXGVLkJ/view).
- [Seven-time source contact](https://drive.google.com/file/d/1gwklICJcFsgcwfXltzNaQ_fs3WYDDL7r/view).

## Insights and next steps

This is an initial sampled-view candidate, not proof of the mathematically
minimum box, hidden geometry or continuous-time containment. Check source masks,
feet/hands/hair/sticks and fallen performers in cutouts and orthographic views.
Ray-box intersection is necessary only; a photograph cutout is not a depth render.
Wireframe overlays must be labelled X-ray if no scene depth is available.

A box fitted using performer masks may still contain floor, mats or crew where
these occupy the same spatial envelope. An AABB cannot perform semantic removal.
Before training, verify timing, floor/normalization, further cameras, intermediate
pose extrema and unresolved clipping, then write a separate acceptance receipt.

### Initial review deliverable

Candidate bounds in declared X,Z,-Y rig frame:
`[-3.84, 0.95, -1.72] → [2.54, 4.84, 1.17]`.
Seven qualified volumes do not touch the search boundary. Three-view-or-more
anchor counts: 74,96,110,102,116,98,156. Source/production-mask contacts,
orthographic volumes, six-camera wireframes and cutout contacts were visually
inspected. No clear performer-image clipping seen in those contacts. Six mask
pixels miss the box in KA1088 at x480–481,y986–988 (bottom raster edge); the saved
crop shows no identifiable performer feature. No masks were changed to improve
this count. Ray coverage and contact inspection do not establish surface depth.

One important limitation: some actor parts do not have eight-camera silhouette
support. E.g. the maximum-X 1208 anchor has four-camera SIFT support with
0.14–0.25 pixel errors, outside the stricter retained cells, and is explicitly
unioned into the envelope. Anchor extents protect observed points, not every
unobserved foot/hair boundary; the candidate is not final geometry acceptance.

Uploaded via Google Drive connector and metadata readback verified:
- [Review folder](https://drive.google.com/drive/folders/11c2NG2kV4NYG6iuw8FjTPYLMfKISGjBv).
- [Seven-time AABB overlay contact](https://drive.google.com/file/d/1vyCNUVfJWv8LvJMu5p9zH9MzdMouXtlI/view).
- [Ray cutouts](https://drive.google.com/file/d/1SqVmiT4zpuD4isYBUsiE7MBFDeMFb9nF/view).
- [Orthographic geometry](https://drive.google.com/file/d/1Ul9DedlqvH2gm1vpjkZfWBSyLPFnDjLn/view).
- [Full review ZIP, 49,643,258 bytes](https://drive.google.com/file/d/1A9SLbuRAvtt8V4SKrrRpoDSeFC06YRgv/view).

`AABB_REVIEW.json` preserves transforms, bounds, assumptions, receipt hashes and
limitations. It is deliberately not named AABB_ACCEPTANCE: no training clearance
or mathematical minimality/continuous-time claim is made.

### User rejection and depth-review rebuild

User rejected both initial box and ambiguous wireframe/ray-cutout visualization.
The initial candidate is withdrawn. User authorized quick diagnostic checkpoints
if reconstruction without training cannot provide reliable depth.

Current-image floor probe: triangulated 16 lower-image, non-person tracks using
12 cameras; plane consensus 12 points, median residual .00454 rig units. Floor
normal in original world `[-.0518764,-.9288484,-.3668098]`, offset `1.8163471`.
Grid overlay inspected on DC/HC/KA/AB. This replaces the weak rig-cloud plane.
Top/bottom faces are constructed parallel to this estimated floor, not original
COLMAP axes. No independent ceiling-plane measurement is claimed.

Simple calibrated stereo probe (HC/HB, DC/DB, KA/KB) did not provide useful actor
depth coverage (<.1% after its checks). Rejected as evidence for dense cutouts;
implementation/rectification and matching limits remain unresolved. It is not
proof that stereo methods generally cannot work for this scene.

Diagnostic bounded Instant-NGP teacher on frame908: 72 Full-HD views, 69 train,
DC/HC/KA held out, full photos without performer masks or candidate-box clipping.
Wide training bounds in floor coordinates `[-5,-3,-4] → [5,7,6]`, scale1/5,
center[0,2,1]; hash21, 6000-step technical budget, checkpoints every2000. This is
not the frozen LookCloser scientific recipe or a production pilot acceptance.

First launch failed at CUDA extension setup (Ninja not on PATH). Used the
already-built GB300 nerfacc_cuda.so explicitly; no recompile or model equation
change. Next launch exposed an existing dynamic-batch odd-ray-count assertion
with variable-resolution image lists. Fixed rounding upward to an even count
(minimum2) in repository dynamic_batch.py and identically in the isolated
technical launcher; parity cases checked, training restarted from scratch.
Runtime uses the installed GB300 kit, whose code is not overwritten.

Candidate for depth review in floor units is `[-3.94,.33,-.05] → [2.46,4.49,1.77]`.
The bottom near the measured floor rejects old below-floor silhouette aliases;
it is not accepted until reconstructed feet and source evidence are checked.

### Seven-frame technical checkpoints and independent seed check

Finished 6k-step full-scene teachers for all seven sample frames (908–1268,
60-frame spacing). WS07 completed908/968; WS08 completed the remaining five.
No OOM; local queue deliberately drained after968 to use idle WS08. Selected
checkpoints and PSNR/SSIM/LPIPS are recorded in per-frame complete.json files.
These short teachers have unreliable depth in some regions: HC1208 shoe depth
was near head height despite plausible RGB. Do not enlarge the box from these
near-camera artifacts. Source-depth blackouts are conditional diagnostics.

Downloaded the seven original 72-camera COLMAP seed clouds from S3 read-only.
Projected into all72 calibrated Full-HD production masks. At least24 foreground
votes, >=3 COLMAP tracks, reprojection error<2 gives 2616–3899 sparse candidate
performer points/frame. This vote threshold is a heuristic, not identity proof;
low-vote floor/background aliases persist. All high-support actor points fit
current candidate; 14 lower-support outside points are near floor/background
and require observation-level interpretation. Sparse points do not certify
untextured limbs or continuous-time containment.

Floor-aligned candidate v2 uses a 49.555-degree in-plane rotation, rotated-floor
bounds `[-.41,-.93,-.15] -> [3.41,5.14,1.77]` (rig units). Bottom was lowered from
-.05 using908 shoe evidence; top/bottom stay parallel to the estimated floor.
Full transform and units are in floor_yaw_box_v2.json. Candidate not certified.

Additional1208 teacher: production performer masks blacken background, conservative
training-stage bounds in floor coordinates `[-4.5,0,-.4] -> [3,5,2.3]`, a separate larger-volume
stage domain. It is not a strict superset of every empty corner of the rotated
candidate; it contains the observed performer region. Goal: exclude camera-vicinity billboard artifacts.
Isolated dataparser patch reads geometry_teacher_bounds.json for both training
and rendering. These masked-background metrics are not comparable to full-scene
teacher metrics. Active supervisor records processes/progress/GPU/OOM every30s.

### NC finds a real v2 boundary problem; v3 correction

The three held-out views partly miss the fourth performer's near-side body.
Added NC as a **labelled training-view diagnostic**. At1268, v2 discarded2.92%
of foreground RGB contribution; the corresponding ray peaks lie near rotated
box Y=-1.06, beyond lower Y=-.93. This is a real diagnostic-box failure, not
just the earlier learned-depth issue. Earlier dismissal of low-vote seed
witnesses near this boundary as floor/background was premature and is withdrawn.
The initial continuous sparse audit's proximity-to-strong-support filter excluded
these witnesses; zero outside among that subset was insufficient for acceptance.

V3 moves only lower Y from-.93 to-1.20rig (~25cm), retaining floor alignment and
all other faces. Re-rendering all28 views against unchanged selected checkpoints.
Repeating all361-frame seed checks with >=24/72 mask votes and >=3 COLMAP tracks,
error<2, **without** the proximity filter. The separate98-point floor check agrees
with the initial estimated normal within0.342degrees (median residual.00696rig).

### Review-ready V3 result

Original-rig box metadata: [aabb_original_rig.json](marshawn_908_1268_aabb/aabb_original_rig.json).
Rotated-floor bounds `[-.41,-1.20,-.15] -> [3.41,5.14,1.77]`, dimensions
3.5354 ×5.8677 ×1.7770m at1.0805rig units/m. This is a practical review candidate,
not a proof of global minimum volume or hidden-surface containment. Assumed120fps
means908–1268 spans3seconds; FPS remains unverified on this machine.

Seven short masked teachers finished; all28 actual-field partition views completed.
Three held-out cameras DC/HC/KA plus NC training-view geometry diagnostics.
Checkpoint selection: highest eval-all PSNR; LPIPS tie-break within0.07dB.
Numbers below use masked backgrounds and are not comparable to full-photo runs.

|Frame|Selected step|PSNR|SSIM|LPIPS|
|---|---:|---:|---:|---:|
|908|1000|33.639|0.9326|0.0979|
|968|2000|34.262|0.9490|0.0778|
|1028|2000|35.660|0.9572|0.0741|
|1088|2000|35.876|0.9605|0.0667|
|1148|2000|36.357|0.9608|0.0620|
|1208|2000|35.789|0.9638|0.0650|
|1268|2000|35.480|0.9540|0.0779|

Visual inspection: all7 moments in all4 views; NC recovers the fourth performer
partly beyond the other camera rasters. V3 removes the visible near-side clipping
that V2 introduced. Max outside RGB contribution under actor masks is0.09783%
for held-out views; NC1088 has0.74983%, NC1268 .21312%. These are field-contribution
fractions, not anatomical coverage percentages. The1088 ray audit finds incorrect
learned positions around X4.6/Z1.53rig for a pants region whose nearby COLMAP
witnesses are X.75–.90/Z.14–.27rig; three other patches are background floor.
Do not inflate a physical box to include these teacher artifacts.

The revised all361-frame sparse audit retains1,147,645 candidates without the
old proximity filter:1,147,634 inside,11 low-support outside near the floor.
Several outside coordinates recur unchanged across motion frames; representative
multiview overlays show floor/background. These11 are preserved explicitly in
the audit rather than silently removed; >=40-vote witnesses have0 outside.
This corrects the overly reassuring earlier proximity-filtered count.

Original-rig/teacher coordinate-membership parity checked on10,000 random points;
rigid transform and floor-parallel faces checked. Render receipts verify unchanged
checkpoint hashes, native-compositor parity and additive partition sums. Diagnostic
training and rendering are finished; no training process remains on WS08. Original
mask workers were not killed. Production LookCloser training has not started.

Drive delivery:
- [Folder: all overviews, geometry and Full-HD archive](https://drive.google.com/drive/folders/1n70r-bKJ2fEnRUIa4ytPZbVXr6lT0jZG)
- [DC overview](https://drive.google.com/file/d/13F3itHzlMRaZ4BdIoExsA2XYWskA6hfs/view)
- [NC overview including the fourth performer](https://drive.google.com/file/d/1ll4G0V156WE1B9fFgT8X3_RIWQRMKIHH/view)
- [28 Full-HD comparisons, receipts and audit](https://drive.google.com/file/d/1vMLc-ecQ7JosAN9h1R3oTflJ5xTmIQJ4/view)

Compare FULL MODEL against INSIDE AABB. Teacher backgrounds were masked before
training: disappearance of crew relative to the source photo does not establish
that crew are geometrically outside the box. A single AABB is not a semantic filter.
