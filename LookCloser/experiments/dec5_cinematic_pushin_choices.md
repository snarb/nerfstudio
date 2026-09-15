# Cinematic dynamic push-in choices

## What was tested

Four restrained, purposeful camera moves replace the earlier broad wandering
paths. All retain the unchanged production 150 actor times `000899..001197`
(step2), production meshes/source masks, incidence2 hard unwarped single-source
RGB, fixed camera profiles and fixed exposure. No experimental jaw/fringe mesh,
new source prior, refiner or inpainting is used. Existing videos are untouched.

Every path ends at the **exact real train extrinsic H004_C005_1210SZ**. In the
selected v4, camera and intrinsics hold from118 onward. Following the user's
explicit subsequent request, indices118..125 dissolve from3D to actual train
RGB, and126..149 (`001151..001197`) are entirely real train footage. All actor
times continue changing. **This ending is not a3D reconstruction result.**
The original source background appears during the deliberate dissolve; no
new mask hides it. Raw mesh predictions remain separately under `frames/`;
the hybrid sequence is under `presentation/frames/`. H/C was chosen using real train
RGB and previous production render evidence, not a held-out prediction image.
The parent agent's independent GT sheet and path audit are retained separately.

## Results

Full publication is pending. The current opt-in root is
`/mnt/data/dec5_cinematic_pushin_v4`.

| Variant | Physical radial approach | Physical angular span | Ending optics |
|---|---:|---:|---|
| locked_arc |20.88%|8.87°|1.65× real H/C focal, whole head|
| free_arc |20.88%|8.87°|1.90× focal, +400px principal-x shift, beauty crop|
| rising_arc |18.31%|9.91°|1.65× focal, whole head|
| soft_diagonal |5.36%|18.71°|1.90× focal, +400px principal-x shift, beauty crop|

The last option trades radial approach for a broader single arc. A supported
pair-mixture search found approximately19° required substantially less radial
approach than17–20%; no extrapolation was used to inflate those numbers.
All centers are convex combinations of train-camera centers and stay inside
the measured train hull. Contributing columns are D..L, avoiding outermost
A/N and penultimate B/M. Actual vertical coordinates remain strictly between
B/D (range0..0.6 in the chosen controls); the rig has only five rows A..E.

V3 timing integrated a quintic velocity envelope:10-frame acceleration,
84-frame cruise and32-frame deceleration to rest at126. V4 continuously
evaluates that same curve at `q=i*126/118`, reaching rest at118:9.365 frames
acceleration,78.667 cruise,29.968 deceleration. Position derivatives are C2
at the joins. V3 first24-frame physical path fractions were17.28%,17.28%,
14.13%,24.87%, independently checked from saved poses; v4 begins slightly
faster after its short acceleration. Thus the opening is not
the nearly stationary first second produced by the rejected global ease.

Focal changes and off-axis sensor composition are **optical framing**, not
physical dolly. All begin at0.85× real H/C focal. A disclosed quintic increase
to1.45× by v3 coordinate95 excludes the broken lowering forearm. Additional
late tightening over v3 coordinates80..126 creates the above endings; all
these optical coordinates are also retimed by126/118 in v4. The two beauty options shift native
principal x by+400px, lifting the projected face in the portrait image and
intentionally placing upper hair outside the virtual sensor window. This is
not a post-render tracked crop and does not manufacture new detail.

The v1 clay gate is retained: global quintic movement began too slowly and
lower-forearm holes remained visible at029/037. V2 used the revised physical
timing and earlier focal ramp; all48 RGB canaries completed. Its head-and-
shoulders ending retained substantial headroom and emphasized crown lace.
V3 retains exactly the same physical paths, with the explicit late optical
framing distinction above. All48 v3 RGB canaries completed; native1123 still
showed the thin nose-side crack, and whole-head endings retained rough crown
lace. No full v3 render was launched after the user chose actual train footage.
V4 retains the physical curve and optics but completes both earlier. All four
v4 clay sheets and its locked radius/speed plot and broad-arc fixed-actor,
fixed-intrinsics contact were visually inspected. V4 RGB gate is in progress.

- [Independent v2 saved-pose audit](/mnt/data/dec5_cinematic_pushin_v2/independent_path_audit.json)
- [Real train endpoint evidence](/mnt/data/dec5_cinematic_endpoint_sources/001197_contact.png)
- [Locked physical radius/speed vs optics](/mnt/data/dec5_cinematic_pushin_v4/locked_arc/motion_plot.png)
- [Fixed-actor, fixed-intrinsics broad arc](/mnt/data/dec5_cinematic_pushin_v4/soft_diagonal/motion_probe/contact.png)

## Insights

Changing composition can avoid a conspicuous lower-forearm gap or cut the top
hair out of a beauty close-up, but does not repair geometry. Existing crown
lace, small lipstick/hand membranes and source/depth seams may remain visible,
especially when enlarged. V2 native1123 showed a thin dark seam near the nose
and a crown opening; these are not declared hidden without RGB review.
The train ending intentionally avoids mesh defects by using original measured
RGB, not by fixing or validating the mesh. Fixed centered camera profiles and
the production exposure remain applied; virtual-lens bilinear resampling is
explicit and supplies no generated detail. Novel camera choices have no matched held-out target image, so PSNR/SSIM/LPIPS
between different shots would not constitute a valid quality comparison.

Replay: initialize `cinematic_pushin_choices.py`, `cinematic_pushin_framed.py`,
then `cinematic_pushin_beauty.py` and `cinematic_pushin_live_ending.py` in
sequence; each request is immutable. `supervise_cinematic_live.py --canary`
checks48 key renders; its default renders only missing raw indices0..125,
at most6 fresh worker processes. `audit_cinematic_pushin_motion.py`
separates fixed-lens physical parallax from actor/lens movement.
`compose_cinematic_train_ending.py` (main-agent owned) prepares32 actual train
frames and composes118 raw3D +8 explicit display dissolve +24 pure train.
`finalize_cinematic_live.py` provides raw126 and independent presentation150
pixel/provenance audits, chronological sheets,24fps encode/all150-frame
decode/ZIP and explicit hybrid publication. Three cinematic path tests pass.
