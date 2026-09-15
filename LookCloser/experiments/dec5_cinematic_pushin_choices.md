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

All four choices are fully rendered, audited, visually reviewed and encoded.
The opt-in root is
`/mnt/data/dec5_cinematic_pushin_v4`.

The bounded full-render supervisor finished at 2026-09-15 12:25:23 UTC:
20/20 fresh worker shards exited normally, with no pending/active workers or
OOM/error evidence in their logs. The campaign contains 504 distinct raw
render instances and 600 presentation frames. All 60 chronological ten-frame
overview groups and four decoded MP4 overviews were actually inspected.
Every video has 150 unique decoded 1080×1920 frames at 24 fps, lasting 6.25s;
all four 150-PNG ZIPs pass integrity checks. No slow version is published.

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
at the joins. V4 first24-frame physical path fractions are18.62%,18.62%,
15.38%,26.99%. Physical reference-space radii decrease1.03362→0.81784
(locked/free),1.00111→0.81784 (rising),0.86412→0.81784 (broad arc).
These are normalization units, not an asserted metric distance. Speed of the
last incoming interval at118 is below0.000008 reference units/s, then exactly
zero. Thus the opening is not
the nearly stationary first second produced by the rejected global ease.

Fixed-intrinsics, depth-separated relative landmark parallax extents are
11.91×102.20px (locked/free), 91.91×68.14px (rising), and 207.83×48.08px
(broad arc). These diagnostics isolate physical viewpoint change from optical
enlargement and actor movement; they are not whole-shot image-travel numbers.

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
fixed-intrinsics contact were visually inspected. All48 v4 RGB canaries
completed normally and all four contacts were reviewed. Native995/1007
retains a lipstick fin,1029 a hand membrane,1123 a thin nose-side seam and
whole-head crown lace. The lowering forearm gap is outside the inspected
1029/1037 framing. Whole-frame opening torso truncation remains visible.
The selected paths proceed unchanged with these explicitly logged residuals.

The locked/free transition contacts and all four chronological transitions
were inspected: face/ear/eye positions
match without a viewpoint jump. A short old-mesh silhouette is visible during
the dissolve. Whole-head real endings include a studio stand; the beauty
framing excludes it. Native real1197 beauty detail was inspected directly.
`free_arc` is the preferred stronger radial push, `soft_diagonal` the preferred
broader arc. Neither is an artifact-free3D claim.

Locked completed126 distinct raw meshes/renders, seven fresh native depth
recasts, unchanged150-time production inventory and fixed radiometry checks.
The separate hybrid audit replayed every presentation pixel:118 exact raw,
8 exact display dissolves,24 exact prepared train RGB. All15 chronological
contact sheets and the decoded MP4 overview were actually inspected. MP4
decodes150 unique1080×1920 frames,24fps,6.25s; maximum full-frame uint8 MAE
against presentation PNGs is2.165 (encoding fidelity, not reconstruction
quality). The150-member PNG ZIP also passes its integrity test.

Locked look-at choice:
[locked video](/mnt/data/dec5_cinematic_pushin_v4/locked_arc/presentation/video.mp4),
[frames ZIP](/mnt/data/dec5_cinematic_pushin_v4/locked_arc/presentation/frames.zip),
[raw audit](/mnt/data/dec5_cinematic_pushin_v4/locked_arc/raw_integrity_audit.json),
[presentation audit](/mnt/data/dec5_cinematic_pushin_v4/locked_arc/presentation/integrity_audit.json).

```bash
scp clever-shadow:/mnt/data/dec5_cinematic_pushin_v4/locked_arc/presentation/video.mp4 cinematic_locked_arc.mp4
```

`free_arc` also passed the same raw126/presentation150 audit, all15 actually
inspected chronological sheets and decoded overview, and150-frame MP4/ZIP
checks. Its maximum encoding MAE is2.132. Preferred stronger-push beauty choice:
[free video](/mnt/data/dec5_cinematic_pushin_v4/free_arc/presentation/video.mp4),
[frames ZIP](/mnt/data/dec5_cinematic_pushin_v4/free_arc/presentation/frames.zip).

```bash
scp clever-shadow:/mnt/data/dec5_cinematic_pushin_v4/free_arc/presentation/video.mp4 cinematic_free_arc.mp4
```

`rising_arc` passed the same complete raw/presentation/decode/ZIP checks and
all15 chronological sheets plus decoded overview were inspected. Maximum
encoding MAE2.165. Its whole-head ending retains the real studio stand:
[rising video](/mnt/data/dec5_cinematic_pushin_v4/rising_arc/presentation/video.mp4),
[frames ZIP](/mnt/data/dec5_cinematic_pushin_v4/rising_arc/presentation/frames.zip).

```bash
scp clever-shadow:/mnt/data/dec5_cinematic_pushin_v4/rising_arc/presentation/video.mp4 cinematic_rising_arc.mp4
```

`soft_diagonal` passed the same raw126/presentation150 audits, all15 actual
chronological reviews, full150-frame decode and PNG ZIP checks. Maximum
encoding MAE is2.132. Preferred broader-arc beauty choice:
[broad-arc video](/mnt/data/dec5_cinematic_pushin_v4/soft_diagonal/presentation/video.mp4),
[frames ZIP](/mnt/data/dec5_cinematic_pushin_v4/soft_diagonal/presentation/frames.zip).

```bash
scp clever-shadow:/mnt/data/dec5_cinematic_pushin_v4/soft_diagonal/presentation/video.mp4 cinematic_wide_arc.mp4
```

Each variant's `publication.json` binds the final shared report, raw/presentation
audits, request, motion audit, terminal manual visual review, MP4 and frames ZIP
by SHA-256. All inspected images have explicit hash records. The composer's
`visual_status=pending` is its frozen production-stage receipt, superseded for
delivery by the separate terminal manual/publication verdict; it is not silently
rewritten after auditing. The earlier locked raw-audit producer is retained with
its exact executed hash as `script_snapshot/finalize_cinematic_live_raw_audit.py`.
The unchanged production recipe and eight earlier published choices remain intact.

- [Independent v2 saved-pose audit](/mnt/data/dec5_cinematic_pushin_v2/independent_path_audit.json)
- [Real train endpoint evidence](/mnt/data/dec5_cinematic_endpoint_sources/001197_contact.png)
- [Independent real-ending EXR replay](/mnt/data/dec5_cinematic_pushin_v4/independent_real_ending_audit.json)
- [Beauty transition](/mnt/data/dec5_cinematic_pushin_v4/free_arc/train_transition_review/contact.png)
- [Raw canary visual findings](/mnt/data/dec5_cinematic_pushin_v4/canary_visual_review.json)
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
