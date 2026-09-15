# DEC5 hand prior: cross-view consistency before temporal geometry transfer

## What was tested

Three controls follow the [temporal palm registration](dec5_temporal_wrist_registration.md).
None edits the production mesh, source EXRs, texture settings or published movie.

1. `diagnose_temporal_wrist_stages.py` follows the same endpoint-selected track
   population through 001031/33/35/37, starting at 001029. It checks when the
   H/A discrepancy grows, rather than choosing a different population per time.
2. `study_geometry_reset_wrist_tracking.py` reprojects the shared rigid 3D pose
   before each adjacent flow step. Final validation uses the original independent
   endpoint tracks, not the newly reset tracks used during fitting.
3. `study_hand_landmark_prior.py` runs MediaPipe 0.10.21 locally on five times
   and six native train views. Its isolated Python 3.12 environment leaves the
   reconstruction environment unchanged. The public
   [Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
   model predicts joints, not measured surface depth. Its model hash and package
   versions are retained. No images are uploaded; no MANO assets are required.

`triangulate_hand_landmarks.py` fits each joint from at least three in-image
predictions in five train cameras, using calibrated pinhole projections and
robust two-pixel residual scale. H004_C005_1210SZ is excluded from fitting.
The model's predicted world coordinates and handedness score are **not** used
as metric geometry or point confidence. Missing detections and off-image points
are not invented. This tests cross-view agreement of predictions, **not error
against annotated ground-truth joints**. It does not compute face/full-frame
quality metrics or use the held-out F/B view.

## Results

### Flow controls: no accepted improvement

The H/A median rigid reprojection discrepancy grows 1.92 → 3.51 → 5.67 → 7.34 px
across the four steps. H/C, excluded from fitting, grows .91 → 1.62 → 2.41 →
3.10 px. Growth is gradual, consistent with drift; it does not isolate calibration,
source shape error, nonrigidity or optical-flow bias as the sole cause.

Resetting tracks to shared geometry gives attractive last-step local residuals
(H/C .83 px), but the **frozen independent endpoint** residual worsens from
3.10 to 3.33 px. H/A changes 7.34 → 7.04 px. It is rejected as a remedy for
the cross-camera inconsistency; local fitting numbers cannot justify promotion.

Roots: `/mnt/data/dec5_temporal_wrist_stage_diagnosis` and
`/mnt/data/dec5_geometry_reset_wrist_tracking`.
Visual review covers H/A at the first/last stage and H/A + H/C reset endpoints,
not every saved overlay.

### Anatomical landmark prior: inconsistent on the difficult interval

28/30 input images produce one hand detection; H/B has no detection at 001035
and 001037. Detection alone is not a geometry-quality gate.

| Time | Fit median, px | Excluded H/C median, px | H/C p90, px | H/C valid joints |
|---|---:|---:|---:|---:|
| 001029 | 6.25 | 7.48 | 19.18 | 21 |
| 001031 | 8.85 | 6.87 | 19.57 | 21 |
| 001033 | 11.57 | 12.10 | 26.83 | 21 |
| 001035 | 15.93 | 69.66 | 95.41 | 21 |
| 001037 | 15.89 | 17.92 | 41.11 | 14 |

The seven excluded endpoints at 001037 are not silently treated as successes.
All five native H/C reprojection crops were directly inspected. Six initial
landmark overlays were also inspected: G/B, H/A and H/C at 001029 and 001037.
Joint assignments differ visibly across cameras, especially under motion blur
and finger occlusion. At 001035, large cross-view mismatches are obvious.
This cannot safely anchor subpixel surface transfer or repair a narrow hand gap.

![Excluded camera, failed 001035 correspondence](/mnt/data/dec5_hand_landmark_triangulation/001035/H004_C005_1210SZ_reprojection.png)

Roots: `/mnt/data/dec5_hand_landmark_prior` and
`/mnt/data/dec5_hand_landmark_triangulation`.
Synthetic tests check calibrated 3D recovery, an unused view, invalid inputs and
degenerate camera baselines. A separate replay checks saved real-data points
and residuals, bindings to source RGB, model and producer scripts. This replay
does not rerun neural inference or independently validate anatomical accuracy.

## Insights

Do not use this landmark model as a direct missing-depth prior on this interval.
A plausible-looking 2D hand skeleton is not necessarily multiview-consistent.
The failure is not proof that every learned hand/body prior fails, and the
landmark pixel errors are not directly comparable with the denser RAFT population.

Both proposed registration fixes are rejected, not silently inserted into the
video. The useful local protected-forearm geometry remains as documented in
[the earlier control](dec5_protected_forearm_surface.md), with unresolved wrist
and finger coverage. The full objective remains incomplete: there is no new
artifact-free video and no accepted temporal hand-surface replacement here.
