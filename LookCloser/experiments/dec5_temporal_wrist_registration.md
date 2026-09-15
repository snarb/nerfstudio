# DEC5 temporal palm registration before hand repair

## What was tested

The previous local forearm completion does not model the whole wrist/hand.
This experiment introduces temporal shape transfer rather than changing another
single-time forearm threshold. It does **not** yet modify a production mesh.

`study_wrist_observations.py` saves six train views at each of
001029/001031/001033/001035/001037 with the existing fixed exposure and fixed
physical-camera RGB gains. Held-out cameras are excluded. Native 001029 and
001037 six-view sheets were inspected: the later hand is substantially blurred
while adjacent cloth remains sharp. This is consistent with motion blur; it
does not establish exposure timing, rolling shutter or synchronization error.

![001037 actual train observations](/mnt/data/dec5_wrist_observations/001037/six_train_views_native.png)

`study_temporal_wrist_registration.py` uses cached RAFT-Large C_T_SKHT_V2,
12 updates, native 640×768 portrait crops and forward/backward consistency
within two pixels. Flow is used for correspondences, not synthesized RGB.
771 source mesh samples come from an inset palm polygon traced on 001029 G/B
RGB. Source-mesh visibility is checked in each view. Camera normalization is
inverted/reapplied for 3D point transfer between times.

One rigid transform is fitted to five cameras with robust reprojection residuals;
H/C remains outside the fit. This is a **withheld train-camera flow check**, not
held-out RGB geometry truth or face PSNR/SSIM/LPIPS. No evaluated camera RGB is
used in the experiment. Initial direct flow and chained neighboring-time flow
are retained separately, with exact producer snapshots.

## Results

### Direct versus adjacent-time tracking

Direct 001029 → 001037 tracking leaves only one reference-camera match after
the fixed consistency gate. The failed PnP initialization is retained; it did
not create a fitted surface. A read-only palm-flow diagnostic found median
motion about −124 horizontal / +228 vertical pixels and median forward/backward
error about 25 pixels. That diagnostic patch is not a new evaluation ROI.

Chaining four adjacent-time flows keeps 556 reference matches under the same
two-pixel final forward/backward gate. All intermediate paths must remain inside
the flow crop; out-of-frame paths cannot become valid by re-entering it.

| Camera | Consistent tracks | Rigid median / P90 error, px | Affine median / P90 error, px |
|---|---:|---:|---:|
| G/A | 472 | 5.578 / 8.047 | 3.564 / 5.741 |
| G/B | 556 | 3.764 / 5.396 | 2.593 / 4.649 |
| G/C | 474 | 2.171 / 5.360 | 2.158 / 4.803 |
| H/A | 510 | 7.344 / 15.482 | 8.457 / 14.389 |
| H/B | 267 | 4.539 / 5.861 | 2.421 / 4.246 |
| H/C, not fitted | 491 | 3.100 / 6.275 | 2.554 / 4.866 |

The bounded affine control allows matrix coefficient changes up to 0.1 and
translation components up to 0.003, then checks sampled-point displacement
against 0.003 normalized units. Maximum displacement is 0.0008955. Singular
values are 1.0903/1.0025/0.8676: coefficient limits must **not** be described as
a 10% singular-value deformation bound. Two coefficients reach their bounds.
The affine control improves H/C but worsens H/A median residual; no automatic
mesh acceptance follows from its aggregate improvement.

All six rigid reprojection crops and all six affine comparison panels were
directly inspected. Points broadly track the palm, but H/A has a visible
systematic offset. Native overlays bind cyan fitted positions to yellow flow
residuals; they are not novel-view textured mesh renders.

![H/A unresolved registration discrepancy](/mnt/data/dec5_temporal_wrist_affine/H004_A005_1210M6_comparison.png)
![H/C withheld from fitting](/mnt/data/dec5_temporal_wrist_affine/H004_C005_1210SZ_comparison.png)

### Camera-consensus diagnosis

Five leave-one-fit-camera-out rigid solves reuse exactly the same tracks.
Omitting H/A yields fitted-camera medians 3.43/2.37/2.42/3.31 pixels and H/C
validation median 3.02 pixels, while omitted H/A worsens to 10.53 pixels
(P90 18.86). The other exclusions do not isolate an equivalent discrepancy.
This identifies an inconsistent set of temporal correspondences, **not** proof
that this physical camera's calibration is wrong. Blur-related flow drift,
source-surface error and nonrigid skin motion remain plausible explanations.
No camera is silently discarded from a reconstruction or accepted evaluation.

Four focused tests pass: gauge transfer, rigid recovery/withheld-camera
projection, flow-chain border validity, and the existing camera-path phase test.
The final audit binds raw EXRs, derived observations, model weights, flow fields,
source mesh, correspondence arrays, fits, overlays and exact helper sources.
It independently replays the rigid fit. No full-frame quality metric, main
face CSV update, production mesh replacement or new full movie is produced.

Artifacts:

- `/mnt/data/dec5_wrist_observations`
- `/mnt/data/dec5_temporal_wrist_registration` — failed direct transfer
- `/mnt/data/dec5_temporal_wrist_chain_registration` — six-view temporal fit
- `/mnt/data/dec5_temporal_wrist_affine` — bounded nonrigid control

## Insights

Adjacent-time tracking is materially more useful than jumping directly across
the rapid hand movement. It supplies a new route to transfer real neighboring
geometry, but correspondence consistency is not anatomical certainty. The
remaining camera-specific discrepancy should be diagnosed before combining
this prior with current geometry; otherwise a transferred hand can create the
same doubled/extended shape problem seen around the lipstick.

A specialized hand mesh predictor was considered but not run: official
[HaMeR](https://github.com/geopavlakos/hamer#installation) and
[WiLoR](https://github.com/rolpotamias/WiLoR#installation) setup requires MANO
assets with separate access/license steps. No account was created or license
accepted. The existing cached flow model allowed the temporal test without
altering the working environment or downloading the LookCloser paper.

The full objective remains unfinished. The previous best local mesh and the
dynamic-camera/dynamic-actor movie are unchanged, with known residual artifacts.
