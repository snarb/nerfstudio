# Marshawn 1660: bounded TRAIN-only floor plane sweep

## What was tested

A read-only HD photometric-depth pilot on the existing calibrated TRAIN images. Each MA reference-pixel patch remains fixed. Candidate 3D points move along the reference rays as parallel-plane height sweeps ±0.03 prepared-world units at 0.001 spacing. NA/NB fit the depth; disjoint LB validates interior candidates, with reverse NA-to-MA/NB matching as an additional consistency check. No eval RGB, NeRF fitting, new training constraints, or carving was used.

ZNCC uses fixed 11×11 patches of linear Rec709 luminance with a fixed 0.7-pixel prefilter. It removes patch scale/offset differences without fitting camera gains. Acceptance requires joint and independent-view scores ≥0.6, agreement/roundtrip ≤0.005, a unique interior peak with ≥0.03 margin over alternatives outside ±0.005, near-peak width ≤0.01, and sufficient patch contrast. The prior normal defines a local patch orientation and search scope; its depth is not imposed.

## Results

| Known floor witness, primary 11px | Height error | Disjoint OA ZNCC | Result |
|---|---:|---:|---|
| 8290, white fleck | 0.000085 | 0.935 | Pass |
| 253, floor tape | 0.002510 | 0.987 | Pass |
| 3738, crease | 0.000331 | 0.379 | Reject |

The 15px sensitivity check passes the crease, but was not promoted. Root accepted the two flat-feature sanity checks for the bounded interior pilot while retaining the crease as a point-level rejection.

**52/128 interior candidates pass unchanged primary criteria**, including 51 outside the original 16-point support hull. All 52 centers were manually inspected in four actual TRAIN-source patches across 13 contact sheets; the match patches show floor covering, without actors or hardware. Heights relative to the old prior plane range from −0.006 to −0.001, with median −0.004. Median joint ZNCC is 0.818; independent LB validation is 0.776. Median near-peak width is 0.003, p90 0.005, maximum 0.007; this is a photometric score width, not a calibrated physical confidence interval. Maximum reverse height difference is 0.002.

The new candidate hull contains six of the nine investigated KA ray/plane intersections (q0/q1/q2/q3/q7/q8); q4/q5/q6 remain outside. No interpolation or planar depth claim is made at those uncovered points.

[Known-depth profiles](/fsx/tmp/marshawn_001660_hd/floor_plane_sweep_HD_PILOT/HD_known_ZNCC_profiles.png), [all 128 scores and rejections](/fsx/tmp/marshawn_001660_hd/floor_plane_sweep_HD_PILOT/HD_128_pilot.json), [52 reviewed source-bound candidates](/fsx/tmp/marshawn_001660_hd/floor_plane_sweep_HD_PILOT/HD_52_reviewed_candidates_NOT_PRIOR.json), [manual review receipt](/fsx/tmp/marshawn_001660_hd/floor_plane_sweep_HD_PILOT/HD128_manual_review_geometry.json).

## Insights

HD patch matching recovers floor depth where sparse feature matching left a gap. Seventy-six samples still fail one or more checks and remain unknown. The accepted patches overlap and share four calibrated cameras, so the 52 points are not independent physical measurements. Target coordinates are calibrated photometric-sweep samples, not independently detected SIFT coordinates; zero reprojection error must not be presented as additional validation. Calibration uncertainty, wrinkles, view-dependent appearance and the limited ±0.03 search remain material limitations. No model, prior manifest, or grid was changed.
