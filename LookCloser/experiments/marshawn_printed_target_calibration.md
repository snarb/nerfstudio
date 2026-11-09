# Independent Marshawn printed-target calibration audit

Current decision: **retain the original calibration and stop this printed-target search**. The original two BA candidates and the later bounded shared-k1 test failed independent held-out validation. No original calibration, AABB or radiance-field model was changed.

## What was tested

Independently inspect and decode background printed targets in **raw distorted 8192×4320 TIFFs**, then test whether they improve the existing rig for global 1660/local408. Frames 1700/448 and 1740/488 supply additional, independently extracted observations. Existing rig intrinsics/poses provide the baseline and gauge; **no existing SfM 2D observations are used as ground truth**. Original rig, prepared images/transforms, and training data remain untouched. No training was launched.

The prints are individual taped **ArUco 6×6 binary markers**, rather than a checkerboard or known ChArUco board. A dictionary sweep supports 6×6: other standard families produce no repeated, consistent detections (one isolated 4×4 candidate). All final IDs are below 100; decoding with the nested 6×6_1000 dictionary does **not** establish which original dictionary cardinality was printed. Physical dimensions and inter-marker layout are unknown. Detection follows OpenCV's [ArUco detection/subpixel refinement API](https://docs.opencv.org/doc/doxygen/html/d5/dae/tutorial_aruco_detection.html).

Full-resolution and half-resolution detection are merged, with corners refined on full-resolution grayscale. Duplicate IDs at incompatible positions are excluded. Intrinsics account for COLMAP's half-pixel origin. All66 cameras were processed at all 3 frames: 198 raw images. Frame 408 yields 208 camera-marker detections, 31 IDs across 41 cameras;448 yields236/30/40 and 488 yields201/32/42. The detector is conservative and does not recover every visible print.

Two isolated candidates were tested:

- Joint camera pose + independent square-marker pose/unknown-size refinement. No invented metric size or rigid wall-board layout. Fourth corner excluded from fitting every training marker.
- Joint camera pose + free3D marker corners, removing the ideal-square/planarity assumption. Intrinsics and distortion fixed. Quadratic Gaussian pose priors: 0.0003 rad rotation-component sigma and 0.003 original-rig translation-unit sigma, with hard3 sigma component bounds. HC anchored; cameras with fewer than 5 training markers frozen. Only 11 cameras can move; maximum final update 0.0257° and 0.00354 original rig units.

**Frozen validation:** marker IDs divisible by 5 are excluded entirely from camera fitting. For each such marker, sort cameras and use alternating subsets: triangulate/refine each corner only in one subset (at least 3 cameras), score only the other subset, then swap. Baseline and candidate use identical detections, subsets, and scoring support. Thus held-out reprojection never fits target geometry using the camera observation being scored. Frame 448/488 never enter camera optimization; they repeat static scene targets and are not independent3D truth. Their common-corner temporal displacement medians are 0.347/0.380 raw pixels, so temporal repeatability is finite.

## Results

**Neither candidate supports replacing the existing calibration.** The conservative free-corner candidate worsens median and90th-percentile held-out errors on all 3 frames. All values below are full-sensor raw pixels; epipolar values use symmetric normalized-point line distance converted using the cameras' geometric-mean focal length.

| Global/local frame | Held-out scored corners | Cross-view median: baseline → candidate | Cross-view p90: baseline → candidate | Epipolar median: baseline → candidate | Epipolar p90: baseline → candidate |
|---|---:|---:|---:|---:|---:|
|1660/408|144|0.521 →0.570|1.686 →1.913|0.334 →0.422|1.115 →1.417|
|1700/448|192|0.597 →0.667|2.001 →2.173|0.319 →0.455|1.518 →1.759|
|1740/488|176|0.498 →0.572|2.553 →2.670|0.343 →0.439|1.277 →1.580|

For408, reprojection RMS changes 2.900→2.882px, despite worse median/p90. A tiny RMS improvement dominated by a few outliers is insufficient evidence for improvement. Unfiltered mean/RMS and all sample counts are retained in the JSON.

The square candidate is worse:408 held-out cross-view median 0.521→1.198px, p90 1.686→3.144px; epipolar median 0.334→0.701px. Its withheld fourth-corner error is already 7.39px median under the baseline (7.17px after fitting). Ideal rigid-square constraints are unsupported here. Visual inspection shows visibly curled/bowed paper and apparent-corner ambiguity in problematic marker 90/95 patches; this does not prove every shape error is paper warp.

| Eval camera | Detections408/448/488 |408 held-out cross-view validation | Treatment |
|---|---:|---|---|
|HC|12/15/12|12 scored corners; baseline median 0.406 px → candidate 0.458 px|Anchored|
|DC|4/5/4|No scored held-out marker corners at408|Frozen in conservative candidate|
|KA|0/1/0|No supported held-out validation|Frozen; no calibration claim|

Full 66-camera coverage, update magnitudes and per-camera validation are in `refinement_summary.json` and `freecorners_summary.json`. Target points cover only the back wall: normalized training-corner bounding box approximately[-0.705,0.870,0.032] to[0.457,1.214,0.369]. They do not establish camera accuracy at the fighting actors' depth or provide adequate independent constraints for KA.

[Raw target identification crop](/fsx/tmp/marshawn_001660_hd/printed_target_calibration_INDEPENDENT/HC_identified.jpg) · [DC detection overview](/fsx/tmp/marshawn_001660_hd/printed_target_calibration_INDEPENDENT/overlay_DC.jpg)

![Held-out problematic patches: green observed, red baseline prediction, cyan candidate prediction](/fsx/tmp/marshawn_001660_hd/printed_target_calibration_INDEPENDENT/heldout_problem_patches.jpg)

## Insights

Retain the existing rig. Its typical held-out background error is already approximately 0.3–0.6 raw pixels (roughly0.07–0.14 HD pixels). The targets do not support calibration as the current explanation for weak actor clarity. Curved prints, detector uncertainty, sparse wall-only coverage, and missing KA constraints prevent treating this as a complete calibration certificate.

Further refinement needs reliably localized static features around the actors' depth or measured rigid targets spanning that volume, with independent held-out support. Freeing distortion or inventing print dimensions would add unsupported flexibility. No candidate has been applied.

Reproduce with `/home/pavlo/.conda/envs/nerfstudio/bin/python`:

```bash
python scripts/marshawn_printed_target_independent.py
python scripts/marshawn_printed_target_refine_independent.py
python scripts/marshawn_printed_target_freecorners_independent.py
python scripts/marshawn_printed_target_evaluate_independent.py
python scripts/marshawn_printed_target_visualize_independent.py
```

[Complete final comparison and source/script hashes](/fsx/tmp/marshawn_001660_hd/printed_target_calibration_INDEPENDENT/final_comparison.json). Artifact root: `/fsx/tmp/marshawn_001660_hd/printed_target_calibration_INDEPENDENT`; NPZ candidates are explicitly marked `DO_NOT_APPLY`. Cached detections include source paths, IDs, ordered corners and duplicate exclusions; per-point held-out prediction files include the exact camera subset used to estimate geometry.

## Bounded continuation: shared radial-distortion residual

After the user resumed calibration work, one new low-dimensional alternative was tested rather than repeating failed pose/board BA: **one shared residual Δk1**, with all camera poses, focal lengths, principal points and other distortion coefficients fixed. This tests a systematic radial-model error without imposing flat/square printed geometry. The original rig remains untouched.

The protocol was saved before fitting: use frame408, marker IDs not divisible by5, and exclude cameras DC/HC/KA entirely from fitting. Whole held-out IDs and the camera-excluded observations are scored on frames408/448/488 with identical support before/after. Cross-view corner geometry is estimated from alternating camera subsets and scored only in the other subset, as before. Acceptance required ≥0.05 raw-pixel cross-view median gain on all3 frames, no worse median/p90 projection/epipolar errors, no worse held-out-camera median/p90, ≥60% held-out-ID medians improved, and a non-boundary optimum.

The fit converged at **Δk1=+0.0001313994**, using2460 training corner-pair residuals. It **failed independent validation**:

| Local frame | Held-out-ID cross-view median: original → candidate | Held-out-camera epipolar median: original → candidate | Held-out-camera epipolar p90: original → candidate |
|---|---:|---:|---:|
|408|0.521 →0.500 px|0.359 →0.364 px|1.079 →1.100 px|
|448|0.597 →0.590 px|0.294 →0.300 px|0.939 →0.955 px|
|488|0.498 →0.506 px|0.318 →0.325 px|1.033 →1.077 px|

Small projection gains on two frames reverse on the third, while median/p90 epipolar error worsens for observations from cameras excluded during fitting on all3 frames. These mixed changes are also small relative to the previously measured0.347/0.380px temporal corner-displacement medians; this comparison is repeatability context, not a formal confidence interval. KA still lacks useful multi-view marker support, so the excluded-camera aggregate does not certify KA.

![Rejected shared-k1 held-out comparison](/fsx/tmp/marshawn_001660_hd/printed_target_calibration_INDEPENDENT/shared_radial_k1_HOLDOUT_20261106/heldout_comparison.png)

**Stop decision: retain the original calibration.** No further coefficients, rolling-shutter terms or board-flexibility parameters were fitted. The observed curved prints/corner ambiguity and sparse back-wall coverage do not supply evidence to justify those extra parameters, and static-marker observations do not independently identify moving-actor rolling-shutter effects. This is a limit of the tested evidence, not proof that every camera is perfect at actor depth. No conclusion about the correctness of the actor AABB follows from this audit.

[Pre-fit protocol](/fsx/tmp/marshawn_001660_hd/printed_target_calibration_INDEPENDENT/shared_radial_k1_HOLDOUT_20261106/protocol_frozen.json) · [Full results](/fsx/tmp/marshawn_001660_hd/printed_target_calibration_INDEPENDENT/shared_radial_k1_HOLDOUT_20261106/result.json) · [Stopping/hash receipt](/fsx/tmp/marshawn_001660_hd/printed_target_calibration_INDEPENDENT/shared_radial_k1_HOLDOUT_20261106/stopping_receipt.json). Both original rig-file SHA256 values match the first audit. The CPU-only helper exited0; no background job, NeRF training or AABB change was started. Reproduction: `python scripts/marshawn_printed_target_radial_holdout.py` (its isolated output directory must not already exist; original outputs are never overwritten).
