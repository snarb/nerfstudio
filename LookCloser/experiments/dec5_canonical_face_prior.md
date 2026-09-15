# Regularized canonical face prior: 001193

## What was tested

One bounded CPU-only human-shape experiment following the [negative independent-landmark triangulation study](dec5_multiview_face_prior.md). A coherent canonical surface can stabilize ambiguous 2D face predictions; this is a different hypothesis from treating every predicted jaw-contour index as a fixed 3D correspondence.

Result: regularization improves the face fit and is stable across camera splits, but the template does not provide a boundary-compatible repair for the requested jaw hole. No triangles were added, no existing geometry changed, and nothing was promoted into the 6K or video baseline. Frame 001195 was not run after this decisive local failure.

The public [MediaPipe canonical face asset](https://github.com/google-ai-edge/mediapipe/blob/87f8074eb976e59655f243a8715c373e76ce3abb/mediapipe/modules/face_geometry/data/canonical_face_model.obj) is pinned to upstream commit `87f8074eb976e59655f243a8715c373e76ce3abb`, under the retained [repository Apache-2.0 license](https://github.com/google-ai-edge/mediapipe/blob/87f8074eb976e59655f243a8715c373e76ce3abb/LICENSE). The OBJ has 468 vertices and 898 triangles; SHA-256 `8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618`. Assets and license are retained under [/mnt/data/dec5_canonical_face_prior](/mnt/data/dec5_canonical_face_prior/protocol.json). No image uploads, model training, production-environment mutation, new neural inference, or paper download.

Frozen fitting protocol:

- Reuse calibrated, current-display train RGB landmarks from the preceding study. Use only image xy, never predicted z, predicted world coordinates, or independently triangulated landmarks.
- The same eight reserved TRAIN cameras remain excluded from fitting and from depth-vote support. None of the three dataset held-out cameras enters this experiment. Of 62 train cameras, 57 have detections: 49 fitting and eight validation cameras.
- Fit 54 interior/cheek indices, not fixed jaw-contour correspondences. Initialization comes from original COLMAP depth anchors, not triangulated neural positions.
- A depth anchor must agree with the unchanged original TSDF within 0.001 and with at least three other fitting-camera geometric depth maps within 0.001, 1.5 pixels round-trip, and over one degree parallax. This yields 6,684 fitting and 1,071 validation anchors. They are correlated estimates with geometric support, not independently measured ground truth.
- Control: seven-parameter similarity transform. Regularized arm: that transform plus eight smooth graph-Laplacian modes, each with three spatial coefficients. Affine components are removed from the shape basis. Coefficient prior scale is 1.5% canonical face width; bounds are ±8%. Joint robust fitting balances 3-pixel landmark residuals and 0.001 point-to-plane residuals. Both use the same inputs and initialization recipe.
- Freeze validation limits before fitting: core median/P90 at most 4/8 pixels; point-to-plane P90 at most 0.002; maximum deformation 15% face width; split-camera lower-face disagreement P90 at most 0.0015. Local geometry also requires distance to original surface at most 0.002 and to original boundary at most 0.003. These limits were not widened.

All 3D distances below use the calibrated, normalized original COLMAP scene gauge—not an assumed physical millimeter scale. Camera intrinsics/extrinsics remain fixed. The target novel camera and previously selected jaw component are read only after fitting for evaluation.

## Results

| Reserved-train measurement | Similarity only | Regularized 8 modes |
|---|---:|---:|
| Core landmarks, median / P90 pixels (432 observations) | 5.316 / 10.004 | 3.673 / 7.625 |
| Cheek landmarks, median / P90 pixels (272 observations) | 4.682 / 9.198 | 3.482 / 7.157 |
| All supported anchors, absolute point-to-plane median / P90 | 0.000420 / 0.000996 | 0.000239 / 0.000810 |
| Supported cheek anchors, absolute point-to-plane median / P90 (247 observations) | 0.000357 / 0.002171 | 0.000391 / 0.002054 |

The regularized fit passes the aggregate numerical gate. Maximum deformation is 5.66% of fitted face width. Independent fitting-camera halves disagree by median/P90 0.000112/0.000258 on 55 lower-face vertices: the remaining error is stable bias, not large split instability. Cheek anchors have median 38 other fitting-camera depth votes. Nevertheless, their P90 point-to-plane discrepancy remains slightly beyond 0.002, and image reprojection error is not boundary precision. Validation measures agreement with reserved model predictions and depth estimates, not manually annotated ground truth.

Native visual review of five fixed train cameras (G_A, G_B, M_A, M_B, E_B) confirms a coherent human face but coarse jaw, cheek, mouth and nose detail. Original COLMAP preserves identity and detail substantially better. The canonical surface ends at the front chin/face perimeter; it does not model the under-chin or neck. Review images are prior-only diagnostics, not replacement renders or accepted completions.

![Reserved G_B: RGB, original, similarity, regularized](/mnt/data/dec5_canonical_face_prior/review/G004_B005_1210FG_clay.png)

![Reserved M_B projection: observed red, fitted topology green](/mnt/data/dec5_canonical_face_prior/review/M004_B005_12109O_projection.png)

Dense locality check uses 9,728 uniformly subdivided triangle-center samples on a topology-defined lower-face region, without selecting geometry from the target image. Only 240 samples pass both frozen surface/boundary distance limits. This is preliminary geometric proximity, not semantic acceptance or evidence of missing anatomy.

The pre-existing requested 001193 jaw component contains 44 original-mesh misses. Unrestricted template rendering hits all 44 rays, but **zero of those 44 points passes the frozen locality gate**:

| Requested-hole diagnostic | Result |
|---|---:|
| Prior-hit point distance to old surface, median / P90 | 0.001429 / 0.001511 |
| Prior-hit point distance to old open boundary, median / P90 | 0.006652 / 0.006909 |
| Original visible rim points evaluated | 24 |
| Rim-to-prior distance, median / P90 | 0.009993 / 0.010170 |
| Rim-to-prior signed original-normal offset, median / P90 | +0.007999 / +0.008245 |
| Rim points within the frozen 0.002 surface limit | 1 / 24 |
| Rim nearest-prior points lying on the template's open edge | 24 / 24 |

Rim distances range from 0.001979 to 0.010202; signed offsets use the original triangle winding normals. The small near-face part and the farther underside/neck-side part are not interchangeable. The nearest model point is on the open perimeter for every rim sample, not in an interpolating face interior. The apparent image coverage comes from a front-face surface overlapping the requested rays, not a surface that connects to the measured under-cheek/neck-side boundary. Replacing or extending the template to bridge that difference would violate the frozen local-completion premise.

![Original and canonical-only target crop; red marks the preselected hole](/mnt/data/dec5_canonical_face_prior/requested_hole_marked_native.png)

No combined candidate mesh or RGB prediction was generated after this failure. PSNR, SSIM and LPIPS are N/A: prior-only clay is a geometric diagnostic, not an image prediction to score against photographed RGB. No hair claims are made.

## Insights

This experiment supports a limited positive conclusion: a low-frequency canonical shape is much more coherent than independently triangulated ambiguous landmarks. It does **not** support the requested local jaw completion. Low split uncertainty and many agreeing anchors on existing facial skin cannot certify an unmodeled underside, and filling image rays alone is not evidence of correct geometry.

The bounded test stops here with the negative local result preserved. A future full-head/neck parametric model with appropriate surface coverage would test a materially different hypothesis; it was not tested, and its success is not implied. Existing geometry, original triangles, source RGB, calibration, and production/video defaults are unchanged.

### Reproduction and audit

Use `/home/brans/repos/nerfstudio/.venv/bin/python` with `OPENCV_IO_ENABLE_OPENEXR=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2`. In a fresh root, run `study_canonical_face_prior.py assets`, `stage`, and `fit`, followed by `review_canonical_face_prior.py`, `probe_canonical_face_locality.py`, and `audit_canonical_face_prior.py`. The first helper supports `--output`; the three companion helpers intentionally share its fixed `OUT` constant, so change that constant consistently for a separate full rerun. Existing frozen roots are not overwritten by stage/fit/review/locality commands.

Optimization alone took 0.063 seconds for similarity and 0.132 seconds for the regularized full-camera fit; split fits were also CPU-only. These are optimizer timings, not total asset loading, raycasting, diagnostics, or prior neural inference time. No GPU job was launched.

Six synthetic tests cover calibrated similarity recovery, zero-shape identity, affine-excluded smooth modes, excluded jaw correspondences, open topology boundaries, and exact segment distance. Run `python -m pytest -o addopts='' -q tests/test_canonical_face_prior.py` (the inherited parent pytest configuration otherwise requests unavailable xdist).

The [audit](/mnt/data/dec5_canonical_face_prior/audit.json) verifies 155 file bindings including all 62 source geometric depth maps, and replays 21,666 saved landmark/point-to-plane residuals. Original-mesh SHA is `316dc2b6a82c0d99a6e399d15d941a69d491896921bc82e4d88ea307bb358b03`. Eleven native diagnostic images were inspected; the parent independently inspected the G_B and M_B clay panels. The initial review manifest accidentally has an empty `review_cameras` field: the supplementary audit explicitly derives and verifies the actual five cameras from the ten image files without altering that frozen producer or its outputs.

Primary artifacts: [frozen protocol](/mnt/data/dec5_canonical_face_prior/protocol.json), [fit summary](/mnt/data/dec5_canonical_face_prior/fit_summary.json), [locality results](/mnt/data/dec5_canonical_face_prior/locality/result.json), [review inventory](/mnt/data/dec5_canonical_face_prior/review/manifest.json). The isolated prior-only meshes and failed diagnostics are retained.
