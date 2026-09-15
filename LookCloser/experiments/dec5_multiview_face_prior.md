# DEC5: face landmarks do not support direct jaw completion

## What was tested

A specialized face prior was tested on jaw-gap times **001193 and 001195**.
This is not DA3, hair completion, or the earlier hand model. The question is
whether matching predicted face-landmark IDs across calibrated real cameras can
supply a precise missing-cheek/chin surface prior while retaining trusted COLMAP
geometry. The correspondence gate fails, so **no face mesh or completion was
created and no production geometry, renderer, environment, or video was changed**.

The official [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
bundle estimates 478 face landmarks. Its
[Python IMAGE API](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
ran locally using the CPU delegate in the existing isolated
`/home/brans/lookcloser_temp/mediapipe_hand_env` (MediaPipe 0.10.21, Python 3.12).
No images were uploaded. The graph initialized an EGL context, but inference
explicitly used CPU/XNNPACK; no CUDA inference/training job was launched.

Model: [official float16 bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task),
downloaded once and pinned by SHA-256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
The URL is a mutable upstream `latest`; reproduction must match the retained hash.
Model normalized z and optional face transforms are **not used as metric depth**.
Only 2D coordinates are triangulated through the actual calibrated cameras.

Inputs are the current uint8 display RGB from `calibrated_depth_witness.load_images`
and normalized cameras from `joint_temporal_texture.cameras`: 62 real train
cameras per time, frozen centered camera gains/exposure, no eval camera input.
One fixed native portrait crop `(0,400,1080,1550)` includes the face and jaw.
An earlier `(0,0,1080,1100)` staging crop cut off several chins; it was visually
rejected **before inference**, retained separately, and never scored.

Eight train cameras were reserved before inference: E/B, F/D, G/B, H/D, I/B,
K/B, L/D, and M/B. They are excluded from fitting, RANSAC/consensus selection,
and all candidate construction. This is validation against independent-camera
**model predictions**, not manually annotated ground-truth landmarks. F/B and
the other two eval cameras are never loaded.

Two fixed controls triangulate the 468 non-iris landmarks independently:

- `all_fit_robust`: calibrated DLT initialization followed by robust nonlinear
  reprojection fitting, two-pixel soft-L1 scale, using every available fit view.
- `train_consensus`: at most 256 fixed-seed calibrated camera-pair hypotheses;
  three-pixel inlier threshold, at least three and 30% of available fit cameras,
  followed by the same robust refinement. Pair parallax must be at least one degree.
  Validation cameras never select the consensus.

The predeclared gate requires each point to have at least three fitting views,
two validation views, fit median ≤2 px, validation median ≤2 px, and validation
p90 ≤4 px. At least 80% of **each** lower-face group must pass before surface
completion: 21 jaw-contour and 34 paired cheek IDs. These explicit index groups
and constants are frozen in `request.json`; no threshold was relaxed after the
result. Twenty central eyes/nose/mouth points and all 468 points are retained as
diagnostics, not pooled to hide a jaw failure.

## Results

One face was detected in **57/62** views at 001193 and **59/62** at 001195;
all eight validation cameras detected a face at both times. The fitting arms
therefore have 49/51 detected fit views. CPU inference took 11.4 seconds total.
A detected face is not evidence of precise cross-view surface correspondence.

All errors below are native pixels. Fit values for the consensus arm use its
selected inliers; validation values use all available reserved-camera predictions.
Group statistics pool landmark-camera residuals, while the last column counts
points passing the stricter per-point gate.

| Time | Region | Fit control | Fit median | Validation median / p90 | Triangulated | Passing |
|---|---|---|---:|---:|---:|---:|
| 001193 | Jaw | All-view robust | 9.11 | 7.39 / 16.63 | 21/21 | 0/21 |
| 001193 | Jaw | Train consensus | N/A | N/A | 0/21 | 0/21 |
| 001195 | Jaw | All-view robust | 9.70 | 7.71 / 15.39 | 21/21 | 0/21 |
| 001195 | Jaw | Train consensus | N/A | N/A | 0/21 | 0/21 |
| 001193 | Cheek | All-view robust | 3.14 | 2.49 / 6.88 | 34/34 | 1/34 |
| 001193 | Cheek | Train consensus | 1.86 | 2.34 / 5.72 | 30/34 | 6/34 |
| 001195 | Cheek | All-view robust | 3.46 | 2.74 / 6.67 | 34/34 | 0/34 |
| 001195 | Cheek | Train consensus | 1.74 | 2.31 / 4.75 | 27/34 | 6/34 |

Every jaw point fails the fixed consensus requirement (at least 15/16 inlier
fit views at these times). Those absent triangulations are **N/A, not zero
error**. The six passing cheek points per time are too sparse to justify the
requested dense cheek/chin completion. Even central diagnostic points pass only
4/20 and 3/20 under consensus. The per-point records and rejected populations
are retained, not silently dropped from the denominator.

![Plausible single-view face topology, 001195 G/B](</mnt/data/dec5_multiview_face_prior/001195/landmarks/G004_B005_1210FG.png>)

![Independent G/B lower-face reprojection, 001195 all-view fit](</mnt/data/dec5_multiview_face_prior/triangulation/001195/all_fit_robust/G004_B005_1210FG_reprojection.png>)

![Independent M/B reprojection, 001195 all-view fit](</mnt/data/dec5_multiview_face_prior/triangulation/001195/all_fit_robust/M004_B005_12109O_reprojection.png>)

Cyan is the projected triangulated point, red is the unused view's prediction,
and yellow joins the discrepancy. Single-view topology looks plausible, but
jaw-contour points visibly move relative to the predicted surface boundary.
Consensus cheek points are closer; they do not restore a coherent jaw contour.
Both train preview sheets, three initial landmark overlays, and six final
reprojection crops were directly inspected (11 images). `visual_review.json`
lists the exact images and hashes; not every saved overlay was inspected.

The separate audit verifies all **124 source EXR hashes**, cropped inputs,
calibration, frozen display parameters, original meshes, model/topology, and
producer-script bindings. It reprojects **1,694** saved triangulated landmarks
and rechecks residuals, excluded-camera membership, denominators, and gate
results. This replays arithmetic, not neural inference or anatomical truth.
Five synthetic tests pass: calibrated recovery for both arms, an unused view,
outlier rejection, degenerate/missing inputs, and portrait/native coordinates.

PSNR/SSIM/LPIPS are **N/A**: the correspondence gate failed before any new surface
or rendered RGB prediction. Comparing decorative landmark overlays to RGB would
not measure reconstruction quality. No full-video rerun or optional temporal
transfer was needed to establish this bounded negative result.

## Insights

Reject this direct per-index triangulated landmark route for the current narrow
jaw repair. Silhouette contours are especially problematic: the visible outline
can correspond to different surface locations across viewpoints, even when each
2D prediction looks reasonable. The measured disagreement does not establish
that calibration alone or model error alone caused every residual.

This is **not** evidence that fitting a parametric head model is impossible.
A visibility-aware parametric shape/expression fit with silhouette constraints
is a different hypothesis and was not tested. Likewise, a few useful cheek
landmarks do not validate an inferred dense surface between them. Trusted COLMAP
geometry remains unchanged; no unsupported anatomy was invented to force a pass.

Reproduction: `study_multiview_face_prior.py stage` in the reconstruction `.venv`,
place the hash-matching official model at `OUTPUT/face_landmarker.task`, then
`infer` in the isolated MediaPipe environment. Run `triangulate_face_prior.py`,
`audit_multiview_face_prior.py`, visually inspect the listed native views, and
only then `audit_multiview_face_prior.py --finalize`. Use one fresh `--output`
for staging/inference and matching `--root` for analysis/audit, EXR enabled,
and two OpenBLAS/OMP threads. Tests: `pytest -o addopts='' -q
tests/test_triangulate_face_prior.py`. Artifacts live at
`/mnt/data/dec5_multiview_face_prior`; the initial rejected crop is retained at
`/mnt/data/dec5_multiview_face_prior_initial_crop`.
