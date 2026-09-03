# DEC5 5A-3 first-50 fixed-pose PatchMatch-TSDF campaign

## What was tested

The campaign applies the frozen `fixed-pose COLMAP PatchMatch -> TSDF mesh -> hard nearest-fill`
recipe to the first 50 six-digit DEC5 5A-3 frame directories in numeric order. Geometry uses 62
train cameras; the only held-out camera is physical camera `F004_B005_1210O9`. Fixed GLOMAP
extrinsics and intrinsics are transferred by unique `physical_camera`; there is no per-frame
feature matching, bundle adjustment, pose optimization, learned refiner, RGB averaging, or mask
in geometry, fusion, source selection, or prediction.

JPEG ingest is temporary and uses per-image exposure, Reinhard, sRGB, middle gray `0.18`, quality
98, and 4:4:4 chroma. PatchMatch uses the verified CUDA COLMAP `3.13.0.dev0` commit `5509fffe` in
photometric and then geometric passes. The TSDF output is an extracted `.ply` mesh plus
manifests; Open3D's raw VoxelBlockGrid is not serialized.

Metrics are display-domain face-only PSNR, SSIM, and Alex-LPIPS. A human-drawn polygon on the
held-out GT defines the face/ear/lips pixels. The prediction is not consulted while drawing it,
and missing/black predicted pixels inside it remain errors. This protocol is intentionally not
numerically identical to the older `000899` table that intersected an independent Splatfacto
surface mask.

## Results

Campaign execution is in progress. The mandatory initial visual gate is `000899`, `000901`, and
`000903`; remaining frames will not start until their native ear and lipstick crops show the same
coherent surface quality. Final metric distributions, worst frames, visual pass/fail counts, and
contact-sheet links will be written here after the independent 50-frame audit passes.

## Insights

The campaign is deliberately fail-closed: an immutable request binds ordered source transforms,
calibration, recipe, and script hashes; a frame is published atomically only after remote and
local checksum validation, face scoring, and visual review. Regression thresholds are not fixed
from the historical surface-mask score. They will be derived from the three newly scored and
visually accepted initial frames using the larger of the requested signal floor and three robust
MAD scales, then compared with the last five accepted frames.
