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

The original campaign ROI (`face_roi_v1`) was subsequently invalidated: although its metadata
declared neck and background excluded, its static lower boundary included a moving
neck/external-silhouette wedge. The official campaign values were atomically rescored from the
unchanged retained predictions with per-frame held-out-GT-only `face_roi_v2` polygons. The
correction preserved the old artifacts in a checksum-bound audit archive and did not modify any
mesh or render. The causal analysis is recorded in [`../lpips_temp.md`](../lpips_temp.md).

After the ROI correction, a separate three-frame render canary (`000951`, `000953`, `000955`)
localized small grey/colour lipstick shards to connected fallback visibility components in the
hard nearest-fill texture pass. A campaign-wide opt-in hard-source continuation rule now replaces
only small (`20..1000 px`) photometrically discontinuous fallback components with the nearest
train camera's RGB at the same target-depth 3D reprojection. It uses no masks or eval RGB and does
not average sources. The frozen PatchMatch/TSDF geometry remains unchanged.

## Results

Campaign execution is in progress. At the 29-frame checkpoint (`000899..000955`), all 29 native
ear/lipstick reviews pass and the independent partial audit reports no full-frame metrics. The
initial three-frame gate passed and fixed regression thresholds at `1 dB` PSNR, `0.03` SSIM, and
`0.05` LPIPS relative to the median of the last five accepted frames.

| Face-only metric | Minimum | Median | Maximum |
|---|---:|---:|---:|
| PSNR (dB) | 24.6147 | 28.4015 | 29.7733 |
| SSIM | 0.827472 | 0.888099 | 0.898313 |
| LPIPS | 0.050266 | 0.057630 | 0.090413 |

The hard-source correction was first prepared for every published frame before any replacement,
then reviewed and atomically published. Across those 29 frames, PSNR changed by
`[-0.0531, +0.0968] dB`, SSIM by `[-0.000142, +0.000488]`, and LPIPS by
`[-0.000759, +0.000018]`; mesh hashes remained identical. Final distributions, worst frames,
visual pass/fail counts, and contact-sheet links will replace this checkpoint after the 50-frame
audit passes.

## Insights

The campaign is deliberately fail-closed: an immutable request binds ordered source transforms,
calibration, recipe, and script hashes; a frame is published atomically only after remote and
local checksum validation, face scoring, and visual review. Regression thresholds are not fixed
from the historical surface-mask score. They were derived from the three newly scored and
visually accepted initial frames using the larger of the requested signal floor and three robust
MAD scales, then compared with the last five accepted frames.

The apparent early LPIPS rise was not a numerical LPIPS failure or a face-render collapse. A
same-prediction control on `000941` changed only the ROI and moved PSNR/SSIM/LPIPS from
`19.8437 / 0.807832 / 0.141004` to `27.4698 / 0.868682 / 0.061396`. The removed wedge accounted
for 84.56% of the old ROI squared error, whereas the corrected face ROI contained only 0.0154%
invalid prediction pixels. The real black support gap remains visible in the actor overview and
is reviewed as an actor/silhouette geometry issue; it is outside the declared face-only metric
population and must not be relabeled as face degradation.
