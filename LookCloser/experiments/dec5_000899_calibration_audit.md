# DEC5 frame 000899 calibration audit

## What was tested

Independent JPEG SIFT matching, fixed-camera triangulation, extrinsics-only BA, and full camera BA on 65 cameras.
The source dataset was not modified. Full-intrinsics BA is treated as a diagnostic because every physical camera contributes only one image.

## Results

- SIFT features per camera: 5998 min, 12763.5 mean, 22457 max.
- Verified image pairs: 1712; fixed-model observations: 420665.
- Median of per-camera median reprojection errors: fixed 0.6436px, extrinsics-only BA 0.6073px, full BA 0.3982px.
- Extrinsics-only rotation correction: median 0.0788°, p95 0.1941°, max 0.2342°.
- Extrinsics-only center correction / camera-array scale: median 0.002379, p95 0.006117, max 0.008962.
- Full-BA maximum focal change: median 6.014%, p95 13.876%, max 15.843%.

### Potentially problematic cameras

| Rank | Physical camera | Image | Flags | Score | Features | Pairs | Inliers | Fixed median px | Rotation shift ° | Center shift frac | Focal change % |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `C004_D005_12107I` | `frame_train_00011.jpg` | 5 | 0.826 | 6571 | 38 | 18098 | 1.1324 | 0.2005 | 0.006142 | 1.902 |
| 2 | `C004_B005_1210ER` | `frame_train_00009.jpg` | 3 | 0.882 | 9185 | 43 | 17577 | 1.2105 | 0.2153 | 0.006021 | 9.414 |
| 3 | `C004_E005_1210X7` | `frame_train_00012.jpg` | 3 | 0.748 | 5998 | 44 | 16088 | 0.6883 | 0.0669 | 0.002448 | 10.630 |
| 4 | `A004_D005_1210CH` | `frame_train_00003.jpg` | 3 | 0.701 | 7462 | 32 | 12124 | 0.6353 | 0.1087 | 0.003152 | 1.073 |
| 5 | `M004_D005_1210R6` | `frame_train_00058.jpg` | 3 | 0.681 | 15826 | 46 | 46214 | 1.0249 | 0.2342 | 0.006371 | 12.122 |
| 6 | `A004_C005_121008` | `frame_train_00002.jpg` | 3 | 0.621 | 7701 | 41 | 16937 | 0.8849 | 0.0548 | 0.001308 | 5.036 |
| 7 | `J004_B005_1210GR` | `frame_train_00041.jpg` | 3 | 0.569 | 22457 | 60 | 67647 | 1.1261 | 0.1986 | 0.008962 | 10.266 |
| 8 | `M004_A005_1210WZ` | `frame_train_00055.jpg` | 2 | 0.824 | 8943 | 49 | 30091 | 0.8489 | 0.1680 | 0.004399 | 15.843 |
| 9 | `N004_A005_121003` | `frame_train_00060.jpg` | 2 | 0.728 | 9719 | 48 | 26892 | 0.5762 | 0.1765 | 0.004524 | 13.444 |
| 10 | `L004_A005_1210YO` | `frame_train_00050.jpg` | 2 | 0.699 | 10158 | 50 | 36915 | 0.9688 | 0.0849 | 0.003101 | 14.072 |
| 11 | `B004_B005_1210Z3` | `frame_train_00005.jpg` | 2 | 0.674 | 9392 | 44 | 15348 | 0.7781 | 0.0608 | 0.002972 | 7.172 |
| 12 | `A004_E005_1210OU` | `frame_train_00004.jpg` | 2 | 0.621 | 8265 | 29 | 12376 | 0.8945 | 0.0661 | 0.001216 | 3.240 |

## Insights

The ranking is a screening list, not proof that a camera is wrong. A camera is stronger evidence only when it is simultaneously weak in the match/track graph and requires a large extrinsics-only correction. Large full-BA focal changes alone may be one-image-per-camera overfitting.

Machine-readable results: `/home/brans/lookcloser_temp/calibration_audit_000899_colmap/calibration_camera_ranking.json`.
