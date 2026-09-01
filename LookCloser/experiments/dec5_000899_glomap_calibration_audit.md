# dec5_000899 GLOMAP / COLMAP 4.1 calibration audit

## Outcome

The maintained GLOMAP implementation in COLMAP 4.1.1 reconstructs all 65/65 cameras from the temporary full-frame JPEGs. This broadly supports the existing `transforms.json`: with a single common focal initialization, robustly aligned camera centers differ by a median 0.123 scene units (0.127× the median nearest-camera spacing), orientations by 0.101°, and mean focal length by 2.32% absolute median.

There is no evidence here for replacing the calibration wholesale. There is strong, repeatable evidence to inspect a small set of physical cameras, especially `J004_B005_1210GR` and `N004_C005_1210AA`. Conversely, standalone view-graph self-calibration is unstable for a few cameras because every physical camera has only one image; it improves internal reprojection error but degrades agreement with the supplied calibration.

The latest incremental COLMAP mapper is not a viable replacement under the same unknown-per-camera initialization: it registers only 6/65 images and produces implausible focal estimates.

Machine-readable results: [calibration_audit.json](/home/brans/lookcloser_temp/dec5_000899_glomap_colmap411_audit/calibration_audit.json). Exact options: [settings.json](/home/brans/lookcloser_temp/dec5_000899_glomap_colmap411_audit/settings.json).

## What was tested

### Version and installation audit

As of 2026-08-30, the official latest stable COLMAP release is [4.1.1](https://github.com/colmap/colmap/releases/tag/4.1.1), dated 2026-07-17 in the [official changelog](https://colmap.github.io/changelog.html). Standalone GLOMAP's latest release is [1.2.0](https://github.com/colmap/glomap/releases); that repository was archived on 2026-03-09. COLMAP 4.0 integrated GLOMAP as `global_mapper` and states that it is maintained in COLMAP going forward ([official 4.0.0 release](https://github.com/colmap/colmap/releases/tag/4.0.0)). Therefore the primary audit uses COLMAP/pycolmap 4.1.1 `global_mapping`, not the archived standalone binary.

Local state:

| Component | Local result |
|---|---|
| System COLMAP | Ubuntu `3.9.1-2build2`; unusable because `/usr/local/lib` Qt shadows the expected Qt and causes a `QOpenGLWidget::event` symbol lookup failure. |
| Standalone `glomap` | Not installed. |
| Isolated Conda COLMAP 4.1.1 CLI | Installed only under the audit temp directory. The package first omitted `libOpenImageIO.so.3.1`; adding it then exposed a FAISS ABI symbol mismatch. No system packages or libraries were changed. |
| Executed implementation | Official `pycolmap==4.1.1` manylinux wheel in the isolated prefix; CPU build (`has_cuda=False`). It exposes the same maintained `global_mapping` pipeline. |

The official GLOMAP guidance recommends assigning intrinsics according to physical-camera identity and relaxing epipolar error for high-resolution or blurry images ([getting-started guidance](https://github.com/colmap/glomap/blob/main/docs/getting_started.md)). Here each of the 65 frames has a unique `physical_camera`, so extraction used 65 separate PINHOLE cameras. PINHOLE is equivalent to the source OPENCV model because all source distortion coefficients are zero.

### Reconstruction configurations

- Input: `/home/brans/lookcloser_temp/dec5_000899_triangle_fullframe_jpeg`, 65 images at 1920×1080; immutable EXR source was read only.
- Features: full-resolution CPU SIFT, up to 16,384 features/image; observed 1,698–11,629, median 6,702.
- Matching: exhaustive 2,080 possible pairs, ratio 0.85, cross-check, guided matching, 4 px geometric-verification threshold.
- Camera initialization: 65 independent PINHOLE cameras, principal point fixed at `(960, 540)`. The uncalibrated-init runs used one common 10,479.19 px focal derived from the supplied calibration's median. This makes their individual focal and pose solution independent, but not their approximate focal scale.
- `global_uncalibrated_init`: COLMAP 4.1.1 global mapper directly on that initialization.
- `global_view_graph_calibrated`: copied database, then COLMAP 4.1.1 `calibrate_view_graph` with defaults before global mapping.
- `incremental_uncalibrated_init`: COLMAP 4.1.1 incremental mapper on the unchanged common-focal database.
- Pose comparison: 80%-trimmed Umeyama Sim(3) for centers and a separate 80%-trimmed global SO(3) alignment for orientations. Center errors are therefore robust to a few calibration outliers; orientation errors do not include an arbitrary global-frame offset.

No source image, source transform, or temporary JPEG input was modified. The source `transforms.json` SHA-256 recorded by the audit is `ab99b704fb12e0745348d200fafe6ccd3e6599c53711a9a4613bfd32f0af27ad`.

## Results

### Match graph and reconstruction coverage

The view graph is strong rather than marginal: 1,747 verified pairs have at least 15 inliers, median 385 inliers/pair, and all 65 images form one connected component. Thus full global registration is not being rescued by a tiny fragile chain.

| Mapper | Registered | Points | Observations | Median track length | Median recomputed reprojection | Mean COLMAP point error |
|---|---:|---:|---:|---:|---:|---:|
| Global, common focal init | 65/65 | 17,392 | 158,354 | 6 | 0.550 px | 0.707 px |
| Global, view-graph calibrated | 65/65 | 18,625 | 151,794 | 5 | **0.493 px** | **0.602 px** |
| Incremental, common focal init | **6/65** | 2,908 | 9,048 | 3 | 0.488 px | 0.567 px |

The incremental model's low reprojection error is not evidence of success: it is computed only on a six-camera fragment. After alignment, that fragment has 6.80° median orientation disagreement and 728% median absolute focal disagreement, so it failed geometrically.

### Agreement with supplied `transforms.json`

| Global mapper | Center median / p95 | Center median in nearest-camera spacings | Orientation median / p95 | Absolute focal disagreement median / p95 |
|---|---:|---:|---:|---:|
| Common focal init | **0.123 / 0.669** | **0.127×** | **0.101° / 0.344°** | **2.32% / 9.81%** |
| View-graph calibrated | 0.192 / 1.209 | 0.198× | 0.112° / 0.545° | 3.07% / 15.40% |

View-graph calibration estimates focal priors spanning 3,331–18,863 px, versus 8,703–12,688 px in the supplied calibration. Its median initial focal deviation is -35.7%. Bundle adjustment recovers most cameras, but the calibrated global result still has worse external center, orientation, and focal agreement despite its better internal reprojection error. This is the expected degeneracy risk when each independently calibrated camera contributes only one narrow-FOV image.

The two successful global models nevertheless agree closely with each other overall: median cross-model center difference 0.122 scene units, orientation difference 0.057°, and absolute focal difference 1.06%. The few large cross-model outliers are therefore informative.

### Ranked physical-camera triage

The machine JSON contains all 65 ranked identities. The score below combines within-dataset percentiles for robust center/orientation/focal disagreement, reprojection error, observation support, view-graph degree, and cross-global instability. It is a triage score, not ground-truth error.

| Rank | Physical camera | Frame / old COLMAP id | Common-init center, focal | VG-calibrated center, focal | Evidence and interpretation |
|---:|---|---|---:|---:|---|
| 1 | `C004_B005_1210ER` | train 00009 / 9 | 0.403, -5.6% | 0.825, -12.2% | Both runs disagree; relatively high median reprojection (0.785/0.708 px), graph degree 46. Inspect locally. |
| 2 | `C004_D005_12107I` | train 00011 / 11 | 0.701, -9.6% | 0.585, -8.5% | Repeatable focal/center disagreement and highest local median reprojection (0.996/0.876 px). |
| 3 | `A004_D005_1210CH` | train 00003 / 3 | 0.355, -5.6% | 1.726, -25.0% | Primarily view-graph-calibration instability; lower graph degree 36 and about 1,000 observations. Medium confidence. |
| 4 | `N004_C005_1210AA` | train 00062 / 65 | 1.388, -13.5% | 1.708, -15.5% | Repeatable under both global methods with degree 54 and 3,014–3,386 observations. High-confidence target. |
| 5 | `J004_B005_1210GR` | train 00041 / 44 | **2.111, +24.3%** | **2.348, +28.0%** | Largest repeatable pose/focal outlier, yet 11,629 features, degree 61, 3,366–3,663 observations, and low 0.540/0.498 px median reprojection. Highest-actionability target. |

Visual inspection of frames 00041, 00062, 00003, 00009, and eval 00001 found no gross blur or blank imagery. Frames 00041 and 00062 are sharp and richly textured, increasing confidence that their repeatable discrepancy is geometric rather than feature starvation. Frames 00003 and 00009 have more difficult lighting/noise, consistent with their weaker evidence or higher reprojection residual.

Eval cameras:

- `J004_D005_1210TA` and `L004_B005_12106A` are stable across both global runs.
- `F004_B005_1210O9` is stable in the common-init run (0.112 center units, 0.107°, -0.37% focal) but becomes a large outlier only after view-graph calibration (1.297 units, 0.616°, +20.6%). This is evidence against changing that eval camera from view-graph calibration alone.

## Insights and next steps

1. Keep the current transforms as the working calibration; global COLMAP 4.1.1 supports them at dataset scale, while unconstrained per-camera view-graph calibration is less stable.
2. Audit `J004_B005_1210GR` and `N004_C005_1210AA` first against original camera metadata and pairwise face/background epipolar overlays. Their discrepancies survive both global initializations and have strong feature/track support.
3. Then inspect `C004_B005_1210ER` and `C004_D005_12107I`; their signal is repeatable but accompanied by higher local reprojection residual. Treat `A004_D005_1210CH` mainly as a self-calibration-instability case.
4. For a corrective experiment, seed exact current intrinsics as priors, fix principal point and distortion, optimize poses first, then selectively loosen focal length only for the flagged cameras. Compare held-out epipolar residuals and rendered eval crops before accepting any change.
5. Do not use the latest incremental mapper's six-camera fragment as calibration evidence. The maintained global/GLOMAP path is materially more robust for this capture.

Audit directory: `/home/brans/lookcloser_temp/dec5_000899_glomap_colmap411_audit`. Key logs are [global_mapping.log](/home/brans/lookcloser_temp/dec5_000899_glomap_colmap411_audit/global_mapping.log), [view_graph_calibration.log](/home/brans/lookcloser_temp/dec5_000899_glomap_colmap411_audit/view_graph_calibration.log), [global_view_graph_calibrated.log](/home/brans/lookcloser_temp/dec5_000899_glomap_colmap411_audit/global_view_graph_calibrated.log), and [incremental_mapping.log](/home/brans/lookcloser_temp/dec5_000899_glomap_colmap411_audit/incremental_mapping.log).
