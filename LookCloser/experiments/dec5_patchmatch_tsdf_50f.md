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

All 50 requested frames (`000899..000997`, numeric order) were reconstructed, scored, reviewed,
and published. The strict audit reports `complete_with_failures`: 36 visual passes and 14 visual
fails. There are no pending or uncertain verdicts, duplicated CSV rows, non-finite metrics, or
full-frame metric fields. Every frame has 62 full-resolution `1080x1920` geometric depth maps, a
finite eval render, and one connected mesh component after filtering.

Lower LPIPS is better. The final face-only distributions are:

| Metric | Minimum | Q1 | Median | Q3 | Maximum |
|---|---:|---:|---:|---:|---:|
| PSNR (dB) | 21.3884 | 22.4075 | 26.4721 | 28.7704 | 29.7733 |
| SSIM | 0.813163 | 0.821997 | 0.844242 | 0.889823 | 0.898313 |
| Alex-LPIPS | 0.050266 | 0.056852 | 0.072667 | 0.102346 | 0.114956 |

| Structural quantity | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Mean depth coverage | 0.377931 | 0.384390 | 0.395595 |
| Minimum per-camera depth coverage | 0.244989 | 0.251372 | 0.272461 |
| Mesh vertices | 78,785 | 80,537.5 | 83,627 |
| Mesh triangles | 152,344 | 155,611 | 161,781 |
| Mesh connected components | 1 | 1 | 1 |

The worst metric frames were `000979` for PSNR/SSIM (`21.3884 dB`, `0.813163`) and `000981`
for LPIPS (`0.114956`). The next-highest LPIPS frames were `000983` (`0.110781`), `000979`
(`0.108936`), `000985` (`0.108011`), and `000973` (`0.107042`). Metric regression flags begin
at `000951`; 24 frames are flagged, but a flag is diagnostic and does not replace visual review.

The 14 visual failures are the contiguous range `000971, 000973, 000975, 000977, 000979,
000981, 000983, 000985, 000987, 000989, 000991, 000993, 000995, 000997`. No frame was marked
with an ear artifact. All 14 failures were marked with a lipstick/hand artifact: the actual-ear
crops remain sharp and single-valued, but the moving hand/lipstick-tube region has a polygonal
hard-source seam and a black cut through the actor/neck silhouette. Background missing outside
the actor was ignored as specified. Because these are real visual failures, this campaign is not
fully successful despite the complete inventory.

The final checksum-bound contact sheets are:

| Frames | Face/ear/hair | Ear crop | Lips/hand | Actor overview |
|---|---|---|---|---|
| `000899..000917` | [face](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000899_000917_face_ear_hair_gt_pred.png) | [ear](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000899_000917_ear_native_gt_pred.png) | [lips](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000899_000917_lipstick_lips_hand_gt_pred.png) | [overview](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000899_000917_actor_overview_gt_pred.png) |
| `000919..000937` | [face](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000919_000937_face_ear_hair_gt_pred.png) | [ear](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000919_000937_ear_native_gt_pred.png) | [lips](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000919_000937_lipstick_lips_hand_gt_pred.png) | [overview](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000919_000937_actor_overview_gt_pred.png) |
| `000939..000957` | [face](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000939_000957_face_ear_hair_gt_pred.png) | [ear](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000939_000957_ear_native_gt_pred.png) | [lips](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000939_000957_lipstick_lips_hand_gt_pred.png) | [overview](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000939_000957_actor_overview_gt_pred.png) |
| `000959..000977` | [face](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000959_000977_face_ear_hair_gt_pred.png) | [ear](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000959_000977_ear_native_gt_pred.png) | [lips](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000959_000977_lipstick_lips_hand_gt_pred.png) | [overview](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000959_000977_actor_overview_gt_pred.png) |
| `000979..000997` | [face](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000979_000997_face_ear_hair_gt_pred.png) | [ear](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000979_000997_ear_native_gt_pred.png) | [lips](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000979_000997_lipstick_lips_hand_gt_pred.png) | [overview](/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/contact_sheets/000979_000997_actor_overview_gt_pred.png) |

Because the source images are stored rotated relative to the anatomical labels implied by the
fixed crop coordinates, the late-frame review also used full-resolution actual-ear
`[760,300,1040,540]` and actual-hand/tube `[400,390,800,780]` GT/prediction crops under
`.diagnostics/visual_spot_checks`. These supplements did not alter the mandated crops, metrics,
or verdict rules.

## Insights

The campaign is deliberately fail-closed: an immutable request binds ordered source transforms,
calibration, recipe, and script hashes; a frame is published atomically only after remote and
local checksum validation, face scoring, and visual review. Regression thresholds are not fixed
from the historical surface-mask score. They were derived from the three newly scored and
visually accepted initial frames using the larger of the requested signal floor and three robust
MAD scales, then compared with the last five accepted frames.

The apparent early LPIPS rise in the original ROI-v1 scores was not a numerical LPIPS failure or
a face-render collapse. A
same-prediction control on `000941` changed only the ROI and moved PSNR/SSIM/LPIPS from
`19.8437 / 0.807832 / 0.141004` to `27.4698 / 0.868682 / 0.061396`. The removed wedge accounted
for 84.56% of the old ROI squared error, whereas the corrected face ROI contained only 0.0154%
invalid prediction pixels. This establishes the ROI-v1 semantic bug, but it does not erase the
real later ROI-v2 regression: the final series still reaches LPIPS `0.114956`, and those 24
metric flags remain recorded. The full causal report is [`../lpips_temp.md`](../lpips_temp.md).

The repeated hand/tube failure is a geometry-support problem rather than stale state or a texture
averaging problem. A clean retry of `000971` was byte-identical. Diagnostic canaries on `000973`
showed that the thin moving hand/tube surface does not survive the required two-view geometric
consistency into the TSDF mesh. Global trials with extraction weight 1, disabled component
filtering, photometric-only depth, one-view consistency, and smaller SDF truncation either failed
to recover the surface or introduced worse fragments/components; no safe uniform fix was found.
The evidence is retained under
`/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/.diagnostics/lipstick_missing_geometry_systemic_20260904`.
The frozen recipe was therefore retained for every frame and all recurrences were reported as
failures instead of receiving per-frame exceptions.

The final audit independently revalidates the 50-frame ordered inventory, CSV/result equality,
immutable source-transform/calibration hashes, all 3,150 selected source-EXR hashes, the 63-camera
JPEG gain receipts and exact 62/1 fixed-calibration split, retained-file hashes, render-revision
provenance, 62-map depth inventory and shape, finite face-only metrics, held-out GT EXR hashes,
absence of eval RGB from the 16 texture sources, all 20 final contact-sheet hashes, and the absence
of full-frame metric keys. The persistent 3D artifact is the extracted TSDF mesh plus manifests;
it is not a serialized raw TSDF volume.
