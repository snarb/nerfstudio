# Separate native-derived4K EVAL visual diagnostics

## What was inspected

Read-only feasibility for DC/HC/KA at frame1660. No images exported, no new
masks, no fits, no models rendered, and no TRAIN/EVAL descriptors changed.
Source inspection is recorded under
`/fsx/tmp/marshawn_001660_hd/tuning_20261106/resolution_4k_eval_diagnostic_feasibility/source_inspection.json`.
All three native TIFFs exist at
`/fsx/oregon/gta_marshawn_2_1/cedar/tiff16/by_local/{camera}/{camera}.000408.tif`,
8192×4320 RGBuint16,212,336,804 bytes each. Local TIFF frame408 corresponds to
this frozen scene's frame1660 lineage; retain that exact mapping.

The RGB decoder `prepare_marshawn_color_variants.linear_hd` accepts any supplied
calibrated camera row and target dimensions, including these three cameras.
However, `build_marshawn_4k_pilots.corrected` looks up only TRAIN `camera_rows`;
using it unchanged on DC/HC/KA would fail. The existing TRAIN builder also
intentionally rejects EVAL rows and regenerates semantics, so do not broaden it.
Create a separate minimal RGB/valid diagnostic exporter, reusing the frozen
RGB/calibration primitives and an explicit frozen EVAL gain lookup.

## Exact photometry and calibration

The frozen k4 fit states `uses_eval_RGB=false`. The EVAL corrections already
exist in `fit['eval_predictions']`, derived from TRAIN/pose neighbours:

| Camera | Frozen EVAL scalar EV |
|---|---:|
| DC | -0.13310256981327664 |
| HC | 0.04627162102184486 |
| KA | 0.10237014191253105 |

Use these existing scalar values once. Do not fit to the new high-resolution
EVAL RGB or substitute the nearest single TRAIN camera's correction.
The exact path remains native Log3G10/RWG decode → positive Rec709linear →
AREA prefilter at target dimensions → one calibrated remap → float32 gain4 →
`2**(radial_ev(k4)-frozen_global_center_EV+frozen_eval_EV)` → float32 →
Reinhard/sRGB8. It is the same operation ordering as the reviewed TRAIN exporter,
with explicit EVAL gains rather than TRAIN lookup. Preserve the vendor decoder,
RAW, rig cameras/images, k4 fit, colour contract and helper hashes.

For4K use3840×2024 and double `fl_x, fl_y, cx, cy` from the existing FHD row.
Keep pose, floor-frame transform, AABB and distortion-free target convention
unchanged. Double COLMAP principal centres before the existing `cx-.5,cy-.5`
raster conversion; do not double the already-subtracted raster coordinates.
Generate geometric validity from the same raw-coordinate remap bounds
`0≤mx≤8191, 0≤my≤4319`. This is calibration coverage, not a new semantic mask.
Do not regenerate actor/hair/worker/veto masks or use diagnostics in training.

## Proposed isolated artifact and acceptance checks

New generation only:
`/fsx/tmp/marshawn_001660_hd/tuning_20261106/resolution_4k_eval_diagnostic_v1`.
Suggested contents: `images/{DC,HC,KA}.png`, `valid/{camera}.png`,
`render_views_4k.json`, `replay/{camera}.json`, `receipts/{camera}.json`,
`source_manifest.json` and an explicit `DIAGNOSTIC_ONLY.json`.
The descriptor contains three view rows and scaled intrinsics, no TRAIN split,
no READY, frequency maps, semantic scores or training eligibility. Later model
renders must be separately released and never overwrite primary V1 renders.

Before any4K export, replay each FHD RGB through the same decoder/photometry:
require byte-exact decoded uint8 RGB arrays and exact geometric-valid arrays
against the frozen FHD sources. Record all three original PNG file hashes as
well as replay pixel-array hashes; PNG container encoding is not the pixel
comparison. Hash the original9EVAL RGB/valid/semantic files before/after and
require them unchanged. The original poses, crops, box,63TRAIN+3FHD transforms
and ready descriptor must remain byte-identical. If replay fails, stop and
investigate the operation order, interpolation or EVAL gain source; no waiver.

Then export4K RGB/valid with per-file hashes, raw TIFF SHA and scaled camera
receipts. No semantic fitting or map fitting is needed. Preserve the original
FHD crops unchanged; optional visualization-only4K crop rectangles are2× copies
in this separate descriptor, explicitly not a new metric domain. Assert that
normalized rays at doubled COLMAP coordinates match the original camera rays.

## Resources and interpretation

CPU only, at most2 workers with OpenCV1 thread/worker. Native RGB decoding needs
approximately0.4GiB float32 per camera plus remap/photometry temporaries; reserve
up to16GiB total headroom and about1GiB scratch/output space. Comparable reviewed
TRAIN exports including extra semantic work took14–16 seconds per camera;
allow a few minutes with hashing and I/O rather than promising an exact runtime.

Primary V1 remains the exact original FHD RGB/crops/masks and its LPIPS/PSNR/SSIM.
Separate4K GT can reveal folds/hair/detail that FHD evaluation suppresses, but
it does not automatically supply a validated4K semantic metric protocol. Show
GT and later model renders at matched4K pixels/crops; label them supplementary
visual evidence and keep geometric-valid pixels explicit. No quality claim or
model render is authorized by this feasibility document.

## Authorized CPU export result

Root subsequently authorized the isolated exporter. The frozen generationv1
completed in16.08 seconds with2CPU workers. All three FHD uint8 RGB arrays replay
byte-exactly, all three geometric-valid arrays match, and all9 original EVAL
files plus original poses/crops/box/transforms/ready remain unchanged. The
separate TRAIN63 4K source metadata/transforms also match their existing frozen
completion hashes. Two CPU tests cover EVAL-only gain lookup and pixel-centre
scaling; the real export supplies the required three-camera replay gate.

Spec SHA `80e00ce2c4c8ecb07c3801fdf38a37e531c3f3279b2dd19b34b5a8bb9cc8f6e0`;
render descriptor SHA `f83aa03d9c148e05014907034d94d0f77f4e3d3282fa8169d9bf562a3cadda29`.
Original GT4K PNGs and geometric coverage are under `images/` and `valid/`.
No model was rendered and no new semantic masks or metrics were generated.

Visually inspected all three `review/{camera}_hair_detail.png` comparisons:
GT4K preserves finer dreadlock boundaries, shirt markings and calibration-target
edges than the enlarged FHD source; coloured fine noise is also more visible.
These are source-image observations, not model-quality claims. Primary V1
remains unchanged and the supplementary rendering protocol awaits its own review.
