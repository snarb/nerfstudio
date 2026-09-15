# Native 6K textures for the selected wide spiral

## What was tested

The exact `wide_spiral_free` presentation from
`/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_free`, with native RGB sampled
from the immutable 6144×3072 PQ16/AP1 originals on `dev3`. Camera matrices,
changing actor times, mesh geometry, graph labels, visibility decisions,
source IDs, fixed camera color profiles and exposure stay frozen. This is an
opt-in texture replay, with no model/default changes or new calibration solve.

The original ingest cropped `[341, 0, 5802, 3072]`, then performed two Lanczos
resizes to 2560×1440 and 1920×1080. This replay skips both resizes. It maps each
selected HD pixel-center coordinate into the original cropped pixel lattice:

```
u_crop = (u_HD + 0.5) * (5461 / 1920) - 0.5
v_crop = (v_HD + 0.5) * (3072 / 1080) - 0.5
u_raw = u_crop + 341; v_raw = v_crop
```

The two scale factors intentionally differ. Equivalently, the cropped native
intrinsics are `fx,cx *= 5461/1920` and `fy,cy *= 3072/1080` in the calibrated
pixel-boundary convention. No camera centers or rotations change.

For every output pixel, only its already selected train camera contributes RGB.
The remote worker reads that camera's original PNG and decodes its four native
bilinear taps using the frozen ST2084 inverse, floor treatment, gain
`356.95123731403817`, AP1→Rec709 matrix and chromatic adaptation. Sampling occurs
in scene-linear RGB, before the existing frozen camera gains/display exposure.
There is no HD RGB intermediate, invented detail, sharpening or per-frame gain.
Raw source SHA-256 hashes and dimensions are retained for each temporal frame.
Only coordinate/sample arrays are transferred; each successful scratch workspace
is removed after local hash verification. Original images remain read-only.

The 150-frame presentation keeps 118 pure 3D images, eight explicit dissolve
images and 24 actual changing train images from `H004_C005_1210SZ`. The ending
uses the same 1.9× lens and principal point, now sampling approximately
2874×1618 native source pixels. Delivery remains portrait 1080×1920, 24 fps,
6.25 seconds.

## Results

Initial canaries passed before the full sequence was launched:

| Time | Role | HD replay mean absolute code difference | Maximum code difference | Native replay time |
| --- | --- | ---: | ---: | ---: |
| 001083 | Intermediate 3D hair/face close-up | 0.00000177 | 1 | 62.4 s |
| 001123 | Late 3D close-up | 0.00000129 | 1 | 57.6 s |
| 001197 | Actual dynamic train ending | Not applicable | Not applicable | 10.2 s |

The HD controls independently project the identical raycast surface points and
resample the saved source IDs. Their tiny numerical differences affect 11 and
eight pixels respectively. Target depth, face labels and source-ID files are
byte-identical to the baseline for both 3D canaries. Native A/B crops show finer
hair, eyebrow and skin texture without visible framing or global color drift.
Existing hair-contour triangles remain visible. No independent target RGB is
used, so this is not a PSNR/SSIM/LPIPS reconstruction-quality claim.

Native-size comparisons:

- [001083 hair](/mnt/data/dec5_cinematic_wide_spiral_6k_v2/review/001083_hair_AB.png)
- [001083 face](/mnt/data/dec5_cinematic_wide_spiral_6k_v2/review/001083_face_AB.png)
- [001123 hair](/mnt/data/dec5_cinematic_wide_spiral_6k_v2/review/001123_hair_AB.png)
- [001197 actual train hair](/mnt/data/dec5_cinematic_wide_spiral_6k_v2/review/001197_hair_AB.png)
- [001197 actual train face](/mnt/data/dec5_cinematic_wide_spiral_6k_v2/review/001197_face_AB.png)

Full-sequence output root: `/mnt/data/dec5_cinematic_wide_spiral_6k_v2`.
All 150 images are complete and distinct. All 126 3D/dissolve images passed
independent target-depth equality and retain byte-identical source-ID/face-label
files. The replay read 7,791 original PNGs (674,140,395,885 bytes) on `dev3`;
median times were 55.06 s for pure 3D and 10.05 s for pure train images. Minute
supervision is recorded in `supervision.json`; no v2 OOM or worker failure occurred.

Both encoded movies decode as 150 frames, 1080×1920, 24 fps, 6.25 seconds. The
150 archived PNG hashes match the verified frame receipts. All 15 chronological
overview sheets and ten decoded movie samples were visually inspected, together
with native canaries and early/middle seven-frame hair strips. Motion, actor
animation and the requested background transition remain intact. Native hair
detail remains visibly finer after normal MP4 compression in the
[codec comparison](/mnt/data/dec5_cinematic_wide_spiral_6k_v2/review/001197_hair_codec.png).

| Deliverable | Bytes | SHA-256 |
| --- | ---: | --- |
| [Compatible MP4](/mnt/data/dec5_cinematic_wide_spiral_6k_v2/presentation/video.mp4), CRF16/yuv420p | 21,574,652 | `32563b01a493253e9d78b4cf2cfaab1378dc808b8b295cc63cd53c8ba81a8d76` |
| [Higher-quality master](/mnt/data/dec5_cinematic_wide_spiral_6k_v2/presentation/master_444.mp4), CRF10/yuv444p | 56,272,795 | `5f9110508495dd74bf53d8a2cca7910634fced677e4130ddd0f9e3e0f48de096` |
| [150 PNG frames](/mnt/data/dec5_cinematic_wide_spiral_6k_v2/presentation/frames.zip) | 327,115,279 | `c57f651a0cd69e474434dbe2887ab90f0a16e3fe51507df519bb5c17c7501b45` |

The normal MP4 is the default delivery; the 4:4:4 master may need a player with
that H.264 profile. Both have the same 1080×1920 resolution. `delivery.json`
binds the report, visual review, manifests, source metadata and script snapshot.

The initial v1 run stopped at 000905 because four threads shared a lazy NPZ
reader on remote Python 3.8. Local/remote UV archive SHA-256 hashes matched,
and all 60 arrays read successfully in sequence, isolating the failure to
concurrent ZIP access. v2 materializes the UV arrays before starting threads;
RGB sampling is unchanged. The failed v1 scratch and logs remain intact.

Frozen provenance:

| Input | SHA-256 |
| --- | --- |
| Authoritative HD video request | `a463a44537b9089b91a8b78106b2f1b58f5cd1c6e213c98014da8c07c053ca5d` |
| Native source `meta.json` | `676fe66283498eaac0bb59af03428970f348a2306cec064aa1113cddc274288b` |
| PQ/AP1 decoder | `29f4fe982d33b7259383cf44ece2ffea3ea6031681492d989f467063f53a535b` |
| Fixed camera profiles | `d516d874f645819bd2c3d212fed8f921fdc75cc915eb5c6a6fc50be4421ffd95` |
| Final original `H004_C005_1210SZ_001197.png` | `5c29ab808119399c4e5b1dc72ab49db0ef38d45ff28b074add992177033505fe` |

The display exposure gain is the unchanged `10.570320292648677`. Per-frame
`source_provenance.json` files contain every actually used original PNG path,
hash, byte count and dimensions. The final train original is 85,719,244 bytes.

## Insights

Native source sampling removes the old HD crop's enlargement softness. The
improvement remains bounded by optical focus, geometry, source seams and the
1080×1920 delivery raster. The reviewed early/middle strips show no new broad
brightness or shape instability. Fine temporal aliasing and native sensor grain
are not ruled out; no extra blur was applied. Existing tan hair fringes, jagged
hair/neck/chin boundaries and small source seams remain. This is a completed
texture-quality replay with disclosed residuals, not an artifact-free geometry
result or a promoted reconstruction change.

Run in the local environment with bounded CPU threading:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python \
  scripts/render_cinematic_6k_texture.py render
../.venv/bin/python scripts/review_cinematic_6k_texture.py review
../.venv/bin/python scripts/review_cinematic_6k_texture.py package
```

The renderer resumes only matching completed frames and preserves failed scratch
workspaces. The packaging command verifies all frame receipts and camera records,
encodes the compatible H.264 CRF16/yuv420p movie plus a CRF10/yuv444p master,
and writes the lossless PNG archive. Visual acceptance is separate from file
integrity and must be recorded after review.
