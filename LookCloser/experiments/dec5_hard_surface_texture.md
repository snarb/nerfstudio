# DEC5: sharp single-source mesh texture

## What was tested

User requested item 2 only; item 3 (motion-compensated temporal geometry,
mask-assisted body/lipstick repair or sculpting) is explicitly deferred until
the user's go-ahead. No new geometry, mask, SfM or PatchMatch job is run here.

Frozen inputs are the prior joint temporal calibration and full-block TSDF mesh
for `000973`. The EXR images, 62 train physical-camera inventory, fixed exposure,
camera RGB profiles, native UV registration and actual UV atlas are unchanged.
F/J/L RGB is excluded from source selection and baking and read only afterward
for independent review. The 155,052 mesh triangles are unchanged; this is an
extracted TSDF surface, not a serialized raw TSDF volume.

The opt-in `hard_surface_texture.py` helper selects one source label per triangle
on **geometric mesh adjacency**, including across atlas chart boundaries. Metric
alpha expansion encourages connected source regions and inexpensive color seams.
The unary quality term favors near-normal, sufficiently resolved observations;
the refined candidate also uses train-only 9×9 low-frequency color consensus to
reject inconsistent source appearance. Neither that consensus nor low-pass RGB
is written into the texture. Actual source samples remain full frequency, with
the frozen color/registration correction and ordinary single-image bilinear
sampling. One camera supplies each texel; there is no cross-camera RGB average.

Per-texel depth-footprint checks may choose a different valid camera if the
face's preferred source is occluded. All unsupported texels stay explicitly
unsupported. This fallback can still produce small boundaries; hard selection
does not guarantee a seamless texture or recover wrong geometry. Atlas gutters
alone are padded, as in the previous baker. Hair can retain source detail, but
the double source-to-atlas-to-render sampling still limits absolute sharpness.

### Reproducible workflow

From `LookCloser`, with the existing environment plus `PyMaxflow==1.3.2`:

```bash
../.venv/bin/python scripts/bake_joint_temporal_mesh.py bake \
  --calibration-root /mnt/data/lookcloser_dec5_5a3_joint_texture \
  --output /mnt/data/lookcloser_dec5_5a3_hard_texture_v2 \
  --frame 000973 --hard-source
../.venv/bin/python scripts/bake_joint_temporal_mesh.py review \
  --calibration-root /mnt/data/lookcloser_dec5_5a3_joint_texture \
  --output /mnt/data/lookcloser_dec5_5a3_hard_texture_v2 --frame 000973
../.venv/bin/python scripts/review_joint_temporal_texture.py score \
  --output /mnt/data/lookcloser_dec5_5a3_hard_texture_v2 --frame 000973 \
  --face-roi /mnt/data/lookcloser_dec5_5a3_joint_texture/config/face_roi_000973.json
../.venv/bin/python scripts/audit_hard_surface_texture.py \
  --calibration-root /mnt/data/lookcloser_dec5_5a3_joint_texture \
  --output /mnt/data/lookcloser_dec5_5a3_hard_texture_v2 --frame 000973
```

The hard-source mode requires a distinct output root and pins scripts, geometry,
calibration and cache receipt hashes. Use a new root after code/config changes.
The atomic completion receipt is written after all textures, GLB and OBJ archive
exist and the embedded GLB geometry/UVs survive a round trip. Completion means
artifact validity, not visual approval. The post-hoc audit validates source
labels, unchanged geometry/atlas/calibration, independent GT hashes, identical
manual face ROI and finite face-only PSNR/SSIM/LPIPS. It builds native old/new/GT
comparison crops and records the separate manual `visual_review.json` verdict.

## Results

The first angle-only source-selection candidate is retained at
`/mnt/data/lookcloser_dec5_5a3_hard_texture`, with source snapshots and its explicit
failed visual verdict. It restores visible hair strands compared with the
multi-camera atlas but creates conspicuous bright patch boundaries on the face.
This motivates a general train-only color-consistency source cost, not a
per-frame pixel exception. The geometry defect beside the lipstick is unchanged.

Face-interior manual GT ROI, identical display exposure and polygon across all
variants; ears/hair/hand are reviewed separately, not included in this metric ROI.
These values are not comparable to the original per-image-exposure campaign.

| Variant, 000973 | Face PSNR ↑ | Face SSIM ↑ | Face LPIPS ↓ |
|---|---:|---:|---:|
| Previous joint, multi-camera averaging | 27.05335 | 0.90087 | 0.11761 |
| First hard source, angle-only | 24.22557 | 0.86017 | 0.14464 |
| Hard source + train-color consistency (delivered) | 26.33602 | 0.87690 | 0.11374 |

The delivered candidate improves face LPIPS by about 3.3% over the averaged
joint atlas, but PSNR declines by 0.72 dB and SSIM by 0.024. This is not an
across-the-board improvement. Native ear/hair inspection and all three held-out
overviews show less clumped, more detailed hair. The first hard candidate's
prominent face highlight boundaries are substantially reduced by train-only
source-color consistency. Residual patch boundaries, wrong hand/tube/neck
surface, under-chin cuts and hair-silhouette fringe remain: strict visual verdict
is **fail**, delivery is **experimental texture complete with known defects**.
No default is promoted, and no claim about generalization across video times is
made from this single delivered frame.

- [Self-contained GLB, about 24 MB](/mnt/data/lookcloser_dec5_5a3_hard_texture_v2/frames/000973/dec5_000973_hard_source.glb)
- [OBJ with texture archive](/mnt/data/lookcloser_dec5_5a3_hard_texture_v2/frames/000973/dec5_000973_hard_source_obj.zip)
- [Native ear/hair: GT / averaged / hard](/mnt/data/lookcloser_dec5_5a3_hard_texture_v2/frames/000973/comparisons/ear_hair_native.png)
- [Native lipstick/hand comparison](/mnt/data/lookcloser_dec5_5a3_hard_texture_v2/frames/000973/comparisons/lipstick_hand_native.png)
- [Native face comparison](/mnt/data/lookcloser_dec5_5a3_hard_texture_v2/frames/000973/comparisons/face_native.png)
- [Three-way F overview](/mnt/data/lookcloser_dec5_5a3_hard_texture_v2/frames/000973/comparisons/F004_B005_1210O9_overview.png)
- [Visual verdict](/mnt/data/lookcloser_dec5_5a3_hard_texture_v2/frames/000973/visual_review.json)
- [Independent artifact audit](/mnt/data/lookcloser_dec5_5a3_hard_texture_v2/frames/000973/audit.json)

The embedded GLB texture is 4935×4937. Round-trip vertex error is zero; exact
original mesh/atlas arrays are independently compared with the previous bake.
The export only recenters and rotates Z-up to glTF Y-up and duplicates UV seam
vertices; it does not modify the surface. The GLB uses an unlit material, so a
viewer should not add a second lighting model to this baked appearance.

Validation: **53 tests passed**, covering exact single-source gathering,
visibility-only fallback, mesh adjacency, color outlier rejection, exhaustive
two-label graph-optimum checks, immutable request guards and prior calibration/
campaign regressions. Numerical checks and native visual checks are separate.

## Insights

Single-source baking avoids one cause of smeared hair but exposes remaining
view-dependent source color, highlights and bad surface visibility. A fixed
per-camera profile is not sufficient to eliminate all such differences.
More coherent source choice cannot repair the lipstick/neck surface, and no
geometric improvement is claimed. The old renderer/model defaults and the
150-frame video are untouched. Temporal reconstruction awaits the user's signal.
