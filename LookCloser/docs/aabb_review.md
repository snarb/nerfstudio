# AABB visual review

`scripts/render_aabb_review.py` loads an explicit immutable checkpoint and exports
both a depth-aware box overlay and the actual reconstructed volume inside the box
on black. Source images are comparison panels only; masks and source RGB never
modify the rendered object. No optimizer or training loop is created.

```bash
python scripts/render_aabb_review.py \
  --config /path/to/config.yml \
  --checkpoint /path/to/step-000020000.ckpt \
  --checkpoint-sha256 CHECKPOINT_SHA256 \
  --box /path/to/box.json \
  --camera HC \
  --output /path/to/new_review_directory
```

The box file is `{"aabb": [[xmin,ymin,zmin],[xmax,ymax,zmax]]}` in the existing
prepared scene coordinates. The tool does **not** renormalize the field or move
cameras to the new box. Camera can be an eval index, filename stem or unique
underscore-delimited suffix. Supply repeated `--python-path /path/to/code` in
priority order for historical custom models/parsers. There are no dataset paths,
camera names, crops or foot coordinates embedded in the tool.

Optional controls:

- `--crop X0 Y0 X1 Y1`: a comparison crop in native array coordinates. The actual
  full-raster renders are always retained; otherwise the comparison frames the box.
- `--queries queries.json`: labels mapped to `[x,y]` pixels. Saves actual density,
  color, weight, XYZ and AABB membership for every sample on those rays.
- `--display-transform identity` for an encoded PNG-trained field; choose
  `linear-reinhard-srgb` explicitly for linear HDR fields. Raw float buffers always
  retain the original encoding and values; no implicit PQ or exposure conversion.
- `--depth-tolerance`: world-distance allowance for line visibility, default one
  native march step. `--hidden-edges ghost` explicitly enables an x-ray diagnostic;
  default `omit` hides the edges behind scene surfaces, including under the floor.

Outputs use the selected camera name as prefix:

| File | Meaning |
|---|---|
| `*_box_visible.png` | Red dashed edges depth-tested against the full scene |
| `*_inside.png` | Only density inside the actual AABB, integrated over black |
| `*_inside_box_visible.png` | Inside render plus edges using original scene visibility |
| `*_full.png` | Unmodified full-scene reconstruction |
| `*_outside.png` | Exterior density reintegrated, including newly revealed surfaces |
| `*_outside_contribution.png` | Exterior contribution with original scene transmittance |
| `comparison.png` | Source/full/inside/exterior-contribution comparison |
| `raw_partition.npz` | Float RGB/opacity for all partitions and unit-ray surface depth |
| `receipt.json`, `code/` | Bounds, runtime camera, hashes, checks, limits and source snapshot |

The full scene uses exactly the native samples and field evaluations. Interior
and exterior density are partitioned by sample-center XYZ, then independently
integrated. Their **original-transmittance contributions** sum to the full raw
RGB and opacity. Independently reintegrated interior/exterior images need not sum
because their occlusion differs. The receipt checks native compositor parity,
partition arithmetic and unchanged model/grid/checkpoint hashes.

Depth-tested lines use the first absolute accumulated-opacity 0.5 crossing,
interpolated within its density interval. Projected edge depth is perspective
correct and measured along the unit ray. Each line pixel is tested separately;
segments are never connected across an occluder. Rasters must be undistorted
pinhole images and share the camera's exact dimensions. These are rendered-depth
estimates; incorrect reconstructed geometry can still give incorrect occlusion.

The field adapter currently supports Nerfstudio's packed `VolumetricSampler`
(bounded Instant-NGP), with black or random/eval-black backgrounds. Other samplers
fail explicitly. The independent NumPy/PIL functions in `aabb_visualization.py`
can be used with any model that supplies a calibrated pinhole camera and a
unit-ray surface-depth map. Changing a volume's AABB does not produce an ideal
semantic mask: floor, props and uncertain density inside it remain visible.

Example bounds for the reviewed frame are in
`configs/aabb_review/marshawn_001660_mat_trim.json`. This contains only scene
geometry; it does not embed a host, checkpoint or training command.

Validation: `python -m pytest -q tests/test_aabb_visualization.py` checks perspective
depth, opaque/unknown visibility, occluder gaps, near-plane clipping, camera/raster
validation and packed opacity crossings. Actual GPU runs additionally exercise
the native compositor and checkpoint immutability checks.
