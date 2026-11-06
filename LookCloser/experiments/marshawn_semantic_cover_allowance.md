# Semantic source support and traversal cover

## What was tested

Read-only audit of semantic R0 at step 60,752. The source grid already has one conservative 26-neighbour dilation. The native traversal mask admits every LC cell that touches allowed source space, including closed-boundary contact. The current density selector uses this same whole-cell mask. No training, checkpoint, AABB, or source mask changed.

## Results

| Quantity | Cells |
| --- | ---: |
| Source before padding | 78,056 |
| Source with pad1 | 105,049 |
| Whole-cell traversal admission | 131,911 |
| Actual learned active grid | 49,583 |
| Active cell centers outside source pad1 | 10,435 (21.05%) |

For these aligned 128³ grids, the whole-cell cover is exactly another 26-neighbour dilation of source pad1 (zero XOR cells). This is a consequence of conservative closed-boundary overlap, including coordinate tolerance, rather than evidence of an accidental coordinate transform. Admitted spatial volume grows by approximately 25.6%. Native R1 dilation is a separate operation on the learned core.

Artifact: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cover_allowance_60752.json`. Source SHA256 `e804d736dd766e8adf11df07e346a6317af87e0b106a94dfc524ab32f1690d71`; actual saved grid SHA256 `cae73793d5ce3020ab9ea1b64ca106e6a0a708d80f4e5e4450bac71a9eb22fe7`.

## Insights and next steps

This identifies additional allowed boundary space, **not** 10,435 physically empty cells. Some conservative allowance can protect imperfect masks, hair, and feet. It is not safe to erase these cells solely from this audit.

A separate experiment could keep conservative traversal admission unchanged while querying original source pad1 for point density, preserving coordinate tolerance and UNKNOWN regions. Before launch it needs source-point and whole-ray positive coverage checks, actual warmup/ARM/resume parity tests, and render/depth review. It must not silently replace the current support. Existing runs remain unchanged.

Reproduce with `scripts/audit_marshawn_admission_cover.py --source SOURCE.npz --archive SAVED_GRID.npz --output REPORT.json`. The audit recomputes and verifies the conservative cover against the actual archive; it does not reconstruct the learned grid from EMA.

### Saved TRAIN ray evidence at matched step15,188

`scripts/audit_marshawn_source_point_trace.py` measures original saved native-ray weights outside source pad1, without changing a model. All103 witness XYZ points and the corresponding points along their camera rays remain admitted. No saved sample within the ±0.02 witness-depth band lies outside this point support. Mean original foreground weight outside support is1.48% for old18 supervised rays,2.34% for new66,0.12% for old8 reserved, and0.052% for new9 reserved. Most removed weight on the supervised groups is in front of the witness band; the largest individual outside weight is20.44%.

Artifact: `independent_occupancy/semantic_point_support_weights_15188.json` under the campaign. These are weights from the original integration, **not** predicted metrics or opacity after removal: transmittance would change. The103 rays are sparse clothing evidence, not a hair/sole or whole-volume guarantee. The result supports preparing a controlled point-support test while retaining the existing conservative traversal cover.
