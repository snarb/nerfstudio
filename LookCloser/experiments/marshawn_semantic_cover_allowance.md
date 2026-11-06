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
