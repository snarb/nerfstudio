# Marshawn export support feasibility, 2026-11-07

What was tested: one finite CPU study of whether a render-only subset of the
selected frame1670 R0 occupancy could remove unsupported floor/shell fragments.
Production R0, hybrid profile2bd9bca5…, model weights and source inputs stayed
unchanged. No GPU was acquired and no checkpoint was loaded.

The proposed cumulative-mass tail criterion is insufficient by itself. This
model uses `corrected_arm_allocator: false`: total occupied interval length/count
controls the per-ray sampling budget. Removing a far interval can change front
sample positions even if its original mass was tiny. A bound on dropped original
mass therefore does not establish a bound on the new rendered RGB.

The finite conservative alternative projected all eight corners of each original
occupied cell into all66 calibrated HD training cameras, using the existing
qualified outward pixel-footprint projection. Any intersection with literal
`W64 > 0`, including the protective positive band, made the cell mandatory at
any depth. Near-plane crossings were retained. The union received one cell of
padding, clipped to original R0. Fully behind-camera cells cannot intersect its
forward supervised rays. No mask erosion or alpha cutoff was used.

| Result | Cells |
|---|---:|
| Original selected R0 | 37,821 |
| Mandatory full source-ray projection union | 37,816 |
| Union plus one-cell halo, clipped to R0 | 37,817 |
| Potentially removable | 4 |
| Retained | 99.9894% |

The66-camera test took4.42s. The branch stopped at the requested95% retention
threshold; no GPU rerender, additional branch or production cleanup was made.
This conservative test does not establish that the retained floor fragments are
physical surfaces. It establishes that preserving all positive supervised-ray
occupancy intervals leaves essentially no pruning opportunity. The old packed
quantile cache is also not an absence test: its depths are intentionally NaN
below alpha0.95.

The checkpoint binding is
`ccd0da52ad069beaaf93c7ebd3e9b79343242b0d7d0f7e6ce63e9c332cc42015`.
The source-bound receipt, all66 semantic hashes and per-camera counts are at
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/background_support_feasibility_001670/receipt.json`.
The repeatable CPU helper is
[`marshawn_export_support_feasibility.py`](../scripts/marshawn_export_support_feasibility.py).
This is a footprint overestimate, with no measured rerender pixel-parity claim.

A separate read-only storage measurement corrected an earlier unit concern.
The completed1660 float package is92,773,290 bytes logical,98,646,016 bytes
allocated, with35 sealed files plus its receipt. At that measured compression,
all72 packages total6.68GB. The shape-derived uncompressed upper bound, with1%
metadata allowance, is79.13GB for all72; changing scene complexity can increase
compressed sizes. The first PNG pair plus transfer audit is6.20MB. Production
does not persist donor warp caches, and CPU concurrency is at most3. Present
free space was222.68GB; model/NN growth remains separately monitored. No cache
eviction, archive transfer or deletion was introduced. The measurement receipt
is `operations/export_capacity_20261107/receipt.json` under the same campaign.
