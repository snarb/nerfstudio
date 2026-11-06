# Marshawn 1660: mat exclusion and depth-aware AABB review

What was tested: move only the minimum X/Y faces of the accepted `common002`
box, then inspect actual density-clipped renders from the unchanged R1 20k
checkpoint. Training remains stopped. The box stays axis-aligned in the existing
prepared world; field normalization and cameras are unchanged.

| Variant | Minimum XYZ | Maximum XYZ | Result |
|---|---|---|---|
| Previous | `[-.20,.02,-.20]` | `[.15,.33,.26]` | Raised mat edge visibly retained |
| X only, aggressive | `[-.14,.02,-.20]` | same | Rejected: clips black trouser contour in FA/FC/FE |
| Selected | `[-.15,.07,-.20]` | same | Raised mat edge removed in reviewed renders; actors/visible boots retained |

Results: root manually inspected the final HC/DC/KA black-background renders,
source comparisons and depth overlays. The renderer also checks unchanged model
and checkpoint hashes, native compositor parity and additivity of original
interior/exterior contributions. Four HC boot-center rays retain at least
`.999973` original opacity inside. The three inspected mat surface density peaks
are outside; some diffuse learned density remains at the cut, especially beside
the black boot. This is not complete semantic deletion of every mat-associated
density sample, nor a certified occupancy grid.

All 66 cameras were checked using 87,645,065 selected native mask rays. All
231,960 accepted hair-addition rays still intersect the box; the 12 independent
boot query rays also intersect. There are 34,049 nominal-mask misses in
MA/MB/MC/NA/NB/ND. These views' missed regions were actually inspected: the masks
include part of the loose hat beside the front grey boot. They are preserved and
reported as mask errors, not silently counted as successful coverage or erased
from the source archives. Ray intersection is still not a surface-depth guarantee.

The selected bounds are a manually reviewed face adjustment. The old automatic
`.02` minimum margin against the padded visual hull does not apply to the two
overridden faces; the upper and lower-Z bounds remain unchanged. Retained sparse
actor witnesses fit inside, but sparse witnesses do not certify full contours or
future animation frames.

The reusable tool is documented in [AABB review](../docs/aabb_review.md).
It exports both real 3D clipping on black and dashed edges tested against the
original full scene's first opacity-0.5 surface. Edges beneath the floor or behind
actors are hidden by default. The depth-aware overlay is displayed on a model
render so its depth and color refer to the same reconstructed geometry.

Artifacts are under
`/fsx/tmp/marshawn_001660_hd/two_actor_isolation/automatic_aabb/mat_trim_v1/`:

- `final/{HC,DC,KA}/`: PNGs, raw partition buffers, density traces, source snapshots and receipts.
- `native66_final/coverage.json`: complete per-camera native/hair/boot coverage.
- `y070_mask_misses.jpg`: actually reviewed hat contamination in six source views.
- `failed_mask_pixels_xmin014.jpg`: rejected trouser clipping evidence.

The reviewed box is committed as
`configs/aabb_review/marshawn_001660_mat_trim.json`. The archived previews are in
`/fsx/tmp/marshawn_001660_hd/downloads/aabb_mat_trim_review_20261106.zip`.
