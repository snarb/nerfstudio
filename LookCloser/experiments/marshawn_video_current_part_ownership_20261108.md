# Marshawn: current-image anatomy and actor ownership

## What was tested

The all-camera contour admission check rejected source frame 1705 before
training. We traced the failing rays back to the original native photographs,
the current neural segmentation components, and the immutable pre-semantic
masks. An independent CPU audit also checked ray/box intersection and every
admitted cell along the failing rays.

## Results

Only GC/black failed: 48 base-grid, 44 padded-grid, and four mapped-grid
contour misses out of 152922 tested contours. No failing mapped ray was outside
the fixed AABB. The four rays crossed cells with independent clear-camera
evidence; changing the box or adding occupancy would conceal a source error.

The current `hand` and `hands` predictions agree on a seated crew member's
hand. Their agreement identifies anatomy, not one of the two target actors.
Prior complete-actor optical-flow support incorrectly admitted this newly
disoccluded hand. The same false component first appears at 1695 and persists
at 1700 and 1705. Current-photograph review of the preceding GC frames found
no corresponding false hand through 1690; apparent overlap at 1690 belongs
to foreground actor hair.

| Source frame | Wrong current-NN hand pixels retracted in GC |
| --- | ---: |
| 1695 | 4334 |
| 1700 | 4380 |
| 1705 | 4344 |

Original published sources and model artifacts remain preserved. The old
1695 model is withheld from delivery despite completing 140000 iterations;
the old 1700 model was stopped gracefully and retained its checkpoints.
Both require independent cold reruns after corrected-source qualification.
No 1705 model had started. Seven preceding completed models remain eligible
subject to final sequence review.

Corrections, native before/after contacts, and fresh source QA are under
`/fsx/tmp/marshawn_video_source_gcrole_g1_20261108/operations/`.
Root personally inspected `GC_corrected_native_visual/ALL3_GC_native_correction.jpg`;
the exact component removals preserve visible actor and braid boundaries.
`ROOT_finite_GC_current_photo_visual_review.json` binds this observation to
the correction and contact hashes. It does not waive fresh grid checks.

Cold replacements use the separate model campaign
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_gcrole_g1_20261108`.
The versioned fleet uses three physical hosts and ten GPU slots; old fleet
CPU processes remain held because their in-memory assignments are stale.
The new dispatcher must bind each frame's source and model generation and
require current source, sparse-grid, and all-camera dense-grid receipts.

Independent native-photograph checks confirmed that detached MD1695 boots,
FB1705 top-border hands, and GD1705 boot pixels are actual actor parts.
Consequently, rejecting every component lacking current body-mask overlap
would remove true detail. Further ambiguous components and a general role
qualification rule are still under investigation.

## Insights

Anatomy confidence and optical flow cannot independently establish actor
identity after disocclusion. Preserve current-image evidence, test both
confirmed crew negatives and detached actor positives, and qualify any
replacement ownership rule before using it on later sources. Recompute
affected guidance from corrected masks; do not relabel stale inference as
a new run or reuse learned model state after changing its inputs.
