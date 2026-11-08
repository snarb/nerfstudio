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

Corrected1695 passed143535 dense contour rays in all three admission forms;
corrected1705 passed152874, both with zero misses. The latter removes the
48 false hand-boundary rays responsible for its old failure. Both frames
have now started genuine cold140000-step models; actual audits show step0,
zero occupancy/EMA/Adam, no resume and66 TRAIN cameras. Seven preceding
models remain complete. These are operational counts, not a quality claim.

Native review of NC1700's8390-pixel dark fragment could not establish actor
or crew ownership. Its unsupported foreground assertion was withdrawn while
preserving physical UNKNOWN and the existing confidence falloff:95 covered
HD pixels retain positive W64, with total mass4.78188. A separate198-pixel
HE1700 fragment also lacks proven target identity; its proposed UNKNOWN
treatment retains positive W64 on all37 covered HD pixels (mass7.600524).
That input revision is being rebuilt before any1700 model consumes it.
Original source markers, arrays and CPU admission controls are archived;
no1700 model or assignment exists for the retired revision. The three changed
cameras require63-camera invariance and all66 RGB equality checks.

An independent audit reproduced all2097152 cells and the fresh26-neighbor
pad1 array exactly, with zero mismatches. It found35860 admitted cells with
no W64-positive projection; actual source footprints explain their UNKNOWN
protection. This is conservative sampling allowance, not proof that these
cells contain objects. No justified safe carve was established. Evidence:
`operations/independent_grid_review_g1_1695_20261108/report.md` under the
corrected source campaign.

For repeatable manual inspection, `scripts/export_marshawn_admission_slices.py`
verifies the manifest-bound grid payload and exports every orthogonal slice.
Gray denotes base allowance, blue the extra pad1 allowance, and black cells
excluded from admission. The helper changes no source arrays and makes no
physical occupancy claim.

## Insights

Anatomy confidence and optical flow cannot independently establish actor
identity after disocclusion. Preserve current-image evidence, test both
confirmed crew negatives and detached actor positives, and qualify any
replacement ownership rule before using it on later sources. Recompute
affected guidance from corrected masks; do not relabel stale inference as
a new run or reuse learned model state after changing its inputs.
