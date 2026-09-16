# DEC5 000995: local subfaces under the frozen train-gap constraints

## What was tested

The [coarse gap constraint with foreground veto](dec5_train_gap_carving.md)
removed much of the false barrel-side wedge but retained a small blue fringe.
The hypothesis was that a coarse triangle is protected as a whole when one
vertex touches the finger/tube safety band, although part of its interior lies
in the reviewed gap.

`study_train_gap_subfaces.py` makes two conforming midpoint subdivision rounds
on 203 original faces having at least one sample with three negative views.
Adjacent edges are split consistently; no original vertex moves. Plane,
orientation, barycentric containment and total area per parent are verified.
The refined mesh has 147858 triangles and 76351 vertices.

The four train masks, context polygons, .003-expanded measured depth slabs,
three-same-view rejection rule and any-view positive veto remain unchanged.
No rendered defect ROI selects subdivision or deletion. Geometry starts from
the same quorum parent as the coarse experiment; it is not a refinement of the
already pruned 79-face subset. Both a subdivision-only mesh and a subdivided,
carved mesh are rendered to distinguish texture relabeling from geometry changes.

Root: `/mnt/data/dec5_train_gap_subfaces`.
Six matched 1080x1920 diagnostic renders cover a moving-path view and the H/C,
K/B train cameras. This is not a new 6K output or temporal sequence.

## Results

**Further local improvement, but not artifact-free.** The barrel/finger gap is
cleaner and the pink tip remains intact. The residual narrow blue fringe is
reduced but not completely gone. A small black cluster appears beside the hand
in the moving view. Existing crown/fringe and chin/neck problems remain.

The rule removes 1537 subfaces affecting 125 original parents. Removed surface
area is `8.5385e-6` normalized square units, versus `7.0140e-6` in the coarse
control. New midpoint evidence protects some previously deleted area
(`6.6544e-8`); hence the fine result is not strictly a subset of the coarse
result. No geometric surface is added outside the original parent mesh.

| View | Subdivision alone: RGB changes / new black | Fine carving vs refined: RGB changes / new black | Fine result vs coarse: RGB changes / new black |
|---|---:|---:|---:|
| Moving | 12 / 0 | 367 / 5 | 155 / 6 |
| H/C | 429 / 0 | 723 / 453 | 596 / 99 |
| K/B | 207 / 0 | 992 / 871 | 394 / 155 |

Subdivision alone loses/adds zero depth hits in all three views; maximum
common-hit depth differences are at most `4.77e-7`, consistent with float32
intersection arithmetic. Nevertheless, texture labels change in some pixels.
The separate control prevents attributing every RGB difference to carving.
Fine carving adds no hits versus the refined mesh. Compared with the coarse
carved mesh it restores 3 H/C and 10 K/B hits because of the extra positive
sample evidence described above.

Newly black pixels relative to coarse do not enter the reviewed hand/tube masks
in H/C or K/B (zero each). Those same masks constrain geometry, so this is
**not independent evaluation**. Pixel counts are not image-quality metrics or
counts of missing anatomy; no PSNR/SSIM/LPIPS is computed here.

[K/B comparison](/mnt/data/dec5_train_gap_subfaces/000995/review/K004_B005_1210DS/lipstick_native.png)
contains actual GT, quorum parent, coarse carving, subdivision-only control,
and refined carving. [Moving comparison](/mnt/data/dec5_train_gap_subfaces/000995/review/moving/lipstick_native.png).

### Remaining surface and new black pixels

The previously established 171-pixel post-hoc blue core has 61 surviving hits
after coarse carving and **23** after refined carving. Actual ray-hit points
are recreated from float32 mesh barycentrics and checked against the masks.
Eleven satisfy the pointwise deletion rule but belong to retained faces; twelve
have positive-mask protection. Eight have fewer than three negative views
(these categories are not all disjoint). The remaining hits have unchanged
depth to within `1.2e-7` relative to coarse: they are not new deeper layers.
This core is diagnostic only, not an input to subdivision or rendering.

The five new black moving-view pixels versus subdivision-only still have mesh
intersections. Four hit newly exposed surfaces about `.0060..0066` normalized
depth units behind the previous ones. One has unchanged depth. Reusing the
old pixel color for all five would therefore texture different surfaces as if
they were the same; the existing same-surface fallback cannot justify that.
The source-visibility failure on those four newly exposed surfaces remains to
be diagnosed, rather than painted over.

The main agent viewed all three native lipstick comparisons, three head panels
as overviews, and three native new-black sheets covering every one of the
32 components relative to coarse. The independent audit replays all mask
projection footprints and verifies exact refined/carved mesh relationships.
Its arithmetic validation does not establish physical truth of the masks.
The [visual-review receipt](/mnt/data/dec5_train_gap_subfaces/000995/visual_review.json)
records the limited scope and does not promote either mesh to video production.

Finalization verifies 182 artifact bindings. All 37 focused tests pass, including
two-round surface preservation, evidence-driven subdivision selection and the
existing negative/positive mask, depth and texture guards. Both render batches
and audits are terminal; compact logs and test output are retained under the
study root. Passing tests do not override the residual visual defects.

## Insights

Finer geometry sampling helps **after** correcting the evidence model; earlier
subdivision under near-depth voting could not remove correlated false depth.
It also reveals why increasing resolution alone is insufficient: part of the
remaining region is deliberately protected by silhouette uncertainty, while
newly revealed surfaces may lack an admitted texture source.

Do not repeatedly subdivide the entire actor or relax the foreground veto to
chase a few pixels. The next useful check is source visibility on the newly
revealed hand-side surface and then transfer of the silhouette workflow to
other actor times. The separate cheek-hole reconstruction objective and the
artifact-free dynamic video are still incomplete; the existing 6K delivery
has not been replaced.
