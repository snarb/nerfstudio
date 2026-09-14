# DEC5 residual jaw holes: topology and camera-independent closure

## What was tested

Two confirmed original-mesh misses in the previous elevated movie, 001193 and
001195, were traced to their exact mesh boundary. The published phase+30 movie
and every source/production mesh remain unchanged. This is a diagnostic geometry
pilot, not an accepted replacement or a new movie.

`diagnose_residual_jaw_topology.py` binds the prior spot audit and mesh/render
hashes, projects actual boundary edges, and compares connected boundary components
with simple biconnected cycles (`boundary_cycle_blocks.py`). The original
loop-filling routine rejects branchy connected components.

`study_jaw_boundary_notches.py` then tests a **camera-independent** local 3D prior:
short arcs of the existing boundary may be closed by an isolated planar
triangulation. The frozen two-time rule limits extent to .003, endpoint gap to
.0015, plane RMSE to .0002, and at most 18 vertices; arc/chord length must be at
least 2.5. Units are normalized scene coordinates, not metres. Original vertices
and triangle prefixes stay unchanged. Proposal construction uses no target image,
camera, hand-drawn spot coordinates, learned image or held-out view.

`guard_jaw_boundary_notches.py` tests an intentionally conservative diagnostic:
every triangle vertex and centroid must lie within at least two train person
masks, with no available-mask contradiction. All 62 train cameras then veto
proposals that occlude an old mesh ray by more than .001 on either integer or
half-pixel grids. These raycasts are **not independent observed PatchMatch depth**;
the old surface is not certified correct. Final recasts test newly exposed layers.

Roots: `/mnt/data/dec5_residual_jaw_topology_v2`,
`/mnt/data/dec5_jaw_3d_boundary_notches`,
`/mnt/data/dec5_jaw_3d_boundary_guarded_attribution`, and
`/mnt/data/dec5_jaw_3d_train_evidence`.

## Results

The small image holes are **not isolated small 3D boundary loops**. Their nearest
edges belong to branchy components of 2,763 / 2,850 vertices. Biconnected analysis
still puts them on the large outer cycles (2,759 / 2,848 vertices), spanning
roughly .117 normalized units. Thus the proposed articulation-only explanation
was insufficient: extracting small cyclic blocks does not recover these holes.
Closing the whole outer loop would invent substantial surface.

![001195 boundary trace: RGB / actual boundary edges](/mnt/data/dec5_residual_jaw_topology_v2/001195_boundary_native.png)

| Time | Raw local proposals / triangles | Selected old-spot misses | After raw 3D closure | After conservative guard |
|---|---:|---:|---:|---:|
| 001193 | 80 / 465 | 45 | 4 | 45 |
| 001195 | 72 / 418 | 73 | 0 | 73 |

These are selected finite-depth diagnostic counts, not PSNR/SSIM/LPIPS or proof
of correct anatomy. At 001193 the prior selected black component included one
later-removed pixel; at 001195 one black pixel had geometry but no RGB. Therefore
the old-mesh depth counts are not interchangeable with the original 45/74 RGB
black-pixel counts.

The raw addition closes most/all of the two spots in the old problem view, but
also adds geometry elsewhere. In five diagnostic views it occludes old surfaces
by more than .001: 12 / 27 pixels in the old moving view and as many as 58 / 128
in individual nearby train views. It is not accepted on one-view appearance.

![001193 raw closure, additions red](/mnt/data/dec5_jaw_3d_boundary_notches/001193/spot_added_native.png)
![001195 raw closure, additions red](/mnt/data/dec5_jaw_3d_boundary_notches/001195/spot_added_native.png)

The conservative guard retains only 42 / 61 triangles and removes all useful
spot-closing proposals. A final two-grid recast still finds 43 / 17 summed old
surface occlusions across train cameras: removal of front proposals reveals
deeper proposal layers. Thus even this guard is **not a passing preservation
result**, and neither candidate is promoted or RGB-rendered as a new deliverable.

Exact admission attribution is more informative than the aggregate rejection.
All five spot-closing triangles at each time have 62 mask supports and **zero
mask vetoes**. Their vetoes come from old-mesh occlusion in lower D/E-row views,
including F/E, G/D–E, H/D–E, and I/D–E. A person-mask count is not a measured
depth-support count and does not establish shape accuracy.

Real train crops were rendered with the unchanged fixed exposure/profiles for
F/E, H/D and M/B at both times. All six native triplets were directly inspected.
The proposed pixels lie on shadowed neck skin close to the chin/neck boundary;
they are not simply projections onto obvious empty room background in these
crops. The cyan old-hit overlap in lower views is real as a raycast observation,
but the old surface can be farther neck geometry revealed by a missing foreground
surface. Treating every such overlap as a true free-space contradiction would
beg the reconstruction question.

![001195 real train / proposal footprint / old-hit overlap](/mnt/data/dec5_jaw_3d_train_evidence/001195.png)

Six focused tests pass: boundary articulation and interior-edge handling, closed
surface rejection, synthetic-notch completion with exact original preservation,
size/head bounds, and independent integer-ray pixel-center convention. The first
evidence-manifest write rejected NumPy integer crop coordinates; serialization
was fixed and rerun before auditing. No long job remains active.

## Insights

This establishes why ordinary hole filling misses the specific two defects and
provides a view-independent local proposal that actually covers them. It does
**not** establish that the proposal is anatomically correct or that the goal is
complete. The blanket old-mesh occlusion veto is too weak a model of confidence:
it may forbid correct occlusion of a farther surface, while one-pass pruning
also fails to guarantee its own conservative preservation condition.

The next decisive test needs retained, calibrated, measured stereo depths at
001193/001195. Classify old and proposed surfaces by independent depth agreement,
return reprojection and viewing separation; distinguish an observed free-space
contradiction from an old ray hitting a farther surface through the defect.
Do not simply loosen the old-mesh tolerance or equate 62 silhouette supports
with 62 depth observations. Only then compare matched production-wrapper RGB,
held-out fixed-region metrics and multiple moving/train views before any temporal
promotion. The already published video stays unchanged meanwhile.
