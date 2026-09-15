# Removing a proposal heuristic that leaves narrow jaw cracks

## What was tested

Follow the [view-consistent texture replay](dec5_head_source_quality.md) with
actual mesh repair, not camera avoidance or painting over black pixels.
In movie frame 001193, the isolated under-jaw spot has seven black pixels;
six lack mesh depth. Previous observed-neighborhood Poisson completion reduces
those six misses to two, but the local raw proposal mesh itself misses the
remaining two rays.

Tracing into the full Poisson surface identifies triangles 85055 and 85054.
Both satisfy the maximum distance, boundary proximity, edge length and normal
tests. Their centroid distances to the original mesh are approximately
8.68e-6 and 2.13e-6 normalized units. The proposal's **minimum** centroid-gap
threshold of 2e-5 wrongly labels these narrow hole-crossing triangles redundant.
A triangle can straddle a crack even when its centroid is near existing skin.

`study_close_boundary_completion.py` removes only this minimum-gap heuristic.
Maximum surface/boundary distances, head-only scope, maximum triangle edge,
normal agreement, semantic admission, measured-depth sample support, verified
neighborhood certificates, and final native free-space checks are unchanged.
Original geometry remains exact; additions are inferred, not new observations.
No renderer/model defaults change. No held-out RGB or target pose constructs
or selects the geometry (moving views are diagnostic only).

Test actual times 001193 and 001195 with identical latest texture settings:
incidence power 2, fixed color profile/exposure, zero registration, hard
single-source RGB and corrected native footprint. For each time render both
the movie pose and real F/E train pose, with production/previous/new meshes.
The geometry-only controller does not reuse a different time's surface.

Repeated Poisson solves have different binary hashes. A second counterfactual
therefore applies the old 2e-5 gate to the **same new raw mesh, semantic samples
and neighborhood certificates**, then reruns native safety pruning. This rules
out attributing changes merely to Poisson nondeterminism.

## Results

| Fixed diagnostic region | Production | Previous completion | No minimum gap |
|---|---:|---:|---:|
| 001193 movie spot: missing depth | 6 | 2 | **0** |
| 001193 movie spot: black RGB | 7 | 3 | **1** |
| 001193 real F/E skin interior: missing depth / black RGB | 30 / 30 | 3 / 3 | **1 / 1** |
| 001193 real F/E edge-inclusive skin ROI | 45 / 45 | 3 / 3 | **1 / 1** |
| 001195 real F/E edge-inclusive skin ROI | 14 / 14 | 4 / 4 | 4 / 4 |

These are geometry/RGB-support counts in fixed diagnostic regions, **not
full-frame quality metrics**. The larger movie rectangle used for 001195 includes
real background; its thousands of zero pixels are not an anatomical hole count.
The new mesh gains one depth pixel there versus previous completion.

The identical-raw-surface counterfactual reproduces both old completion RGBs
byte-for-byte, despite different mesh serialization. Relative to that matched
control, removing the minimum gap gains 2/2 visible depth pixels in the
001193 movie/train views and 1/2 in 001195, with zero lost depth pixels in all
four views. It changes only 4/19 and 5/9 RGB pixels respectively. This is a
verified incremental geometry fix, not a large new whole-head improvement.

New completion retains 2007/1989 added triangles for 001193/001195; the matched
old-gap control retains 1602/1636. Independent audits recompute 4508/3591 seed
queries, 4101/3478 certificates, and 124 native ray checks per new mesh.
Both pass with original triangle coordinates preserved. The two matched controls
also converge with zero remaining qualified free-space violations. Surfaces are
still appended, not welded; watertightness is not certified.

Held-out 001193 GT-only face scores are exactly unchanged:
PSNR **30.858383**, SSIM **0.943257**, LPIPS **0.072251**. This is one-view
non-regression, not proof of improved full-face geometry. The original texture
scorer correctly rejected a changed-mesh comparison; the separate
`score_close_boundary_heldout.py` explicitly permits the intended mesh change
while requiring identical pose, sources and exposure. The failed initial scorer
log is retained; no metric was accepted from that failure.

Fifteen new RGB controls were rendered. The main agent inspected eight broad
head/detail panels against production/previous completion, four native detail
panels from the same-raw counterfactual, and the held-out comparison. Under-jaw
spot improvement is visible, especially in the lower train view. One black
texture-support pixel remains in the movie spot; rough hair/crown, thin jaw
edges and larger hand/lipstick failures are **not fixed** by this test.

- [Movie detail comparison](/mnt/data/dec5_close_boundary_completion/001193/review/moving_detail.png).
- [Real train GT and jaw comparison](/mnt/data/dec5_close_boundary_completion/001193/review/F004_E005_1210FP_detail.png).
- [Same-raw control](/mnt/data/dec5_close_boundary_completion/001193/matched_review/moving_detail.png).
- [Held-out face scores](/mnt/data/dec5_close_boundary_completion/heldout/metrics.json).
- [001193 geometry](/mnt/data/dec5_close_boundary_completion/001193/interpolated/001193/mesh.ply).
- [001195 geometry](/mnt/data/dec5_close_boundary_completion/001195/interpolated/001195/mesh.ply).

Six focused tests passed. Sources, previous experiments and published 150-frame
video are unchanged. No two-frame-only substitution was silently published as
a general temporal repair.

## Insights

The remaining tiny crack was not caused by insufficient PatchMatch support:
its filling triangles were excluded before support checks. Minimum distance
from old geometry is not a reliable proxy for redundancy near a boundary.
Removing that heuristic, while preserving independently measured vetoes,
closes the diagnosed mesh gap without changing image synthesis.

The isolated black pixel now has valid geometry and needs a separate texture
visibility diagnosis. Full temporal deployment must apply the same method to
the affected sequence and audit transitions. This local success does not finish
the requested artifact-free dynamic video or fix the much larger hand defects.
