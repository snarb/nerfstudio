# DEC5 000995: measured free space and actual lipstick-fin RGB sources

## What was tested

The [full-block TSDF transfer](dec5_full_block_transfer.md) did not remove the
000995 lipstick fin. This follow-up asks whether the **current production**
surface is actually supported by the 62 measured train depths, and where its
blue color comes from. It is not a new claim that all TSDF artifacts share one
cause: similar missing-confidence failures were already found at 000973/000971.

`diagnose_lipstick_fin_depth.py` uses a manually inspected polygon in real train
camera K/B to identify 86 diagnostic triangles. It evaluates three vertices and
the centroid against all 62 freshly reconstructed, hash-verified native depth
maps. Missing depth is unknown. Agreement requires a real depth anchor, 0.001
normalized depth tolerance, 1.5-pixel roundtrip and at least one degree parallax.
A farther observation is trusted only after agreement with three other cameras.
Existing source masks are examined descriptively, not used as truth.

`trace_lipstick_fin_rgb_sources.py` then reads the **actual** `source_ids.png`
from the matched K/B production render, unprojects each selected target pixel,
and checks its chosen source against measured source-camera depth and train RGB.
The current RGB renderer is hard-source, fixed camera profiles/exposure,
incidence2 and zero registration warp. No held-out RGB is loaded.

Two separate opt-in pruning arms test a uniform whole-mesh rule, **without the
diagnostic polygon, target camera, RGB, masks or learned shape**. At all four
triangle samples require zero near measurements, six stable farther 5×5 native
footprints, and six farther observations corroborated by three other cameras.
Far gap exceeds both 0.005 normalized units and 1% depth; at least 20/25 taps
must be farther and their middle depth spread no greater than 0.5%. Near depth
tolerance is 0.0015. The sole ablated factor is near-footprint radius:

- Radius 2: any near tap in the native 5×5 footprint protects the whole face.
- Radius 0: only the actual nearest native pixel supplies that protection.

Original vertices never move. Retained triangles are an exact subset, with no
component cleanup, filling or smoothing. Both arms start from the same current
production mesh, including its existing repairs, not the raw fusion controls.
Three new matched renders per arm use the actual wide left-arc pose and H/C,
K/B train poses. No full video was regenerated or replaced.

## Results

**A false surface and incorrect selected texture layer are directly confirmed;
both pruning arms improve it only partially and are not promoted.**

66/86 diagnostic faces have zero near measurements at all four samples;
80/86 have at least six independently corroborated farther observations at
all four. The deterministic largest projected representative face, 48941,
has zero coherent depth votes at all four samples and 13/11/15/15 trusted
farther-camera votes. All 62 train crops for its centroid were visually checked.

In the 743-pixel diagnostic polygon, **559 pixels actually select J/A
(`J004_A005_121014`)**. Of these, 558 have measured depth farther than the
rendered surface by more than 0.005 normalized units; only one is within the
0.0015 near tolerance. The reviewed J/A example projects onto blue shirt fabric,
with measured camera-z minus mesh camera-z **+0.04136 normalized units**. A K/A
example likewise lands on shirt fabric (+0.04246); J/B lands on farther skin
(+0.04495). This is not evidence of two-camera RGB averaging or changing exposure.

- [GT, prediction and selected diagnostic faces](/mnt/data/dec5_lipstick_fin_depth/000995/selected_native.png)
- [Actual dominant J/A source: blue shirt](/mnt/data/dec5_lipstick_fin_depth/000995/rgb_trace/case_15.png)
- [Actual K/A source: blue shirt](/mnt/data/dec5_lipstick_fin_depth/000995/rgb_trace/case_17.png)
- [Actual J/B source: farther skin](/mnt/data/dec5_lipstick_fin_depth/000995/rgb_trace/case_16.png)

Only these three of 19 generated per-source examples were individually reviewed;
the separate four contact sheets cover all 62 cameras for the representative
mesh point. The diagnostic polygon is not itself a proof of false geometry.

| Near footprint | Removed / 144,918 faces | Removed among 86 diagnostic faces | Moving lost / deeper-hit pixels | H/C lost pixels | K/B lost pixels |
|---|---:|---:|---:|---:|---:|
| 5×5, any near tap protects | 76 | 24 | 0 / 70 | 86 | 239 |
| Native center only | 196 | 48 | 0 / 253 | 333 | 538 |

These are geometry diagnostics, **not quality metrics**. Deletion produced no
new or nearer ray hits, as independently checked in all six RGB comparisons.
Native lost rays can mean removal of false foreground; they do not alone prove
anatomical holes. Original mesh had 39 index-connected components; both subsets
have 38. No small-component cleanup was used to cosmetically improve counts.

Main LLM inspected all six lipstick and six head comparison panels. Radius 0
removes more lateral blue fragments, but the main blue strip behind the tube
and the H/C bridge remain. Moving-pose changes are modest. No conspicuous new
face/hair defect appears in these matched panels; the existing crown opening
remains. This is **not** held-out or temporal quality certification.

[5×5 K/B comparison](/mnt/data/dec5_measured_free_surface_pruning/000995/review/K004_B005_1210DS/lipstick_native.png),
[center-only K/B comparison](/mnt/data/dec5_measured_free_center_pruning/000995/review/K004_B005_1210DS/lipstick_native.png).

Independent audit replays the footprints of every removed triangle in both
arms, verifies every farther observation's corroboration, and checks unchanged
vertices and exact triangle subsets. Footprint implementation is separate;
the established inter-camera corroboration helper is explicitly reused.
Three focused tests pass, including missing/invalid data, any-near-tap protection,
the radius-zero control, and the requirement that all four samples satisfy both
farther-depth gates. All six GPU render workers completed normally (about
22–24 seconds per first-arm render); no production/default changes or deletions.
Final seal rechecked **331 SHA-256 bindings** across the diagnostic, both arms,
raw depth maps, source RGB and dependencies. A trace-field naming correction
replaced `gt_depth` with `measured_depth`: these are PatchMatch estimates, not
ground-truth depth. Its original output and executed script are archived;
all 19 example PNGs and numeric observations are byte/numerically identical
after replay. The three focused tests were rerun successfully in 1.22 seconds.

Artifacts:
`/mnt/data/dec5_lipstick_fin_depth/000995`,
`/mnt/data/dec5_measured_free_surface_pruning/000995`,
`/mnt/data/dec5_measured_free_center_pruning/000995`.

## Insights

The color failure is now attributed to a concrete source camera and a different
physical layer, not just suspected from appearance. The renderer's visibility
test raycasts the candidate mesh itself (`render_smooth_temporal_mesh_video.py`,
`query`). A false surface can therefore appear self-consistent in that test
while actual stereo depth puts the RGB-bearing surface much farther away.
Fixed color calibration cannot correct this correspondence error.

Removing strongly contradicted faces is useful, but whole-triangle protection
is conservative near real object boundaries; decreasing its pixel radius alone
does not eliminate the fin. Do not infer that more aggressive deletion is safe.
The next relevant controls are measured-depth-aware texture visibility and/or
confidence trimming at finer surface resolution, with explicit checks for new
lipstick, hand and jaw holes. A pre-extraction voxel constraint is another
candidate, but was **not run here**. The numerical TSDF zero-crossing mechanism
at 000995 remains unisolated, despite the measured contradiction being clear.

The requested artifact-free dynamic video and broadly corrected jaw geometry
remain incomplete. Existing published videos and all source data are unchanged.
