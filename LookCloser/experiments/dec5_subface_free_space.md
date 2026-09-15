# Subface measured-free-space pruning

## What was tested

Frame `000995`, following the [near-protection diagnosis](dec5_lipstick_near_protection.md).
Hypothesis: valid near-depth evidence at one vertex overprotects a whole false
triangle; smaller faces allow supported and unsupported parts to be separated.

The opt-in `subdivide_conflicted_surface.py` performs two rounds of conforming
midpoint subdivision, preserving the surface, original vertices, orientation,
parent ancestry and per-parent area. Red refinement selects all 1,300 original
faces having any near sample and any sample with at least six stable farther
views, across the entire mesh. Green neighbour splits avoid T-junctions. No RGB,
ROI, semantic mask, held-out image, or learned prior selects refinement/deletion.

The exact preceding native-center deletion gates remain frozen: all three child
vertices and centroid must have no near observation within `0.0015`, six stable
farther 5×5 footprints, and six farther observations each corroborated by at least
three other cameras. No displacement, smoothing or component cleanup is added.
Depth tolerances are in the established normalized coordinates, not claimed meters.

Three matched render controls use the original current-recipe mesh, subdivision
only, and subdivision plus pruning. Exposure, camera profiles, masks, poses,
incidence-2/angle-prior hard texture and registration-off policy are identical.
These diagnostic RGB renders use CUDA on clever-shadow; 6K delivery is untouched.

## Results

| Geometry | Vertices | Triangles |
|---|---:|---:|
| Subdivision only | 85,486 | 166,362 |
| Subdivision + pruning | 85,486 | 164,518 |

Original mesh: 144,918 triangles. Of 1,850 candidate child faces, 1,844 pass all
deletion gates, affecting 443 original parents; 260 are only partly removed.
The representative fin parent `48941` loses **12.5%** of its area, rather than
being wholly protected. 62/86 diagnostic parents lose some area, versus 48/86
whole parents in the preceding native-center control. These counts are not a
quality score or proof that all remaining geometry is correct.

Independent native-footprint audit: 7,376 removed samples × 62 cameras = 457,312
queries; maximum near count 0, minimum stable/corroborated farther count 6/6.
Corroboration deliberately reuses the established helper. Exact subdivision
replay and independent surface/area checks pass; all 4,658 resulting boundary
edges lie on pre-existing boundaries. Ten focused subdivision tests cover all
eight edge patterns, conforming shared edges, orientation and repeated ancestry.

| View | Subdivision-only changed RGB | Pruning lost depth hits | Pruning changed RGB | Newly black RGB after pruning |
|---|---:|---:|---:|---:|
| Moving | 134 | 0 | 672 | 11 |
| H/C | 687 | 386 | 749 | 394 |
| K/B | 552 | 629 | 841 | 642 |

The pruning comparisons above are against subdivision-only. Subdivision alone
loses/gains no hits and has no depth difference exceeding `1e-6`; nevertheless
its new adjacency graph changes some source labels (including one newly black
K/B pixel). Pruning adds no hits or nearer surfaces. Black RGB counts are not
missing-anatomy metrics: the largest native-view components remove false surfaces
over real background. In the moving view, however, 11 isolated black pixels
appear near the lipstick/skin boundary despite retained depth hits.

Main-agent inspection covered all six head/lipstick panels and all 32 newly-black
components on native-size sheets:

- [Moving lipstick](/mnt/data/dec5_subface_free_space/000995/review/moving/lipstick_native.png), [side effects](/mnt/data/dec5_subface_free_space/000995/review/moving/new_black/all_native.png).
- [H/C lipstick versus train GT](/mnt/data/dec5_subface_free_space/000995/review/H004_C005_1210SZ/lipstick_native.png), [side effects](/mnt/data/dec5_subface_free_space/000995/review/H004_C005_1210SZ/new_black/all_native.png).
- [K/B lipstick versus train GT](/mnt/data/dec5_subface_free_space/000995/review/K004_B005_1210DS/lipstick_native.png), [side effects](/mnt/data/dec5_subface_free_space/000995/review/K004_B005_1210DS/new_black/all_native.png).

The false bridge/fin is partly trimmed, but the prominent blue-fabric protrusion
behind the metal tube remains. Existing crown hole and coarse hair fringe also
remain. **Partial cleanup, not promoted.** No PSNR/SSIM/LPIPS claim is made for
these train/moving-view controls, and no full-frame quality metrics were computed.

Artifacts and explicit visual verdict:
`/mnt/data/dec5_subface_free_space/000995/`.
The two branch meshes/render receipts are under sibling `refined/000995` and
`pruned/000995` directories. `final_audit.json` seals retained hashes and the
post-inspection verdict; earlier numerical receipts retain their pre-review state.

Replay: `study_subface_free_space.py geometry`, then `prepare` and `render`
with `--control refined|pruned` and each of the three `--view` values; geometry
refuses an existing root. `audit_subface_free_space.py` independently checks the
removed faces. Review/localization/sealing helpers are separate from geometry.

## Insights

Confidence granularity matters, but subdivision alone does not solve the fin.
Even at two refinement rounds many child samples remain within the fixed near
tolerance of real measured surfaces. Corroboration of those observations is not
equivalent to agreement with the exact queried mesh surface. Simply increasing
subdivision or deleting more faces is not yet justified as a robust fix.

The actual unresolved mechanism is mixed surface support near an occlusion:
the mesh can be near a measured layer at some samples yet project onto unrelated
RGB elsewhere. Further work needs an uncertainty-aware surface admission or
measured-depth texture visibility check, with the same native train comparisons;
the current experiment provides no permission to loosen production thresholds.
