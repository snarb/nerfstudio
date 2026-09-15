# Independent positive-clearance contact review

## What was tested

Read-only review of `mesh_contact_clearance.py` and `run_mhr_clearance_correction.py`, including inherited anatomical/conic driver and `AllContactGuard`. No fitting, geometry edits, target-ray inputs, guard changes or production changes. CPU threads limited to two. Image metrics are not applicable to this geometric/mathematical diagnostic.

Artifacts: [synthetic results](/mnt/data/dec5_mhr_clearance_independent_review_v2/result.json), [detector localization](/mnt/data/dec5_mhr_clearance_independent_review_v2/detector_probe_v3/result.json), [two actual-output checks](/mnt/data/dec5_mhr_clearance_independent_review_v2/actor_pair_probe/result.json). Reproduce with the corresponding `review_mhr_contact_clearance.py`, `probe_clearance_detector_discrepancy.py`, and `probe_actor_contact_clearance.py` scripts, supplying fresh output roots. Audit: `audit_mhr_clearance_review.py`.

## Results

### Constraint mathematics and integration

For selected unit axis `n`, every vertex pair enforces `n·(δb−δa) ≥ clearance−n·(b−a)`. Row and lower-bound normalization divide by the same positive coefficient norm. Consequently every endpoint pair projects at least `1e-8` scene units apart. The signs are correct. Added normal-cross-edge directions supply coplanar in-plane axes; additional axes are safe because any verified positive separating projection proves convex triangle separation. Fixed/fixed projection inequalities are checked before zero-variable rows are omitted. This is a necessary feasibility screen, not proof of feasibility under all other constraints or an exhaustive search over alternative planes.

108 seeded cases cover coplanar, parallel and skew triangles, 12 rotations/translations, alternating winding and three active-vertex patterns. Independent SLSQP solves the returned linear constraints. All 108 satisfy the original-coordinate pairwise clearance; minimum gap is `9.999999911e-9`. Separated all-fixed triangles accept with zero rows; touching and overlapping all-fixed cases reject. Three existing unit tests pass.

Four skew contact-only test outputs collapsed a triangle: cross-product magnitudes `0..3.32e-21`, versus `4e-8` on the other triangle. Open3D reports an intersection despite an 80-digit Decimal positive separating witness. Those tests deliberately contain no area constraint; this is **not** an area-preserving solver failure. Four matching rigid-translation controls preserve nonzero area and satisfy clearance with no Open3D pair. The initial assertion-failure root is retained; two subsequent detector-probe provenance failures are also retained, followed by the successful `detector_probe_v3` output.

The wrapper changes proposed-pair detection to all Open3D pairs and retains final `AllContactGuard`. No target inputs were introduced. **Clearance applies to the full proposed endpoint, not necessarily the accepted iterate:** maximum-step scaling/backtracking can reduce the margin toward the current separation. Final acceptance guarantees no new detector-reported pairs, not a universal accepted minimum gap of `1e-8`, and not continuous collision-free motion. Adjacent shared-vertex pairs are outside this detector's proposal domain; the helper should not be treated as a general positive-clearance contract for such pairs.

### Open3D discrepancy and real stored outputs

Installed version is Open3D 0.19.0; the loaded binary and official source snapshots are hash-bound. Official code centers and scales each triangle pair per coordinate using standard deviation plus `1e-12`, then calls the triangle predicate. Thus its epsilon is not simply a fixed scene-unit clearance. [IntersectionTest.cpp, v0.19.0](https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/geometry/IntersectionTest.cpp).

The inner predicate zeroes plane-expression values below `1e-6`; normals in those expressions are not unit normals. Degenerate/near-plane cases therefore require care. The four collapsed synthetic false positives persist at coordinate scales 1, 10 and 1000, both centered and uncentered. [Official opttritri.h](https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/3rdparty/tomasakeninemoeller/include/tomasakeninemoeller/opttritri.h). The outer routine excludes shared-vertex triangles before testing. [TriangleMesh.cpp](https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/geometry/TriangleMesh.cpp).

Parent-specified actual cases were checked independently using centered long-double axis construction, then 80-digit Decimal projections of the original stored coordinates and the resulting axis. Pair inventory comes from retained topology receipts, not a new full-mesh enumeration; each reported pair was independently tested.

| Stored case | Pairs | Float64 separating gap | Float32 result |
|---|---:|---:|---|
| Certified conic final, new versus warm | 11 | `5.458945e-14..1.052461e-13` | 5 retain positive `4.816e-10`; 6 lose the witness (`−2.429e-11`) |
| Anatomical last proposal, 11510/11596 | 1 | `1.33313134e-10` | Loses positive witness (`−9.212e-10`) |

All actual triangles are nondegenerate: certified cross norms are at least `4.945e-8`; anatomical pair cross norms are `2.194e-6` and `2.640e-6`. Open3D reports all 12 original float64 pairs despite strict positive witnesses. Float32 quantization changes coordinates by up to `1.74..1.85e-9`, greater than those separations: the five positively separated quantized pairs stop being reported; the remaining seven remain reported. A nonpositive sampled-axis result alone is not presented as an exact intersection certificate. [Per-pair axes, projections and coordinates](/mnt/data/dec5_mhr_clearance_independent_review_v2/actor_pair_probe/result.json) are preserved; normalized plane expressions are saved separately without claiming a complete C++ replay.

## Insights

No sign/axis error was found within the intended disjoint-vertex proposal domain. Contact constraints alone do not preserve triangle area; inherited area/topology guards remain necessary. Detector positives are not exact geometric-contact proofs, but tiny float64 separating gaps are also not robust float32 rendering clearances. Neither observation justifies relaxing the actor guard. This review makes no repair, convergence, promotion or continuous-collision claim.
