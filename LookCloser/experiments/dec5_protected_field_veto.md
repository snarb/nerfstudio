# Protected pre-extraction TSDF free-space control

## What was tested

At `000995`, test whether measured free space can remove the false lipstick fin
inside the TSDF scalar field, before marching cubes, without cutting observed
skin. This follows the negative [full-block transfer](dec5_full_block_transfer.md)
and [conditional depth-peeling study](dec5_measured_source_depth_peeling.md).
Earlier unprotected three-camera field veto at `000973` caused an extra hand
notch; it was not accepted and is not silently repeated here.

Both fresh local fusion arms use the same 62 geometric depth arrays, fixed
calibration, full bounded block union, voxel .0005, truncation .004, extraction
weight 2 and original component filter. The experiment changes only already
observed negative TSDF samples with zero nearby native depth observations and
at least six stable farther 5×5 footprints. One near observation (within .0015)
protects the sample. Farther evidence requires gap max(.005,.01z), 20/25 taps
and middle-depth spread at most .5%. Native depth queries use integer centers.
Integration's existing lookup convention is unchanged. Unknown weights and
positive field values are not changed, and no integration weights are added.

The control computes identical evidence but leaves the field untouched.
No RGB, ROI, semantic mask or held-out image chooses the geometry. Rendering
retains production's fixed profiles/exposure, source masks and hard single-source
unwarped RGB. Production's later head/silhouette repairs are **not** reapplied
to either raw fusion arm. Meshes and sampled field evidence are saved, not a
serialized raw TSDF volume. Existing scripts and model defaults are unchanged.

## Results

| Arm | Vertices | Triangles | Components | Negative field samples changed |
|---|---:|---:|---:|---:|
| Current-runtime full-block control | 79,695 | 153,829 | 1 | 0 |
| Protected field veto | 79,673 | 153,785 | 1 | 149 |

Both integrations allocate 9,097,216 voxels. Evidence queries cover 339,363
observed negative samples; 141,683 have at least one protecting near observation.
The sorted physical sample positions, weights, eligibility and pre-veto field
match exactly between runs (maximum pre-field difference **0**). Independent
NumPy/float64 replay verifies all 149 eligible samples against all 62 native
depth maps. All 62 imported normalized arrays equal their original raw COLMAP
arrays exactly. No new PatchMatch/SfM run or input color change was involved.

The fresh local control has two more vertices/triangles than the older dev3
full-block extraction. Therefore the causal comparison uses this fresh control,
not an assertion of bitwise equivalence with the old remote mesh.

| Fixed view | Changed RGB pixels | Lost first-hit pixels | Added first-hit pixels |
|---|---:|---:|---:|
| Moving left arc | 3038 | 98 | 0 |
| Native H/C | 811 | 463 | 1 |
| Native K/B | 2585 | 537 | 0 |

These are diagnostic counts, not quality metrics. Marching-cubes re-extraction
is not an exact triangle-subset deletion: there are also 10/34/74 nearer-depth
pixels beyond 1e-6. A lower triangle count is not a quality improvement.

The main agent inspected all three native lipstick panels and all three head
comparison overviews. **0 pass / 3 fail** for the intended repair. The main blue
shirt-textured fin behind the tube remains especially clear at K/B. H/C retains
an irregular tube and the hand/neck hole or membrane. Moving-view changes are
minor and do not correct the tube/hand form. Crown openings and neck silhouette
artifacts persist. No convincing broad facial benefit was observed.

- [Moving lipstick comparison](/mnt/data/dec5_protected_field_veto/000995/review/moving/lipstick_native.png)
- [H/C train GT and comparisons](/mnt/data/dec5_protected_field_veto/000995/review/H004_C005_1210SZ/lipstick_native.png)
- [K/B train GT and persistent fin](/mnt/data/dec5_protected_field_veto/000995/review/K004_B005_1210DS/lipstick_native.png)
- [Field replay audit](/mnt/data/dec5_protected_field_veto/000995/field_audit.json)
- [Visual verdict](/mnt/data/dec5_protected_field_veto/000995/visual_review.json)
- [Retained artifact hashes](/mnt/data/dec5_protected_field_veto/000995/artifact_manifest.json)

Six full-resolution renders completed on clever-shadow, with three view workers
running concurrently. An initial launch failed before rendering because the
new preparation wrapper omitted the parent `frames/` directory. Its requests,
code and logs are retained in `rgb_attempt1`, `review_controller_attempt1.py`
and the initial logs. Preparation was fixed and all six new hash-bound requests
rendered successfully. Geometry and source recipes were not changed by that fix.
Three focused tests pass for native-center lookup, near protection, preservation
of unknown/positive field samples, and NumPy agreement.
The combined focused/existing free-space suite passes **7 tests**. The final
artifact seal rechecks **288 SHA-256 bindings**, including all six completed
renders, paired geometry, evidence, failed-attempt records and native panels.

## Insights

This conservative field veto does not solve the fin and is **not promoted**.
Changing field extraction location alone is insufficient under these confidence
rules. Only 13 additional far-qualified negative samples have exactly one near
camera, so merely changing protection from one to two raw observations has weak
evidence for a material repair. It is not tested as another cosmetic threshold
sweep here. The next useful hypothesis should address surface reconstruction or
object shape constraints, not assume that more global carving will repair the
unsupported object. No held-out metric or temporal-video pass is claimed.
