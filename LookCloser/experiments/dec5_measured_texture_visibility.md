# Measured-depth texture visibility: wrong layer versus wrong geometry

## What was tested

Frame `000995`, three matched views, original production mesh and the unchanged
hard-source incidence-2/angle-prior/zero-registration renderer. The new opt-in
`measured_texture_visibility.py` rejects a candidate RGB source when its measured
PatchMatch depth shows a stable farther layer: gap greater than both `0.005`
normalized units and 1% query depth, at least 20/25 valid farther native taps,
middle-quantile spread no greater than 0.5% depth. Missing depth stays unknown.
No RGB blending, color refit, mesh edit or production change occurs.

This is deliberately a **negative per-source texture test**, not the stronger
multi-camera geometry-deletion rule. Additional inter-camera corroboration is
not computed here. Nearer occluders are not tested by this specific farther-layer
ablation. Passing it does not establish that the source is correct.

The first K/B canary accidentally indexed raw depth with display UV (`-.5`).
It is preserved, not used for conclusions. The corrected `native_depth_v2` worker
projects actual world samples with the established `project_integer` raw-depth
convention. Its explicit test demonstrates the half-pixel distinction. The old
executed worker is hash-bound under the initial root's `config/`; no existing
renderer or calibration convention was changed to accommodate the new test.

## Results

Independent replay reconstructs the actual float32 triangle/barycentric query
points and tests **every selected colored source** with separate native-footprint
arithmetic. Mesh hashes and target depths are exactly unchanged in all views.

| View | Selected sources contradicting stable farther depth: before → after | Changed RGB pixels | Newly black RGB pixels |
|---|---:|---:|---:|
| Moving | 938 → 0 | 1,517 | 18 |
| H/C | 1,329 → 0 | 2,658 | 27 |
| K/B | 2,998 → 0 | 3,536 | 94 |

K/B diagnostic polygon: 755 mesh-hit pixels (743 initially colored), 617
contradicting selections before and none after. Uncolored source-ID-255 pixels
increase **12 → 57**. The earlier 743-pixel trace counted colored pixels only;
the different denominator is explicitly retained, not treated as a metric change.

Main LLM inspected all three head/lipstick pairs and all three lipstick
new-black overlays, nine panels total:

- [K/B: real GT / old / guarded](/mnt/data/dec5_measured_texture_visibility_native_depth_v2/000995/K004_B005_1210DS/review/lipstick_native.png).
- [H/C: real GT / old / guarded](/mnt/data/dec5_measured_texture_visibility_native_depth_v2/000995/H004_C005_1210SZ/review/lipstick_native.png).
- [Moving comparison](/mnt/data/dec5_measured_texture_visibility_native_depth_v2/000995/moving/review/lipstick_native.png).

The conspicuous blue-fabric fin becomes mostly skin-colored, but the same wrong
protruding surface and jagged bridges remain. In the moving view, a darker,
more sharply bounded patch near the tube/finger is still visible. Missing RGB
increases at some boundaries. No obvious broad new face/hair distortion is seen,
but this is not comprehensive temporal or fine-fringe acceptance.

**Rejected as a standalone artifact fix.** It confirms the wrong-layer source
mechanism and removes those measured contradictions, but cannot repair geometry.
Nine tests pass, including unknown/near/nearer data, singleton farther outliers,
eligibility preservation and the raw-depth/display coordinate distinction.
No quality metrics were computed; counts above are diagnostic inventories, not
full-frame PSNR/SSIM/LPIPS. No 6K video frame is replaced or rerendered by this study.

Corrected root: `/mnt/data/dec5_measured_texture_visibility_native_depth_v2`.
Initial preserved canary: `/mnt/data/dec5_measured_texture_visibility`.
Each corrected view has request, source/depth replay, render receipts and crops;
`000995/visual_review.json` records the actual review. `final_audit.json` seals
the corrected study and initial-canary ancestry. Earlier numerical receipts
retain their pre-review `pending` field rather than pretending they performed
visual inspection.

Replay: `study_measured_texture_visibility.py --view VIEW`, then
`review_measured_texture_visibility.py --view VIEW`; use a fresh root for another
recipe. The immutable run rejects an existing view directory.

## Insights

Self-mesh raycast visibility can accept RGB from another real layer when the
candidate mesh is wrong. Fixed per-camera color calibration is not the cure for
that correspondence error. A measured-depth guard is useful negative evidence,
but accepting an alternative camera with unknown/weak depth is not positive
surface validation. The remaining protrusion requires a geometry-level repair;
changing its color merely makes a different-colored artifact. Do not loosen
the guard or silently return to contradicted sources to hide its new black pixels.
