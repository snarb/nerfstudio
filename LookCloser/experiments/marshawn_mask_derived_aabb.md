# Programmatic mask-derived actor AABB

What was tested: derive a conservative bounding hypothesis from the same two reviewed fighter IDs in all66 cameras, using the complete scene search domain instead of the rejected hand-sized actor boxes. No 3D model was trained or changed.

The source is `two_actor_isolation/confidence_native_hair_v2/manifest.json` (SHA`1f7f28751aac5ae7486ff8392606cf6c09bd2118eef934dbf64c1171d96aabc3`). The corrected identity master is SHA`f5971248ab1c1dd92d6503ff98a8131f4fa2705d934909b31e164cb2c0846fa0`. Names/instance indices are mapped to stable `fighter_black` and `fighter_grey` identities; detector rank does not select people. The optional `marshawn_person_ids.py` association module can supply future frame mappings; actual1661/1662 tracking is not validated here.

`scripts/mask_visual_hull_aabb.py` adaptively subdivides the calibrated full-scene domain. Every cell projects all8 corners; integral images test its outward-rounded pixel footprint. At least3 foreground views are required, with at most1 known-clear background contradiction. Verified occluders, ambiguous/unselected person instances, invalid pixels, out-of-FOV cells and near-plane crossings are UNKNOWN, not certified empty. Foreground votes remain valid where ambiguous duplicate masks overlap a selected actor. Unsupported unknown volume is reported separately rather than called physical free space. This is an observed visual-hull hypothesis, not exact occupied geometry.

Selected foreground uses any positive calibrated RAW-area-prefiltered nominal coverage, including accepted native hair additions. Independent mask uncertainty pads64/128/256 native pixels were tested. Whole retained voxel extents are unioned with364 semantically vetted TRAIN SfM anchors before the 3D margin. No assumed floor/MVS depth was used. The leaf extent is approximately `[.006872,.007322,.002863]` in prepared-world units; the fullscene boundary was not touched.

Results:

| Native mask pad | Final minimum | Final maximum | Volume (prepared units³) | Native actor/hair ray misses | Additional boot ray misses |
|---|---|---|---:|---:|---:|
|64|[-.19,.04,-.19]|[.14,.32,.25]|.040656|0/87,645,065|0/12|
|128|[-.20,.02,-.20]|[.15,.33,.26]|.049910|0/87,645,065|0/12|
|256|[-.22,0,-.22]|[.17,.35,.28]|.068250|0/87,645,065|0/12|

The native audit covers all66 cameras:87,413,105 coarse selected pixels plus231,960 accepted hair-addition pixels, with verified source-occluder vetoes excluded. All5,077,692 positive HD footprint rays also intersect each box. The12 AA/GA/OA boot-scope pixels were independently selected on source images; their depths are not ground truth. Geometry's separate review inspected17 pages and11 additional source-toe rays, all with zero misses, and recommended128 as the robust baseline. Root approval and actual separate clipping review remain required.

The computation is transductive geometry: **eval segmentation is used for the AABB**, as explicitly requested by the user for all66 cameras. Eval RGB is not fitted photometrically, used for gain optimization or used as training reconstruction targets. The same fixed box and frozen eval semantic domain must be used for all confidence policies.

Artifacts under `/fsx/tmp/marshawn_001660_hd/two_actor_isolation/automatic_aabb/`:

- `input_v2/manifest.json`: calibrated foreground/UNKNOWN/valid rasters and all source hashes.
- `experiment_v2/summary.json`, `coverage.json`, `padding{64,128,256}/{aabb.json,retained_cells.npz}`: frozen actual calculation, full native ray counts and per-camera misses.
- `full_context_v2/`: all66 full-source box overlays and3 full **unclipped** teacher views. No fit claim is inferred from an already clipped teacher. The3 source comparisons and3 pad256 full-teacher images were actually inspected by the preparer; geometry owns the independent whole-source foot review.
- `teacher20k_prepost_pipeline_proof.json`: all38 NN/field/EMA/grid tensors are exactly equal between the full-render pre-eval20k checkpoint and selected post-eval20k checkpoint. This permits reuse of the original full teacher context; checkpoint RNG/timing bytes differ.
- `independent_boot_source_review_retry1/manual_review_geometry.json`: independent source-foot review. CE feet are out of FOV; several foreground occlusions remain explicitly unknown.

The frozen carve uses a common minimum3D margin `.02` before outward snapping to a fixed `.01` world grid. Units are **prepared-world coordinates**, not assumed centimeters. The normalization scale is `.22172035973820498`; conversion to original rig coordinate distances is division by that scale, without asserting the rig's physical unit.

The updated helper additionally supports a guaranteed scalar minimum, absolute XYZ minimums and an absolute lower-Z minimum for feet. Effective XYZ margins are `max(common, axis)`; the lower-Z margin is additionally `max(..., bottom)`. Mask padding is an optical pixel uncertainty margin and is independent of these world-space margins. Receipts report requested and actual post-snap gaps for all6 faces and verify the minimums. Negative/NaN margins are rejected.

Margin-only rebuild, without re-carving or changing masks:

```bash
python scripts/mask_visual_hull_aabb.py \
  --rebuild-from /fsx/tmp/marshawn_001660_hd/two_actor_isolation/automatic_aabb/experiment_v2/padding128/retained_cells.npz \
  --source-receipt /fsx/tmp/marshawn_001660_hd/two_actor_isolation/automatic_aabb/experiment_v2/padding128/aabb.json \
  --margin-world .02 --axis-margin .02 .02 .02 --bottom-margin .02 \
  --snap-step .01 --output /path/to/new_margin_review
```

`temporal_envelope` unions actual per-frame extents and applies guaranteed margins/outward snapping on one fixed world lattice. Report per-frame boxes and window union separately. No real1660/1661/1662 envelope or temporal rendering has yet been validated; this is a control for coordinate-frame jitter, not proof of no flicker.

Validation:17 CPU tests pass, including actual synthetic calibrated projections of a cuboid/slender leg, stable-ID crew exclusion, occluded camera, mask false negatives/padding, cell-edge support, unknown duplicate overlap, near/FOV unknowns, domain clipping, temporal envelopes and post-snap anisotropic minimum margins. A low-support synthetic diagnostic exposed crew-shadow/unknown-compatible lobes: filtering masks by person ID cannot guarantee that all retained3D volume is actor geometry. Stronger multi-view support rejects that synthetic ambiguity; the actual66-view result still requires independent source/teacher inspection.

Insights:128 is a proposed conservative compromise;64 is merely the smallest tested candidate with no known ray misses. Ray intersection is necessary but **does not prove the real boot/hair surface lies inside the box**. Hard masks undersegment some dark toes beside detached props; source review remains essential. UNKNOWN/hidden volume is not an ideal occupancy certificate, and a box can still contain floor/props or other people along some camera rays. Separate clipped images and an actual actor-only2k training/occupancy gate are needed before release.

### DC foot complaint: independent depth diagnostic, 2026-11-06

What was tested: the user reported three feet outside `DC_common002_FULL_SOURCE.jpg`.
Root inspected the actual DC/HC/KA sources and all four boot crops. The wireframe
draws rear and front edges over actors without scene-depth occlusion; crossing an
edge in the image is not an outside-volume test. The bounds were **not changed**:
`[-.20,.02,-.20] -> [.15,.33,.26]`.

`diagnose_marshawn_feet_aabb.py` provides a stronger, independent check: intersect
the calibrated positive-depth frusta of manually reviewed whole-boot source ROIs
and solve six unconstrained-world linear programs for their extrema. The tested
AABB is never used to restrict that calculation. DC/HC constrain all four boots;
KA additionally constrains the far black boot and rear grey boot. KA near black
foot is occluded, and KA front-grey-foot/hat overlap is ambiguous, so both are
unused. Root actually viewed all source ROI crops and all three final diagnostics.

|Boot|Minimum face clearance, original ROIs|With +10 HD pixels on every ROI side|
|---|---:|---:|
|Black near|.039032|.025662|
|Black far|.045628|.033941|
|Grey rear|.039777|.027870|
|Grey front|0.015629|.001873|

Clearances are prepared-world units, not meters; JSON preserves full precision. All possible points satisfying those visible-boot ROIs
remain inside the unchanged box, including the expanded-ROI sensitivity test.
This is conditional on calibrated cameras and the reviewed source rectangles;
it is not a certificate for hidden geometry, other cameras or the occupancy grid.

Additional independent checks: 24 geometrically consistent SIFT tracks on the
grey boots, each observed in at least three cameras, are inside the box. No reliable
black-boot tracks passed; do not extrapolate SIFT evidence to those boots. Actual
`nerfstudio.Cameras.generate_rays` agrees with the overlay's source-pixel
projection for DC/HC/KA within `7.48e-8` maximum direction-component error.
Four synthetic tests verify LP extrema, outward padding and unbounded-frustum
failure. `render_marshawn_foot_diagnostic.py` exports labelled lower vertices and
three orthographic envelope plots beside unchanged source RGB.

Artifacts: `/fsx/tmp/marshawn_001660_hd/two_actor_isolation/automatic_aabb/foot_diagnostic_root_v1/`;
`roi_frusta.json`, `native_camera_ray_parity.json`, `independent_tracks/receipt.json`,
`whole_boot_ROI_manual_review.png`, `{DC,HC,KA}_feet_depth_diagnostic.png`.
NeRF training remains stopped. The old unoccluded wireframe alone is not an
adequate visual containment review.

### HC actual learned-field partition after repeated visual objection

What was tested: render the exact same historical R1 20k model with density
retained only at sample centers inside `common002`, and separately outside it.
`render_marshawn_aabb_partition.py` preserves the original field normalization,
camera, grid and native packed sample locations. No segmentation or GT RGB is
used for compositing. It also partitions the ORIGINAL visible contributions,
preserving full-scene transmittance, to show which reconstructed surfaces the
box removes; these contributions sum to the full image.

Results: root viewed full HC, both actor/foot comparisons and the inside-only
render. Both actors and all four visible boots remain after the actual 3D cut;
no visible boot cutting was observed. Four explicitly sampled boot rays retain
at least `.99999958` original opacity inside the box; their outside opacity is
at most `1.20e-7`. This is representative-ray evidence, not a whole-silhouette
numerical certificate. Native compositor parity is `7.16e-7`; partition sum
error is `1.97e-6`. Model/grid tensor digest and immutable checkpoint hash are
unchanged. The source-camera reprojection of all four actual density peaks
agrees with their queried HC pixels within `.001 px`.

Insight: the previous unoccluded wireframe draws the bottom over the physical
floor even though the bottom lies below the soles; that presentation does not
communicate depth adequately. Actual clipping provides a clearer inspection.
The model still retains floor, a hat and part of a mat inside the AABB. This
does not release actor isolation or certify the occupancy grid. Historical
learned geometry remains an imperfect reference; midpoint partitioning has
up to `.0015` world-unit boundary discretization at the native `.003` step.

Artifacts: `automatic_aabb/HC_actual_partition_v1/` includes raw float buffers,
four sample traces, receipts and manually viewed comparisons. Downloadable
`/fsx/tmp/marshawn_001660_hd/downloads/HC_actual_AABB_cut.png` contains source,
full model, inside-only and outside original contribution. Training stayed stopped.
