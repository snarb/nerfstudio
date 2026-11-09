# Marshawn1660: near-hardware novel-view diagnosis

## What was tested

Selected radius0 checkpoint8000 produced broad nearfield veils in matched-intrinsics novel views03,06 and21. This was treated as a release failure even though sampled BODY rays and surface-witness coverage passed. The earlier common-focal810 path is a deliberately widened stress trajectory with valid vertical-FOV serialization. Its source is preserved; it was not a serialization bug. The matched path uses the same24 poses with independently interpolated actual fx/fy/cx/cy.

Independent CPU projection used the actual density-peak and50% termination XYZ from the occupancy agent's frozen native ray trace, rather than camera origins or expected-depth buffer points. Those XYZ were projected through all63 calibrated training cameras and sampled from the real scalar SDR/linear EXR source images.

## Results

[Actual trace](/fsx/tmp/marshawn_001660_hd/final_pair/release_audit/novel_cloud_traces/audit.json) reproduces saved pixel RGB within3.58e-7 without changing model state. The dominant nearfield samples begin around0.0115 prepared-world distance from the novel camera, close to the0.01 near plane.

The actual peak and50% termination points consistently project onto the same luminous octagonal softbox faces in real training views:12 cameras for03,11 for21 and8 for06. I manually inspected the three peak sheets and three50% sheets. Example evidence: [03 face](/fsx/tmp/marshawn_001660_hd/final_pair/release_audit/cloud_GT_visibility/view03_x100_y280_peak_GT_1.jpg), [21 face](/fsx/tmp/marshawn_001660_hd/final_pair/release_audit/cloud_GT_visibility/view21_x100_y280_peak_GT_1.jpg), [06 face](/fsx/tmp/marshawn_001660_hd/final_pair/release_audit/cloud_GT_visibility/view06_x320_y280_peak_GT_1.jpg), [projection audit](/fsx/tmp/marshawn_001660_hd/final_pair/release_audit/cloud_GT_visibility/audit.json).

For each problematic view, samples in three real cameras have PNG codes253–254, zero255 channels in the central5×5 samples, and finite linear Rec709 values approximately59–139. These are bright diffuser faces, not metal edges at the projected sample. This measures SDR-code saturation only; it does not prove RAW sensor unsaturation. [Radiometric samples](/fsx/tmp/marshawn_001660_hd/final_pair/release_audit/cloud_GT_visibility/bright_face_samples.json).

The camera-origin audit independently finds actual softbox/rig SfM points within0.017–0.055 of the problematic origins. Together with the actual peak projections, this supports hardware proximity as the cause of these nearfield views. Frustum inclusion and white-face projection alone do not establish exact surface depth or a closed physical volume. Optional local-plane RANSAC was rejected as diffuser depth truth after manual GT inspection: fitted inliers mixed fixture edges with adjacent cameras and metal mounts. No such plane was used for carving or training.

New inner-standoff candidates translate every origin by `offset * (-rotation[:,2])`, retaining bitwise-identical rotation and fx/fy/cx/cy. All24 translations were independently checked after reloading the bundles; the actual camera-bundle loader also passed.

| Offset | Minimum distance to9022 scene witnesses | Minimum distance to495 BODY witnesses |
|---:|---:|---:|
|0.00|0.01726|0.93521|
|0.10|0.09777|0.84085|
|0.15|0.11191|0.79419|
|0.20|0.13986|0.74796|

The criterion was recorded before candidate renders: scene clearance≥0.10 and BODY clearance≥0.70 for all24 origins. Offset0.15 is the minimum tested offset satisfying both. These are sparse-point distances in prepared-world units, not a physical collision certificate. [Exact pose/K checks](/fsx/tmp/marshawn_001660_hd/review_path_inner_standoff/independent_pose_contract_check.json), [clearance audit](/fsx/tmp/marshawn_001660_hd/review_path_inner_standoff/clearance_audit.json), [selection before images](/fsx/tmp/marshawn_001660_hd/review_path_inner_standoff/selected_before_render.json).

Occupancy rendered the selected inner0.15 path with radius0 learned weights and the separately declared `witness_union_halo1` inference mask: actual exported GPU core UNION9022 witness cells, then exactly one radius1 guard. I launched no render or additional training and changed no model/grid state. [All24 actual outputs](/fsx/tmp/marshawn_001660_hd/final_pair/release_audit/inner015_witness_candidate/contact24.jpg) remove the broad former03/06/21 veils. Root separately reviewed all three640px page sheets and passed initial visual quality with remaining detail/periphery softness.

## Insights

Do not infer false occupied air from a foggy novel image without locating its actual opacity contributions and checking calibrated source geometry. Here the problematic samples lie near real luminous hardware, and original interpolated camera origins approach that hardware closely. The original trajectories remain explicit stress tests; the new inner path has a declared, independently verified clearance rule.

The inner-path result does not prove an ideal global grid, physical visibility of every projected point, or hidden-volume correctness. Witness union prevents known sampled surface-cell misses; guard cells remain conservative traversal admission. Global atlas review is owned by root/recipes and remains distinct from the physical source/projection audit.
