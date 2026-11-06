# Immutable native20k semantic-gate reference

What was tested: Read-only rendering on dev3 of checkpoint `fd56e541d91abde26b776ffe03b91caf997b3c0ada0044cebf6407728c988fa2`. An independent 128³ gate uses the unchanged accepted floor AABB. A cell is rejected only when its entire outward projected eight-corner rectangle is clear of padded actor support, invalid pixels and occluder UNKNOWN in at least two of 63 TRAIN cameras. Native field coordinates, hash normalization, original R1 occupancy, weights and sample intervals remain unchanged. The gate only zeros density at original sample midpoints.

Results: **The earlier W64 artifact is visually cleaner but has a demonstrated source-visibility defect. The partial correction below preserves the measured toe feature while restoring substantial floor/mat geometry. Neither is an artifact-free release or a matched LookCloser winner.** Worker/mat geometry is removed, but floor patches remain around shoes, including in ±12° HC orbit views. Hair, clothing folds and hand contact remain recognizable. All three views and hair/hand/feet crops were visually inspected.

| Variant | PSNR v1 | SSIM v1 | LPIPS v1 | PSNR v2 | SSIM v2 | LPIPS v2 |
|---|---:|---:|---:|---:|---:|---:|
| Native accepted-box clip |30.14774|0.827527|0.054467|30.14977|0.827641|0.054283|
| Corrected semantic gate0, diagnostic |28.50480|0.816661|0.065449|28.58366|0.817608|0.065308|
| Corrected semantic pad1, preferred provisional |29.03658|0.819663|0.064609|29.08238|0.820104|0.064405|
| Pad1 +136 contributor cells, diagnostic |29.08389|0.819916|0.064393|29.12983|0.820358|0.064188|
| Blind gate0 witness repair +padding, rejected |29.38830|0.821973|0.062646|29.38748|0.822065|0.062453|

Both versions use original fixed crops, full-HD float RGB, identical neutral 0.5 mask exterior, AlexNet LPIPS without resize and equal actors/views. EVAL-v2 independently removes 334 grey-labelled bag pixels in KA; HC/DC are unchanged. EVAL-v2 was used only for rescoring every variant, never gate construction or the main campaign selector.

TRAIN-only temporal proof from MC/NA identifies the stationary bag. Separate auxiliary corrections remove 6,522 confirmed bag positives and withhold 3,032 uncertain contact positives in six TRAIN views. Another 97 bag-tip positives in IA/LA/LB were source-reviewed separately. Original campaign masks/data are untouched. Exact RAW 64-pixel support was regenerated, with contact uncertainty remaining UNKNOWN. The corrected binary gate happens to equal its original counterpart: 37,647 base cells, 50,790 after exactly one semantic padding. This padding is distinct from the unchanged historical native R1 occupancy.

The preferred pad1 gate misses 31 of 4,657,323 qualified TRAIN actor rays at tiny black shoe/contact edges. This is a ray-intersection audit, not retained object volume. Blindly retaining native contributors restores floor fragments and is rejected. The 136-cell alternative eliminates those ray misses but has median 37 whole-footprint TRAIN background contradictions. One independent native-RAW shoe feature triangulates from HE/IC/ID at floor Z=0.01457 (maximum reprojection 0.314 HD pixels); only 19/136 cells are within 0.02 of that feature. One toe-upper point cannot certify an entire sole, so this repair remains diagnostic.

The native field places 99.99896% of termination mass within ±0.02 of the 28 observed TRAIN shirt depths; independent 4096/8192 sampling agrees. **All 28 are known native training-supervision labels.** This is a supervised fit diagnostic: the historical native run used SfM free-space/surface-band and MVS supervision, unlike the current LookCloser baseline. Its result cannot isolate an architecture effect or establish held-out geometry.

The original red-box overlay incorrectly used removed full-scene floor depth to hide lines on the isolated image. The preferred export now uses exact first-0.5-opacity depth from the current semantic render. All eight corner transforms and all twelve near/viewport-clipped edges were independently checked; worst projection-route disagreement is 1.82e-12 pixels. The accepted box extends outside the original HC raster; a labelled half-focal virtual HC view shows all eight corners. Clean HC RGB bytes were preserved.

- [Full-HD HC](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/corrected_v2/export/HC_inside.png)
- [HC with audited red dashed box](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/corrected_v2/export/HC_accepted_box_red_dashed.png)
- [Complete box in virtual wide HC](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/corrected_v2/export/HC_wide_accepted_box_depth.png)
- [Three-view comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/corrected_v2/review/final_three_view_comparison.jpg)
- [Hair, hand and foot review](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/corrected_v2/review/detail_review_sources_vs_pad1.jpg)
- [HC orbit](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/corrected_v2/export/HC_orbit_contact.jpg)
- [Download package](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/corrected_v2/native20k_semantic_w64_review.zip)
- [Reusable renderer](/home/pavlo/repos/nerfstudio/LookCloser/scripts/render_marshawn_native_semantic_gate.py)

Insights: Semantic gating removes most unrelated scene geometry without optimization. Padding protects uncertain boundaries but also retains floor density. Native contributor weights alone cannot identify real shoe geometry. The completed 32/16 native-pixel factor below rejects globally smaller margins. Keep TRAIN identity qualification separate from EVAL annotations; inspect hair, shoe edges and remaining floor in every candidate.


What was tested next: The same corrected TRAIN nominal masks and UNKNOWN handling, 128³ gate, whole-cell rule and one semantic padding, with native-pixel support widths 32 and 16. No contributor exceptions, new labels, EVAL-based fitting, optimization or checkpoint/grid mutation. The W64 download package remains unchanged.

| Support width +one semantic padding | Admitted cells | TRAIN ray misses /4,657,323 | PSNR v1 | SSIM v1 | LPIPS v1 | PSNR v2 | SSIM v2 | LPIPS v2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W64, retained provisional reference |50,790|31|29.03658|0.819663|0.064609|29.08238|0.820104|0.064405|
| W32, rejected replacement |41,332|3,927|28.59317|0.817277|0.065553|28.65233|0.817959|0.065259|
| W16, rejected replacement |36,785|9,097|28.41552|0.816244|0.065777|28.48500|0.817161|0.065576|

All three hair, hands and feet comparisons were visually reviewed. Smaller margins reduce floor-patch extent but visibly truncate black shoe toes/soles; floor islands still remain around grey shoes. Hair and upper-body hand contact remain visually similar. W32 misses comprise 3,365 black and 562 grey mask pixels; W16 has 8,086 black and 1,011 grey. Neither has misses in the upper third of actor image height. Source crops FD/HE/JB/JC/MA/IE show actual toe/sole edges among the misses; IA grey also includes residual bag-tip label ambiguity. Image-height bands alone are not anatomical certification, and not every mask-positive miss is a real actor pixel. Previously withheld bag/contact positives remain excluded and UNKNOWN; no further mask edits were made for these factors.

The direct preferred-W64 ray audit now explicitly binds `candidate_grid.npz` and `semantic_gate_pad1`, confirming 31 misses on all 4,657,323 finally qualified rays. The older `final_train_ray_receipt.json` refers to the rejected contributor-union diagnostic despite its legacy total-key name; it must not be cited as zero misses for the preferred W64 gate. Margin receipt `semantic_gate_names` and saved raw arrays contain no contributor union. A copied historical `protected_gate_rule` prose field in early margin renders was stale; a separate errata receipt records this without changing rendered bytes or original receipts. The reusable renderer now describes optional contributor arrays as diagnostics and verifies both gate coordinate frames.

- [W32/W16 metrics, original fixed crops, v1/v2](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/margin_comparison_metrics_v1_v2.json)
- [All source/native/W64/W32/W16 detail crops](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/margin_comparison/all_details.jpg)
- [HC shoe comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/margin_comparison/HC_feet.png)
- [W32 TRAIN source miss regions](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/margin_W32/train_miss_region_summary.json)
- [W16 TRAIN source miss regions](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/margin_W16/train_miss_region_summary.json)
- [Explicit preferred W64 audit](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/corrected_v2/preferred_train_ray_audit/final_train_ray_receipt.json)

Validation: All six margin renders retained checkpoint/model-state hashes and passed native compositing parity. The exact voxel-plane ray audit crosschecked 40 synthetic rays against the independent CPU DDA. Two packed-depth regression checks verify transparent/missing rays and that removing foreground moves box occlusion to the retained surface. Compilation passed; the checks ran directly in the historical dev3 environment because pytest is not installed there. The unchanged W64 ZIP remains the provisional deliverable, not an artifact-free final result.

Independent source audit found a concrete occlusion-policy defect: the independently triangulated toe-upper feature falls in cell `[93,73,15]`; W32/W16 gate0 rejects it solely from DE/EE votes whose source projections land on a foreground mat hiding the shoe. Both padded gates still retain this particular point, and W64 mask padding suppresses those votes, but this does not make the clear-space rule correct. The audit also found 3,505 confirmed bag pixels permitted to vote clear despite the opaque bag; no independent hidden-shoe counterexample was located for that second vulnerability. Existing variants and the delivered W64 archive are preserved as provisional appearance filters. A separately versioned occlusion-aware correction is required before claiming a physically valid gate. [Independent source witness](assets/marshawn_native_grid_independent_20261106/witness_clear_votes.png), [projection votes](assets/marshawn_native_grid_independent_20261106/witness_votes.json).


What was corrected: `occlusion_v2` separately unions 60 conservative TRAIN mat envelopes and six independently qualified bag interiors/contact shells into UNKNOWN, preserving the existing three bag-tip UNKNOWN regions and actor labels. CE/JE/NE have unqualified mat visibility; their entire raster is UNKNOWN, so these cameras supply no clear votes. This adds 14,405,452 UNKNOWN source pixels at each width. The frozen original masks, W64 ZIP, accepted AABB and native field/grid remain unchanged. `occlusion_v1` is an unrendered intermediate diagnostic lacking the three-camera exclusion.

Independent verification rebuilt all 2,097,152 W64 cells and all 126 W64/W32 source archives with zero discrepancies. New gates are monotone supersets of their predecessors and use fresh, non-cumulative semantic padding. The independently triangulated toe feature `[93,73,15]` previously received DE+EE clear votes at W32/W16; both votes are now UNKNOWN and **gate0 retains the feature without padding at both widths**. This is one toe-upper feature, not a proof of complete sole geometry. See the [independent audit](marshawn_native_grid_independent_audit.md).

| Partial visibility correction | Cells | Qualified TRAIN ray misses | PSNR v2 | SSIM v2 | LPIPS v2 |
|---|---:|---:|---:|---:|---:|
| W64 gate0 |78,056|528 grey, 0 black|28.97942|0.819543|0.064489|
| W64 pad1 |105,049|0|29.17838|0.820786|0.063784|
| W32 gate0 |65,970|1,084 grey, 0 black|28.85687|0.818654|0.065074|
| W32 pad1 |91,703|237 grey, 0 black|29.03169|0.819814|0.064379|
| W16 gate0, grid-only diagnostic |60,303|1,428 grey, 4 black|—|—|—|
| W16 pad1, grid-only diagnostic |84,636|483 grey, 0 black|—|—|—|

The denominator remains 4,657,323 qualified TRAIN rays. Remaining grey misses concentrate in previously identified bag/contact views; they are not automatically classified as non-actor pixels. All three full actor comparisons and hair/hand/feet crops were viewed, plus full-HD HC and its current-depth box overlay. Actor detail remains similar; **large foreground mat and floor remnants return**, including a broad mat to the left of the actors in HC. Better masked actor scores do not establish a cleaner isolated scene.

For evaluation only, pixels with alpha >0.1 outside the EVAL-v2 actor union dilated by three HD pixels increase from 11,908 to 177,032 in DC, 11,928 to 85,622 in HC, and 10,944 to 57,048 in KA for old-W64-pad1 versus partial-W64-pad1. This measures mask-exterior opacity, not certified non-actor volume; actual full renders confirm the mat/floor tradeoff. All six new renders preserve checkpoint/model hashes and exact camera parameters from the independently audited box projection. Two additional source-occlusion regression checks pass without pytest: an opaque mat/bag prevents a clear vote behind it, while uncertain contact preserves actor support and original arrays.

**Remaining scope:** mat envelopes intentionally include neighboring floor and have no independent depth. Bag identity/extents are independently qualified only in six views plus three tips, not every TRAIN camera. Other opaque clutter may still generate invalid clear votes. This is a bounded, partial visibility correction; no gate here is certified physical occupancy. Independent depth-qualified occlusion and actor-surface classification are the next required distinctions for preserving feet without retaining the mat. Native contribution weights alone are insufficient depth truth.

- [Partial-v2 full-HD HC](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/occlusion_v2/W64/renders/HC/HC_semantic_gate_pad1.png)
- [Partial-v2 source/native/old-W64/new-W64/new-W32 details](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/occlusion_v2/review/all_details.jpg)
- [Metrics v1/v2](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/occlusion_v2/metrics_v1_v2.json)
- [Source-scope and visual-review receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/occlusion_v2/partial_scope_receipt.json)
- [W32/W16 toe regression before padding](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/occlusion_v2/toe_occlusion_regression.json)
- [Partial visibility-fix diagnostic bundle](/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000/occlusion_v2/native20k_partial_visibility_fix_review.zip)
