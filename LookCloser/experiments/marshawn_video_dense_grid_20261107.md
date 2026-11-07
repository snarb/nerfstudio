# Marshawn video: source admission and learned-grid edge audit

## What was tested

Keep the accepted common three-second AABB and selected cold-training recipe.
Check whether positive actor contours have any retained cell along their camera
rays, using exact voxel traversal rather than render-alpha heuristics. Source
admission and learned occupancy are different masks and are audited separately.
A retained cell on a ray does not establish the true surface depth or certify
all hidden geometry.

The read-only dense helper is committed in `076a1cf9`. Its exact DDA traversal
passed 250 random comparisons with exhaustive cell/slab intersection. It binds
the source RGB/masks, poses, AABB and actual exported grid snapshots. Native
canary rays use the source camera's calibrated inverse distortion; forward
roundtrip error was below 2e-12 native pixels.

## Results

### Frame 1675: genuine source-label failure, before field publication

The initial sparse witness found 55 failures in 918 rays. Every failure was
the black actor's topmost flying braid; body, feet and grey actor samples passed.
Exact traversal confirms an inside-box ray, with no transform/indexing problem.
For the best cells, 58 cameras support foreground but AB/AC/AD/BC/BD incorrectly
label the visible braid tips as known background. Those tips are 127–218 native
pixels from the existing nominal masks, beyond the 64-pixel RGB falloff; their
W64 and both UNKNOWN masks are zero. The classifier follows incorrect inputs.

| Original 1675 mask | Boundary rays | Rays with no retained cell |
| --- | ---: | ---: |
| Source base | 162757 | 6238 |
| Source exact one-cell halo | 162757 | 4701 |
| Frozen mapped source support | 162757 | 2867 |

All mapped failures are black-actor boundaries in the same 55 views; grey has
zero. Additional native contours from the five fresh hair canaries also expose
the missing support. The weak AD seed is not anatomical ground truth.

No 1675 production model was published. The old source release and all
transaction/grid/ready proofs were archived on both hosts; only the input
transaction's CPU controllers were retired. Original checkpoint and field
processes were untouched. The common AABB remains unchanged.

Root manually reviewed all five current native candidate contacts. Qualified
additions are AB blond-dreadlocks14, AC dreadlocks16, BC dreadlocks29 and BD
dreadlocks26. AD's weak dreadlocks7 is permitted only as a geometry-uncertainty
seed. Even the strong NN candidates miss the witness by 11–46 native pixels,
so foreground additions alone do not establish safe free-space labels.

The correction adds a separate regulated 128-native-pixel geometry UNKNOWN
envelope around current anchored hair/head support. This does not force density,
RGB supervision or occupancy; it prevents confident background carving near
unresolved hair. Original RGB/W64/UNKNOWN arrays remain separate. All 66 cameras
receive current native head/hair queries; previous tracking proposes a search
region only. The new source revision requires its own full grid and contour
audit before publication. Corrected-grid results are pending.

Evidence: original source campaign
`operations/witness1675_diagnostic/{HC_ray_trace.json,five_clear_camera_exact_margins.json,dense_all66/}`;
native selection `operations/1675_topbraid_ROI_v6/root_five_native_candidate_selection.json`.

### Completed frame 1670: actual learned occupancy, checkpoint 139999

The checkpoint SHA is
`ccd0da52ad069beaaf93c7ebd3e9b79343242b0d7d0f7e6ce63e9c332cc42015`;
training completed exactly 140000 iterations. All 168805 nominal actor-boundary
rays were checked against the saved real grid, without loading or modifying the
NeRF during CPU traversal.

| Mask | Retained cells | Boundary-ray failures |
| --- | ---: | ---: |
| Source base | — | 302 |
| Source halo1 / mapped support | — | 0 |
| Runtime semantic admission | 441304 | 0 |
| Saved learned core / active R0 | 37821 | 3 |
| Fresh runtime R1, clipped to same admission | 92460 | 0 |

The three R0 failures are ND black pixels (1313,535), (1314,536), (1315,536).
Current native NN labels say black=true and UNKNOWN=false. Their largest saved
occupancy EMA is .00936276, below the .01 core threshold. Root inspected actual
native GT and matched R0/R1 crops: R0 has stepped black holes at the coat edge;
R1 partially fills them. Alpha increases from zero to .0910/.1364/.1292. This
is partial coverage, not proof of opaque or physically correct depth.

All registered binary-grid aliases were restored after the render. Pipeline
tensor SHA before/after is identical, and restored R0 HD RGB is byte-identical.
The original checkpoint, config and learned EMA were never changed.

| Matched in-sample DC/HC/KA actor monitoring | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| R0 | 46.12168 | 0.991924 | 0.00094554 |
| Full R1 | 45.86622 | 0.991849 | 0.00097706 |

All six actor/view LPIPS values worsen slightly under blanket R1. Root reviewed
the actual V4 record2 full 4K R1 image: floor fragments and the hair shell remain.
The extra support does not justify global promotion on these measurements.

A narrower render-only candidate keeps every R0 cell and additionally admits
saved EMA >= .009 inside unchanged semantic admission. It uses 39278 cells
(1457 extra, +3.85%) and restores 2/1/1 cells on the three failed rays. The R0
superset property proves zero failures on the same observed boundary-ray set.
Matched actual render/metric evaluation is pending; no training setting changed.

Evidence: capped campaign `operations/contour_runtime_001670/`, including
`dense/report.json`, `ND_three_ray_proof.json`, `render_RGBA_root_replay/`,
`R1_full4K_and_HD_metrics/`, and `root_EMA_threshold_candidate_counts.json`.

## Insights

The image-derived one-cell halo prevents real edge omissions in the tested
source admission. It cannot repair confident background labels caused by a
mask missing an entire braid tip. Native head/hair uncertainty must be handled
before carving. The learned grid can independently omit low-EMA edges after
source admission passes. Blanket runtime dilation restores those edges but
currently worsens image metrics, motivating a smaller source-checked correction.
None of these tests certifies every empty or occupied cell in unseen geometry.
