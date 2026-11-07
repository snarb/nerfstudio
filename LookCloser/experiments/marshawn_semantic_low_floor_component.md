# Isolated semantic support below the fitted floor

## What was tested

Root personally inspected all384 X/Y/Z slices of the p=.01, two-UNKNOWN128³, hard≥.1 occupancy candidate. The major floor cloud is removed, but a seven-cell26-connected component remains near the lower AABB face. Its bounds are approximately `[-.298594,.228047,-.070000]..[-.275391,.242812,-.059062]`. R1 expands the same component to51 cells. The accepted AABB remains unchanged.

The strongest retained cell center is `[-.286992,.240352,-.061797]`, with actual CPU head probability.952743. The read-only support audit restricts analysis to that point's trilinear basis and the immutable native LC TRAIN contribution cache. It assigns no new labels or physical free-space truth.

## Results

| TRAIN query | Class | Shared-node native-visible exposure |
|---|---|---:|
| ND66502, pixel[1402,809] | Actor |.01654833|
| MC61101, pixel[1384,675] | Nonactor |.000023119|
| ND66665, pixel[1399,806] | Actor |.000017961|
| ND66496, pixel[1400,805] | Actor |.000001688|
| MD61499, pixel[1344,992] | Actor |.00000000287|

These are basis-weighted source visibility contributions, **not** whole-ray alpha, optimization gradients or direct membership measurements at the point. Root inspected the positive source crops: ND observations are on the black actor's shiny boot/sole and should retain their actor labels. They are unlike the separately qualified floor-bag annotation errors.

[Actual source support and hashes](/fsx/tmp/marshawn_001660_hd/tuning_20261106/low_island_source_support_CPU_v1/support.json), [positive source crops](/fsx/tmp/marshawn_001660_hd/tuning_20261106/low_island_source_support_CPU_v1/positive_TRAIN_support.png), [all384-slice root review](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_cell_candidates_cpu_v1/prior001_fresh2064/hard_theta01/root_all384_slice_review.json), [connected-component inventory](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_cell_candidates_cpu_v1/component_floor_inventory_v1/report.json).

The audit completes with all69,888 query identities unchanged, all source hashes verified and the head unchanged. CUDA remains uninitialized. It does not query native sigma at these cell centers; selected opacity and exact depth responsibility require the separate GPU trace.

## Insights

Correct image class labels can propagate actor membership to a deep or misplaced density tail. Relabeling a valid boot pixel would conceal the geometry issue. More semantic fitting alone therefore cannot certify an ideal occupancy grid. Investigate exact ray contributions and explicitly separated depth-responsibility or floor-prior experiments; do not silently carve the component, raise the fixed AABB, or treat sparse positive observations as a whole-scene certificate.
