# Frozen semantic occupancy R0/+1 replay

The completed frozen-field comparison favors **freshR1 as traversal for this specific p01/theta.1 head**, with substantially better coverage and a small V1 improvement. It does not produce an ideal occupancy grid or clean isolated asset: contour halos, shoe attachments and low fragments remain. All source/head states were restored unchanged.

## What was tested

This package prepares a **read-only, separately released** comparison of native traversal grids for the fixed quarter-step / distortion .1 source at step126316 (checkpoint `340e794…cd641`) and the completed **two-UNKNOWN direct-LC p01 head** `bdb2cc0…aa98bc`. The physical field is always native density multiplied by the same hard `p>=.1` selector, with transmittance recomputed. No training, field default, masks, source occupancy or head values are changed.

The active executable generation is [v3 plan.json](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_grid_replay_v3/plan.json), explicitly released by root. Earlier v1 is an unlaunched preparation snapshot; the v2 failed attempt is preserved. See the actual runtime status below.

| State | Active native cells | Construction |
|---|---:|---|
| source_grid | 32,046 | Actual saved source binaries |
| semantic_r0 | 23,730 | Actual learned core AND original admission AND conservative head support |
| semantic_r1 | 38,319 | Fresh 26-neighbor dilation of original learned core AND original admission AND conservative head support |

The 128³ semantic grid contains **nodes**, whereas native occupancy contains **cells**. The already exported bridge uses every trilinear logit node basis intersecting each actual cascade cell, with declared coordinate/numerical guards. No indexwise node-to-cell substitution occurs. R1 is computed afresh from the original learned core, not cumulatively from R0. Its additional traversal can expose density not constrained by original sampling.

## Results

**Actual replay and CPU evaluation completed successfully.** The preparation evidence below remains part of the record; final measured results follow. The 298 exact file bindings include qualified P3 source/observer/stable exclusive FP64-reference helpers, candidate arrays, head, source provenance, sealed130 rays, P3 physical references and fixed V1 evaluation inputs. The immutable source loader independently checks its full scientific branch and seven serialized source states.

CPU tests: **53 passed**, covering this replay (21), the node-cell bridge and candidate exporter. Tests include binary-only swaps and restoration before source assertions on exceptions, non-binary drift rejection, eval-dilation/admission rejection, axis/boundary handling, private diagnostic RNG, omitted dense selected-opacity mass, revealed background recomposition, immutable source/head plan rejection, and false-release refusal. These are not CUDA or complete physical-geometry certificates.

Proposed GPU scope: local physical GPU0 UUID `GPU-7ed25c4d-10d1-cbe3-d0b2-76c21211bd46`, exclusive shared lease, 8GiB allocator / 10GiB process limit, 3600s deadline, 5s monitoring. The new controller does not launch without the exact root release and refuses duplicate output attempts. Signals terminate only its owned worker group. V3 acquired the exclusive lease, completed in711.74s, peaked at2924MiB process GPU memory and released the GPU before CPU metrics.

The planned execution performs:

1. Qualified source seven-state restoration and original full3FHD parity; fixed head restoration.
2. Actual CUDA head-envelope checks at109 source observations, uniform points, semantic-node planes/native-cell planes and adjacent FP32 values.
3. Source-grid physical RGB/alpha and V1 parity against P3, followed by full3FHD semanticR0 and freshR1 renders. Only `occupancy_grid.binaries` is temporarily replaced. Original binaries are restored **before** every full source/RNG/LUT/FR check. All other pipeline tensors remain exact.
4. Native and independently queried uniform8192/16384 **selected-density** profiles on103 clothing,6 shoe,12 DC and9 contour rays. Dense references explicitly evaluate native density and the actual head at each uniform position, then recompute optical weights. Native sampling counts, cap hits, tail extent, interval sizes, dense convergence and selected-opacity mass omitted by each grid/actual intervals are retained. Sampled positive selected density outside the conservative head envelope fails the gate. Native sample centers outside active cells are separately flagged.
5. CPU-only fixed V1 PSNR/SSIM/LPIPS, full-frame PNGs, and GT/source/R0/R1 body, hair, hands and feet contact pages. CPU metrics run after the GPU lease releases. Source V1 numeric parity is mandatory; candidate metrics may differ because traversal changes quadrature and allocation.

## Insights and next steps

Root reviewed and released v3. The false template remains an unmodified example; the actual release and terminal receipts are bound in the final summary. This experiment can establish observed consistency of a **chosen head selector** with an actual traversal grid; it cannot certify physically empty space, complete hair/foot coverage or correctness of the original learned core. The109 observations contain repeated views of sparse geometry, including the documented5568 hair/chest ambiguity. Scene-coordinate units are not certified metres.

The previous P3 numerical tolerance remains2e-5, with existing savedRGB8/alpha16 parity limits and unchanged fixed V1. A support proof cannot repair raw density omitted by the base learned grid; R1 may expose that density. Numerical/reference failures will be preserved rather than fixed by changing tolerance, sigma or the head during this comparison. No automatic quality promotion is implemented.

## Actual runtime qualification and isolated low-cell inventory

V2 ran for approximately81s and exited before the first grid override. Original three-view native source parity, exact seven-state restoration and the actual26733-point CUDA envelope test passed; zero sampled positive selected-density or positive head-membership points escaped the conservative envelope. All109 source observations were retained by source/R0/R1 at their actual cells. Peak observed GPU use was approximately2.6GiB, below the10GiB limit.

The failure was a bookkeeping assumption: the actual52-tensor pipeline registers the same active buffer as both `_model.occupancy_grid.binaries` and `_model.adaptive_sampler.occupancy_grid.binaries`. CPU checkpoint inspection proved identical storage, offset, stride and values. V3 requires exactly those two keys, the identical grid object and shared storage; both aliases must equal the temporary replacement, and only those exact keys are excluded from the non-binary digest. Unrelated grid-like tensors remain protected. The worker now saves this actual alias proof. Twenty-four focused CPU tests passed, including new negative cases for distinct grid objects, cloned alias storage and unrelated-state mutation. No rendering math, support, threshold or tolerance changed.

Root released v3 with planSHA `685b7970870d5b6e5a0a1539dcf57f0d9e01fc4306af8d3a82e7bd3d910c8f22`, releaseSHA `2373ee800083cf34e2fb491c9a2724a61f5db55263e0e533ca48678c91a0c62d`. Controller2194778 was launched once and initially queued behind the independent four-UNKNOWN physical render. This replay continues to use the original two-UNKNOWN head. The subsequent R0/R1 replay completed successfully; final results are below.

The independent [CPU component inventory](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_cell_candidates_cpu_v1/component_floor_inventory_v1/report.json) provides all6/26-connected components and exact low-cell coordinates without changing any grid:

| Grid | Cells in Z-index0/1 | Cells entirely below Z−.02 | Isolated low26-connected component |
|---|---:|---:|---|
| Original |45|605|See full coordinate inventory|
| SemanticR0 |7|7|7 cells; index[70,89,0] through[72,91,1]|
| SemanticR1 |34|51|51 cells; index[69,89,0] through[73,92,2]|

R0 island bounds are `[-.298594,.228047,−.070000]` to `[-.275391,.242812,−.059062]`; R1 extends the same7 original core cells to `[-.306328,.228047,−.070000]` through `[-.267656,.247734,−.053594]`. No109 observed actor point falls in this island; the lowest observed sole point is Z=.005085. CPU head-center probabilities reach .952743 there, so part of this island passes the physical selector rather than surviving only because of the conservative node-basis envelope.

The14 photo-qualified floor witnesses have maximum fit residual.004637 in these scene coordinates. The island lies substantially below that fitted plane, but this residual is not a spatial floor uncertainty bound, and local floor coverings may warp. The CUDA support receipt did not save localized raw sigma for these island cells, so its global sigma maximum cannot establish their optical importance. Sparse sole points do not certify all feet. Actual selected-density contributions and source-image projections are required before any removal; no physical empty-space or safe-carving claim is made.

## Completed three-grid comparison

[Terminal proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_grid_replay_v3/replay/complete.json), [compact source-bound summary](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_grid_replay_v3/replay/final_summary.json), [fixed V1 metrics](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_grid_replay_v3/replay/visual_review/metrics.json), and [manual review with image hashes](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_grid_replay_v3/replay/manual_review.json). Source checkpoint, all seven serialized states, actual pipeline/core/EMA/admission, RNG, LUT, FR and head remained unchanged. Occupancy/frequency updates and optimizer steps were all zero. The actual CUDA support test retained all109 source observations and found no sampled positive selected density outside the conservative semantic envelope.

| Traversal | PSNR↑ | SSIM↑ | LPIPS↓ | Full3FHD samples | Native cap-hit pixels DC / HC / KA |
|---|---:|---:|---:|---:|---|
| Original source grid |30.887904|.828332|.02216182|124,082,045|0 / 0 / 630|
| SemanticR0 |30.887912|.828332|.02216197|81,783,120|0 / 0 / 436|
| Fresh semanticR1 |30.895068|.828455|.02207324|104,547,899|3 / 0 / 847|

R0 removes34.09% of actual full-frame samples while preserving major structure. On the130 sealed diagnostic rays, source-versus-R0 physical selected RGB differs by at most4.77e−7 despite removing up to152 samples/ray. Full images are not claimed bitwise equal: rare cap-affected pixels change and KA reaches the1024 limit. R1 improves aggregate LPIPS by about0.40%, largely in HC, but PSNR falls slightly in DC and KA. This is a small image-metric improvement, not evidence that the broad halos are solved. No timing speedup is claimed from sample counts.

| View | Source LPIPS | R0 LPIPS | R1 LPIPS | Source PSNR | R1 PSNR |
|---|---:|---:|---:|---:|---:|
| DC |.02304587|.02304587|.02302014|31.61718|31.59652|
| HC |.01953893|.01953893|.01932658|31.19286|31.24660|
| KA |.02390066|.02390111|.02387301|29.85368|29.84208|

Source selected/native RGB/alpha reproduced P3 within the existing saved-code limits, with worst floatRGBA error2.93e−6. The separate CPU V1 implementation reproduced all six P3 actor records within4.51e−6 for PSNR/SSIM/LPIPS. Fixed GT masks/crops, Alex LPIPS without resizing and identical RGB.5 exterior were unchanged.

### Dense selected-density reference and sampling limitations

Dense8192/16384 explicitly queried the same native field and head, then recomputed selected-density transmittance. The worst8192→16384 difference across130 rays was RGB.001456 and opacity.001221. These are finite quadrature references, not exact continuous integrals. All130 native diagnostic rays were uncapped: maximum sample counts were736 /608 /726 for source/R0/R1. Their favorable cap behavior cannot certify the full images, which contain the cap-hit pixels shown above.

The following entries are **mean fractions of unit incident selected-opacity mass**, expressed as percentages, not relative percentages of each ray's opacity. R0 matches source-grid omission on these130 rays because the additional pruned regions have zero selected density.

| Ray group | Dense selected mass outside source/R0 grid | Outside R1 grid | Outside actual source/R0 native intervals | Outside actual R1 native intervals |
|---|---:|---:|---:|---:|
| Clothing103 |1.24375%|.01120%|1.36165%|.01198%|
| Shoe6 |.78302%|.00407%|.76563%|.00499%|
| DC12 |.98693%|.00804%|1.12276%|.00647%|
| Contour9 |1.48871%|.01285%|2.21722%|.41421%|

The contour residual demonstrates that conservative cell support is not an exact integration guarantee. Native intervals/phase, finite reference resolution and sharp selected-density boundaries remain relevant. Positive selected native sample centers outside active cells were explicitly flagged; this does not contradict the semantic envelope bound, because the original learned-core restriction can omit selected density and interval placement can cross cell boundaries. No such measured head-support escape was found.

Better field coverage also exposes existing geometric errors. In the predeclared old18 clothing group, mean mass before the ±.02 witness band rises2.454%→3.403% under R1, close to the dense3.371%; band mass falls94.534%→93.633%. For old8 local reserves, early mass rises4.432%→5.494% (dense5.605%), and band mass falls93.937%→92.946%. New9 whole-patch reserves retain99.949% in-band (R0 99.961%; dense99.933%). Thus R1 integrates the frozen field more faithfully; it does **not** correct its early density. Historical group names come from earlier prefix experiments, not new fitting in this replay. The5568 hair/chest ambiguity and sparse-depth limitations remain.

### Visual result and remaining unwanted geometry

Both agents independently inspected all three body comparisons and all nine hair/hands/feet contact pages; this agent also inspected all six fullFHD R0/R1 outputs. There is no obvious new gross body, hair or shoe deletion relative to the selected source. Thin gray/beige contour halos, interleg fragments, shoe/toe attachments and sparse distant specks remain. This does not certify every hair or anti-aliased contour pixel.

- [DC body](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_grid_replay_v3/replay/visual_review/DC_body.png), [HC body](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_grid_replay_v3/replay/visual_review/HC_body.png), [KA body](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_grid_replay_v3/replay/visual_review/KA_body.png).
- [DC feet](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_grid_replay_v3/replay/visual_review/DC_feet.png), [HC feet](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_grid_replay_v3/replay/visual_review/HC_feet.png), [KA feet](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_grid_replay_v3/replay/visual_review/KA_feet.png).

Root's independent [low-island source-support audit](/fsx/tmp/marshawn_001660_hd/tuning_20261106/low_island_source_support_CPU_v1/support.json) traced the high-probability cell to valid black boot/sole pixels, especially ND66502. Root personally reviewed those source patches: they are legitimate actor positives. Their contribution follows an existing deep native-density tail below the fitted floor. They must not be relabeled UNKNOWN merely to hide this geometric error. A future below-floor prior would be an explicit, separately qualified geometry assumption, with foot/occlusion protection; it is not implied by this semantic-grid replay.

Root accepted R1 as the experimentally preferred traversal **under this frozen two-UNKNOWN p01 head**, while withholding final geometry/asset promotion. Source masks, AABB, head, trained field and live defaults remain untouched.
