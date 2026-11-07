# Frozen semantic occupancy R0/+1 replay preparation

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

**Preparation only; no CUDA replay or new render quality result.** The 298 exact file bindings include qualified P3 source/observer/stable exclusive FP64-reference helpers, candidate arrays, head, source provenance, sealed130 rays, P3 physical references and fixed V1 evaluation inputs. The immutable source loader independently checks its full scientific branch and seven serialized source states.

CPU tests: **53 passed**, covering this replay (21), the node-cell bridge and candidate exporter. Tests include binary-only swaps and restoration before source assertions on exceptions, non-binary drift rejection, eval-dilation/admission rejection, axis/boundary handling, private diagnostic RNG, omitted dense selected-opacity mass, revealed background recomposition, immutable source/head plan rejection, and false-release refusal. These are not CUDA or complete physical-geometry certificates.

Proposed GPU scope: local physical GPU0 UUID `GPU-7ed25c4d-10d1-cbe3-d0b2-76c21211bd46`, exclusive shared lease, 8GiB allocator / 10GiB process limit, 3600s deadline, 5s monitoring. The new controller does not launch without the exact root release and refuses duplicate output attempts. Signals terminate only its owned worker group. No GPU has been acquired by this package.

The planned execution performs:

1. Qualified source seven-state restoration and original full3FHD parity; fixed head restoration.
2. Actual CUDA head-envelope checks at109 source observations, uniform points, semantic-node planes/native-cell planes and adjacent FP32 values.
3. Source-grid physical RGB/alpha and V1 parity against P3, followed by full3FHD semanticR0 and freshR1 renders. Only `occupancy_grid.binaries` is temporarily replaced. Original binaries are restored **before** every full source/RNG/LUT/FR check. All other pipeline tensors remain exact.
4. Native and independently queried uniform8192/16384 **selected-density** profiles on103 clothing,6 shoe,12 DC and9 contour rays. Dense references explicitly evaluate native density and the actual head at each uniform position, then recompute optical weights. Native sampling counts, cap hits, tail extent, interval sizes, dense convergence and selected-opacity mass omitted by each grid/actual intervals are retained. Sampled positive selected density outside the conservative head envelope fails the gate. Native sample centers outside active cells are separately flagged.
5. CPU-only fixed V1 PSNR/SSIM/LPIPS, full-frame PNGs, and GT/source/R0/R1 body, hair, hands and feet contact pages. CPU metrics run after the GPU lease releases. Source V1 numeric parity is mandatory; candidate metrics may differ because traversal changes quadrature and allocation.

## Insights and next steps

Parent/root must review the exact plan and publish its release before execution. `release.template.json` remains false. This experiment can establish observed consistency of a **chosen head selector** with an actual traversal grid; it cannot certify physically empty space, complete hair/foot coverage or correctness of the original learned core. The109 observations contain repeated views of sparse geometry, including the documented5568 hair/chest ambiguity. Scene-coordinate units are not certified metres.

The previous P3 numerical tolerance remains2e-5, with existing savedRGB8/alpha16 parity limits and unchanged fixed V1. A support proof cannot repair raw density omitted by the base learned grid; R1 may expose that density. Numerical/reference failures will be preserved rather than fixed by changing tolerance, sigma or the head during this comparison. No automatic quality promotion is implemented.

## Actual runtime qualification and isolated low-cell inventory

V2 ran for approximately81s and exited before the first grid override. Original three-view native source parity, exact seven-state restoration and the actual26733-point CUDA envelope test passed; zero sampled positive selected-density or positive head-membership points escaped the conservative envelope. All109 source observations were retained by source/R0/R1 at their actual cells. Peak observed GPU use was approximately2.6GiB, below the10GiB limit.

The failure was a bookkeeping assumption: the actual52-tensor pipeline registers the same active buffer as both `_model.occupancy_grid.binaries` and `_model.adaptive_sampler.occupancy_grid.binaries`. CPU checkpoint inspection proved identical storage, offset, stride and values. V3 requires exactly those two keys, the identical grid object and shared storage; both aliases must equal the temporary replacement, and only those exact keys are excluded from the non-binary digest. Unrelated grid-like tensors remain protected. The worker now saves this actual alias proof. Twenty-four focused CPU tests passed, including new negative cases for distinct grid objects, cloned alias storage and unrelated-state mutation. No rendering math, support, threshold or tolerance changed.

Root released v3 with planSHA `685b7970870d5b6e5a0a1539dcf57f0d9e01fc4306af8d3a82e7bd3d910c8f22`, releaseSHA `2373ee800083cf34e2fb491c9a2724a61f5db55263e0e533ca48678c91a0c62d`. Controller2194778 was launched once and initially queued behind the independent four-UNKNOWN physical render. This replay continues to use the original two-UNKNOWN head. Actual R0/R1 image results remain pending.

The independent [CPU component inventory](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_cell_candidates_cpu_v1/component_floor_inventory_v1/report.json) provides all6/26-connected components and exact low-cell coordinates without changing any grid:

| Grid | Cells in Z-index0/1 | Cells entirely below Z−.02 | Isolated low26-connected component |
|---|---:|---:|---|
| Original |45|605|See full coordinate inventory|
| SemanticR0 |7|7|7 cells; index[70,89,0] through[72,91,1]|
| SemanticR1 |34|51|51 cells; index[69,89,0] through[73,92,2]|

R0 island bounds are `[-.298594,.228047,−.070000]` to `[-.275391,.242812,−.059062]`; R1 extends the same7 original core cells to `[-.306328,.228047,−.070000]` through `[-.267656,.247734,−.053594]`. No109 observed actor point falls in this island; the lowest observed sole point is Z=.005085. CPU head-center probabilities reach .952743 there, so part of this island passes the physical selector rather than surviving only because of the conservative node-basis envelope.

The14 photo-qualified floor witnesses have maximum fit residual.004637 in these scene coordinates. The island lies substantially below that fitted plane, but this residual is not a spatial floor uncertainty bound, and local floor coverings may warp. The CUDA support receipt did not save localized raw sigma for these island cells, so its global sigma maximum cannot establish their optical importance. Sparse sole points do not certify all feet. Actual selected-density contributions and source-image projections are required before any removal; no physical empty-space or safe-carving claim is made.
