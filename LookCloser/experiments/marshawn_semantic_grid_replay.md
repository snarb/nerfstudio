# Frozen semantic occupancy R0/+1 replay preparation

## What was tested

This package prepares a **read-only, separately released** comparison of native traversal grids for the fixed quarter-step / distortion .1 source at step126316 (checkpoint `340e794…cd641`) and the completed **two-UNKNOWN direct-LC p01 head** `bdb2cc0…aa98bc`. The physical field is always native density multiplied by the same hard `p>=.1` selector, with transmittance recomputed. No training, field default, masks, source occupancy or head values are changed.

The executable proposal is [plan.json](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_grid_replay_v2/plan.json). The earlier v1 directory is an unlaunched preparation snapshot; v2 includes additional sampled-support checks and the source candidate formula verification. Neither has a GPU release.

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
