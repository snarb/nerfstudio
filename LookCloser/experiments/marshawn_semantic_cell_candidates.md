# Direct-LC semantic head: separate CPU occupancy candidates

## What was tested

The completed independent LC heads with initial class priors `.99` and `.01` were exported through the conservative [node-to-cell bridge](marshawn_semantic_node_cell_bridge.md). Both have 2064 actual head Adam updates and the same immutable native source126316, checkpoint `340e794e…cd641`, full-pipeline digest `c83bc7db…d46636`, capture-v4 `fb8b875d…9073d`, and identity floor transform. The `.99` head SHA is `f2eb8982169728556fe956fdfaac313f47a52d2aeb628ff0f0cbb122dae8a341`; `.01` is `bdb2cc0d8138e4c22c61b4d411227b272b95ceac9e8295d474eecc4a84aa98bc`.

Only CPU candidate arrays and diagnostics were written. No native model, renderer, density hook, optimizer or GPU was used; source occupancy/EMA, AABB, colours, masks and labels stayed unchanged. Candidate selection is opt-in and no candidate is promoted.

For an envelope `E` from trilinear **logit nodes**, original learned core `C`, and original admission `A`, the primary order is:

`candidate_R = fresh_dilate(C, R) & A & E`, for native-cell radius 0 or 1.

The separately recorded alternative is `fresh_dilate(C & A & E, 1) & A & E`. It is narrower and is not silently substituted. The source's actual cascade bounds and 128-cell spacing are preserved; semantic128 nodes have 127 intervals. Explicit coordinate slack `2e-7` and arithmetic logit slack `1e-5` remain separate. Hard thresholds include equality; smoothstep support is strictly above `.1`, with conservative arithmetic slack near equality. These are scene units, not certified metres.

## Results

The unmodified source has 32,046 active R0 cells; fresh source R1 before semantic filtering has 64,268, under unchanged 131,911-cell admission.

| Head prior | Physical selector support | Semantic envelope | Candidate R0 | Candidate R1 | Filter-first R1 |
|---:|---|---:|---:|---:|---:|
| .99 | hard p≥.1 | 2,096,083 | 31,056 | 63,203 | 63,199 |
| .99 | hard p≥.5 | 2,093,610 | 28,944 | 60,773 | 60,668 |
| .99 | smoothstep(.1,.5), p>.1 | 2,096,083 | 31,056 | 63,203 | 63,199 |
| .01 | hard p≥.1 | 45,644 | 23,730 | 38,319 | 38,111 |
| .01 | hard p≥.5 | 42,128 | 22,979 | 36,495 | 36,334 |
| .01 | smoothstep(.1,.5), p>.1 | 45,644 | 23,730 | 38,319 | 38,111 |

All **109 observations / 41 unique XYZ points** retain CPU-reference class support and candidate R0/R1 cells for every variant. The weakest `.01` point is black sole6862 at **p=.51107014**, repeated in JA/JB/KB, versus `.99266633` for the `.99` head. This narrowly exceeds the `.5` threshold and does **not** certify the complete sole, foot, hair or adjacent contours. The next weakest unique point is grey jacket8492 at `.87750262`. No threshold or source target was adjusted in response.

Each candidate has all 384 XYZ slices on six contact pages, with exact per-slice bounds/counts in the receipt. Cyan denotes retained R0, gold additional R1 traversal, magenta original-active cells removed by R0. Blank is not a certified physical-free label. The inspected [lower-Z .99 page](assets/marshawn_semantic_cell_candidates_20261107/prior099_fresh2064/hard_theta05/lower_Z_slices.png) and [.01 page](assets/marshawn_semantic_cell_candidates_20261107/prior001_fresh2064/hard_theta05/lower_Z_slices.png) show stronger removal of low islands for `.01`; no source-photo or physical-density conclusion follows from those colours. Every full page is available in the campaign directory below. The hard-.1 and smooth support arrays happen to be identical for these two fitted heads; their **density multipliers are still different**.

[Comparison](assets/marshawn_semantic_cell_candidates_20261107/comparison.json), all109 per-candidate records, exact input bindings, slice inventories and [array-loader checks](assets/marshawn_semantic_cell_candidates_20261107/actual_candidate_API_proof.json) are archived. Thirty-two CPU tests pass. They cover wrong source/capture/frame/spec/labels, technical checkpoints, threshold choices, metadata tampering, exact logit interpolation, noncumulative dilation/order, unchanged source arrays, read-only loading and complete384-slice enumeration. The actual technical64 checkpoint was independently rejected. CUDA stayed uninitialized. An independent actual-head CPU FP32 `grid_sample` check on all109 points agrees with the FP64 reference at every declared threshold: maximum probability differences are 1.46e-7 (.99) and 2.75e-6 (.01). The latter sole probability is .51106739 in CPU FP32. This is not a GPU arithmetic certificate.

## Artifacts and API

Campaign root:
`/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_cell_candidates_cpu_v1`

Each `{prior099_fresh2064,prior001_fresh2064}/{hard_theta01,hard_theta05,exactzero_smoothstep01_05}` directory contains `candidate.npz`, `metadata.json`, `witnesses.json`, and `slices/`. The NPZ preserves original core/active/admission alongside independent candidates; EMA remains untouched in the hash-bound source grid. Completed head files remain in the immutable `frozen_LC_semantic_head_archives/recovery1_20261107_v1` archive.

`export_marshawn_semantic_cell_candidates_cpu.py` first seals a JSON role→path list with `--seal-paths`, then validates a sealed binding by default. Only `--execute` writes new candidate arrays to a new output directory. Required roles are head, completion, spec, contract, original capture, adapter, labels, alias equivalence, source grid JSON/NPZ, and the exact clothing103/shoe6 query files. Missing/incomplete heads fail rather than being assumed ready. Example once the binding is sealed:

```bash
python scripts/export_marshawn_semantic_cell_candidates_cpu.py \
  --binding /path/prior099_fresh2064_binding.json --output /path/new_candidate_generation \
  --selector hard_theta01 --selector hard_theta05 \
  --selector exactzero_smoothstep01_05 --execute
python scripts/render_marshawn_semantic_cell_slices_cpu.py \
  --candidate /path/new_candidate_generation/hard_theta05 --output /path/new_slice_pages
```

The future replay API `load_candidate(...)` requires exact metadata, head and source-grid hashes, explicit selector and radius. It returns read-only arrays plus the density-selector contract, **never installs or changes φ**. Future replay must install the exact bound physical selector separately and recompute transmittance.

## Limits before integration

The node bound certifies the chosen selector's mathematical support under declared arithmetic slack, not actual physical emptiness. Actual CUDA interpolation and threshold arithmetic still need a bound/check. A semantic envelope cannot repair raw density omitted by the learned base grid. Fresh R1 can expose untrained/skipped density and alter stepping phase/tail coverage; +1 cannot restore actor density already removed by the class head. Both require actual native/dense rays and full renders before integration.

The separate physical observer's independent FP64 prefix check is outside this CPU export and is not assumed fixed or passed. No PSNR/SSIM/LPIPS, optical-alpha correctness or artifact-free asset claim is inferred from these class probabilities or cell counts. Sparse point retention is not a replacement for hair/feet/contour inspection, especially the `.511` sole point.
