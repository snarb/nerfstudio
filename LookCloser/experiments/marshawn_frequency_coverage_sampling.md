# Frequency coverage and native integration diagnostics

## What was tested

CPU-only inspection uses the actual selected semanticR0@111316 checkpoint and its saved103 TRAIN clothing plus12 DC diagnostic ray arrays. Native sample weights are preserved exactly. No field query, optimizer update or occupancy reconstruction occurs. Frequency lookup uses the exact stored float32 frequency AABB/size and `(resolution-1)` indexing from the frozen FrequencyGridManager, not occupancy-cell bounds. Independent Torch/NumPy float32 indices agree.

## Results

None of115 native rays saturates the1024-sample cap. Clothing counts have median317/max843; DC median270/max517. Thus these native traces do not demonstrate legacy allocator far-tail truncation. Fresh halo1 saturates one clothing ray and all-cells saturates two; those artificial profiles cannot be substituted for actual native saturation.

All native traces contain a maximum interval of approximately.00625. Native versus dense16384 RGB maximum-channel differences have median.01052/p90.02817/max.04659 for clothing, and median.00740/max.05057 for DC.89/103 clothing and9/12 DC rays exceed one8-bit code; dense8192/16384 converges in all cases. All-cells versus dense clothing median difference is still.00781. This supports testing the integration step, while occupancy omissions also contribute. Dense probability after the native last interval is small: max.00180 clothing/.00273 DC.

The stored frequency grid contains only levels0/15. More importantly, original native rendered mass is not confined to level15:

| Domain | Level0 share of termination | Level0 share of witness±.02 mass |
|---|---:|---:|
|All103 clothing rays|51.27%|44.95%|
|Old18|72.37%|71.11%|
|New66|49.65%|42.19%|
|Old8 reserves|58.35%|54.74%|
|New9 patch reserves|7.44%|6.54%|
|DC12|40.62%|No depth truth|

These are unrenormalized original-weight partitions, with relative shares reported separately. Frequency at a refined sample position need not equal the level used at its parent coarse-interval midpoint. Level0 is minimum stored frequency, not evidence of physical emptiness. Frequency-gridSHA `81a1c6ec7866c4d18609260801d5c357b1b1a74fc83248714a27f75ad3cbe4d5` matches the independent cell-center inventory.

[Sampler report](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/selected_R0_sampling_review/report.json), [original-weight partition](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/selected_R0_frequency_weight_mass/report.json), [query/source provenance](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/selected_R0_frequency_weight_mass/source_provenance.json).

## Controlled follow-up

A read-only selectedR0 comparison changes only scratch inference coarse_step.00625→.003125, preserving the trained field, actual occupancy, sources and all other settings. It uses the same103+12 rays, dense convergence and explicit actual-import bindings, under an idle-exclusive GPU lease. This is not a training or quality promotion.

A separate proposed training factor is grid_update_batch_size2048→8192, keeping interval1024, global uniform TRAIN camera/pixel sampling, original maps/projection and all other recipe settings. It is not launched. The frozen parent has independent_rng_streams=false: the CPU prefetch barrier and larger image/y/x draws necessarily change subsequent global RNG/pixel trajectories. The proposal must report that consequence rather than silently enable another RNG mode or claim matched training rays.

Required scratch proof: actual global camera IDs must map to all and only the63 TRAIN cache entries; record coordinates, rendered-depth positions and retrieved frequencies; measure CPU/CUDA RNG deltas, peak GPU memory, finite outputs, new15-level coverage and the original native-weight frequency coverage. Parameters and actual learned core/EMA remain unchanged; only a scratch frequency grid may update before restoration. Legacy integer-coordinate projection and lack of opacity filtering remain unchanged for attribution, with the explicit risk that extra rays reinforce background or biased depth. A larger batch is not a substitute for validating source geometry.

## Read-only half-step and actual update results

The selectedR0 field was replayed at `.00625` and `.003125`. Dense16384 RGB is bit-identical across all115 rays, proving the field stayed fixed. Native cap saturation remains0/115. All-cells versus dense median RGB error falls from.007813 to.004354 on clothing and.006742 to.002324 on DC; native median depth error also falls. However DC12 mean GT maximum-channel error rises.049227→.052005: more accurate integration does not guarantee improved images from a field trained with coarse integration. [Exact replay](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/selected_R0_halfstep/complete.json).

A separate cold half-step arm therefore starts from seed42, changing only `adaptive_coarse_step_size=.003125`; legacy allocator, original semanticR0/W64, body80, .01 distortion, maps/projection/LR and stopping remain unchanged. The actual config comparison has one field difference. Warmup and ARM32 pause/resume pass with successful Adam16→32 and15→30, exact RNG restoration, finite six-view technical outputs, peak7952MiB. It launched on silverGPU3 under global locking, controller1479253/trainer1479267. Minimum106316 and unchanged plateau apply; established selection, raw LPIPS minimum and terminal checkpoints are preserved separately. First15188 audit uses actual trained coarse values without inference overrides. [Source/config/technical release](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_cold_halfstep_candidate_20261107/release.json). Technical validity is not quality approval.

The single-update2048/8192 scratch proof also passed: exact original fullstate/RNG restored per arm, all model/core/EMA state unchanged, only scratch frequency grid updated. It took20.3s with2108MiB process peak; initial compilation makes per-arm time incomparable. New level15 cells440→1652, but only115→450 receive any admission-qualified projected position. Most additions are not actor-supported;36→127 new cells also receive an outside-AABB projected position via legacy clamping. Original clothing level0 termination share improves only51.269%→51.172% with8192, versus no change with2048. The larger update consumes different global CPU/CUDA RNG. No8192 cold training is launched. [Actual update proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/selected_R0_coverage_update_proof/results/complete.json), [new-cell attribution](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/selected_R0_coverage_update_proof/new_cells_attribution.json).

## CPU proposal: actor-focused frequency updates at the original budget

The existing `dec5_frequency_sampler(camera_indices)` hook runs **after** the original global Torch camera/y/x draws. It can replace only pixel coordinates while preserving all2048 global TRAIN camera IDs. Under the current legacy projection, `support_valid` is ignored; merely adding this hook does not introduce the normalized-Jacobian opacity threshold. No coordinate, map, depth or projection correction is proposed.

The isolated CPU prototype uses1229 existing core pixels,410 existing protective-band pixels and409 exact original uniform pixels per2048 update. Camera assignments are untouched. Existing upper-body32px/lower-body16px proposal bands preserve a dedicated contour budget; the20% original uniform stream can still discover support outside these masks. This does not guarantee missing hair coverage. Original core/band includes the known bag falsepositives; no simultaneous bag correction is hidden in this factor.

Because the hook does not receive original y/x, `prepare()` shadows the exact historical Torch draws using a cloned CPU generator state. The unchanged pipeline consumes its original draws normally; the hook asserts camera order and resulting CPU RNG bytes before substitution. Additional permutation/pool draws use a private, serialized NumPy generator. No changes to the main body80 sampler, RGB weights/normalizer, FAS, occupied cells or positive geometry targets occur. The private generator state must be saved/restored alongside the existing runtime state, including the75941 historical seed reset policy; it must not be reseeded implicitly on resume.

Three CPU tests pass: original global RNG and uniform pixels are exact, private/global state reproduces the next update after resume, and changed camera ordering or an extra global draw fails closed. Across32 simulated updates using all63 hash-verified original TRAIN masks, core hits rise2531→39827 and protective-band hits1550→13428 out of65536 rays; no empty-pool fallback occurs. Camera histograms are identical by construction. No field, grid or GPU was touched. [CPU proof/source hashes](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/actor_frequency_CPU_proposal/report.json).

Source inspection suggests CUDA RNG consumption can also remain identical: native nerfacc stratification draws one `rand_like(near_planes)` per input ray before traversal, while the fixed warmup marcher is deterministic. Keeping2048 rays therefore preserves the apparent draw shape, unlike8192. This is an inference from code, **not yet an actual GPU RNG proof**. Before integration, compare original versus focused single updates from identical selected-model/RNG state, assert CPU and CUDA post-update RNG byte equality, exact camera IDs and original-uniform coordinates, unchanged non-frequency tensors, finite depth/output and actor-admitted new-cell attribution. Record source `nerfacc` path/hash. No speculative manual RNG advancement or duplicate baseline render should hide a mismatch; if actual CUDA states differ, report it and revise the causal design explicitly.

A viable isolated cold arm would retain originalR0/W64/.01distortion/.00625coarse/global2048/interval1024 and change only this frequency pixel proposal. It would first require actual warmup/ARM and private-state Adam pause/resume checks, plus full106316→unchanged plateau and matched15188 visual/geometry gates. Neither this proposal nor8192 is launched. Sparse reserves are diagnostic only, never sampling targets.
