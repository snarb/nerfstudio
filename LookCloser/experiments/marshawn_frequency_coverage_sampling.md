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
