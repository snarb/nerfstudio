# Mature focused model: native cap1024 versus2048

## What was tested

Immutable focused-frequency checkpoint75940, separately at FR1 and counterfactualFR.3. Actual native legacy sampler cap1024→2048→restored1024; coarse.00625, field weights, fixed semantic admission, core/EMA/active/frequency grid and all source state stay unchanged. No optimizer, grid refresh, live producer change or trained-cap claim.

## Results

| Fixed FR | Cap | PSNR | SSIM | LPIPS | Black LPIPS | Grey LPIPS | Native cloth cap hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| focus_FR1 | cap1024 | 31.360956 | 0.823754 | 0.02101198 | 0.02290023 | 0.01912374 | 7 /103 |
| focus_FR1 | cap2048 | 31.360908 | 0.823794 | 0.02102067 | 0.02290633 | 0.01913501 | 0 /103 |
| focus_FR03 | cap1024 | 31.350705 | 0.823860 | 0.02096377 | 0.02289562 | 0.01903192 | 7 /103 |
| focus_FR03 | cap2048 | 31.350660 | 0.823899 | 0.02097449 | 0.02290257 | 0.01904641 | 0 /103 |

Actual native cloth cap hits7→0 and maximum samples1024→1528. DC native max1011 has no cap hit in either phase. All-cells diagnostic cap hits30cloth/1DC also disappear; these are separate from actual native traversal. Counts refer to the115 declared rays, not an all-FHD cap census.

Native witness-band group changes are at most.0009 percentage points. The maximum dense weight beyond the native final interval remains.004798(FR1)/.003928(FR.3): this maximum is not caused by the seven capped rays. Whole-interval dense16384 arrays (positions,σ,RGB,weights) are bitwise unchanged between caps, proving that only traversal/integration changed.

All7serialized states, actual pipeline/global RNG/private focus state and feature LUT restore. Original1024 native arrays restore bitwise; all3FHD RGB/alpha and aggregate/view/actor/all6 metrics pass the existing strict numerical replay policy. Root policy limits and global exclusive GPU4 lease were respected; lease released after both arms.

## Visual inspection and interpretation

Personally inspected all6 GT/1024/2048/restored1024 body contacts (DC black, HC grey, KA black for both FR values). No visible cleanup or detail benefit; broad colored halos, floor/bag remnants remain. Both actors have slightly worse LPIPS with2048. These results do not support raising the live cap as the explanation or fix for current halo. They also do not establish training with2048 is equivalent; no such training was run.

Evidence: `C=/fsx/tmp/marshawn_001660_hd/tuning_20261106`; `C/independent_occupancy/focus_75940_cap2048_v1/{comparison.json,report.md,focus_FR*_comparison.jpg}`. Arrays and source/restoration receipts remain on silver.
