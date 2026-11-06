# Late distortion on the semantic actor field

## What was tested

The v2 diagnostic pair resumes the exact evaluated semantic R0/W64 checkpoint30376, including parameters, Adam/scheduler/scaler, RNG, sampler and actual occupancy state. It runs4000 attempts with distortion coefficient.01 or1, preserving native exp(h+1), corrected SH, body80, original confidence and coarse traversal admission. This is separate from the earlier full-objective fork60752 and from source-point/bag experiments.

The independent final34376 replay uses the same103 TRAIN clothing observations and12 posthoc DC RGB diagnostics, actual schema2 core/active/EMA/admission, native traversal, fresh native+1 and dense8192/16384 convergence. No optimizer or training callbacks execute. Root-released idle-exclusive GPU5 audit took180seconds,1758MiB peak process memory; the lease is released.

## Results

| Final34376 | PSNR ↑ | SSIM ↑ | Masked LPIPS ↓ |
|---|---:|---:|---:|
|Coefficient.01|31.51389|.844005|.02673186|
|Coefficient1|31.70173|.849399|.02909081|

Coefficient1 worsens LPIPS8.82% while improving PSNR/SSIM. Geometry becomes much more concentrated near the recovered clothing observations:

| Source group | Native witness±.02 mass,.01→1 | Median10–90 width,.01→1 |
|---|---:|---:|
|Old18|81.89→99.38%|.03169→.009865|
|New66|82.56→98.48%|.02302→.006238|
|Old8 reserves|84.98→99.79%|.03483→.008602|
|New9 patch reserves|97.43→100.00%|.01491→.001335|

The reserve gains survive dense replay: old8 dense band77.05→99.48%,new9 93.01→99.9998%. All206 clothing and24 DC rays pass convergence, and native DC RGB matches saved PNGs. This is not an inference-only occupancy improvement.

New9 median signed residual changes+.000617→+.000345; HA5556 changes+.002148→+.001070 with width.01839→.00650. Old8 median absolute residual instead worsens.000601→.001067, despite its narrower distribution. New66 has four versus two rays with unknown t90 because opacity does not reach.9. Units are reconstruction world units, not certified meters. Point5568 remains annotated hair/clothing overlap. A±.02 mass band and narrower distribution do not establish an exact thin surface or complete physical occupancy.

[Full clothing/mass report](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_late_pair_geometry_34376/clothing103/report.md), [signed residuals and widths](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_late_pair_geometry_34376/clothing103/depth_residuals.json), [DC replay](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_late_pair_geometry_34376/DC12/report.md), [matched source/control/treatment visuals](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_late_visuals_v2/step-000034376/report.md).

## Insights and next control

The measured tradeoff supports an intermediate coefficient.1 from the same original30376 full state. Existing.01 control can be reused only after exact zero-state/cache/sampler/RNG and numeric render parity; the future4000-ray stream must match too. No new control trajectory is implied by reuse, and no automatic quality promotion follows from geometry concentration.

A separate prepared generation extends only the two frozen v2 coefficient-admission guards to allow.1; all training/math/cache/restore code and original1495 parent files are unchanged. The old v2 code/runs remain untouched. The new zero-update gate and subsequent diagnostic each require a separate exact release. No repeated64-Adam proof is requested solely for this coefficient guard extension. Prepared root: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_late_lambda0p1_candidate_20261106`.

Manual final HC-black and KA-grey contact-sheet review confirms smaller exterior haze with coefficient1, while rough floor pieces and floating material above the hair remain. Better source-ray concentration therefore does not certify a clean full scene.

The prepared.1 zero-only policy binds14 artifacts and1499 source files, independently verified on silver-sable. Actual CPU configuration differs only in distortion_loss_mult.01→.1; invalid coefficients and booleans remain rejected. The proposed idle-exclusive GPU5 gate has300-second deadline and12GiB process/10GiB allocator caps. It compares seven restored state hashes plus semantic support, cache, sampler, next-step, Adam and FR against the existing v2.01 zero receipt, then checks all three RGB/alpha renders and V1 numeric parity. Both zero and diagnostic releases are still pending; no GPU launch has occurred for.1.
