# Cold semantic R0/W64 intermediate distortion

## What was tested

The late30376→34376 coefficient.1 intervention modestly concentrated clothing depth with a1.76% LPIPS penalty. This separate cold trajectory tests whether learning with the original4096→12288 ramp ending at.1 allows more texture adaptation. Its reference is original semantic R0/W64, native exp(h+1), corrected SH, body80, legacy frequencies and original confidence. No bag correction, source-point density lookup or4K inputs are combined.

The1495-file frozen generation changes only the worker's exact coefficient validator from.01 to.1. All model, sampler, optimizer, ramp and plateau implementations remain byte-identical. Source manifestSHA `1d879b66779bd566ebb612968c429ffe89271bbc184aca1ddf4f4606709b1694`; specSHA `d7d63f5d9a4d3587217a105dc1c3d1d89ce273dd2605bea018e17f2631a8e380`.

Actual configuration comparison is identical after path normalization; the sole scientific spec difference is distortion_final. The actual runtime expression gives.01 through4096,.055 at8192 and.1 from12288. Invalid bundled width/radius/objective changes are rejected. An initial config-import attempt discovered that visible-CUDA TCNN imports initialize a context; the preserved log records this, and the successful CPU proof reran with CUDA hidden, without a model or optimizer.

## Technical validation and launch

Idle-exclusive silverGPU5 validation passed in143.1seconds with7948MiB peak process memory. Successful Adam counts are16→32 for warmup and15→30 for ARM, with exact RNG/sampler resume, finite parameters/moments, valid actual occupancy and complete three-view RGB/alpha. ARM starts cold at4096 only to exercise the ramp/integration path; it is not a quality trajectory or a new final-coefficient gradient proof. The identical field's all-ones/density/positive-support GPU proof is explicitly inherited, and the separate actual4000-step.1 continuation already exercises the final coefficient. Both provenances remain distinct.

All six technical output images were inspected: finite coarse silhouettes and initial noise only. [Config/ramp proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_cold_lambda01_candidate_20261106/config_parity.json), [native Adam proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_cold_lambda01_candidate_20261106/technical_complete.json), [technical images](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_cold_lambda01_candidate_20261106/technical32_overview.jpg).

Root explicitly delegated scoped technical and production execution after these checks, without another approval loop. ReleaseSHA `1ed74ad0a69c80165fbebf850e502aaf9d70b5bdc38895b8308796e4aaad47e0` binds28 artifacts and all1495 source files. Controller1336517, supervisor1336526 and trainer1336534 started on idle globally locked silverGPU2; first actual progress802 was healthy. Mandatory106316 followed by the existing validation plateau remains unchanged, with no arbitrary iteration ceiling.

## Review and retention

Local watcher2103917 preserves ongoing evaluation/grid/render evidence. First15188 auditor1339481 protects the weights and compares the exact103TRAIN observations plus12 posthoc DC rays against R0 at the same15188 gate, using idle-exclusive GPUs and zero updates. Its audit-only v2 corrects report title metadata before any trace; old generation evidence and production source/spec/release remain unchanged. Local fetcher2104115 prepares six GT/control/treatment crops automatically.

A separate read-only watcher retains established-selector, raw minimum-LPIPS and terminal checkpoints with actual grids and renders. This does not change either selector or plateau. The minimum-LPIPS Pareto point and terminal are already protected by worker retention; the explicit archive makes them reviewable together. Future selected reviews also include root's CPU saved-alpha exterior diagnostic, with unequal checkpoint ages stated.

Production root: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_cold_lambda01_candidate_20261106`. First-gate report root: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_lambda01_15188_v2`. Production quality is pending; technical validity is not a clean-asset or ideal-grid certificate.

## First15188 gate

Cold.1 gives PSNR31.62870/SSIM.846893/LPIPS.03237706 versus reference31.41439/.841056/.03117262: LPIPS worsens3.86%, while PSNR/SSIM improve. Native witness-band mass improves80.23→89.92% old18,83.95→95.15% new66,84.48→94.39% old8 reserves and95.95→99.71% new9 reserves. Old8 early mass falls9.76→2.20%; median10–90 widths improve.03433→.02453/.01928→.01186/.02668→.02015/.01139→.00791. All206 clothing/24 DC convergence and PNG checks pass.

Manual DC-black,HC-grey and KA-black review shows narrower torso haze in places, with visible hair/shoes retained, while floor and hair-adjacent contamination persists. The measured tradeoff supports continuing the unchanged full trajectory, without early quality promotion. [Matched source geometry](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_lambda01_15188_v2/clothing103/report.md), [actual crops](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_lambda01_15188_v2/DC_black_gate_comparison.jpg).

## Natural completion and separate checkpoint choices

Training completed naturally at126316. Trainer1336534 exited and the controller recorded completion; no healthy process was interrupted. The established selector remains91128, while raw minimum LPIPS equals terminal126316. Both distinct weights and actual schema2 grids are protected. The selector and plateau were not changed posthoc.

| Choice | Step | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Established selector |91128|31.652626|.840094|.02228724|
| Raw minimum / terminal |126316|31.651726|.837686|.02135285|

All three GT/reference/selected/raw-min/terminal actor contacts were manually inspected. Relative to originalR0 selected111316, torso and hair-adjacent halos are narrower, while broad floor/bag contamination remains. Feet and hair are visible in these views; this is not a full contour-retention certificate. At the fixed16px exterior margin, alpha>.1 counts DC/HC/KA fall from185261/107305/74830 to121644/79593/54451 at selected91128, and120767/79280/54414 at terminal126316. Checkpoint ages differ explicitly. Exterior opacity is not a physical free-space label.

The durable final watcher passed grid-hash checks and generated all0/3/8/16/32px diagnostics without model queries. [Final choices](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_lambda01_final_choices/complete.json), [exterior evidence](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_lambda01_final_exterior/receipt.json), [DC visual](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_lambda01_final_exterior/DC_black_comparison.jpg).

Final103TRAIN+12DC qualification of both selected91128 and terminal126316 completed under the existing idle-exclusive read-only scope. ActualFR.3 and the original radius/field/admission were preserved. OriginalR0 selected111316 traces were reused by CPU comparison, without duplicate model queries. All309 clothing and36DC dense-convergence/native-PNG checks pass.

| Source group | Native band R0 / cold.1 selected / terminal (%) | Median10–90 width R0 / selected / terminal |
|---|---|---|
| Old18 |80.03 /91.31 /91.58|.02962 /.02192 /.02111|
| New66 |83.12 /95.66 /96.14|.02646 /.00926 /.00803|
| Old8 reserves |84.22 /92.39 /92.54|.03480 /.01889 /.01842|
| New9 patch reserves |98.76 /99.97 /99.998|.003224 /.002481 /.002154|

These are accepted-world coordinates, not certified physical units. Narrower distributions do not uniformly improve exact surface alignment: old18 median absolute witness residual.001533→.001794→.001895, old8.000372→.000749→.000857. New66 improves.002685→.001782→.001873. Inherited source uncertainty, hair/clothing ambiguity5568, unobserved surfaces and broad floor contamination remain. No global clean-asset or ideal-grid certificate follows from these local gains. [Final baseline comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_lambda01_final_vs_R0/clothing103/comparison.json), [signed depth and widths](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_lambda01_final_vs_R0/clothing103/depth_residuals.json).
