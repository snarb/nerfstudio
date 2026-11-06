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
