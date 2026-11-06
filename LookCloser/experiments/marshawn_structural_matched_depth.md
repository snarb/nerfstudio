# Marshawn structural changes: matched15188 geometry audit

## What was tested

The exact same28 TRAIN shirt rays were replayed at checkpoint15188 for SH-corrected softplus, SH-corrected `exp(h+1)`, and SH-corrected softplus with normalized-image Jacobian frequency projection. The two structural checkpoints were protected by immutable hardlinks before pruning. The audit used their actual exported GPU occupancy cores, native traversal, one noncumulative halo, all-cell traversal, and uniform8192/16384 integration. No optimizer step, producer mutation or training callback ran.

All actual SH flags were true and FR was explicitly restored to1 for15188. Structural density and projection flags were checked against their saved configs. A numerical activation probe verified `exp(h+1)`, not `exp(h)`, for the exponent candidate. Actual camera rays matched the independent prepared-floor query chain. Model state hashes were unchanged after replay.

## Results

**The exponent improves actual rendered termination geometry on these shirt rays. Its dense field still contains substantial density in skipped space.** These are separate observations. A learned field is not independently constrained in every skipped voxel; density there alone does not prove that occupancy falsely calls physically empty space free.

Unconditional mean mass percentages, retaining transmission:

| All28 rays, native traversal | Before shirt band | Inside ±.02 | After band | Transmission |
|---|---:|---:|---:|---:|
| SH softplus | 21.741 | 65.445 | 12.787 | 0.027 |
| SH exp(h+1) | **2.191** | **88.613** | 9.194 | 0.002 |
| SH normalized Jacobian | 26.692 | 63.741 | 9.530 | 0.037 |

| All28 rays, dense16384 | Before shirt band | Inside ±.02 | After band | Transmission |
|---|---:|---:|---:|---:|
| SH softplus | 52.538 | 39.777 | 7.671 | 0.014 |
| SH exp(h+1) | 62.506 | 34.428 | 3.066 | 0.001 |
| SH normalized Jacobian | 58.734 | 36.092 | 5.156 | 0.017 |

The8 original nearby reserved rays have native band mass64.321% for SH,97.590% for exponent and61.811% for Jacobian. Their native early mass is26.705%,0.156% and33.154%, respectively. These groups were reserved from the separate prefix objective; **none of the three models compared here was trained with that prefix objective**. All28 are still TRAIN RGB observations and correlated SfM evidence, not held-out scene-depth ground truth. The ±.02 width is a diagnostic policy in prepared-world units, not a calibrated uncertainty interval.

The inference-only halo reveals substantial sensitivity:

| All28 rays | Native early / band | Halo+1 early / band |
|---|---:|---:|
| SH softplus | 21.741 /65.445 | 37.889 /52.048 |
| SH exp(h+1) | 2.191 /88.613 | 15.907 /76.212 |
| SH normalized Jacobian | 26.692 /63.741 | 45.557 /47.399 |

Exponent reserved-ray band mass falls97.590%→84.154% under this same-field halo replay. This shows that admitting neighboring cells can expose latent density. It does not predict a separately trained R1 model, whose gradients and learned field will differ. No inference-only halo was promoted to production.

All28 rays for all three models pass8192→16384 RGB/opacity convergence. Maximum change in any before/band/after/transmission component is0.001379 for SH,0.000199 for exponent and0.000636 for Jacobian. These are much smaller than the reported model/profile differences.

- [Native, halo and dense mass comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/structural_matched15k/native_halo_dense.png)
- [Dense cumulative mass traces at representative source points](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/structural_matched15k/comparison.png)
- [Matched held-out actor crops](/fsx/tmp/marshawn_001660_hd/tuning_20261106/structural_review/matched15188_actors.jpg)

Both plots and the actor montage were visually inspected. Held-out image quality remains imperfect, with visible fog/sheets around actors:

| Matched15188 held-out actor metrics | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| SH softplus | 26.4216 | 0.78052 | 0.069681 |
| SH exp(h+1) | 27.1029 | 0.79867 | 0.056944 |
| SH normalized Jacobian | 26.4445 | 0.78518 | 0.065198 |

Exponent's black-actor PSNR falls26.5285→25.7160, while grey-actor PSNR rises26.3146→28.4898. Therefore its better aggregate LPIPS and native shirt termination do not establish a uniform improvement for both actors or certify the whole occupancy grid.

## Evidence and next gates

Root: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/structural_matched15k`.

`comparison.json` contains all per-ray mass components, grouped results, hashes, actual flags and coordinate checks. `halo_sensitivity.json` contains all traversal profiles. The complete protected checkpoints remain on silver under `independent_occupancy/structural_exp_hplus1_015188` and `structural_normalized_jacobian_015188`; local traces and receipts are archived, avoiding an unnecessary multi-gigabyte transfer.

Read-only audits ran sequentially with an explicitly authorized10GiB allocator cap. Exponent producer remained alive and advanced35913→36704; Jacobian producer remained alive and advanced36693→37632. Before/after PID identities, GPU processes and progress are recorded. The first audit attempt stopped because the query JSON had not been staged remotely; the original query was then SHA-verified and staged. This did not indicate a training failure. Successful results are in `trace_02`; failed-attempt logs are retained. No timing is used as a speed comparison.

For the A/B/C geometry arms, `independent_occupancy/queries/clothing_103_predeclared.json` is now frozen before their first15188 gate:18 old fitted rays,66 newly fitted rays,8 old local reserves,9 whole-patch reserves and2 historical duplicate diagnostics. Its original28 pixel rays are unchanged. New rays use the75 manually reviewed actual feature pixels, and maximum witness-to-pixel-ray separation is0.000921 world units. Each subset will be reported separately; no pooling will disguise weak reserved-patch or grey-actor behavior. This diagnostic is ready but has not yet been executed on A/B/C checkpoints. All ongoing arms are unchanged.

The exponent result supports a separately trained R1 comparison and continued geometry-source testing. It does not justify carving all positive dense-field values, treating all skipped voxels as errors, or declaring an ideal occupancy grid from these sparse clothing observations.
