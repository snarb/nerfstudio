# Reviewed clothing priors on semantic and focused fields

**What was tested.** Read-only CPU compatibility of the existing 31-point/84-ray TRAIN clothing prior with the matched semanticR0 and focused-frequency checkpoints at 15,188. Source qualification, annotations, AABB, masks and models are unchanged. The configured `depth_loss_mult=.001` is inactive here: both configs set `load_3D_points=false`, the ready receipt states sparse depth is not supplied, and the built-in term requires `depth_image` in the training batch. The historical prefix/band terms are separate auxiliary objectives.

All reserves 63/64/71/72/2219/5556/5568 are absent from the actual sealed 84-ray archive. Point5568 remains a diagnostic hair-boundary reserve, excluded from opaque-clothing acceptance. We reused existing recovered feature graphs and root source reviews; no source points were recovered or relabeled again.

**CPU results.** Source/calibration/semantic hashes match. The actual frozen semantic lookup confirms identical admission in both model snapshots, retaining all 84 witness positions and every band midpoint at 16, 64 and 256 samples per ray. Prefix sampling retains only about 14.0% of positions; 18 of 84 prefixes have no admitted midpoint even at 512 samples. These finite samples establish neither continuous geometric support nor physical emptiness.

The measured feature reprojection residual reaches 0.965814 HD pixels. The corresponding feature ray misses its fitted point by at most 0.00072478 scene units. The existing band half-width is 0.009271–0.011316; all available leave-one-source-view shifts fit the clipped band. These checks preserve a soft prior, not an independent confidence interval. Three-view tracks still lack an independent three-view leave-one-out certificate.

The prior's 84 rays pass exactly through fitted XYZ. Actual source detections have nonzero residual, and the saved 103 diagnostic rays use integer pixel centers. Those three objects must remain distinct. The following table partitions **existing diagnostic pixel-ray weights**, preserving original native weights with normalized constant-density interval fractions; it does not measure a newly queried projected-prior integral.

| Model and source group | Rays | Mass before old prefix endpoint | Additional early mass in unpenalized gap | Mass in existing narrow band |
|---|---:|---:|---:|---:|
| Original R0, old fitted shirt |18|2.41%|5.74%|65.30%|
| Focused, old fitted shirt |18|7.41%|12.47%|55.16%|
| Original R0, nearby reserves |8|3.73%|5.87%|74.51%|
| Focused, nearby reserves |8|9.30%|14.91%|63.51%|
| Original R0, new fitted sources |66|3.06%|2.27%|75.15%|
| Focused, new fitted sources |66|3.02%|3.54%|79.95%|

“Additional early” is after `depth−max(.01,.05×depth)` but before `depth−.02`. The whole unpenalized interval before the band is 0.037083–0.045264 scene units. About 63% of the focused old-shirt premature termination lies outside the old prefix penalty. Weight reconstruction error is zero. The full [CPU receipt](assets/marshawn_semantic_clothing_transfer_20261107/CPU_result.json) and [per-ray partitions](assets/marshawn_semantic_clothing_transfer_20261107/saved_pixel_profile_partitions.json) preserve grouping, hashes and the separate hair qualification.

**Historical limits.** On the earlier full-background objective, P/PB improved fitted geometry but not whole-patch reserve behavior. At matched 15k, PB increased new fitted ±.02 mass from 90.37% to 99.55%, while new reserve mass changed only 82.41%→82.88%; LPIPS worsened .056944→.059599. Those results do not establish that the same regularizer will succeed with actor-only RGB, fixed semantic admission or focused frequency updates. The sparse sources also provide no qualified shoe/hair prior and uneven garment coverage. See [source qualification](marshawn_expanded_clothing_sources.md) and [historical exp geometry arms](marshawn_native_exp_geometry_arms.md).

**Authorized read-only probe.** Root authorized immutable R0/focused15k comparisons using the same 84 projected prior rays: old gap `max(.01,.05d)` versus a closer soft gap `max(.01,.02d)`, with the existing positive band unchanged. The closer gap is not ground truth or a hard carving rule. Prefix quadrature is 128/256/512, adding 1024 if any ray fails the declared 5%-or-0.05 convergence test; band quadrature is 16/64. Auxiliary aggregation is the analytic camera-balanced expectation over all 84 sources, while each model supplies its own actual resumed 4,096-ray body80 gradient reference. Different main-batch identities are recorded rather than claimed identical.

The probe queries the actual gated `exp(h+1)` density and native backward under AMP. It records full-field and geometry-only gradient norms/cosines, source residuals, prefix-to-band clearance, and any unresolved integration. It preserves seven serialized-state hashes, actual sampler/cache/global RNG, the focused private state, and all optimizer/scaler/core parameters. Diagnostic backward-scale backoff does not update the real GradScaler. The unchanged fixed selector remains in every query; learned occupancy skips and early termination do not define these auxiliary integrals.

The executable plan is `SC/independent_occupancy/semantic_clothing_transfer_probe_v1/plan.json`, SHA256 `c191f0f37019259ac44ecc9cece20f47cda6a20209d0f275ff59a0ad3cd16847`. It binds 26 explicit inputs plus both complete frozen code inventories, uses one idle-exclusive silver GPU, and limits allocation/process/time to 12GiB/16GiB/900s. Actual physical UUID isolation precedes model loading; absolute imported helper paths and hashes are recorded. No optimizer update, new annotation or training release is part of this probe.

**Actual probe results.** Both models passed in 134.64 seconds total, peaking at 7,856MiB. Seven initial/final serialized-state hashes match exactly; sampler, cache, global RNG and the focused private state are unchanged. No Adam update occurred. The actual 4,096 main-batch index hash happened to match across models, making this particular gradient comparison use the same pixels. Both main objectives contain only RGB and distortion terms; the depth term is absent. Projected-camera ray replay agrees within `1.19e−7`.

| Quantity | Original R0 | Focused |
|---|---:|---:|
| Old-prefix camera-balanced τ, 512 samples |0.05622|0.04582|
| Closer-prefix camera-balanced τ, 512 samples |0.13935|0.11484|
| Old-prefix gradient / main geometry gradient, coefficient .003 |5.25%|1.27%|
| Closer-prefix gradient / main geometry gradient, coefficient .003 |12.72%|3.26%|
| Closer-prefix gradient cosine to main geometry |+0.230|+0.338|
| Positive-band gradient / main geometry gradient, coefficient .001 |17.59%|3.76%|
| Positive-band gradient cosine to main geometry |−0.369|−0.426|

All 84 prefix rays satisfy the declared 256→512 convergence tolerance for both gaps and models, so 1024 was not needed. Band 16→64 passes only 82/84 original rays and 62/84 focused rays; the focused worst relative τ difference is 10.43%. Although aggregate band-gradient norms change by less than 1%, this does not establish per-ray integration accuracy. The inherited 16-sample positive band is therefore not ready for reuse without further convergence evidence. All tested gradients are finite after read-only backward scaling; the actual GradScaler remains at its source state.

The closer prefix ends 0.009271–0.011316 before the unchanged band. It provides a concrete direction for a separate controlled geometry experiment, but these one-batch gradients do not demonstrate preserved image detail, improved reserves, or a correct physical surface. The band term has an opposing gradient on this batch and unresolved per-ray quadrature; blindly combining it with the closer prefix would obscure attribution. No new training is launched here.

The [completion receipt](assets/marshawn_semantic_clothing_transfer_20261107/complete.json) SHA256 is `e25bf11eeae9e2602b3b9c5386baea9ff2c401e806b19011f798ec1f809ff25f`. [Parent probe](assets/marshawn_semantic_clothing_transfer_20261107/R0_baseline_probe.json) and [focused probe](assets/marshawn_semantic_clothing_transfer_20261107/frequency_focus_probe.json) retain per-ray τ, convergence, gradient scales and complete restoration proofs. All 32 evidence files are hash-verified in `/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_resolution_pair_100000/documentation/semantic_clothing_transfer_v1`; the accompanying archive SHA256 is `0685aff08d67c21cbdf97e253ff779123d49f729cbc3b51d23af011f7e2162c6`.

`SC=/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106`. Code: [CPU audit](../scripts/audit_marshawn_semantic_clothing_transfer_cpu.py), [native gradient probe](../scripts/probe_marshawn_semantic_clothing_transfer.py), [bounded supervisor](../scripts/run_marshawn_semantic_clothing_transfer_probe.py).
