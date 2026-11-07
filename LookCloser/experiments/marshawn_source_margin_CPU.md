# Small geometric buffer around original SourcePad1

## What was tested

CPU-only integration of already observed samples from the mature quarter-step/distortion .1 checkpoint126316 (`340e794e…cd641`). No model query, CUDA context, optimization, source-label change or producer modification occurred. The160 sealed rays contain103 TRAIN clothing witnesses, six TRAIN shoes, twelve EVAL error rays,27 EVAL foot-context rays and twelve EVAL artifact rays. Context rays are not physical floor/empty-space truth.

The new helper queries the original SourcePad1 union using explicit L-infinity safety margins0/.0005/.001/.002 scene units. Numerical contact epsilon remains2e-7 separately. Source pitch is [.007734375,.004921875,.00546875]; all margins plus epsilon are below half the minimum pitch .0024609375. The implementation intersects the query cube with original source cells, not a dilation of the mapped traversal cover. Outside-domain UNKNOWN admission is inherited unchanged.

## Results

Five CPU tests pass: zero-margin equality with the qualified selector, nested admission as margin increases, geometric-distance/contact behavior, half-pitch/nonfinite rejection, integral identity and empty native rays. All109 observed witness occurrences remain admitted. Zero-margin dense density and weights exactly match the previously observed B arrays. Each input NPZ is hashed in the full results.

The table uses uniform16384 analytic integration of the same A field samples. It is a dense counterfactual, not a rerun of the native renderer or an image-quality measurement.

| Diagnostic opacity | Original A | Margin0 | .0005 | .001 | .002 |
|---|---:|---:|---:|---:|---:|
| Head DC[850,85] |1.0000|0|0|0|0|
| Floor-context DC[865,555] |.2057|.0162|.0166|.0171|.0207|
| Feet/bag-context DC[1100,595] |.3204|.2263|.2277|.2292|.2301|
| Remote DC[600,845] |.1947|.0376|.0587|.0842|.1224|

The buffer preserves removal of the sampled head artifact while restoring some remote clutter. In the six TRAIN shoe rays, the maximum RGB difference from A on fixed **native sample positions** decreases from .001761 at zero margin to4.44e-7 at .002. The dense16384 shoe reference still differs by up to .00497 at .002; native sample placement and skipped space matter. These sparse observations do not establish full-foot retention or performance.

Across the twelve artifact rays, the mean removed original-A dense weight is .137524/.132461/.127368/.118932 for the four margins. This is a descriptive statistic dominated partly by the opaque head ray, not a physical empty-space metric. Color differences need not change monotonically even though admission and removed original weight do: revealing deeper differently colored density changes compositing.

## Insights and next measurement

The existing160 queries do not include the two worst grey contour pixels located in the full-image A/B review. Consequently this CPU result cannot establish that any margin recovers the grey actor's appearance. A prepared nine-query **EVAL-only** native plan covers x−1/x/x+1 around KA[1028,331], HC[1052,576] and head DC[850,85]. It remains unlaunched pending root review. The plan keeps all native traversal, field, FR.3, coarse .0015625, cap1024, core/EMA and random-state invariants.

[Compact summary](assets/marshawn_source_margin_20261107/summary.json) and [unlaunched source-bound query plan](assets/marshawn_source_margin_20261107/native9_query_plan.json). Full320 profile records, input hashes and helper: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/source_margin_CPU_v1`. Results SHA `bac592e56475dd07ebbe63512e06db33dcf9df57456f91908f553e4f0505379c`.
