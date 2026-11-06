# Native20k versus frozen LookCloser: geometry attribution

## What was tested

Read-only comparison of the actual native20k configuration, frozen source and
geometry archive against the current campaign's frozen LookCloser configuration.
The native checkpoint is `fd56e541d91abde26b776ffe03b91caf997b3c0ada0044cebf6407728c988fa2`.
No GPU execution, optimization or production mutation occurred for this review.

**The native reference is not an architecture-only NGP control. All28 shirt
rays used in the depth comparison are explicitly supervised native training
geometry.** Exact pointID+camera membership exists in its44938-observation
archive; original XYZ transformed into the accepted floor frame agrees within
5.55e-17. Its pipeline adds observed-SfM prefix optical thickness and positive
surface-band opacity, both coefficient.01, on128 TRAIN rays each iteration
(64 prefix and16 band samples). It also adds distributional MVS depth.01 on a
different measured-floor hypothesis set. The native shirt result is a useful
successful fitted-geometry reference, not independent depth generalization.

[Exact membership receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/native_shirt_training_membership.json),
[native run specification](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_R1_DEV3_STAGED/run_spec.json),
[executed geometry composition](/fsx/tmp/marshawn_001660_hd/native_png_radialK4_COLD_R1_DEV3_STAGED/code/marshawn_final_pipeline.py).

## Results

| Factor | Native20k actual | LookCloser campaign actual | Interpretation |
|---|---|---|---|
|Geometry labels|SfM visible-prefix.01 + opaque-band.01, distributional MVS.01|No depth targets; configured depth coefficient.001 is inactive without `depth_image`|Direct, verified supervision difference on the exact diagnostic rays|
|Density|FP32 `trunc_exp(h)` with initialization scale1|Legacy `softplus(h+1)` on native TCNN precision|Plausible optimization difference, unisolated by this comparison|
|World normalization|Full-scene native AABB extents7.037×7.498×2.931|Accepted rotated actor AABB extents.99×.63×.70|Same per-axis position normalization formula, very different support/physical hash spacing|
|Hash/MLP|16levels,16→8192,2features,hash2^21,64-wide,1 geometry/2 color hidden layers|16levels,16→8192,2features,hash2^23,64-wide,1 geometry/2 color hidden layers|Network hidden depth is effectively matched despite different config layer conventions|
|Frequency|Off|FR1 until75940, then.3; frequency-grid sampling|Additional unisolated representational/optimization difference|
|Directions|Maps unit directions into[0,1] before SH4|Legacy raw unit directions into TCNN SH4|Known implementation discrepancy, not proved cause of front depth|
|Marching|Fixed.003 world-distance step, near.052|Adaptive coarse.00625, frequency min.0001/max.1, near.01|Different sample policy; both use unit rays/world-distance intervals|
|Occupancy|128³,1cascade,EMA.95,threshold cap.01,one-cell halo|128³,1cascade,EMA.95,threshold cap.01,halo0/1|Density multiplier differs: native.003; LookCloser diagonal/1000=.00136638|
|RGB objective/background|Full-scene MSE with random background|Full or actor-weighted Charbonnier, black background, actor AABB|Full-image targets outside the small actor support can encourage compensating density|
|Distortion|.01 packed linear normalized-distance objective|.01 baseline, separate early1 diagnostic|Native already uses the same family of packed regularizer; compactness supplies no absolute surface depth|

At capped occupancy threshold, equivalent density cutoffs would be3.333(native)
and7.319(LookCloser), respectively. Actual cutoffs also depend on EMA mean and
must be read from the actual GPU core; these numbers are **not** reconstructed
binary-grid labels. Independent uniform traces reproduce the large depth gap,
so occupancy skipping alone cannot explain it. Native retains its full-scene
training history even when its render is clipped into the accepted actor box.

The local paper's loss section explicitly includes sparse point-cloud depth
early in training to avoid incorrect geometry. Its recipe motivates geometry
anchoring, but does not establish that the historical native auxiliary weights
are optimal for this LookCloser field.

Softplus remains a viable later ablation:99%opacity over a.01-world-unit slab
requires density≈460.5. Softplus needs preactivation approximately459.5 with
the current+1 bias; exponential needs log-density≈6.13. This is an analytical
conditioning example, not a measured saturated-logit diagnosis. The native
field uses `exp(h)` while LookCloser's built-in exponential option uses
`exp(h+1)`, so an activation toggle also changes initial density. Do not call
it exact native parity or use old exponential-only geometry adapters on softplus.

## One proposed controlled factor

Prefer **adding only a reviewed TRAIN clothing visible-prefix optical-thickness
term** to the chosen unchanged LookCloser recipe, initially candidateλ=.01
as in the native successful recipe. Preserve activation, direction handling,
AABB, FR/sampler schedule, RGB objective, halo and all optimizer behavior. The
first candidate must pass a real AMP zero-update gradient check; lowerλ if
the auxiliary dominates the existing geometry gradient. No coefficient is
accepted from CPU tests alone.

This targets the observed failure directly. A positive-band-only term can
add density behind the fog while its surface remains occluded. Prefix-only
can instead make rays too transparent; this must be measured, and only then
would a separately controlled distribution/band term be justified. Native
success with both terms does not prove either alone suffices.

New isolated helper:
[marshawn_softplus_prefix_candidate.py](../scripts/marshawn_softplus_prefix_candidate.py).
It calls native `field.density_fn`, preserves softplus forward/backward, uses
private step RNG and FP32 integration, and never changes occupancy. It rejects
wrong AABBs, nonunit rays, fewer than3 actual TRAIN observations, unsafe surface
margins, altered artifacts and evaluation supervision. `load_reviewed_artifact`
requires a separately hash-bound manual clothing-visibility review and current
floor-frame data bindings. The old automatically qualified364-point actor
artifact is **not** such a new approved artifact and is not auto-loaded.

Five CPU tests pass: analytic integral and finite-difference gradients,
camera-balanced sampling/global RNG preservation, coordinate/uncertainty
rejection, runtime activation/AABB/nonfinite guards, and reviewed-artifact hash
checks. No trainer integration or actual field AMP test has run; no candidate
training is launched. The helper is deliberately opt-in and production snapshots
remain untouched.

The next comparison needs frozen masks/crops, standard actor PSNR/SSIM/LPIPS,
all3 held-out views, hair/foot and inter-person occlusion crops, and independent
dense ray traces reporting prefix mass, band mass **and transmission**. Reserve
TRAIN tracks outside the auxiliary for geometry evaluation. The current28
shirt rays become fitted diagnostics if their points are used for this prior.
Missing depth/occlusion evidence remains unknown, never free space.
