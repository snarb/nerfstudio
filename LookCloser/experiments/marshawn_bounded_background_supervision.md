# Bounded actor field versus full-image RGB supervision

## What was tested

Read-only audit of original TRAIN feature tracks and the accepted AABB. In eight
TRAIN cameras, recover a unique original feature within its original1HDpx fit
tolerance, use the actual integer training pixel center and intersect its ray
with the box. Require at least3 source views, zero frozen W64 actor weight, valid
nonblack RGB, more than.05world units traversing the box and a measured surface
more than.03 beyond the exit. No EVAL or learned field depths enter this test.

## Results

2,244 sparse observed rays qualify in AA,AB,HE,IC,JA,MC,NE,OA. This is not the
proportion of all training pixels. Root inspected32 spatially separated examples:
wall markers, ceiling, equipment and staff. Their measured surfaces are at least
.3829world units outside the box; the rays traverse at least.07564 inside it.

[Original TRAIN patches](/fsx/tmp/marshawn_001660_hd/tuning_20261106/bounded_background_audit/background_source_examples.jpg)
· [Observed UV, ray intervals, 3D points and hashes](/fsx/tmp/marshawn_001660_hd/tuning_20261106/bounded_background_audit/report.json)

Frozen code confirms that full objective uses RGB Charbonnier without actor
confidence. Body80 retains20% global-valid sampling plus a protective band.
These rays therefore receive RGB supervision although their true surface cannot
be represented at its measured depth inside the bounded field. The existing
actor objective gives these demonstrated rays zero RGB weight and retains its
soft margin elsewhere.

## Insights and next steps

This is concrete pressure to represent background color inside the actor volume;
it does not establish the cause of every artifact or make all non-actor density
physically empty. Native20k uses a larger field domain, clipped to the accepted
box for output, and differs here as well as in architecture and geometry priors.

### Measured actor weighting and TRAIN support follow-up

The actor-objective experiment subsequently demonstrated a second failure:
removing background RGB pressure alone permits incorrect opaque front surfaces.
Its better aggregate LPIPS concealed a large DC regression. The following are
matched15,188-step results with identical frozen V1 masks/crops and equal-actor,
equal-camera aggregation:

| Exponential/body80/R0 recipe | PSNR | SSIM | LPIPS | DC PSNR |
|---|---:|---:|---:|---:|
| Full RGB objective |27.1029|.798674|.056944|25.3546|
| Actor-weighted RGB, W64 |27.5159|.809772|.053507|20.5371|
| Same actor recipe + fixed conservative TRAIN admission |31.4144|.841056|.031173|31.9023|

Root viewed all three matched camera comparisons. The added support removes
the catastrophic yellow surfaces through DC clothing; body and shoe contours
remain coherent at inspected scale. It still leaves fuzzy silhouette halos and
floor fragments, so this is not an ideal occupancy grid or a cleaned actor asset.

The source uses all63 TRAIN cameras and original reviewed8K-derived confidence,
without EVAL masks or RGB. A whole-cell conservative mapping preserves a fixed
131,911-cell admission. Native learned core/EMA remain separate; traversal uses
native dilation intersected with admission. Source semantic pad1 and native
occupancy radius1 are different operations. At15,188, actual R0 active cells fell
from123,817 in the actor-only model to51,894 in the separately trained candidate;
lower cell count alone is not an accuracy metric.

The sealed103 camera-ray geometry audit also improves: old eight reserved shirt
rays increase native mass within±.02 of the measured surface from.34% to84.48%.
The HA5556 gray-clothing front-surface bias becomes+.001155 instead of the
full-objective parent's−.32018. Some broad/partly transparent surfaces remain;
these sparse rays cannot certify the entire volume. The matched slice inspector
also contains39 unique triangulated landmarks;103 ray records are not103 unique
independent points.

Artifacts: [three-view comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_actor_015188/GT_actor_semantic_crops.jpg),
[matched grid inspector](/fsx/tmp/marshawn_001660_hd/tuning_20261106/grid_inspector/semantic_15188/index.html),
[root visual review](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_actor_015188/root_visual_review.json).
Detailed implementation/audits are in `marshawn_training_semantic_admission.md`.

The main follow-up continues improving: at45,564 it reaches31.5320/.843159/.025900
(PSNR/SSIM/LPIPS). R1 and W32-loss-margin comparisons are separate running/queued
experiments; neither changes the fixed W64-derived admission. A distinct late
distortion intervention improves the full-objective ancestor's geometry but
regresses KA; see `marshawn_late_distortion_geometry.md`. Its benefit cannot be
assumed for the semantic recipe without a new paired experiment.

Preserve the accepted output box and V1 evaluation. Do not treat masked mean
metrics, an inference-only deletion, or low density in unobserved space as proof
of physically correct geometry. Separate background modeling remains a possible
later experiment if soft-band RGB pressure and underconstrained regions continue
to leave halos after the present comparisons.
