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

The prepared exponential actor-objective arm tests this cause with corrected SH
and improved density activation. Preserve the full-objective control. Do not
silently enlarge the accepted output box, replace EVAL RGB or treat inference-only
grid deletion as a training fix. Separate background modeling or a larger training
domain remains a distinct later experiment if actor weighting is insufficient.
