# Matched15188 actor objective and half-step geometry

## What was tested

Three immutable native-exp checkpoints at15188: full/body80/R0 parent, actor-only objective, and full objective with half coarse step. Actual loaded steps were.00625/.00625/.003125; corrected SH, native exp(h+1), legacy projection and FR1 were checked. Read-only replay used the same103 predeclared TRAIN witness rays, native traversal, fresh single halo, all-cell diagnostic and8192/16384 dense convergence. All309 ray convergence checks passed; no optimizer or callback ran. Producer identity/progress and checkpoint hashes were recorded before and after bounded GPU overlap.

The historical group names “prefix supervised” describe prospective source partitions. None of these three models received those auxiliary geometry labels. Reserves are excluded from auxiliary labels, not from TRAIN RGB. The separate5568 hair/chest annotation remains in effect.

## Results

| Model | Held-out PSNR | SSIM | LPIPS | Old18 native band ±.02 (%) | Old8 reserve band (%) | New9 reserve band (%) |
|---|---:|---:|---:|---:|---:|---:|
| Full parent |27.1029|.798674|.056944|86.56|97.59|82.41|
| Actor objective |27.5159|.809772|.053507|4.98|.344|93.70|
| Half coarse step |27.0478|.811806|.050777|91.73|98.77|80.80|

Actor-only old18 rays have94.80% front mass and median signed termination-depth bias−.6203; old8 reserves have99.65% front mass and bias−.6354. Half-step improves those regions, but HA5556 retains front bias−.3186 and10–90% width.3602. The selected parent60752 audit is reported separately; these are matched15188 comparisons.

[Numeric comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/exp_followup103_matched15k/comparison.json), [signed depth/width diagnostics](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/exp_followup103_matched15k/depth_residuals.json) and [visually inspected mass plot](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/exp_followup103_matched15k/native_mass.png). Immutable checkpoints and raw profiles remain under the matching `independent_occupancy` folders on silver-sable.

## DC failure diagnosis

Twelve posthoc DC evaluation-only rays were selected and source patches manually reviewed: four black-shirt errors, four grey-garment errors and four healthy pants controls. These are biased debugging examples, never training input or independent selection metrics, and have no asserted depth truth. Native RGB agrees with the corresponding saved PNG within.00197 for all36 model/ray pairs; dense convergence also passes.

Actor-only failure-ray opacity is.998618–1.0. At DC(939,263), actor RGB is(.870,.663,.274) versus target(.125,.149,.114), and median termination depth.5449 versus parent.9866 and half-step.9914. A fresh halo changes RGB only.000816. At DC(1024,379), actor median.5771 versus parent1.0137 and half1.0150, halo change.000354. Healthy pants control DC(855,363) agrees in depth: actor.98358 versus parent.98331.

These yellow patches are opaque, wrong-colored front layers rather than transparent holes caused by skipped boundary cells. [DC query/source review](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/DC_error12_15188/source_patch_review.jpg) and [full trace comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/DC_error12_15188/results/comparison.json).

## Insights

Aggregate LPIPS conceals catastrophic per-view geometry in the actor-only arm. Half-step improves detail and old-shirt geometry but leaves other front layers; neither is geometrically approved. The next diagnostic is conservative TRAIN-only semantic admission during actor training, documented separately. Skipped-space positive density is unconstrained and must not be declared physical occupancy or fixed by blindly activating all cells.
