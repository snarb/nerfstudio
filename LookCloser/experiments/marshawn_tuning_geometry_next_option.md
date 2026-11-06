# Train-only geometry option after the distortion control

## What was tested

This is a read-only proposal, not an implemented or launched experiment. The
current matched15188 shirt audit finds substantial termination before the
triangulated shirt in both early leaders; the independent camera-chain check
rules out a large coordinate mismatch. See
[the occupancy audit](marshawn_tuning_occupancy_audit.md). First remeasure the
separate early-distortion1 control: stronger compactness can concentrate a
wrong front surface without locating the shirt correctly.

If that control remains front-biased, the next isolated factor should be a
small **visible-ray prefix optical-thickness penalty from existing TRAIN-only
actor SfM observations**, before adding a positive surface-depth objective.

## Existing evidence and implementation limits

- `geometry/strict_tracks.json` has9022 points with3–26 original TRAIN camera
  observations. The source builder required positive depth and final fitted
  reprojection below1HDpixel. These are fitted correspondences, not independent
  ground truth or a measured depth confidence interval.
- The previous `actor_geometry_AUTO128_SAFETY` artifact selects364 points and
 1810 observations. Selection requires the same point inside nominal actor
  interiors in at least3 original TRAIN track cameras, distance at least2HD
  pixels from the mask edge, and source eligibility. It excludes soft mask
  bands as geometry evidence. Its old AABB and old coordinate normalization
  differ from the current accepted floor frame: **do not load it directly**.
- `frequency_seed/observations.npz` UVs are projections of fitted XYZ into
  actual track cameras, not saved original feature detections. Its exactlyzero
  projection check establishes consistency only. Recovered original features
  or a fresh TRAIN-only rematch are needed for independent residual and
  leave-one-camera-out depth uncertainty estimates.
- `build_marshawn_free_space_rays.py` and `marshawn_free_space_objective.py`
  already implement camera-balanced128 rays×64 stratified samples, private
  step RNG, FP32 optical-thickness integration and strict source/hash checks.
  The latter deliberately rejects mask-qualified artifacts; do not weaken its
  historical contract. A new explicit adapter would be required.
- `marshawn_actor_geometry.py` and the frozen `SurfaceBandObjective` assume
  exponential density and its cached logits. The current fight-leader recipe
  uses **softplus**. Their exponential derivative/log-tau formulas cannot be
  transplanted. Query the current field's native differentiable density;
  test finite differences and actual AMP gradients without an optimizer step.

## Proposed controlled experiment

1. Rebuild a separate, hash-bound candidate artifact from the original TRAIN
   track graph and points, current63 TRAIN camera identities, current native
   mask provenance, accepted world-to-box transform and current AABB. Preserve
   the existing semantic-interior qualification. Never create observations in
   cameras merely because a point projects into their frustum. DC/HC/KA supply
   no prior rays, depth labels, semantic confidence fit or RGB fitting targets.
   The existing accepted AABB uses evaluation segmentation and remains a
   declared shared preprocessing choice; this is not a new unseen-test claim.
2. Qualify source-camera crops for occlusion, reflections and mixed boundaries.
   Start with opaque clothing-interior points; exclude hair, silhouette edges,
   hands crossing other bodies and uncertain tracks. A mask interior alone
   does not establish depth visibility. For each admitted ray, clip the prefix
   to the current AABB and stop at `depth - max(.01,.05*depth)` in the original
   normalized-world distance policy, transformed consistently if scale changes.
   The5% margin is conservative policy, not calibrated accuracy. Increase it
   when independently recovered uncertainty requires; reject ambiguous rays
   rather than extend a constraint through their uncertainty or another surface.
3. Use `lambda * mean(integral sigma(t) dt)` only on those prefixes, querying
   the field directly, including cells skipped by occupancy. Keep training
   RGB batches, seed, optimizer, core/halo policy and existing objectives
   identical to a matched no-prior control. No hard cell carving, positive
   occupancy admission, semantic-absence penalty or global sparsity term.
4. Select the small coefficient by zero-update geometry-gradient measurements
   on representative real batches; probe .001/.003/.01, and prefer a tested
   value with auxiliary geometry-gradient norm about5–10% of RGB. This is an
   experimental starting criterion, not a proven universal scale. Preserve
   global RNG and prove all parameters/optimizer/grid unchanged by the probe.
5. If clearing the prefix only makes rays transparent, evaluate a separate
   subsequent depth-band arm. Use uncertainty-bounded termination distribution
   rather than a mean-depth target; report unconditional band mass and residual
   transmission as well as conditional depth. A positive opaque-band assumption
   is permitted only for qualified opaque clothing points, never hair or all
   actor-mask pixels. Native differentiable softplus density must be used.
   A broad band cannot certify thinness, and an arbitrarily narrow band risks
   imprinting triangulation error. Do not add it simultaneously to the first
   prefix-only arm, or attribution becomes ambiguous.

## Relevant measured negative results

| Prior evidence | Result | Consequence |
|---|---|---|
|[Marshawn conditional mean depth](marshawn_mvs_depth_continuation.md)|Median normal error .003951→.000133, but thickness p90 .01331→.01608; floor mottling remained unacceptable|Mean depth can fit while separated tails persist|
|[Marshawn distribution-depth preparation](marshawn_mvs_distribution_depth.md)|Samplewise robust error correctly penalizes opposite tails; zero-update proof only|Promising objective form, no demonstrated rendered-quality win|
|[Luster soft cord band](luster_000470_cord_soft_depth_band.md)|Weaker .003 prior improves ROI PSNR but worsens held-out cord LPIPS and training cord|Do not infer detail quality from depth or global PSNR alone|
|[Marshawn LOS gradient forensics](marshawn_los_gradient_forensics.md)|Old trunc-exp backward materially differs from true derivative at low density; historical crash cause unproven|Activation-specific gradient checks and failure-state capture matter; current softplus avoids that specific exp-adapter assumption|
|[DEC5 measured free space](dec5_lipstick_fin_measured_free_space.md)|Strict corroborated-depth pruning partly removes false geometry but leaves main artifact|Missing evidence stays unknown; more deletion is not automatically safe|

## Acceptance and confidence

Repeat the same28 shirt rays plus a predeclared broader set of clothing,
limb/foot and healthy-control rays. Keep a TRAIN-track subset entirely outside
the auxiliary as a geometry diagnostic; the existing28-rays become fitted
diagnostics if their points enter the prior. Compare independent8192/16384
quadrature, early termination, in-band mass, transmission, native/core/one-halo
render differences, and actual source crops. Require unchanged held-out actor
PSNR/SSIM/LPIPS or improvement under the campaign rule, and inspect held-out
hair, feet and inter-person occlusions. No hidden-space certification follows
from these sparse observations. Do not select thresholds by a held-out crop.

Confidence is high that the inspected early leaders contain front-biased
termination on these shirt rays; medium that observed opaque TRAIN prefixes
are useful corrective evidence after requalification; unmeasured that the
proposed auxiliary will improve held-out actor detail. Sparse supervision,
wrong correspondences, calibration bias, moving-scene timing, transparency,
and an oversized uncertainty band remain material risks. No production files,
masks, checkpoints or occupancy states were changed for this proposal.
