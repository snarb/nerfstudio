# Marshawn: reviewed clothing sources and controlled geometry arms

## What was tested

Recover original detected SIFT observations and manually qualify additional opaque clothing across both actors. This prepares a separate geometry experiment; it does not change the running campaign, current images, masks, AABB, calibration or occupancy.

Artifact: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/expanded_clothing_geometry_v1/manifest.json`. It remains `approved_for_training=false` until the source review, native AMP probe and separate worker integration are accepted.

## Results

The initial semantic candidate pool contained308 points. Manual inspection rejected many black-actor hair, face and hand features: semantic interior alone does not identify safe clothing. Twenty-five new points were inspected in three original TRAIN source photographs each, using the actual detected feature location. The selected new support covers black trouser lettering and sleeve cloth; grey jacket shoulder, upper/lower back, forearm, elbow and front; and grey trouser cloth.

| Source group | Points | Reviewed rays used in prior |
|---|---:|---:|
| Existing black upper-back lettering, unchanged | 9 | 18 |
| New black trouser lettering | 6 | 18 |
| New grey jacket regions | 16 | 48 |
| New reserved black sleeve and grey trousers | 3 | 0 |
| Final expanded prefix supervision | 31 | 84 |

New reserves are point5568 (black sleeve) and points2219/5556 (grey knee/hip trousers). No new supervision occupies those garment patches. Existing nearby shirt reserves63/64/71/72 remain excluded, and new black back-lettering points were not admitted. The six trouser lettering points remain one correlated patch, not six independent geometry controls. The reserves are still TRAIN photographs/rig evidence and are not independent scene-depth ground truth.

Review sheets, all visually checked:

- [Black trouser lettering, page1](/fsx/tmp/marshawn_001660_hd/tuning_20261106/expanded_clothing_geometry_v1/review_01.jpg)
- [Whole-patch reserves and grey shoulder, page2](/fsx/tmp/marshawn_001660_hd/tuning_20261106/expanded_clothing_geometry_v1/review_02.jpg)
- [Grey back, forearm and elbow, page3](/fsx/tmp/marshawn_001660_hd/tuning_20261106/expanded_clothing_geometry_v1/review_03.jpg)
- [Grey back, shoulder and front, page4](/fsx/tmp/marshawn_001660_hd/tuning_20261106/expanded_clothing_geometry_v1/review_04.jpg)
- [Grey jacket patches, page5](/fsx/tmp/marshawn_001660_hd/tuning_20261106/expanded_clothing_geometry_v1/review_05.jpg)

Original detector caches and source RGB hashes were verified. The recovery gathers all detected keypoints within the original one-pixel fit radius, reconstructs the original directed0.65 descriptor-ratio edges, and accepts only a connected component with exactly the original camera membership and one feature per camera. This resolves multiple SIFT orientations sharing the same pixel; nearest-keypoint selection alone broke four of these tracks. Triangulation uses original source cameras, then the already accepted rigid floor transform.

All25 recovered graphs reproduce the stored original XYZ to at most1.95e-16. Their actual detected-coordinate fit residuals reach0.966HD pixels. Where at least four source views exist, refitting without each view gives a maximum held-out source-feature residual1.064HD pixels and maximum XYZ shift0.001011 prepared-world units. Three-view tracks do not receive a false leave-one-out three-view confidence claim. These measurements are source consistency evidence, not independent calibration uncertainty; the fixed rig already influenced historical track acceptance.

The builder keeps only the three manually inspected source views per new point for supervision. Other original observations establish its fitted track. Projected witness rays are retained for compatibility with the existing prefix arm; actual detection UV and their nonzero residuals are explicitly preserved in `visibility_review.json`. No projected zero-residual quantity is reported as measured feature accuracy.

The original18 rays are retained verbatim. New prefixes use the same `max(.01, .05*depth)` gap before the witness and current AABB intersection/inset. All84 rays pass the existing coordinate, normalized-direction, bounds, original-observation and source-count checks. No duplicate reviewed camera/pixel pairs were found. No hair, skin, boots, floor or stationary bag point was admitted.

## Controlled arms

Use the same frozen SH-corrected full/body80/R0 recipe and seed42 for all arms. Keep RGB loss, learning rates, frequency schedule, distortion, occupancy, checkpoint gates and validation/plateau policy identical. Do not mix in concurrent exponent or Jacobian experiments.

| Arm | Geometry addition | Controlled comparison |
|---|---|---|
| A | Original18-ray prefix, lambda0.003 | Versus SH-corrected reference: compatibility of the previously tested prefix with corrected view directions |
| B | Expanded84-ray prefix, same lambda0.003 | Versus A: source coverage only |
| C | B plus positive opaque-clothing band | Versus B: positive surface support only |

A and B preserve128 camera-balanced auxiliary rays,64 stratified prefix samples and the same private stateless seed policy. The expanded set necessarily changes which camera groups are available; retain the existing camera-balanced rule and report this consequence of source expansion. Do not silently change to actor or patch balancing.

For C, begin with the previously successful native band-distance policy, halfwidth `max(.005, .01*depth)`, clipped inside the same AABB with inward endpoint protection. Bind measured source residuals and leave-one-view-out shifts to the artifact and require each available shift to fit inside the proposed band; widen or reject a source only by an explicit predeclared policy, never by looking at held-out render quality. A band width is a soft prior, not a confidence interval, especially for three-view sources. All reserves remain excluded from both prefix and band terms.

Evaluate the field's actual `density_fn` on16 stratified samples per selected band. Integrate native density in FP32 to optical thickness, then use a numerically stable opacity negative log probability. Do not reuse historical helpers that infer `exp(raw_logits)` or replace gradients: the current field uses native softplus. The band sampler must use its own deterministic seed stream and leave prefix sample draws unchanged. Band coefficient is pending a real native AMP zero-update gradient probe;0.001/0.003 are probe candidates, not an authorized coefficient. Inspect separate RGB/prefix/band gradients and their combined gradient, finite native-scale backward, model immutability and private/global RNG behavior.

Before any new production launch, hash-bind the artifact and actual code/spec, run the real4096-ray AMP gate, and verify at least one successful Adam update before and after a durable technical restart. Keep the mandatory106316-step trajectory and subsequent validation plateau policy. Score masked actor LPIPS/PSNR/SSIM and evaluate the old nearby reserves and new whole-patch reserves separately; retain transmission in depth-mass accounting. Inspect all three held-out full renders for sheets, holes and changes around limbs.

## Insights and next steps

The simpler SH correction already improved matched15k shirt geometry as well as RGB; prefix-only cleared its fitted rays more than nearby reserved rays. The A/B/C sequence tests whether broader clothing coverage repairs that local behavior and whether positive surface support is needed. Current support remains sparse and uneven. In particular, black pants lettering is one patch, grey jacket fabric dominates new regions, and there is no qualified shoe or hair geometry. None of these artifacts certify an ideal whole-scene occupancy grid.

Ten focused CPU tests pass, including duplicate-orientation graph recovery, rejection when only projected positions exist, nonzero detector residual preservation, and existing prefix gradient/RNG/coordinate safeguards. Actual source preparation validates all84 rays. No additional training was launched for this source qualification.
