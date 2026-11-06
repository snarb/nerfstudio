# TRAIN semantic admission feasibility for the actor objective

## What was tested

CPU-only intersection of the frozen actor-objective checkpoint15188 actual active grid with the reviewed `occlusion_v2/W64/semantic_gate_pad1`. No model or GPU was loaded, and no production grid was changed. The source is63TRAIN cameras; DC/HC/KA masks and RGB never contribute to admission. The existing103 TRAIN witness queries and12 posthoc DC diagnostic rays remain sealed and separate.

The mapper admits a whole LookCloser cell whenever any intersected semantic cell is admitted. It includes boundary contact and retains outside-domain UNKNOWN. XYZ axes and accepted floor coordinates are checked against dataset hashes, AABB, transform and checkpoint receipts. `world_to_box` is deliberately not applied twice. Semantic pad1 is a fresh26-neighbor source dilation; it is distinct from native learned-grid halo. Closed-cell mapping adds a further conservative boundary cover: source pad1 has105,049 cells; mapped admission has131,911 of2,097,152. The original source, rather than the stricter positive-support variant, is used.

## Results

| Check | Result |
|---|---:|
| Actual learned active cells |123,817|
| Hypothetical active ∩ admitted cells |36,438|
| Source admitted-cell centers retained |105,049/105,049|
| Sealed witness XYZ and actual pixel-ray depth points retained |103/103 both tests|
| Independently reviewed toe/sole anchors retained |2/2|
| Existing native sample mass with midpoint in witness±.02 definitely excluded |0 across103 rays|

Conservative whole-interval rejection bounds how much current termination mass lies fully outside admission. Mixed intervals are retained. These are existing weights, not a prediction of a rerender or retraining result. Early/band group membership in this CPU screen uses native sample midpoints; full-interval rejection itself requires the whole segment cover to be excluded. The separate depth-comparison reports integrate interval fractions at band boundaries.

| Diagnostic group | Existing early mass (%) | Definitely excluded early mass (%) |
|---|---:|---:|
| Old18 source rays |94.799|90.282|
| Old8 local reserves |99.655|99.521|
| New66 source rays |4.443|4.443|
| New9 whole-patch reserves |0|0|

For the separate DC evaluation diagnostics, excluded total mass is95.368% over four bad grey-garment rays and78.848% over four bad black-shirt rays. Four healthy pants controls lose only0.00148%. A center-only classification would report98.946% for the black-shirt group; the stricter interval result is reported to avoid overclaiming removal at boundaries.

[Complete numeric report and hashes](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_admission_feasibility_20261106/diagnostic/report.json). [Whole-grid three-axis projections](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_admission_feasibility_20261106/diagnostic/whole_grid_projection.png) and [excluded-mass plot](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_admission_feasibility_20261106/diagnostic/excluded_mass.png) were visually inspected.

## Insights and proposed controlled experiment

The evidence supports testing training-time semantic admission as one change to the failed actor-objective parent. It would remove most of the measured false front sheets while preserving the qualified positive evidence. Source UNKNOWN remains allowed; an excluded cell is a conservative semantic hypothesis, not a proof of physical emptiness. Two sparse shoe anchors do not certify entire feet. Point5568 remains a hair/chest material ambiguity; its sealed query is retained for comparison rather than asserted to be opaque cloth.

A separate candidate should intersect the actual native learned active grid with the fixed admission map during training and evaluation, preserve the learned core independently, and apply noncumulative native halo before the final admission intersection. Warmup and restore must enforce the same rule. Save source/mapping/native-core/active hashes separately; never overwrite the learned core with the semantic mask. Keep the actor parent recipe, sampling, seed, schedule, native exp activation and SH correction unchanged, with no simultaneous prefix or positive-band addition. Compare against the unchanged actor parent and full-objective reference at matched gates, inspecting DC failures, both actors, hair and feet,103 geometry rays, PSNR/SSIM/LPIPS and eventual plateau. This report does not authorize a production launch or promote the geometry.

Implementation: `scripts/marshawn_semantic_cell_admission.py` and `scripts/audit_marshawn_semantic_admission_feasibility.py`. Four CPU tests cover arbitrary anisotropic XYZ overlap against brute force, boundary contact, outside UNKNOWN, center-only failure and full-grid retention. Existing4,657,323TRAIN-ray source coverage and103+shoe receipts are hash-bound in the audit source directory. No model mutation occurred.
