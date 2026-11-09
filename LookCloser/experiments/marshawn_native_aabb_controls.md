# Marshawn1660 native NGP AABB controls

## What was tested

The first native cube±32 sanity used a nominal finest cell of64/2048=.03125
and automatic world ray step.110851. The two-person subject is about.411 units
wide: only13 finest cells and4 ray intervals across it. Native also silently
created32 appearance channels when both appearance flags were false.

Corrected controls use the same sealed gain4 SDR dataset, unchanged prepared
poses/K/splits, native `NGPModel`/`DynamicBatchPipeline`, hash21, explicit
`raw_no_appearance_embedding=True`, constant step.003, and a262144-point target.
Initial ray estimate16384 starts with16 rays; it does not cap ray samples.
Online eval chunk512 is explicit after the corrected cube32 run. Each cold run
is capped at22000 attempts and retains the quiet runner's default early stop.

Corrected cube32 used maxres65536; cube4 used8192, both nominal finest
world cell.0009765625. The exact scene.tight control keeps maxres8192 and
changes only the existing identity-normalization parser's explicit AABB.
Native model/field source hashes remain `c40ce53b…` / `f2853a7b…`.

## Results

These are real all-three-view metrics. Selected final checkpoints received a
fresh native HD eval; online rows are shown at their recorded precision.

| Bound | Step | PSNR | SSIM | LPIPS | Visual result |
|---|---:|---:|---:|---:|---|
| cube±32, corrected SDR |2000|21.1554|.674755|.712853|People largely absent in opaque blur; rejected|
| cube±4 |2000|23.8605|.709544|.579813|People recognizable in all3 views|
| cube±4 |4000|24.4494|.732867|.489979|Sharper silhouettes; residual floaters|
| cube±4 |6000|24.8069|.745260|.445898|Online eval|
| cube±4, selected/fresh |8000|24.923300|.754104|.424170|All3 views manually reviewed at640px|
| scene.tight |2000|23.0547|.697285|.534433|People recognizable|
| scene.tight |4000|24.0735|.727891|.446753|Online eval|
| scene.tight, selected/fresh |6000|24.382591|.740420|.396618|All3 native renders manually reviewed|

Cube32 was stopped at CSV3580 (3581 recorded spent attempts). Cube4 default
early stop ended after CSV9360; scene.tight ended after6460. Conservative
attempt bounds for the latter runs are9370 and6470, both below22000. These
early stops do not establish metric convergence: all-view quality was still
improving. Selection uses saved all-eval PSNR with LPIPS tie-break within.07dB.

Scene.tight volume81.4949 versus cube4 volume512 improves LPIPS at matched6k,
but lowers PSNR/SSIM. It is not an unqualified winner. Sparse containment does
not certify all visible floor/background: approved scene.loose volume154.6652,
bottom−.6741, is being tested against tight bottom−.3922.

Artifacts:

- [cube4 8k640 previews](/fsx/tmp/marshawn_001660_hd/native_ngp4_corrected_sdr/preview_8k_640/receipt.json)
- [exact-tight final GT/pred contacts](/fsx/tmp/marshawn_001660_hd/native_exact_scene_tight/final_best6k_contacts/HC_GT_prediction.jpg)
- [exact-tight live bound receipt](/fsx/tmp/marshawn_001660_hd/native_exact_scene_tight/actual_bounds.json)

Native nerfacc recomputes ROI center/extents even at enlargement factor1;
the exact-tight occupancy AABB differs from the parser by at most2.384e−7.
All model/field/parser bounds match FP32 exactly; the occupancy bound passes
the existing1e−6 tolerance. No grid/field source patch was used.

## Free-space and occupancy evidence

The read-only audit selects the same128 actual train track rays from63 cameras,
using the same9022 SfM witnesses and seed42. Uniform sigma queries include cells
skipped by native occupancy; their opacity is a field diagnostic, not a direct
claim about rendered opacity. Native packed pre-margin weights are also saved.

| Frozen checkpoint | Median free tau | Median free opacity | Median native pre-margin weight | False-free witness points |
|---|---:|---:|---:|---:|
| cube4 8k |.400908|.330288|.257594|15/9022|
| exact-tight6k |.683622|.495214|.280129|77/9022|

The audit excludes3 voxel diagonals or5% of witness distance around the surface.
The cube4 margin is.32476 and tight margin.20241. Reintegrating cube4 over the
same tighter prefixes gives median opacity.35503; step counts still differ,
so this comparison does not isolate AABB causality.

Inference-only noncumulative halo1 on cube4 8k reduced missed witness points
15→1, occupying603606→1296728 cells. Seven missed points have actual sigma24–831
but low cell EMA: source patches show tripod/boot/locker and shoulder/arm edges.
Halo1 admits all7. The remaining point6349 has sigma.00653 and requires geometry
review; expansion cannot recover absent field density.

Candidate-free ray-segment cell admission increased22.5%→61.7% with halo1.
Partial ray evidence does not prove an entire cell empty. All3 default/+1
renders and15 two-view source crops were inspected; field/grid state was
restored to its original hash after the probe.

- [halo report and witness details](/fsx/tmp/marshawn_001660_hd/native_ngp4_corrected_sdr/halo_8k/report.json)
- [missed-witness source crops](/fsx/tmp/marshawn_001660_hd/native_ngp4_corrected_sdr/halo_8k/missed_witnesses_contact.jpg)
- [cube4 free-ray audit](/fsx/tmp/marshawn_001660_hd/native_ngp4_corrected_sdr/free_space_8k/audit.json)
- [pure-tight free-ray audit](/fsx/tmp/marshawn_001660_hd/native_exact_scene_tight/free_space_6k/audit.json)

## Insights and next checks

Large-box native defaults confounded representational and ray sampling density.
The corrected cube4 is a recognizable baseline, while corrected cube32 still
failed the early visual gate. Native unit-density initialization over a much
longer world interval is a plausible additional contributor; it was not changed.

Subsequent measured free-space density and missing surface opacity justified
separate SfM LOS and surface-band opacity controls. These add no neural
parameters, RGB changes, or occupancy labels. The surface auxiliary uses the
native cached log-density to avoid the native truncated-exponential surrogate's
large inverse-density gradients at very negative logits. Its opaque-feature
assumption can fail for reflections or transparent features. Geometry and
actual CUDA gradient/checkpoint probes passed before training.

## Final matched pair

Approved `scene.loose` is the smallest sufficient tested candidate: volume
154.6652, bounds `[-3.53424656,-3.88523223,-0.67410045]` to
`[3.50251317,3.61259731,2.25736250]`. Medium at 8k scored
26.5984/.772654/.361541 versus loose 27.0844/.772500/.362974 and showed no
coverage advantage in all three matched views. This is a discrete experimental
choice, not proof of a continuous global AABB optimum.

The final pair uses sealed scalar-corrected SDR RGB, identical poses/intrinsics,
seed 42, the native architecture/settings above, LOS weight .01 and surface
weight .01. Radius 1 applies exactly one noncumulative expansion after each
fresh native core update every 16 steps. The saved radius-1 mask is not expanded
again on load. Both runs used the same DEV3 runtime and stopped normally under
the default stopping rule, below 22,000 attempts each.

| Training grid policy | Selected checkpoint | Fresh all-view PSNR | SSIM | LPIPS | Conservative attempt bound |
|---|---:|---:|---:|---:|---:|
| Native radius 0 | 8000 | 27.532591 | .780458 | .359162 | 8170 |
| Noncumulative radius 1 | 8000 | 26.243208 | .762026 | .399108 | 8140 |

Radius 0 wins the defined all-view quality selection. All six 640px selected
renders were manually inspected: both main people are opaque and recognizable,
and the mat and floor remain visible. Fine detail and some peripheral geometry
remain blurred. A training-time halo did not improve the quality metrics here.

- [Final summary and exact artifact paths](/fsx/tmp/marshawn_001660_hd/final_pair/run_summary.json)
- [Radius-0 HC render](/fsx/tmp/marshawn_001660_hd/final_pair/radius0/preview_8k_640/eval_HC_rgb.png)
- [Radius-1 HC render](/fsx/tmp/marshawn_001660_hd/final_pair/radius1/preview_8k_640/eval_HC_rgb.png)
- [Radius-0 actual grid receipt](/fsx/tmp/marshawn_001660_hd/final_pair/radius0/grids/step-000008000_grid.json)
- [Radius-1 actual grid receipt](/fsx/tmp/marshawn_001660_hd/final_pair/radius1/grids/step-000008000_grid.json)
- [Independent source/config review](/fsx/tmp/marshawn_001660_hd/final_pair/independent_geometry_review.json)

Each grid archive contains the actual cached GPU core, active mask, EMA values,
world bounds and resolution, bound to its checkpoint and source hashes. CPU
threshold recomputation is diagnostic only. Full independent selected-8k
BODY128, novel-view, and global slice audits remain the release gate; neither
grid is certified ideal from training metrics or sparse containment alone.
