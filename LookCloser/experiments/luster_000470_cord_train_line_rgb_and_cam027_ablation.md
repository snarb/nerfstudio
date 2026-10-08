# Frame 000470: train-line colour and cam027 sampling ablation

## What was tested

The [native 4K prior screen](luster_000470_cord_fullres_termination_prior.md)
put more termination weight near the approximate cord segment without
restoring the lower braid. A read-only audit therefore projected 12 points
along that **train-only** two-landmark segment into each of five annotated
training cameras (025, 026, 027, 097, 098). It compared each original RGB
sample with the rendered colour and with learned colour conditional on field
samples inside `±.003` normalized depth of the segment. These are sparse
diagnostic samples, not a cord mask or a certified surface. The field and
training data were not changed. The [audit receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres33000_train_line_rgb/result.json)
pins the checkpoint, source image hashes, calibration, fit and individual
ray results.

Cam027 was the largest line-colour outlier. To test whether its training RGB
conflicts with the cord reconstruction, one 500-step no-prior continuation
excluded **only cam027** from the sharp-28 whole-camera sampler, leaving 27
training cameras. The matched control retained all 28. Both resumed the exact
same 4K 32.5k field and optimizer, using identical original unmasked RGB,
poses, transferred frequency maps, local cord sampling, evaluation crops,
stride-two native RGB and code hashes. The [selection manifest](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/selection_without_cam027.json)
identifies the one removed camera; held-out views 012/095/150 were untouched.
The first sampled camera/RGB batches differ by design.

## Results

| Train camera | Sparse projected points | Rendered RGB PSNR | Segment-band RGB PSNR | Mean segment-band mass |
|---|---:|---:|---:|---:|
| 025 | 12 | 28.958 | 27.647 | .865 |
| 026 | 12 | 25.272 | 24.872 | .913 |
| 027 | 12 | 20.910 | 16.808 | .776 |
| 097 | 12 | 23.691 | 22.571 | .896 |
| 098 | 12 | 31.716 | 25.003 | .828 |

On the 60 sparse training points, rendered RGB scores `24.609 dB`, while
segment-band colour scores `21.609 dB`. At one lower cam027 point the band
has zero weight. The [original train/eval projection review](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/cord_midpoint_review.png)
shows why this cannot establish cam027 as defective: it sees a thin side view
and the straight segment does not stay centered on every braid twist. The
separate [train-only photometric sweep](luster_000470_cord_multiview_depth.md)
also finds larger per-camera depth disagreements on the lower segment.

| 33k continuation | Eval PSNR | SSIM | LPIPS | Held-out cam095 cord PSNR | SSIM | LPIPS | Train cam097 cord PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sharp 28 control | **19.70369** | .81483 | .23199 | **23.95539** | **.76829** | **.10119** | 26.61852 | .81212 | .08408 |
| Sharp 27, no cam027 | 19.69073 | **.81510** | **.23131** | 23.88924 | .76612 | .10315 | **26.79310** | **.81494** | **.08268** |

Removing cam027 slightly helps the train cam097 cord but worsens all three
held-out cam095 cord metrics; global PSNR falls `.0130 dB`. The
[GT/control/ablation cam095 crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres33000_skip027_cam095_comparison.png)
and [cam097 crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres33000_skip027_cam097_comparison.png)
were inspected: the lower braid remains blurred and fades into the skirt.
The ablation optimized 500 steps in `32.92 s`, finished without OOM, and
selected its 33k checkpoint under the global eval PSNR/LPIPS rule. Its
[metrics](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp27_without_cam027_033000/eval_033000/metrics.json),
[selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp27_without_cam027_033000/result.json)
and [supervision log](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp27_without_cam027_033000/supervision.jsonl)
pin the result.

## Insights

Cam027's weak colour agreement with a projected straight line does not mean
its RGB should be dropped: it supplies a distinct side view, and removing it
does not repair the held-out cord. More importantly, conditional colour at the
assumed segment depth is worse than the rendered mixture on every annotated
training camera. A harder opacity or colour objective centred on this line
would constrain uncertain geometry and can amplify misplaced density. Keep
the 4K no-prior control, retain cam027, and first improve train-only local
correspondence/cord geometry before trying a stronger prior there.
