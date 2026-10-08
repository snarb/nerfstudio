# Frame 000470: source-RGB anchoring inside the cord depth band

## What was tested

The [confidence-filtered depth screen](luster_000470_cord_trusted_ray_and_depth_width.md)
showed that a termination-mass prior moves held-out cam095 ray weight toward
the approximate cord depth, but the learned colour at that depth becomes
worse. To test that specific failure, a weak auxiliary RGB anchor queried
seven field samples inside the `±.003` band on the same 42 train-only,
confidence-selected rays. It compared their density-weighted conditional
colour with **original training RGB** at those source pixels. Density and
geometry features were detached for this auxiliary colour query, so its
direct gradient reached only the colour head; the separate termination-mass
prior still controlled density. The RGB targets came from the hash-pinned
[train-only ray manifest](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/curve_annotation_v1/trusted_rays_32500_rgb012_mass03_with_rgb.json)
made at the common 4K 32.5k source checkpoint. The runner verified the
source audit, original-image hashes, fit, calibration and source checkpoint.
No dataset masks or held-out RGB were used.

Two 500-step continuations changed only the auxiliary colour coefficient:
`.01` and `.05`, compared with the already completed same-source 42-ray
prior at zero. All used the same 28 training cameras, original 4096px RGB,
transferred frequency maps, local cord sampling, opacity coefficient `.003`,
depth tolerance `±.003`, seed, and stride-two native evaluation. The first
32 ordinary sampled pixel/RGB batch hashes matched between the arms. Both
new runs reached 33k, were selected under the global eval PSNR/LPIPS rule,
and ended without OOM.

## Results

| 33k continuation | Eval PSNR | SSIM | LPIPS | Held-out cam095 cord PSNR | SSIM | LPIPS | Train cam097 cord PSNR | SSIM | LPIPS | Held-out cam095 face PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No cord prior | 19.70369 | **.81483** | .23199 | **23.95539** | **.76829** | **.10119** | **26.61852** | **.81212** | .08408 | 27.18092 | .07805 |
| Trusted depth, no RGB anchor | **19.70929** | .81476 | .23182 | 23.76043 | .75776 | .10520 | 25.97103 | .79621 | .09073 | **27.18650** | **.07764** |
| Trusted depth + RGB `.01` | 19.69180 | .81478 | **.23180** | 23.79541 | .75738 | .10569 | 26.00778 | .79658 | .09198 | 27.15515 | .07925 |
| Trusted depth + RGB `.05` | 19.68424 | .81345 | .23244 | 23.62142 | .74598 | .10284 | 25.82828 | .77589 | **.07896** | 27.01046 | .08143 |

The `.01` arm slightly improves held-out cord PSNR versus the 42-ray prior
without colour, but worsens its LPIPS and remains below the no-prior control
on all three cam095 cord metrics. The `.05` arm improves LPIPS versus the
unanchored prior but lowers PSNR, SSIM, global metrics and face quality. The
[GT/control/prior/colour-strength cam095 crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/curve_annotation_v1/fullres33000_color_strength_cam095_comparison.png)
was inspected: neither anchor resolves distinct lower braid fibres. The
[cam097 crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/curve_annotation_v1/fullres33000_color005_cam097_comparison.png)
shows the corresponding train-view appearance. The
[weak-run metrics](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_trustedray003_color001_033000/eval_033000/metrics.json),
[strong-run metrics](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_trustedray003_color005_033000/eval_033000/metrics.json),
and their supervision logs pin the result. The new optimization segments
took `39.68/39.66 s` for `.01/.05`, versus `38.98 s` without the RGB anchor.
An isolated backward pass from the common source checkpoint verified that
the auxiliary RGB term had a nonzero colour-head gradient (maximum `.13745`)
and zero direct gradient on every other field parameter. Its [read-only
gradient receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/curve_annotation_v1/color_anchor_gradient_audit_v2/result.json)
also verifies unchanged field weights. This isolates the direct auxiliary
gradient, not the indirect effects of later ordinary RGB optimization.

The anchor does change the intended quantity. On exactly the same 42
selected training rays, conditional band RGB PSNR at the 32.5k source was
`28.334 dB`. At 33k it falls to `24.511 dB` with the depth prior alone, but
recovers to `27.056 dB` with colour coefficient `.05`. Full rendered RGB on
those 42 rays rises `25.280→27.458 dB` relative to the unanchored prior.
Across all 60 diagnostic train rays, conditional band PSNR rises
`19.870→20.645 dB`; the remaining nonselected rays are still poor. The
[unanchored](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/curve_annotation_v1/trainline_33000_trusted/result.json)
and [anchored](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/curve_annotation_v1/trainline_33000_color005/result.json)
read-only audits preserve source RGB, checkpoint and field hashes.

## Insights

Local colour at the assumed depth can be corrected using source RGB, but
that correction does not generalize to the held-out cord and harms other
person regions at the stronger coefficient. The limitation is not merely an
unconstrained colour head: the two-landmark depth/visibility assignment or
cross-view correspondence remains unreliable. Both RGB-anchor arms are
rejected at the 33k gate. Keep the no-prior 4K control and avoid stronger
colour or opacity supervision on this approximate segment until local 3D
surface correspondence is independently established.
