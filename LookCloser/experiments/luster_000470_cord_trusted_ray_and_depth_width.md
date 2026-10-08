# Frame 000470: confidence-filtered cord rays and depth tolerance

## What was tested

The 4K no-prior field renders the lower hanging braid too softly, while a
`±.003` termination-mass prior moves opacity toward a train-only fitted 3D
segment but worsens cord colour. A read-only sweep first checked whether the
visible cord surface could be represented by shifting that depth uniformly
toward the camera. It reused the actual field samples and original RGB on 60
projected training rays in cams025/026/027/097/098. Candidate colour bands
had halfwidth `.0015`; each offset was compared with the zero-offset band on
the **same eligible rays**. The [sweep receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/curve_annotation_v1/front_surface_sweep/result.json)
pins the 33k checkpoint, calibration and per-ray samples. The
[source-camera projection patches](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/curve_annotation_v1/projected_line_native_patches.png)
show the approximate line on five original JPGs. This was diagnostic; no
held-out RGB entered its calculation.

A second audit repeated the sparse colour measurements at the common 4K
**32.5k source checkpoint**, before continuation. A static confidence
manifest selected a projected ray only if its `±.003` depth-band mass was at
least `.3`, its conditional RGB differed from original training RGB by at
most `.12` Euclidean units, and at least two training cameras passed at that
segment fraction. The [manifest](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/curve_annotation_v1/trusted_rays_32500_rgb012_mass03.json)
contains **42 of 60** rays over all 12 fractions, its thresholds, audit and
checkpoint hashes, calibration and fit hashes. No source masks or evaluation
views were used. The opt-in model control checks those hashes and restricts
only auxiliary cord-prior rays; ordinary RGB sampling remains unchanged.

Two no-prior/strong-prior comparisons resumed that same 32.5k field and
optimizer for 500 steps with identical 28 sharp cameras, 4096px source RGB,
frequency maps, local cord sampling, evaluation crop and stride-two eval.
The first used coefficient `.003` and tolerance `±.003` on the 42 trusted
rays instead of all 60. The second kept all 60 and coefficient `.003`, but
widened tolerance to `±.006`. All first 32 ordinary sampled pixel/RGB batch
hashes matched the no-prior and original prior runs. Both new runs completed
without OOM and selected their 33k checkpoints under the global eval
PSNR/LPIPS rule.

## Results

| 33k continuation | Eval PSNR | SSIM | LPIPS | Held-out cam095 cord PSNR | SSIM | LPIPS | Train cam097 cord PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No prior | 19.70369 | .81483 | .23199 | **23.95539** | **.76829** | **.10119** | **26.61852** | **.81212** | **.08408** |
| All rays, `±.003` | 19.69888 | .81497 | .23200 | 23.71776 | .75606 | .10585 | 25.93474 | .79523 | .09119 |
| Trusted 42 rays, `±.003` | 19.70929 | .81476 | .23182 | 23.76043 | .75776 | .10520 | 25.97103 | .79621 | .09073 |
| All rays, `±.006` | **19.71067** | **.81532** | **.23129** | 23.81641 | .76031 | .10363 | 26.14027 | .80014 | .08868 |

The confidence filter and wider tolerance each reduce the strong prior's
damage, but both still worsen all three cord metrics in held-out cam095 and
train cam097 versus no prior. Their [GT/control/all/trusted cam095
crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/curve_annotation_v1/fullres33000_trustedray003_cam095_comparison.png)
and [GT/control/narrow/wide crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/curve_annotation_v1/fullres33000_band006_cam095_comparison.png)
were visually inspected: neither restores distinct lower fibres. The
[cam097 trusted-ray comparison](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/curve_annotation_v1/fullres33000_trustedray003_cam097_comparison.png)
shows the same persistent blur. The [trusted-run metrics](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_trustedray003_033000/eval_033000/metrics.json),
[wide-band metrics](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_termination003_band006_033000/eval_033000/metrics.json)
and their supervision logs pin terminal results. The new optimization
segments each took about `39 s`, versus `32.77 s` with no prior.

On the same 65 near-line held-out cam095 rays selected by **control** band
mass above `.5`, the no-prior/all-ray/trusted-ray mean band termination mass
is `.83694/.91281/.91315`; back mass is `.11010/.03412/.03390`. The matched
ray RGB PSNR is `22.367/21.687/21.705 dB`, and conditional band RGB PSNR is
`22.739/22.213/22.229 dB`. Thus the confidence filter barely changes the
prior's geometric effect and does not prevent colour regression. This is a
read-only diagnostic, not a checkpoint-selection metric. The
[trusted-ray audit](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/curve_annotation_v1/trustedray003_ray_layers/result.json)
retains the exact pixels and field hash.

A uniform front-surface offset also failed. On 23 training rays with band
mass at least `.15` both at zero and at `−.002`, conditional colour PSNR is
`23.664 dB` at the line but `20.224 dB` closer to the camera. On 43 paired
rays, moving `+.002` farther gives `21.386 dB` versus `22.557 dB` at zero.
Per-ray best offsets span both signs. These results do **not** establish the
true surface depth, because the colour is from the already-trained field;
they refute a single front-face correction as a reliable shortcut.

## Insights

The simple depth segment is close to the rope in source images but does not
provide a trustworthy surface prior for its fine braid. Filtering only rays
whose current band colour agrees with source RGB does not stop the prior from
redistributing opacity and degrading held-out colour. Widening the depth
uncertainty reduces harm but leaves the visible defect. Both 500-step prior
arms are rejected; keep the 4K no-prior field as the cord control. The
[native multiview strips](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/curve_annotation_v1/multiview_cord_strips_native.png)
show repeated twist patterns and view changes, so no new midpoint is claimed
without a demonstrably consistent cross-view correspondence. A credible next
depth prior requires local 3D surface evidence rather than a stronger or
more selective penalty on this two-endpoint line.
