# Frame 000470: prior at the front of a finite-width cord

## What was tested

The previous [guided-stereo test](luster_000470_cord_guided_stereo_prior.md)
failed when it moved a single 3D target away from the approximate cord
centerline. The original train-camera crops show the fitted line running near
the visible braid in cams025/026/027/097/098. A different hypothesis is that
the line is a **centerline**, whereas an opaque cord should terminate rays at
its front surface. A tube radius of `.0011` in normalized scene units gives
an analytic front-depth offset of `.00110–.00116` on the selected rays and
projects to approximate half-widths of `5.3–10.2` original pixels across the
five views. This radius is an approximate image-based screen, not measured
ground-truth surface geometry. The [train-camera source crops](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/guided_stereo_v1/native_cam097_line_annotate.png)
and [front-depth geometry receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/guided_stereo_v1/tube_front_radius0011_geometry.json)
pin the construction. No dataset masks were used.

The opt-in `cord_ray_tube_radius` moves the auxiliary termination target from
each centerline point to the first intersection of its camera ray with an
ideal cylinder around that line. It accounts for the angle between ray and
axis. Camera ray directions, ordinary RGB sampling, 4K source photos, fine
cord sampler, 28-camera allowlist, opacity coefficient `.003`, and `±.003`
depth tolerance stay fixed. It cannot be combined with the rejected stereo
point override. The paired control and candidate used the same eight
train-only rays at segment fractions `.691/.844` and resumed the identical
32.5k no-prior field and optimizer for 500 steps. Their first 32 ordinary
pixel/RGB batch receipts match exactly. Both completed without OOM; the
500-step optimization segments took `38.94/38.89 s` for line/tube.

A read-only check of the source checkpoint on those eight selected training
rays found conditional field-colour PSNR `24.44 dB` at the line and `24.47 dB`
at offset `−.001`, so a forward test was justified. This is sparse
train-view evidence only; its [preflight receipt](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/guided_stereo_v1/tube_front_radius0011_preflight.json)
does not establish the held-out surface.

## Results

| 33k continuation | Eval PSNR | SSIM | LPIPS | Held-out cam095 cord PSNR | SSIM | LPIPS | Train cam097 cord PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No-prior 4K baseline | 19.70369 | .81483 | .23199 | **23.95539** | **.76829** | .10119 | **26.61852** | **.81212** | **.08408** |
| Matched eight-ray centerline | 19.69660 | .81508 | .23168 | 23.93163 | .76598 | .09920 | 26.31441 | .80522 | .08471 |
| Eight-ray front tube `.0011` | **19.71879** | **.81554** | **.23087** | 23.93558 | .76471 | **.09906** | 26.30257 | .80425 | .08463 |

The tube improves global PSNR by `.022 dB` and LPIPS by `.00081` against its
matched centerline control, but held-out cord PSNR improves only `.004 dB`,
its SSIM falls `.00127`, and it remains worse than the no-prior baseline on
cord PSNR/SSIM. The [cam095 GT/baseline/line/tube crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/guided_stereo_v1/fullres33000_tube0011_cam095_comparison.png)
and [cam097 crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/guided_stereo_v1/fullres33000_tube0011_cam097_comparison.png)
were inspected. The lower plait remains a smooth, fading teal strip; neither
view shows a clear recovery of individual strands. The [line selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_u8u10_line_003_033000/selection.json)
and [tube selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_u8u10_tube0011_003_033000/selection.json)
contain full-view and person-crop PSNR/SSIM/LPIPS and checkpoint paths.

A frozen, exact-sample audit compared the same 91 held-out cam095 pixels
within `.002` of the line in both fields. On the 36 lower rays, the tube
candidate **reduces** native termination mass in the `±.003` line band
`.696→.646`, while front mass increases `.053→.083` and back mass
`.252→.272`. Conditional colour PSNR in that band falls `19.96→16.49 dB`;
rendered RGB PSNR falls `21.07→21.02 dB`. The upper 55 near-line rays remain
nearly unchanged. This local diagnostic is more specific than the whole cord
ROI and explains why the small global gain is not a visible repair. The
[line ray audit](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/guided_stereo_v1/tube0011_ray_audit_line/result.json)
and [tube ray audit](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/guided_stereo_v1/tube0011_ray_audit_tube/result.json)
pin the field hashes and exact pixels.

## Insights

A physically plausible front-surface shift avoids the large degradation from
the arbitrary stereo shift, but does not restore the lower braid. A single
radius around this two-landmark line is too weak a description of the twisted
surface and its cross-view colour. The tube candidate is **not accepted** as
the cord recipe. Keep the no-prior native-4K baseline and avoid tightening
this depth/opacity penalty without independently validated local surface
correspondence. The small global metric gain can be studied separately, but
does not outweigh the missing visible detail that motivated the experiment.
