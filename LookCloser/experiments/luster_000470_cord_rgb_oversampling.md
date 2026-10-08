# Frame 000470: train-only RGB oversampling around the cord

## What was tested

The cord occupies few pixels per full frame. To test whether too few source
RGB rays caused the fading braid, an opt-in sampler replaced 64 of each 4096
ordinary train rays with real source pixels near the projected train-fitted
cord line. Five sharp training cameras (025–027, 097–098) were used; samples
were uniform along the line and jittered `±6 px` across it. This is ordinary
RGB supervision from the existing photos, not an image mask or synthesized
colour/depth. The held-out cam095 was excluded.

The candidate and zero-prior local-sampling control resumed the same sharp-28
32k field and optimizer, kept local rendering radius `.005` and step `.001`,
and trained to 33k with identical gates. A [setup audit](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/rgb64_setup_audit.log)
compared the step-32001 pixel batches: all 4032 untouched indices **and RGB
values** matched exactly. The other 64 pixels were read from the cached source
images under a separate deterministic RNG stream. Setup checked projected
camera coordinates; a focused unit test checks source RGB and repeatability.
Both runs completed without OOM.

## Results

| Train sampling | Step | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | SSIM | LPIPS | cam097 cord PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordinary control | 32500 | **19.8485** | **.81657** | **.22094** | **22.2898** | **.77768** | .06761 | 27.2262 | .06825 |
| Cord RGB +64 | 32500 | 19.8073 | .81581 | .22112 | 21.8582 | .77298 | **.06739** | **27.5840** | **.06461** |
| Ordinary control | 33000 | **19.7956** | **.81603** | .22152 | **22.5719** | **.78013** | **.06945** | 27.3386 | .06755 |
| Cord RGB +64 | 33000 | 19.7893 | .81567 | **.22108** | 22.0858 | .77476 | .06993 | **27.5791** | **.06403** |

The global selection rule picks 32.5k for the ordinary control and 33k for
the RGB candidate. The [32.5k GT/control/RGB crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_rgb64_32500_cam095_comparison.png)
and [33k crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/sharp28_rgb64_33000_cam095_comparison.png)
show no clear recovery of the lower strands on the held-out view. A narrow
diagnostic strip along lower rows 1340–1385 changes by at most `+.23 dB`
at tested widths `±4–8 px`; a wider strip covering rows 1250–1385 can worsen
by `.22 dB`. This approximate line strip is evaluation-only, not a train mask.
The first 1000 optimization steps took `65.09 s` versus `62.62 s` for the
ordinary local-sampling control, about 4% longer. The [candidate selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_rgb64_033000/selection.json)
and [supervision log](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_rgb64_033000/supervision.jsonl)
record the completed run.

## Insights

Oversampling makes the braid slightly better in the training camera but
reduces held-out cam095 cord PSNR by `.43 dB` at 32.5k and `.49 dB` at 33k.
The lower braid remains visibly less distinct than GT. This candidate is
rejected; source-pixel scarcity alone does not explain the artifact. Combined
with the complete-occupancy result, the remaining problem points toward
cross-view correspondence or learned geometry/colour separation. The global
person-quality checkpoint remains the earlier sharp-28 30.5k candidate, and
the best tested cord-specific prior-free variant remains the 33k control with
local rendering and `.003` occupancy admission.
