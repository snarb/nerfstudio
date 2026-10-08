# Frame 000470: native 4K frequency-map canary on visible cord cameras

## What was tested

The initial [native-resolution gate](luster_000470_cord_fullres_gate.md) used
4096px source RGB but spatially transferred scalar-resolution frequency maps
fitted at 1920px. This can understate tiny details in LookCloser's
frequency-averaged pixel sampling and feature reweighting. A canary refitted
genuine 4K maps with the pinned 16-level, 1000-steps-per-level recipe on five
train cameras with annotated visible cord: 025, 026, 027, 097, and 098. The
other 157 maps were byte-identical to the transferred control. Original RGB,
train/eval split, calibrated poses, local cord ray sampler, 28-camera training
allowlist, and source 32.5k field/optimizer were unchanged. No masks or
held-out RGB entered map fitting or model training.

The 4K map fitting finished in 510 seconds with four supervised GPU workers,
no OOM. The [fit audit](/home/pavlo/renders/luster_000470_opacity/hybrid5_cord/refit_audit.json)
checks all 162 maps, exact calibration, image hashes, the five native fit
metadata files, and byte identity of the 157 untouched maps. The
[supervision log](/home/pavlo/renders/luster_000470_opacity/hybrid5_cord/logs/supervision.jsonl)
records worker liveness and GPU memory. The paired no-prior continuation
then trained 32.5k→33k with either the transferred or hybrid maps and
evaluated on the same native RGB at stride two. Code hashes match; ordinary
RGB/pixel batch hashes differ as expected because FAS uses the changed maps.

## Results

| Train camera | Projected cord median scalar resolution: transfer → true 4K | Cord patches at level ≥12: transfer → true 4K |
|---|---:|---:|
| 025 | 3838 → 8192 | 100% → 100% |
| 026 | 3699 → 8192 | 85% → 100% |
| 027 | 1585 → 8192 | 44% → 89% |
| 097 | 4097 → 8192 | 100% → 98% |
| 098 | 5753 → 8192 | 94% → 97% |

All five fitted maps raise the cord's median frequency estimate to the
maximum level, while high-frequency coverage across the *whole image*
decreases. The new maps therefore focus high-frequency sampling more tightly
on the rope rather than merely raising the frequency everywhere.

| 33k continuation | Eval PSNR | SSIM | LPIPS | cam095 cord PSNR | SSIM | LPIPS | cam097 cord PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transferred maps, control | **19.70369** | .81483 | .23199 | 23.95539 | .76829 | **.10119** | 26.61852 | .81212 | .08408 |
| Five true 4K maps | 19.65867 | **.81512** | **.23186** | **24.04909** | **.76843** | .10160 | **26.82789** | **.81346** | **.08249** |

The canary gains only `.094 dB` on held-out cam095 cord and `.209 dB` on
train cam097 cord; held-out cord LPIPS becomes slightly worse. Global PSNR
falls `.045 dB`, and the specified global PSNR/LPIPS selection rule retains
the prior 32.5k checkpoint for this hybrid run: its 33k PSNR is more than
`.07 dB` below that parent. The [GT/control/refit held-out crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres33000_refit5_cord_comparison.png)
and [cam097 train crop](/home/pavlo/renders/luster_000470_opacity/reviews/cord_geometry_20261008/lower_source_28/fullres33000_refit5_cam097_cord_comparison.png)
show no visible recovery of the lower braid. The canary's 500 optimization
steps took 32.85 seconds and ended without OOM. Its
[selection](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_refit5_033000/selection.json)
and [supervision log](/home/pavlo/renders/luster_000470_opacity/runs/cord_sharp28_fullres_refit5_033000/supervision.jsonl)
pin the result.

## Insights

The transferred maps did understate cord frequency, especially in cam027,
but correcting the five directly visible training views yields only a small
numeric cord benefit and no clear visual repair. Given the slight global PSNR
regression and checkpoint-selection failure, fitting all 28 maps is not the
next efficient experiment. The remaining lower-braid artifact depends more
on multiview geometry/appearance alignment or layer composition than on
frequency-map resolution alone. Continue from the stronger no-prior 4K
baseline while checking train-view colour at the estimated cord depth before
trying another transparency objective.
