# Marshawn: proposed final frequency stage

## What was inspected

The 3D frequency comparisons remain pending; an optional schedule, synthetic
technical check, and isolated TRAIN map pilot are implemented below. The pilot
does not edit datasets or the approved63-camera queue. Primary references are [the local paper](../Paper%20LookCloser.md)
§4.1 and §7.3, and `nerfstudio/scripts/lookcloser_preprocess.py` lines813–1000.
The inspected source SHA is
`f1a3b3dbc2a72b4173c3e943ed4cd21f19526b97c7e9761e9f8150199c061855`,
identical to the frozen production preprocessing source.

The paper defines frequency as the first progressive level whose patch SSIM
passes a threshold; it does not prescribe the exact .95 threshold, 8-pixel
patch, or the current 2D constant .01 learning rate. The actual fitter uses
one Adam optimizer across progressively enabled levels, random pixels, constant
LR .01, and SSIM window7 on nonoverlapping patch8/stride8. Resolved cells retain
the first passing level; unresolved cells receive level15 as a fallback.
The maximum label is therefore not evidence that a patch passed the criterion.

## Two informative alternatives

First select the completed structural/geometry recipe, evaluation domain and
training resolution. Keep that exact choice, photometry, masks, pose/AABB,
projection mode, 3D optimizer/sampling and plateau rule fixed in all comparisons.
Compare these two independent factors against the same baseline:

| Alternative | Sole intended intervention | Question |
|---|---|---|
| A: per-level 2D LR decay | .01→.001 exponentially across the existing updates within each level; reset LR to .01 at the next level, retain Adam moments | Does settling the fit before each SSIM gate improve useful frequency assignments? |
| B: SSIM threshold .93 | .95→.93; retain the baseline constant .01 LR, budget, patch8 and window7 | Is the strict assignment criterion excluding recoverable actor detail despite adequate fitting? |

For the current 4K candidate, A uses the existing4000 updates/level and
`lr(j)=.01*10**(-j/3999)`, j=0…3999. There is no optimizer reset, extra fit
budget, RNG alteration, full-level training substitute, or source change.
The exact baseline `train_steps_per_level` is retained: HD1000 or reviewed
4K4000. For HD, use `lr(j)=.01*10**(-j/999)`, j=0…999; do not change
1000→4000 while testing the schedule. No resolution/budget change is bundled
into A.
B's lower threshold is an ablation, not an assertion that low censoring is good.
A and B are independent comparisons, not an unlabelled combination.

Historical full-HD preprocessing rejected .93 because it appeared to
under-label detail ([earlier report](lookcloser_frequency_map_preprocessing.md));
this is useful prior evidence, not a verdict for the current scene.
AA4000 and its repeat have materially different censoring, and AA16000 at
constant .01 performed worse than4000. These observations motivate the tests;
they do not establish either numerical noise or excess learning rate as the
cause, and do not predict 3D quality.

## Why physical patch16 is deferred

At4K, patch16 restores the approximate image-plane footprint of HD patch8,
but it is a different locality/statistic experiment. Leaving window7 unchanged
would not restore the SSIM physical support; changing to window13/15 would be
an explicit multi-setting support bundle. Current `compute_patch_starts`
excludes tail patches: height2024 is divisible by8, but patch16 would leave
8 bottom rows outside the regular lattice. It needs reviewed map-edge/UV
handling and downstream projection validation before any production map export.
Thus A and threshold B are the cleaner first two one-factor comparisons.
Physical-support experiments remain a separately documented later option.

## Optional schedule implementation and technical check

The preprocessing config now accepts `lr_final_per_level`; its default is
`None`, retaining the original constant-LR path. Alternative A sets `.001`.
The existing single Adam instance is retained across levels, and only each
parameter group's LR changes before an update. Map metadata records the
optional schedule. Threshold B already uses the existing `ssim_threshold`
option; it must leave the new LR option unset. Neither option changes the
currently frozen maps or production code copies.

Ten CPU checks pass, including exact1000/4000-update endpoints, constant
log-rate increments, and rejection of insufficient level budgets. Specializing
the new fitter to the default `None` produces the original function's exact
AST after removing dead schedule branches and renaming its unused loop index.
Receipt: `tuning_20261106/frequency_lr_technical_v2/default_AST_proof.json`.

An isolated synthetic64×64 CUDA test used one idle, exclusively locked silver
GPU2, one-GiB allocator cap and120-second external timeout. The actual fitter
completed in1.38seconds, peak tensor allocation610304bytes. All four fits
retain identical initial parameters, random pixel streams and final RNG;
every run has one Adam instance and state steps1…12. The decay fit reaches
both endpoints on each of three four-update levels without resetting moments.

The first probe's arbitrary1e-5 final-parameter parity assertion failed and is
preserved under `frequency_lr_technical_v1/failed_probe.log`. The follow-up
separately measured an unchanged-original repeat: its maximum parameter
difference is `.0130891`, exactly the same maximum measured between original
and default candidate in that probe. This demonstrates numerical variation in
the original test; its underlying cause is unproven. Bitwise numerical
equivalence is **not** certified, and this is not a scene-map quality test.
The follow-up receipt explicitly records that limitation. No new scene fit,
NeRF training recipe or frequency-map promotion follows from this probe.

## Evidence required

Preserve initial model/RNG/first-sample hashes, per-level SSIM counts and exact
censor masks, plus the fixed shirt/pants/hair/folds/shoe GT/pred/residual patches.
Use a small declared multi-camera TRAIN pilot before authorizing a full export;
keep EVAL untouched. A repeat on the same camera/runtime is necessary to
interpret the already-observed trajectory variation. These are diagnostics,
not optimization targets.

The decisive comparison is the completed 3D model's unchanged V1 actor LPIPS,
PSNR and SSIM, per actor/view, accompanied by heldout visual and geometry checks.
Keep geometry-supervised/native or semantic-training recipes explicitly labelled;
unchanged V1 measurements remain comparable. No map is promoted from its
censor rate, and a lower rate does not certify better 3D detail or temporal
stability. Additional threshold variants or combinations follow only a
substantial measured gain. Frequency remains the final tuning stage.

## Isolated HD map pilot, 7 November

`scripts/pilot_marshawn_frequency_variants.py` runs full 1920×1012 images from
TRAIN cameras AA, HA and MA, using the original HD1000-update level budget,
patch8/window7, 16 levels, resolutions16–8192, hash23 and batch16384. Each camera
gets a new baseline, an unchanged repeat, LR decay A and threshold B. Baseline
and threshold B leave the new LR option unset. This pilot prepares evidence
for the final frequency stage; no NeRF recipe or training dataset is replaced.

Every input image, source metadata, preprocessing module and worker is bound
by SHA. The worker imports the specified preprocessing module by absolute
path and records its actual path. Read-only SSIM counters retain exact censor
masks; Adam construction and the first eight random draws are recorded with
final RNG. These checks do not certify full numerical equivalence. The repeated
baseline measures existing variation independently of the intended factors.

The isolated generation is `tuning_20261106/frequency_HD_variants_pilot_v1`.
Its plan SHA is `927ab5a99439ab6909893035c4226a7c73c426876208d3fb489cfaaef511431b`;
the worker SHA is `f8be51932edca5084a4c5f26a94bdfc6a3d1ee6547c2eb6cbea6e1986278fb4d`.
It runs under the shared exclusive silverGPU6 lease, a4GiB Torch allocator
cap and an external3600-second timeout. Startup confirmed the timeout and
worker PIDs alive, imported-source identity correct, baseline level3 progress,
and2458MiB process GPU memory. Outputs remain outside all training datasets.
Final maps, counts and source-bound receipts are still pending; no map is
promoted on censor rate alone.

### Completed three-camera pilot

All12 fits finished and GPU6 was released. Peak Torch allocation was1357.6MiB.
Every camera's four runs had identical initial parameter hashes, first eight
random draws, final RNG and16000 updates. This does not imply identical maps:
unchanged repeats differ substantially. The new CPU reporter
`scripts/review_marshawn_frequency_pilot.py` verified all source/map bindings,
SSIM censor counts and discrete schedules, producing24 actor records and18
comparisons. Root inspected AA black and MA grey overlays.

Fraction of actor patches still failing the final SSIM criterion (percent):

| Camera / actor | Baseline | Repeat | LR decay | SSIM .93 |
|---|---:|---:|---:|---:|
| AA / black |58.52|56.11|49.92|22.62|
| AA / grey |8.19|9.22|5.80|3.07|
| HA / black |38.38|29.64|44.78|4.90|
| HA / grey |15.60|9.48|31.80|0.31|
| MA / black |38.86|24.57|29.52|11.05|
| MA / grey |12.88|7.67|20.82|5.21|

These regions contain original actor coverage of at least90% in complete8×8
patches; the four uncovered bottom image rows are excluded. LR decay helps
some regions and worsens others, with no consistent actor-level advantage.
The unchanged repeat changes34–74% of actor patch labels. Its mean level shifts
by−0.18 to−0.63, while LR decay shifts−0.17 to+0.40 relative to baseline. The
lower threshold shifts levels by−1.20 to−2.02, consistently assigning coarser
labels. That is its intended intervention, not evidence of better details.

A formerly censored patch resolved only at the last level still has label15;
the reporter explicitly measures this distinction. Neither lower censoring nor
label change certifies a better radiance field. These results do not select a
winner or replace production maps. Full3D comparisons remain required after
structural recipe selection, including the observed repeat variability.

[Source-bound report and map comparisons](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD_variants_pilot_v1/review/report.json),
[AA black overlay](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD_variants_pilot_v1/review/AA_black_levels.jpg),
[MA grey overlay](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frequency_HD_variants_pilot_v1/review/MA_grey_levels.jpg).
