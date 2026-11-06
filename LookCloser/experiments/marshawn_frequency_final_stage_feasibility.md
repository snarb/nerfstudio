# Marshawn: proposed final frequency stage

## What was inspected

Read-only proposal; no new fits, dataset edits, or changes to the approved
63-camera experimental queue. Primary references are [the local paper](../Paper%20LookCloser.md)
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
If the final structural comparison selects HD instead, retain its existing
fit budget; do not bundle a resolution/budget change into A.
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
