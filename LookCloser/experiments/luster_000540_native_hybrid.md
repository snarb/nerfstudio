# Frozen LookCloser + native photographs: Luster 000540

Status: experiments in progress; no hybrid accepted yet.

**Active user requirement (2026-10-17): fixed LookCloser geometry and exactly
one native TRAIN photograph supplying texture per pixel/patch. No interpolation
of colors between photographs and no optical-flow texture-coordinate shifts.**
All pair-interpolation and registration results below are historical ablations,
not eligible final candidates. The single-source investigation supersedes their
provisional reference status.

## What is being tested

Selective transfer from native TRAIN photographs into the frozen
`cold_body80_fas50_threshold095_zero` checkpoint, step 106316. The goal is at
least 40% lower native validation person LPIPS, without local LPIPS regression
or new visible artifacts. Cameras 012/095/150 are validation for recipe selection,
not an independent test; cam007 is a demonstration with its own RGB excluded
from texture sources. All four views are static. There is no training.

Campaign: `/fsx/tmp/luster_000540_hybrid_20261017T104440Z` (called `C` below).
The isolated worktree is `C/code`; frozen compatible model implementation is
`C/runtime`, copied from the actual body experiment. Neither arbitrary main
model code nor the late_control checkpoint is used.

Checkpoint SHA256:
`f07369d46d29852d9816a9681ac5758e9ecf22fad087ac3aece78e443ae8413b`.
Full input paths, hashes, deadline and runtime inventory are in
`C/preflight/manifest.json`. Intrinsics match the original COLMAP calibration
within 1.8e-12; recomputed normalized poses match all 165 prepared poses exactly.

## Method

The native baseline uses the checkpoint's original adaptive marcher and saved
occupancy. Inference computes conditional termination-depth quantiles in
physical ray-distance bins, then converts them explicitly to camera-z. It does
not change RGB weights or expand occupancy. Complete model-state hashes are
checked before and after rendering.

Source proposals use predicted surface coverage, angular distance and projected
resolution. Warps use native JPEGs directly and check each nonzero bilinear tap,
source/target alpha, depth disagreement, round-trip reprojection and projected
depth-interval uncertainty. A second TRAIN source must corroborate appearance.
The first screen compares direct RGB, single-band detail replacement and
multiband replacement, with an exact original-RGB fallback. Source changes fade
towards the baseline. Evaluation masks, target photographs and evaluation ROI
coordinates are not inputs to transfer.

## Results

Native baseline, original alpha-composed RGB and frozen person masks:

| Camera | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| 007 (demonstration) | 31.196819 | 0.921246 | 0.069444 |
| 012 | 27.744911 | 0.870384 | 0.124925 |
| 095 | 29.272514 | 0.913118 | 0.126975 |
| 150 | 27.707949 | 0.914635 | 0.153221 |
| Validation mean | 28.241791 | 0.899379 | 0.135040 |

The 40% native LPIPS target is **0.0810242310**. HD values from the earlier
experiment use a different sampling protocol and are not compared to these
native values as though they were interchangeable.

All metrics and evaluation-only ROI coordinates are in `C/evaluation`.

The first ten-configuration screen reaches at most 16.61% mean improvement,
but every single-source configuration regresses at least one local ROI. A
view-bracketing pair interpolates observed appearance from cameras around the
target direction. `bracket8_s8` reaches **13.94%** mean improvement (LPIPS
0.11621011) with **zero measured local regressions**. It is a provisional
reference, not completion of the 40% goal. Full-strength pair transfer reaches
18.43% but regresses cam007 eyes/hair and cam012 skin, so it is rejected.

The source-exclusion, finite-output and exact-fallback audit passes for
`bracket8_s8`; its `provenance.json` binds the actual warp root, input receipts,
output hashes and code. Preliminary visual inspection shows clearer fabric and
cuffs; eye/pendant shape and merged lace holes remain. Final whole-image visual
acceptance is still pending.

Other ablations test native area integration, classical bounded texture lookup
alignment, high-frequency corroboration and filters whose support is independent
of selected source labels. Their numerical and visual limitations are recorded
in `C/preflight/progress_review.json` and per-candidate metrics. Source-background
corroboration uses measured empty plates and TRAIN photos only. Both tested
foreground guards reduce useful transfer; the robust-noise guard gives provisional
CPU LPIPS 0.117184 and regresses the cam012 face, so it is not the selected recipe.
A separate bounded classical TRAIN/TRAIN registration ablation aligns texture
lookups before view interpolation. It checks forward/backward consistency, a
4-pixel displacement cap, local Jacobian, source visibility, and improvement of
source/source photometric agreement. It does not reconstruct or modify geometry. The expanded
source pool (up to 32 coverage-selected cameras per target) is being rendered.

The expanded pool contains 76 unique TRAIN cameras and 103 target/source pairs.
All geometry and point/area/foreground caches are complete. Increasing the
per-pixel candidate ranks from 4 to 8 reduces improvement to 12.13%; those
remaining k8 variants are deferred. Four local ranks give 14.23% without local
regression.

The resolution audit found a specific coverage issue: in the cam150 lower
sleeve ROI, geometrically visible sources have native sampling ratios around
0.705–0.744, just below the historical 0.75 cutoff. They still exceed the
model's 1920/4096 training sampling ratio. Lowering the cutoff globally to 0.5
improves the expanded-pool result to **14.629%**, LPIPS **0.11528552**, with zero
measured local regressions on CUDA. This is the current numeric reference;
full visual acceptance remains pending. The legacy cam150 `lace` ROI mostly
contains the sleeve hem because the lace itself is clipped by the image edge;
its metric must not be described as fixing lace holes.

A source-model residual ablation transfers native photo minus frozen rendering
of the same TRAIN source camera. Full strength reaches **20.041%** but regresses
cam012 skin/nails/hem and cam095 nails, so it is rejected. A 4-pixel bounded
TRAIN/TRAIN lookup alignment brings only a small mean gain and a cam007 eye
regression. Current tests investigate fine-band-only strengthening and a wider,
still visibility-checked texture lookup bound. None changes calibration, model
weights, depth arrays or grids.

The provisional 13.94% reference has passed deep input/runtime/cache verification
and is exported to `C/exports/bracket8_s8.tar.gz` (277093152 bytes; all 141 archive
members verified). This archive is a review milestone, not the final 40% result.
Its SHA256 is `21cf78b2953d5587f8ea782b1f58db8e503858dcd270be53cc3a2fd6be43418b`.

CPU pilot metrics are provisional: CPU and CUDA LPIPS may differ by about 1e-5.
All formal acceptance comparisons use CUDA for both baseline and candidate;
no cross-device difference is treated as a quality tolerance.
The eight-source cam007 pilot improves overall LPIPS to 0.065762, but slightly
regresses the eye region; it is **not accepted**. Broader transfer coverage alone
is therefore not the selection criterion.

## Reproduction and validation

Use `/home/pavlo/.conda/envs/nerfstudio/bin/python`, with the same environment
prepended to PATH and CUDA_HOME set to that environment. Model operations must
run sequentially on the GPU. Source warps and combination are CPU operations.

```bash
export PATH=/home/pavlo/.conda/envs/nerfstudio/bin:$PATH
export CUDA_HOME=/home/pavlo/.conda/envs/nerfstudio
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
C=/fsx/tmp/luster_000540_hybrid_20261017T104440Z
S=$C/code/LookCloser/scripts
python "$S/render_luster_hybrid.py" --root "$C" --cameras 7 12 95 150
python "$S/transfer_luster_hybrid.py" --root "$C" sources --count 32
# Render chosen sources sequentially before warping them.
python "$S/transfer_luster_hybrid.py" --root "$C" warp --count 8
python "$S/screen_luster_hybrid.py" --root "$C"
python -m pytest -q -n 0 "$C/code/LookCloser/tests/test_luster_hybrid.py"
```

Twenty focused tests currently pass: native pixel centers and camera-z/ray-range
conventions, source exclusion, analytic termination CDF, multiple density
layers, empty rays, bilinear footprint validity and exact fallback with masked
filter support. Dense diagnostics confirm unchanged model state and grids. Repeated RGB differs
by at most 5.3644e-7, and repeated CUDA baseline metrics are identical. Dense
8192/16384 sampling retains the known geometric defects. Final recipe selection
and full visual acceptance are still pending.

## Insights and next steps

No geometry correction is claimed. Existing merged lace holes, eye-shape and
pendant errors remain potential limits. Final results will include rejected
variants, remaining defects, full-resolution comparisons, source/confidence
maps, reproducible commands and a verified download archive.


## Single-source projection investigation

`single_source_luster_hybrid.py` assigns one camera to each accepted pixel.
The 3D point remains the frozen target conditional median depth. Native lookup
coordinates remain the calibrated projection of that point, without shifts.
Eligibility checks all bilinear taps, source/target opacity, depth consistency,
round-trip error, uncertainty and projected native sampling. Smoothed local
appearance costs against the frozen target render regularize source choice;
no target photograph or evaluation mask enters construction.

The selected photograph supplies direct RGB or its high-frequency component.
Source changes fade to LookCloser, never to another photograph. A second photo
is not mandatory when one source independently passes the geometry and patch
checks. Source labels, exact unchanged fallback and source hashes are saved.

Evaluation-only nail montages are in
`C/preflight/cam{007,012,095,150}_nail_single_sources.jpg`, with numerical
visibility diagnostics in `C/preflight/nail_source_diagnostics.json`. The
montages deliberately show raw projections before rejection to expose why a
source is unsafe. Their coverage fractions include the whole crop, including
background; they are not nail-surface coverage measurements.

Initial inspection: cam007 sources 008/023 align more plausibly than oblique
sources with stretched finger edges. In cam095, sources 094/093/097 exhibit
shifted nail contours; source 096 is closer but has incomplete reliable support.
This supports source-specific visibility and fallback, not unconstrained texture
painting. GT is shown solely for evaluation.

The first automatic patch-cost screen reaches 25.0175% mean validation LPIPS
improvement (`single_patch_s8`, LPIPS 0.10125667). No nail ROI regresses, but
cam007 face/skin/eyes/hair/jewelry and cam012 face regress, so it is not accepted.
Increasing the detail scale to 16 reduces mean improvement to 24.3363%. Direct
single-photo RGB gives 17.6216% and also worsens cam095 nails; preservation of
LookCloser's broad appearance remains useful even with one texture source.

Automatic eligibility and patch/source assignment are an explicit user
requirement. There are no manual semantic region exclusions, per-camera source
assignments, or GT-driven construction masks. `single_consensus_luster_hybrid.py`
uses other TRAIN photographs only as independent phase/appearance witnesses;
`graph_single_luster_hybrid.py` tests spatially coherent hard source labels with
an explicit LookCloser fallback label. Neither shifts sampling coordinates.

22 synthetic tests pass, including end-to-end hard-source selection and
occlusion fallback for both independent and graph-based assignment. The candidate
audit now reconstructs single-source output from its selected native warps and
verifies exact equality with saved float RGB and transfer weights; this passed
for `single_patch_s8`. This reconstruction audit is separate from full input
hash verification and visual acceptance.
