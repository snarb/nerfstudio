# Frozen LookCloser + native photographs: Luster 000540

Status: experiments in progress. Current single-source reference: `single_scale16_s8`, **24.51%** native validation person LPIPS improvement, zero measured local LPIPS regressions. The 40% goal is not achieved.

**Active user requirement (2026-10-17): fixed LookCloser geometry and exactly
one native TRAIN photograph supplying texture per pixel/patch. No interpolation
of colors between photographs and no optical-flow texture-coordinate shifts.**
All pair-interpolation and registration results below are historical ablations,
not eligible final candidates. The single-source investigation supersedes their
provisional reference status.

## Current automatic recipe

`single_scale16_s8` considers 16 automatically proposed TRAIN cameras. It keeps
the median camera-z geometry and all visibility checks of the original frozen
projection. In addition to angular distance, round-trip error and patch mismatch,
it penalizes a large capture-scale mismatch by
`0.3 * max(0, log2(source_resolution / target_resolution))`. The existing score
already penalizes insufficient source resolution. This is a global recipe rule,
not a camera blacklist or a manually selected face patch.

The selected photo is `T`, the frozen render is `B`, and `G8` is a Gaussian
low-pass operation normalized strictly within the selected-source patch.
The merge is `B + w * [(T - G8(T)) - (B - G8(B))]`, clipped to [0,1].
`w` fades over three pixels at source/support boundaries; otherwise it is one.
No other photograph contributes RGB to that pixel. Rejected pixels are an exact
copy of `B`. The code chooses both source labels and transfer support.

| View | Baseline LPIPS | Hybrid LPIPS | Exact LookCloser preserved in person |
|---|---:|---:|---:|
| 007 | 0.06944436 | 0.06287702 | 19.53% |
| 012 | 0.12492531 | 0.10576124 | 21.10% |
| 095 | 0.12697524 | 0.08318930 | 8.72% |
| 150 | 0.15322061 | 0.11689487 | 10.98% |

Validation mean: LPIPS **0.10194847**, PSNR 28.345015, SSIM 0.89996771. No measured local LPIPS row regresses. The **40% goal remains unmet**.

Visual inspection of eyes, skin, nails and embroidery found no obvious new
seams or doubled contours in the reviewed crops. The existing eye shape,
finger-edge halos and lace geometry remain limitations. More detail in a source
is not by itself proof that its appearance fits the target view: the head
close-up diagnostic is `C/preflight/eyes_source61_diagnostic.png`. This is an
evaluation-only comparison, not an input to source selection.

The mean-depth projection ablation (`single_mean4_s8`) converts the saved
conditional mean ray distance to camera-z without changing model density or
calibration. It gives 23.832% improvement and regresses cam150 embroidery, so
the median-depth reference is retained. Both depth readouts come from LookCloser.

Reproduce the current recipe using `scale_single_luster_hybrid.py`, its saved
`C/candidates/single_scale16_s8/config.json`, and a new candidate name.
The review exporter is `review_single_luster_hybrid.py`; it reads GT/ROIs only
for comparisons after prediction. GPU evaluation remains sequential.

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
depth-interval uncertainty. The current reference ranks sixteen automatically
proposed cameras using angular/resolution cost, round-trip error and a weak
patch appearance cost against LookCloser. Source selection is per pixel, with
isolated assignments removed. A second photograph is not required. One selected
photograph supplies detail above a masked Gaussian scale of 8 native pixels;
LookCloser supplies the broad appearance and exact rejected-pixel fallback.
Source changes fade towards LookCloser over 3 pixels. No colors are interpolated
between photographs. Historical screens below also tested other merge methods. Evaluation masks, target photographs and evaluation ROI
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


## Automatic single-source method comparisons

These are recipe-selection measurements, not accepted final results. Every
listed predictor chooses its sources and transfer regions automatically; no
manual source/ROI assignment is used. Native validation person LPIPS:

| Candidate | LPIPS | Reduction | Regressing local/global rows |
|---|---:|---:|---:|
| `single_patch_s8` | 0.10125667 | 25.017% | 6 |
| `single_patch_s16` | 0.10217659 | 24.336% | 6 |
| `single_patch_direct` | 0.11124409 | 17.622% | 10 |
| `single_patch_strict_s8` | 0.10323984 | 23.549% | 7 |
| `single_patch_s32` | 0.10345585 | 23.389% | 7 |
| `single_consensus_30` | 0.12605630 | 6.653% | 6 |
| `single_consensus_60` | 0.13275978 | 1.689% | 8 |
| `single_consensus_85` | 0.13399314 | 0.776% | 8 |
| `single_graph_s8` | 0.10177520 | 24.634% | 6 |
| `single_pool32_s8` | 0.10125939 | 25.015% | 6 |
| `single_residual_s8` | 0.10426928 | 22.787% | 16 |
| `single_surface_s8` | 0.10286743 | 23.825% | 7 |
| `single_appearance_s8` | 0.10837916 | 19.743% | 5 |
| `single_missing_s8` | 0.13084212 | 3.109% | 11 |
| `single_near4_s8` | 0.10932222 | 19.045% | 0 |

The projection differential now measures minimum/maximum native sampling axes
and surface incidence using finite differences of frozen median-depth points.
It changes admission and ranking only, never the geometry or lookup coordinates.
This corrects a limitation of focal-length/depth-only source-resolution ranking,
but did not by itself improve the accepted image quality.

A separate automatic appearance guard estimates frozen-model view dependence
and photo structure. Its geometry and TRAIN-photo inputs are cached with hashes.
The missing-band method retains existing model detail when a photo has no
additional band energy or its pattern disagrees in phase. Other cameras never
contribute RGB to the selected pixel. Its transfer-weight map records the maximum
applied strength across bands, not one common coefficient for the entire RGB
correction. Construction labels are retained separately for exact auditing.

24 tests pass, covering projection anisotropy/back-facing surfaces and missing
texture versus conflicting detail phase. Exact single-source reconstruction
audit also passes for the rejected `single_residual_s8` experiment. These checks
do not imply visual acceptance or completion of the 40% goal.

## Earlier single-source reference: near-view priority

`single_near4_full_s8` uses `single_source_luster_hybrid.py` with `count=4`,
`match_weight=.25`, `match_limit=.06`, `gain=1`, and otherwise its defaults.
The same rule applies to every camera and surface; no semantic masks, manual
source assignment, target RGB or evaluation ROI enter prediction.

| View | LookCloser LPIPS | Hybrid LPIPS |
|---|---:|---:|
| 007 | 0.06944436 | 0.06219758 |
| 012 | 0.12492531 | 0.10591259 |
| 095 | 0.12697524 | 0.08417430 |
| 150 | 0.15322061 | 0.11740734 |
| Validation mean | 0.13504039 | 0.10249808 |

All measured person and detail LPIPS rows improve. This does not mean that
PSNR/SSIM improve in every row; all three are retained in the JSON/CSV.
Eye LPIPS in cam007 improves by 20.7%; the four nail/hand ROIs improve by
8.5%, 1.5%, 5.6% and 8.5%. Some named nail ROIs mainly show palms/fingers,
so these numbers are not claims that a complete nail surface was reconstructed.

Visual checks of face, eyes, fingers, embroidery and sleeves show useful texture
changes without obvious new doubled edges in the inspected crops. Existing eye
shape, jewelry shape, filled lace gaps and finger-boundary defects remain.
The transfer does not repair geometry. Full native crop comparisons are in
`C/evaluation/single_near4_full_s8/`.

Deep input/model/grid hashes, forbidden-source exclusion, exact unchanged
fallback and reconstruction from one selected native photo all pass for this
reference. See `C/candidates/single_near4_full_s8/provenance.json`. The audit
rebuilds float RGB and transfer weights exactly; it does not merely trust a
receipt claiming single-source construction.

Reproduce with the saved candidate config:

```bash
C=/fsx/tmp/luster_000540_hybrid_20261017T104440Z
PY=/home/pavlo/.conda/envs/nerfstudio/bin/python
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" "$C/code/LookCloser/scripts/single_source_luster_hybrid.py" --root "$C" --name reproduced_single_near4_full_s8 --config "$C/candidates/single_near4_full_s8/config.json"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" "$C/code/LookCloser/scripts/evaluate_luster_hybrid.py" --root "$C" --candidate reproduced_single_near4_full_s8 --panels
```

Inputs are the verified frozen native geometry and photo warp caches described
above. New names avoid overwriting historical results. GPU evaluation must be
run after any other GPU campaign job finishes.

## Near-view, patch-consistency and footprint ablations

Automatic four-source selection remains the best locally safe reference.
Additional sources introduce close-up head cameras with roughly 2.1 source
pixels per target pixel. Footprint integration reduces some regressions, but
does not make these sources safe. This is evidence against treating resolution
or angular distance alone as a sufficient suitability criterion.

| Candidate | Native person LPIPS reduction | Regressing rows |
|---|---:|---:|
| `single_near4_full_s8` | 24.098% | 0 |
| `single_near4_area_s8` | 24.068% | 0 |
| `single_near4_s8` | 19.045% | 0 |
| `single_near4_s16` | 18.772% | 0 |
| `single_near4_s32` | 18.326% | 0 |
| `single_near16_area_s8` | 23.947% | 8 |
| `single_near4_graph_s8` | 23.841% | 3 |
| `single_near8_area_s8` | 23.727% | 8 |
| `single_near16_full_s8` | 23.674% | 8 |
| `single_consistent8_s8` | 23.530% | 8 |
| `single_near4_full_s16` | 22.874% | 1 |
| `single_near2_full_s8` | 22.759% | 1 |
| `single_near16_s8` | 18.918% | 8 |
| `single_near8_s8` | 18.750% | 8 |
| `single_near4_direct` | 14.482% | 2 |
| `single_near4_full_direct` | 13.169% | 6 |

The patch-consistency variant compares projected TRAIN-photo structure with
two corroborating sources. Witnesses supply only scalar costs; RGB is always
from one selected source. It remains rejected because face LPIPS regresses.
Graph regularization also remains rejected: smoother source labels alone did
not protect eye detail. Twenty-five synthetic tests pass.

## Automatic selection follow-up (2026-10-17 13:31 UTC)

The best numerical scale-compatible candidate is now `single_scale16_match10`
(25.169% improvement, LPIPS 0.10105200, no measured local LPIPS regression).
Its visual review and final provenance audit are pending. The previously audited
reference remains `single_scale16_s8` until those checks finish.

| Controlled change from scale16/sigma8 | Validation LPIPS reduction | Local regressions |
|---|---:|---:|
| `single_scale16_s4` | 24.507% | 0 |
| `single_scale16_s6` | 24.866% | 0 |
| `single_scale16_s12` | 23.779% | 0 |
| `single_scale16_support6` | 24.956% | 0 |
| `single_scale16_geometry2` | 25.589% | 2 |
| `single_scale32_s8` | 24.506% | 0 |
| `single_scale16_match4` | 23.719% | 0 |
| `single_scale16_match10` | 25.169% | 0 |

The support-mass .6 candidate was visually checked on eyes, skin, hands,
embroidery and whole-view previews; no obvious new seams or doubled contours
were found, while original geometry defects remain. Looser geometric gates
regress cam150 embroidery and are rejected.

Both source-model residual variants regress nails/cuffs: the high-pass version
gives 22.954% improvement with six regressions; full residual gives 20.370% with
seven. Neither is accepted.

Evaluation-only spatial LPIPS maps are saved under
`C/evaluation/single_scale16_match10_spatial`. CPU map means differ slightly from
the CUDA selection metrics and are diagnostic only. On cam012, about 35.2% of
person pixels receive boundary-faded transfer and 22.4% retain LookCloser;
on cam095/150 about 83% receive full-strength transfer. Most remaining error
is inside the person, not confined to the silhouette. LPIPS receptive fields
span neighborhoods, so these strata are not causal per-pixel error estimates.

Next controlled investigations use locally eligible source proposals (including
cameras missed by coverage votes), soft geometric-risk ranking, and alternative
termination-depth statistics read from the exact same frozen LookCloser.
The depth probe appends q25/q75/PDF-mode arrays and verifies the existing median
readout and full model-state hash; it never changes the volume, calibration,
grids or baseline. No experimental depth recipe is accepted in advance.

## Patch-coherent ranking and local review (13:42 UTC)

`single_risk_patch` reaches **26.149%** improvement (LPIPS 0.09972881),
with no regression in the predefined named ROIs. It smooths each source cost
over sigma=3 pixels, before the hard per-pixel choice and native visibility
recheck. Merely adding soft geometric-risk weights gives 24.671%; combining
sigma=6 transfer with support mass=.6 gives 25.365%. Expanded source proposals
give 24.867%, essentially unchanged from the original pool.

**The 26.149% variant is not yet a final visual acceptance.** Evaluation-only
128px tiles reveal remaining local regressions in cloth folds, jewelry and lace
even though larger named ROI means improve. Worst-tile panels are saved in
`C/evaluation/single_risk_patch_spatial`; some small jewelry detail becomes
softer and some creases differ from GT. These observations motivate structural
phase checks and native interpolation tests. The deep single-source audit
passes, which verifies provenance/construction, not visual correctness.

### Additional local safeguards under evaluation (2026-10-17 13:51 UTC)

The worst local regression panels reveal cases where a selected photograph is
softer than LookCloser and where small folds differ despite a favorable named
ROI average. All candidates remain provisional. The early structural phase
inspection found high coarse correlation even in several problematic jewelry
patches, so coarse correlation alone is not treated as a sufficient safeguard.

Queued controlled comparisons cover source-boundary fading, spatial cost scale,
phase agreement, clamped cubic native sampling, and projected surface/sharpness
quality. The surface-quality comparison retains the current angular/risk score;
the previous surface experiment changed that score too and cannot isolate the
benefit of a surface differential gate. The q25 and q75 frozen-density readouts
returned validation LPIPS 0.10273537 and 0.10304938 respectively, below the quality
of the established q50 recipe (lower is better). q25 also regressed cam150
embroidery. Neither is selected. No geometry or saved model/grid state was edited.

Validation: 32 focused transfer tests passed (`tests_projection_quality.log`),
including automatic one-photo choice, occlusion fallback, exact unchanged pixels,
projection conventions, cubic footprint visibility and adaptive seam behavior.
