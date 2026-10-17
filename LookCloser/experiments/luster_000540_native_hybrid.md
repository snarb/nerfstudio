# Frozen LookCloser + native photographs: Luster 000540

Status: experiments in progress. Audited/exported provisional reference: `single_cubic75_seam_adaptive`,
**28.69%** lower native validation person LPIPS. All predefined named ROI LPIPS
checks pass, but smaller patches still show local degradation. **Not visually
accepted; the 40% goal is not achieved.**

**Active user requirement: fixed LookCloser geometry and exactly one native
TRAIN photograph supplying texture per pixel/patch. Source choice, consistency
and rejection are automatic. No color blending between photographs, optical
flow, geometry fitting, manual source assignments or target-GT construction.**
Historical pair-interpolation and registration experiments below are ineligible.

## Audited automatic recipe

1. Unproject native target pixels using conditional median camera-z from the
   exact frozen LookCloser. Project those points into native TRAIN photos.
2. Propose 16 cameras using predicted coverage and view/resolution suitability.
   Reject invalid footprint, alpha < .99, relative depth error > .005,
   round-trip error > 1 pixel, projected uncertainty > 2 pixels, source score
   > 1 or resolution ratio < .5. No evaluation mask is used.
3. Rank geometrically valid sources using angular/resolution cost, patch
   appearance, capture-scale compatibility, round-trip and depth risk. Smooth
   source costs over sigma 3 before hard selection; remove isolated labels.
   Exact settings are in `candidates/single_risk_patch/config.json`.
4. Keep those automatic source labels fixed. Resample the selected native photo
   at the same projected coordinate with clamped Keys cubic interpolation
   (a=-.75). Every nonzero 4x4 footprint tap must pass visibility. If it fails,
   retain LookCloser rather than change to another photo.
5. Let selected-photo RGB be T and frozen RGB be B. Apply
   `B + w * [(T-G6(T)) - (B-G6(B))]`, clipped to [0,1]. Gaussian support is
   normalized over that source’s geometrically visible support, independent of
   final labels. Weight fades over three pixels at unsupported boundaries.
   At internal source seams, full transfer is allowed only when every touching
   source corroborates the high-pass correction within .02 RGB (with a 3x3
   disagreement guard). Witnesses never contribute RGB. Rejected pixels are
   bitwise B.

The recipe transfers visible detail while retaining LookCloser's broad color.
It does not promise full texture replacement where view-dependent appearance
or inaccurate geometry makes it unsafe.

| View | Baseline LPIPS | Hybrid LPIPS | Hybrid PSNR | Hybrid SSIM | Exact LookCloser preserved in person |
|---|---:|---:|---:|---:|---:|
| 007 | 0.06944436 | 0.06186977 | 31.349873 | 0.915177 | 11.83% |
| 012 | 0.12492531 | 0.09895791 | 27.865360 | 0.873815 | 14.15% |
| 095 | 0.12697524 | 0.07930170 | 29.237572 | 0.907893 | 6.47% |
| 150 | 0.15322061 | 0.11061428 | 27.897682 | 0.914091 | 8.66% |

Validation mean: **LPIPS 0.0962912962**, PSNR 28.333538, SSIM 0.89859943.
The 40% threshold is .0810242310. Native measurements are separate from HD.

Deep provenance audit passed: exact checkpoint/runtime/geometry, source
exclusion, cubic array hashes, single-photo reconstruction and exact fallback.
50 focused tests passed (`logs/tests_source_phase.log`), including newer experimental selectors.

Visual review is explicitly **not_accepted_local_detail_degradation**. Embroidery and many
fine details improve, but small jewelry/brocade patches can soften and cloth
creases can change appearance. Dark-hair texture and incorrect lace holes
remain. Named ROI averages do not establish that every detail improved.
Review `evaluation/single_cubic75_seam_adaptive_visual_review.json`, the source panels,
and the worst 128px spatial-regression panels. These diagnostics never enter
prediction. Existing eye and pendant geometry cannot be repaired by texture.

## Reproduce the audited reference

Use the existing verified campaign caches with fresh candidate names. GPU
commands must run after the current GPU owner exits. The immutable runtime is
used by the geometry renderer; no training is invoked.

```bash
C=/fsx/tmp/luster_000540_hybrid_20261017T104440Z
PY=/home/pavlo/.conda/envs/nerfstudio/bin/python
S="$C/code/LookCloser/scripts"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$PY" "$S/risk_single_luster_hybrid.py" --root "$C" --name reproduce_risk --config "$C/candidates/single_risk_patch/config.json"
"$PY" "$S/cubic75_luster_hybrid.py" --root "$C" --count 16
"$PY" "$S/cubic75_fixed_luster_hybrid.py" --root "$C" --parent reproduce_risk --name reproduce_cubic
"$PY" "$S/fixed_sampling_seam_luster_hybrid.py" --root "$C" --parent reproduce_cubic --name reproduce_seam --filter-support geometry --seam-fade 3 --adaptive-seam --seam-tolerance .02
"$PY" "$S/evaluate_luster_hybrid.py" --root "$C" --candidate reproduce_seam --device cuda
"$PY" "$S/review_single_luster_hybrid.py" --root "$C" --name reproduce_seam
"$PY" "$S/freeze_luster_hybrid_candidate.py" --root "$C" --name reproduce_seam --warp-root "$C/variants/cubic75" --deep --require-single-source
```

Saved commands/configuration and full source/model provenance accompany exports.
The expanded <=60-degree TRAIN source search is complete and gave negligible
improvement with the current costs; it is not part of this reference.


Verified review archive (261 members, 183913153 bytes):

```bash
scp olive-lark:/fsx/tmp/luster_000540_hybrid_20261017T104440Z/exports/single_cubic75_seam_adaptive.tar.gz .
```

SHA256: `be7c5ca4e378b192cf178162edb5bf2b5170d2cc0532db1b60548dfc7e0ec729`.
The archive contains the report snapshot at export time; this live report
continues to record subsequent experiments.

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
resolution. The current recipe above supersedes earlier bilinear-only transfer.
All projection coordinates and depths come from the same frozen LookCloser;
source labels and rejection are computed by code. Evaluation photographs,
masks and ROI coordinates are never inputs to construction.

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

## Interpolation, source consistency and local review (2026-10-17 14:15 UTC)

Clamped cubic sampling improves detail. Keeping the original automatic source
labels avoids a small hand/nail ROI regression caused by reranking after cubic
sampling. A bilinear-RGB control with identical wider visibility gains less,
so the benefit is not explained solely by the 4x4 footprint rejection.

| Candidate | Validation LPIPS reduction | Predefined ROI regressions |
|---|---:|---:|
| single_risk_patch | 26.15% | 0 |
| single_risk_sharp_cost | 26.18% | 0 |
| single_cubic_fixed | 27.93% | 0 |
| single_cubic_risk_patch | 27.91% | 1 |
| single_cubic_visibility_control | 26.13% | 2 |
| single_cubic_s4 | 27.49% | 1 |
| single_cubic_match5 | 27.44% | 1 |
| single_cubic_sharp_cost | 27.91% | 1 |
| single_risk_surface_gate | 26.01% | 0 |
| single_risk_surface_cost | 26.08% | 0 |
| single_risk_sharp_gate | 25.81% | 1 |
| single_guided_e3_s8 | 25.12% | 0 |
| single_consensus_low15 | 25.69% | 0 |
| single_consensus_low30 | 24.24% | 1 |

Guided photometric correction and TRAIN-witness broad-color agreement do not
improve the reference. Witnesses contribute confidence only, never RGB.
Unchanged-density q25/q75/PDF-mode depth readouts also underperform q50;
there is no basis to replace the median readout. Larger ROI checks pass for
some variants but do not override visible small-patch defects. Spatial LPIPS
maps use CPU and may differ from CUDA by about 1e-5; CUDA is the selection
metric. Maps have finite receptive-field spill and are not causal attribution.

Verified provisional export: `C/exports/single_cubic_fixed.tar.gz`, 183558628 bytes,
242 archive members rehashed, SHA256
`da9316e4835b85e86a56fba28108a2c2a6f04c183df0d09647974eb156b38099`.
Status is `provisional_local_defects`, not final acceptance.

```bash
scp olive-lark:/fsx/tmp/luster_000540_hybrid_20261017T104440Z/exports/single_cubic_fixed.tar.gz .
```

Two additional fixed-label tests are running: detail-quality rejection using
G1-G4 energy/phase, and multiplicative broad-illumination matching (sigma6/16,
gamma1/2.2). They retain the current cubic source labels and projection centers.
37 focused tests pass, including blur/wrong-phase rejection, exact fallback,
no support growth and isolation of photometric correction between sources.
The new evidence maps and gates are construction outputs, not evaluation masks.

## Fixed-source quality and photometric ablations (2026-10-17 14:23 UTC)

| Candidate | Validation LPIPS reduction | Predefined ROI regressions |
|---|---:|---:|
| single_cubic_fixed_sharp8 | 27.034% | 1 |
| single_cubic_fixed_phase8 | 23.178% | 1 |
| single_cubic_fixed_quality8 | 23.206% | 1 |
| single_cubic_illumination_s6 | 27.862% | 0 |
| single_cubic_illumination_s16 | 26.122% | 0 |
| single_cubic_illumination_g22 | 25.750% | 0 |

The sharpness gate visibly restores small cam007 jewelry detail and reduces
brocade softening, but regresses the cam012 hand/nail ROI by .000307 LPIPS.
The phase gate reduces crease changes but regresses cam012 lace by .005300;
combined gating regresses it by .005110. Rejected as overall replacements.
Hard spatial rejection can itself disturb fine texture continuity. Dark-hair
noise and lace geometry remain. Comparison panels are in
`evaluation/fixed_quality_review`; their tile positions are evaluation-only.

Multiplicative illumination matching does not outperform the reference.
Fine-phase source ranking is the next isolated test: G1-G4 correlation affects
source score, and a capped reward favors additional source detail only when
its phase agrees with LookCloser. Geometry and sampling coordinates remain
unchanged. 39 focused tests pass, including selection of an aligned detailed
photo over a lower-angle but opposite-phase source.


## Fine-phase selection and source-view illumination (2026-10-17 14:47 UTC)

The current **unaccepted numerical leader** is `single_cubic_view_detail`:
LPIPS **.0970667402**, **28.1202%** below the native validation baseline.
The 40% target remains unmet. No predefined ROI regresses, but that does not
establish absence of small local defects. Initial face/hand review still shows
inaccurate soft eyes, finger bridges and edge halos; spatial review is pending.

This candidate keeps `single_cubic_fixed` source labels and projection. Before
transferring its G6 high-frequency component, it scales the selected photograph
by `clip((G16(B)+.02)/(G16(S)+.02), .5, 2)`, where S is the **frozen LookCloser
render in that source camera**, warped at the same surface points. G16 uses
normalized support within the selected source label. This estimates broad
view-dependent illumination without a second photograph supplying RGB.

| Candidate | Native validation LPIPS | Reduction | Predefined ROI regressions |
|---|---:|---:|---|
| single_fine_phase25_cubic | .0972041612 | 28.0185% | none |
| single_cubic_view_detail | .0970667402 | 28.1202% | none |
| single_cubic_view_direct | .1032527263 | 23.5394% | cam007 person |
| single_fine_reward25_cubic | .0972388933 | 27.9927% | cam007 jewelry, cam012 nails |
| single_fine_reward50_cubic | .0972432494 | 27.9895% | cam007 jewelry, cam012 nails |
| single_cubic_band16_phase80 | .0977506489 | 27.6138% | none |
| single_cubic_band16_phase95 | .0974310736 | 27.8504% | none |

Fine-phase ranking compares local G1-G4 bands at the original projected
coordinates. Rewarding higher aligned photo-band energy still worsened jewelry
and nails; those candidates are rejected. Expanding the transferred band from
sigma 6 to 16 only where phase agrees also failed to improve the reference.
No per-ROI switching or target-GT-derived construction masks were introduced.
Results: `evaluation/fine_phase_view_screen.json` and
`evaluation/adaptive_band_screen.json`.

A separate frozen-field readout probe is queued: intrinsic density peak and
density-squared physical-width centroid, each restricted to termination
q10--q90. It changes the depth statistic, **not** weights, density, grids or
calibration. Photographs never influence this depth. First-four-source transfer
uses the exact `single_near4_full_s8` selector settings for comparison. It is a
diagnostic, not a new default. Results are not available yet.

Focused tests: **43 passed** (`logs/tests_density.log`). The new density test
checks exclusion of a hidden high-density layer and centroid invariance when a
constant-density interval is split. Wide-source and cubic a=-.75 experiments
remain in progress; GPU stages are serialized.


## Next isolated checks (2026-10-17 14:55 UTC)

Fixed-label Keys a=-.75 (`single_cubic75_fixed`) measured LPIPS
**.0970088343**, **28.1631%** reduction, with no predefined ROI regression.
Initial eyes/nails/cuffs review retains the same limitations; no visual
acceptance is claimed. `single_cubic_view_detail` passed deep provenance,
source exclusion, immutable-grid checks and exact one-photo reconstruction
(`logs/audit_cubic_view_detail.log`).

Two further automatic tests are running:

- Pairwise G1-G4 luminance phase among geometrically valid TRAIN sources.
  Two distinct witnesses can change a candidate's rank; their RGB never enters
  its texture. Missing witness evidence is neutral, not an automatic rejection.
  Source ranking retains the existing geometry/appearance costs.
- Screened Poisson transfer within fixed selected-source patches:
  `(L + lambda I) delta = L(T-B)`, followed by the parent's boundary fade.
  L contains edges only between pixels assigned to the same photo. The solve
  preserves the broad LookCloser color while integrating source gradients;
  lambda .02 and .05 are tested. Each solve records relative residuals and
  iteration counts. Exact rejected-pixel fallback is asserted.

The combination of cubic a=-.75 and frozen-view illumination is also queued.
All GPU stages wait for current owners to exit. **45 focused tests passed**
(`logs/tests_poisson.log`), including two-witness evidence, constant-offset
rejection, source-boundary isolation and linear-solver convergence.


## Wide proposal result and local review (2026-10-17 14:58 UTC)

Expanding source proposals to 64/54/68/51 cameras (cam007/012/095/150)
produced `single_wide_risk`: LPIPS **.0997196684**, **26.1557%** reduction,
versus .0997288103 / 26.1489% for the same risk selector with 16 proposals.
No predefined ROI regressed. Sources beyond the initial 16 supplied only
.05370%, .01506%, .00591%, .00529% of the union of transfer support,
respectively. The current ranking therefore gains almost nothing from a wider
proposal set. This is a measured result for these costs, not proof that every
possible source-selection rule has reached its optimum.

The full small-patch review of `single_cubic_view_detail` is now
**not_accepted_local_detail_degradation**, despite its favorable named-ROI
averages. Cam007 small brocade motifs soften/change and cloth folds change
contrast; cam150 lace holes remain incorrect and some folds/embroidery soften.
Cam012 hair/jewelry boundaries and cam095 lace/hair boundaries retain errors.
See `evaluation/single_cubic_view_detail_visual_review.json` and all four
`evaluation/single_cubic_view_detail_spatial/*worst_local_regressions.png`.
No coordinates from this evaluation are fed to construction.

`summarize_single_luster_hybrid.py` rebuilds the scoreboard only from canonical
native metrics and receipts declaring one texture source with no lookup shift.
CPU preview aliases are excluded. It records numerical and visual status
separately: at this checkpoint **94 measured eligible candidates, zero visually
accepted candidates**, and no achieved 40% result.


## Completed readout and consistency checks (2026-10-17 15:15 UTC)

| Candidate | Native validation LPIPS | Reduction | Predefined ROI regressions |
|---|---:|---:|---|
| single_witness25 | 0.0996983424 | 26.1715% | none |
| single_witness25_cubic | 0.0973088096 | 27.9410% | none |
| single_witness75 | 0.0993678669 | 26.4162% | none |
| single_witness75_cubic | 0.0971047605 | 28.0921% | none |
| single_cubic_poisson2 | 0.0973432561 | 27.9154% | none |
| single_cubic_poisson5 | 0.0980198011 | 27.4145% | none |
| single_cubic75_view_detail | 0.0967073242 | 28.3864% | cam012 nails |
| single_density_peak | 0.1030058886 | 23.7222% | none |
| single_density_center | 0.1026470686 | 23.9879% | none |
| single_model_bias15 | 0.0979127636 | 27.4937% | none |
| single_model_bias30 | 0.0989035840 | 26.7600% | none |
| single_cubic75_gradient_guard | 0.1246083751 | 7.7251% | none |
| single_witness75_cubic75 | 0.0968182658 | 28.3042% | cam012 nails |

The density peak/center probes lose to q50 under the same first-four-source
settings (q50: 24.098% reduction). All 19 frozen-field renders reproduced q50
to about 5.96e-8 and verified unchanged model/grid hashes. Neither density
readout becomes the default. The model-bias correction and gradient guard also
fail to improve the numerical leader. The latter protects LookCloser's existing
structure aggressively and reduces the mean gain to 7.73%.

Cubic a=-.75 plus view illumination and plus witness ranking each improve the
mean slightly, but regress cam012 nails; both are rejected. Source selection
remains automatic: no per-ROI override is used to repair these scores.

`single_cubic75_fixed` passed deep provenance reconstruction
(`logs/audit_cubic75.log`), but its completed small-patch visual review is
**not_accepted_local_detail_degradation**. It remains the best numerical
candidate with no predefined ROI regression: .0970088343, 28.1631% reduction.
This is not a claim of an accepted final result, and the 40% objective remains
unmet. The already verified downloadable archive is still the explicitly
provisional `single_cubic_fixed` reference documented above.

Next check: replace scalar focal/depth resolution ranking with both singular
values of the frozen-surface projection Jacobian. This tests local directional
sampling, especially for oblique small surfaces. It does not alter the surface
or the sampled coordinates. **47 focused tests passed**, including anisotropic
projection detection and depth-discontinuity support rejection
(`logs/tests_differential.log`).


The directional-resolution follow-up differs from the earlier
`single_surface_s8`/surface-cost trials: it uses the current risk selector,
central differences across four native pixels, no new hard incidence gate, and
separately tests cubic sampling with fixed labels. It is a controlled follow-up,
not a newly discovered untested method family.

A second review bundle is being prepared for `single_cubic75_fixed`, explicitly
marked `not_accepted_local_detail_degradation`. It includes the unfavorable
small-patch panels as well as full images, source maps and favorable ROI scores.


Verified additional review bundle:

```bash
scp olive-lark:/fsx/tmp/luster_000540_hybrid_20261017T104440Z/exports/single_cubic75_fixed.tar.gz .
```

Archive SHA256: `c060f549e3cccc16d5b3166f4d90c7d1203fdb7126472be50b90ab065142a093`. 184036426 bytes; 256 members rehashed.
Status: **not_accepted_local_detail_degradation**. This supplements the older
provisional reference; it does not declare goal completion or visual acceptance.


## Adaptive seams and source potential (2026-10-17 15:30 UTC)

With fixed cubic a=-.75 source labels, computing the detail filter on each
source's geometric support gives LPIPS **.0964590957** (28.5702% reduction).
Removing the LookCloser fade only at mutually compatible source boundaries
gives **.0962912962** (**28.6944% reduction**), with no predefined ROI regression.
The compatibility check requires every touching source in a 3px neighborhood
to agree on the high-frequency correction within .02 RGB, with a 3x3 maximum
disagreement guard. It uses witnesses only for confidence, never RGB blending.
The exterior/support boundary retains its LookCloser fade.

`single_cubic75_seam_adaptive` passed deep provenance and exact reconstruction
(`logs/audit_cubic75_seam.log`). All four worst-tile panels and eyes/nails/cuffs
were inspected. Status remains **not_accepted_local_detail_degradation**: small
brocade motifs and some cloth folds soften/change, and existing hair/lace
geometry errors persist. Better averages do not override this visual finding.

Directional-resolution variants reduce LPIPS by 24.97--25.84%, but regress
face/skin/eyes and other details. They are rejected. The scalar scale-compatibility
penalty remains useful for this data; higher directional sampling resolution
alone is not sufficient evidence that a photograph is a better texture source.

`evaluate_source_potential_luster.py` generates 16 separate GT-free one-camera
texture predictions for each target, then evaluates their spatial LPIPS maps.
It saves only error statistics, **no GT-selected image or minimizing source map**.
Pointwise/tile minima include the baseline and the cubic75 reference. They
compare incompatible image neighborhoods and are **not achievable render
metrics, a rigorous lower bound, or evidence that the 40% goal is met**.

| Evaluation-only comparison | Validation mean spatial LPIPS |
|---|---:|
| LookCloser | .1350403999 |
| Cubic75 reference | .0970088417 |
| Minimum error, 256px tiles | .0886879505 |
| Minimum error, 128px tiles | .0858354505 |
| Minimum error, 64px tiles | .0827753488 |
| Minimum error, 32px tiles | .0800476926 |
| Pointwise minimum of spatial error maps | .0774553977 |

This diagnostic suggests source-selection headroom, while not establishing that
a coherent automatic selection can realize it. Prediction code never reads these
statistics. Next isolated tests vary geometric ranking weights globally and test
a one-photo multiplicative residual against the frozen source-view render.
**49 focused tests passed** (`logs/tests_log_residual.log`).


## Source-view structural reliability (2026-10-17 15:37 UTC)

The log photo/source-render residual does not help: detail-only gives LPIPS
.1027586907 (23.9052% reduction, 11 predefined regressions), and full residual
.1044174805 (22.6768%, eight regressions). Nails/cuffs regress; both are rejected.
Results: `evaluation/log_residual_screen.json`.

The next controlled selector retains all geometric and resolution safeguards,
and adds soft G2-G8 luminance phase costs: TRAIN photo versus the same frozen
source-view render, and that reprojected source render versus target LookCloser.
These are three isolated own/view/both .5-weight tests. Constant radiance gain
does not alter the score; opposite structure is penalized, one-sided texture
is unknown (.5), and flat/flat carries no mismatch evidence. It changes source
labels only, not geometry, photo coordinates, or single-photo construction.
No target GT, masks or ROI enter the selector. CPU preparation runs concurrently;
GPU evaluation waits for all four geometric-ranking evaluations to finish.

Fifty focused tests pass, including gain invariance and reversed-structure
ranking. The archived adaptive-seam reference remains visually unaccepted.

## Directional witnesses and sampling checks (2026-10-17 15:47 UTC)

The completed scoreboard contains 115 eligible measured single-source recipes,
zero visually accepted candidates, and no 40% result. The current audited and
exported numerical reference is still adaptive-seam cubic-.75 at 28.6944%.

Three further controlled checks are queued, not accepted results:

- Opposite-side witnesses: projected camera baselines in each target-ray tangent
  plane must have cosine <= -.25. Two distinct TRAIN witnesses from that side
  must corroborate G1-G4 phase before affecting the selected photo's cost. This
  tests whether same-side cameras were corroborating a shared parallax error.
  Weights .5 and 1.5 retain all earlier geometry and scale safeguards.
- Lanczos-3 interpolation with fixed automatic risk labels: all nonzero 6x6
  taps pass native source visibility; RGB is clamped to the central 2x2 range.
  Coordinates and geometry are unchanged. This isolates loss of source detail
  from the interpolation filter and its larger visibility footprint.
- Frozen pointwise directional color: query the existing field at each selected
  q50 surface point in both target and source directions, using the saved
  frequency grid. Density must be exactly equal for both queries, and the full
  model state must remain unchanged. Add .5 or 1 times the directional RGB
  difference to the single selected photo before the established detail merge.
  This separates view-dependent appearance from the different surface samples
  in a reprojected source-view volume render. It introduces no new model and
  never changes the point or projected photo coordinate.

GPU ownership is sequential: geometric costs, source-view phase, opposite-side
witnesses, Lanczos, then frozen pointwise color and its two evaluations. CPU
preparation can overlap. Fifty-three focused tests pass. The constant-color
point-correction test permits 5e-7 FP32 Gaussian roundoff; equal directional
colors, unchanged fallback, and independence across source labels are checked
exactly. No target photographs or evaluation masks enter any predictor.
