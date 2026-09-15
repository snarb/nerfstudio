# DEC5: zero-weight texture footprints and neighborhood-prior transfer

## What was tested

The [continuous-surface pilot](dec5_poisson_jaw_completion.md) repaired geometry
but exposed a source-visibility failure: 16 valid-depth jaw pixels lost every RGB
source because neighboring depth taps with negligible weights were still required.

`native_texture_footprint.py` adds an isolated opt-in rule:

- Snap projected/registered coordinates within .001 native pixel of an integer
  to that center, bounding the numerical adjustment per axis.
- Ignore **exactly zero-weight** footprint taps after snapping. Every nonzero
  tap still requires foreground depth and the unchanged relative-depth tolerance.
- Gather exact integer samples directly, avoiding normalized-grid roundoff.
  Other fractional samples use the original bilinear sampler unchanged.

There is no cross-camera RGB averaging, inpainting, mask dilation, exposure fit,
depth change or new geometry in this texture control. Static registration,
early target-angle source ranking and hard source selection stay active.
`study_native_texture_footprint.py` checks the original renderer source before
compiling an isolated opt-in variant; production defaults/files stay untouched.

## Results

### Matched 001193 texture control

All three fresh views have byte-equal depth arrays and the same mesh, camera,
train sources and fixed exposure as their matched previous renders.

| Frozen F/E train jaw region | Before | After |
|---|---:|---:|
| Geometry misses | 3 | 3 |
| Black RGB pixels | 19 | **3** |
| Black RGB with valid depth | 16 | **0** |

All 16 recovered pixels select physical F004_E005_1210FP (source index 25) and
equal its calibrated GT RGB exactly. This is a train reprojection diagnostic,
not independent held-out accuracy. A CUDA test places extremely bright values
in neighboring pixels and confirms exact-center output does not include them.

Moving/F/E/held-out RGB changes at 40/4062/68 pixels respectively; no depth array
changes. Some changes are outside the small jaw region, so native head comparisons
were inspected rather than assuming locality from the selected-pixel result.
No new conspicuous defect was seen in these views; existing hair/crown and edge
geometry limitations remain.

Held-out face PSNR / SSIM / LPIPS: baseline **30.2765446 / .93969184 / .07612340**;
new **30.2765465 / .93969178 / .07612446**. Changes are about 1e-6, including a
tiny LPIPS increase; do not describe this as identical metrics or an improvement
in held-out fidelity. No full-frame image-quality metrics were computed.

![Same geometry, corrected source footprint](/mnt/data/dec5_native_texture_footprint/review/F004_E005_1210FP_detail.png)

### Frozen geometry and texture method at 001195

`run_neighborhood_completion_transfer.py` configures the existing prototype
modules in a separate worker for the requested actual time. It checks existing
audited native-depth/mask inputs and never substitutes the 001193 mesh or poses.
This controller currently supports the four times with those retained inputs;
it is not a completed 150-time reconstruction controller.

At 001195, the same Poisson settings, observed-neighborhood certificate and
measured free-space guard are reused without tuning. Four fresh renders compare
production/repaired geometry at moving and real F/E views, with the **new
footprint applied to both variants**. Thus the pair isolates geometry.

The frozen edge-inclusive train jaw region improves from **14 to 4 depth misses**
and **14 to 4 black RGB pixels**. The interior-only region has zero misses in both.
The moving view changes 17 RGB pixels; the F/E view changes 117. Independent
replay recomputes 2858 seed queries, 2662 certificates and 124 native ray checks,
with exact original geometry preservation and no qualified free-space violation.

![001195, same footprint in both geometry variants](/mnt/data/dec5_neighborhood_completion_transfer/001195/review/F004_E005_1210FP_detail.png)

Main-agent visual review covered the 001193 moving/train detail panels, all three
head pairs, held-out comparison, and all four 001195 head/detail pairs. Residual
small spots, ragged silhouette/hair and the wider movie's hand/lipstick defects
remain. This is positive local transfer, **not artifact-free video approval**.

Five tests pass, including CUDA exact-center isolation, unchanged fractional
sampling, bounded snapping and refusing an unexpected renderer implementation.
All workers finished normally. Roots:
`/mnt/data/dec5_native_texture_footprint` and
`/mnt/data/dec5_neighborhood_completion_transfer/001195`.
No published movie, source EXR or existing model default was changed.

## Insights

Zero-weight neighbors should not prohibit a valid color source; fixing this
does not require allowing background contribution or averaging camera colors.
Mesh coverage and texture coverage needed separate corrections. The local
neighborhood prior now helps at two real times with a common recipe, but broader
temporal/view coverage and unresolved body/lipstick defects still prevent
completion of the full dynamic-video objective.
