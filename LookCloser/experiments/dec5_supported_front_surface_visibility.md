# DEC5: unsupported inward hits versus supported exterior visibility

## What was tested

After [gap subface carving](dec5_train_gap_subfaces.md), five moving-view pixels
became black despite retaining target mesh intersections. Four hit newly exposed
geometry; reusing their old colors would paint a different surface. This study
replays every source-admission gate rather than loosening masks indiscriminately.

`diagnose_gap_texture_admission.py` recreates actual float32 ray-hit points,
all 62 full native source-mesh depth maps, unmasked/masked bilinear center and
four-tap depth tests, exact-integer handling, incidence/angle weights, and raw
PatchMatch near observations. No prediction changes in this diagnostic.
`inspect_gap_texture_witnesses.py` adds continuous camera-to-point rays and
five actual train RGB/mask witnesses with the frozen color response.

Root: `/mnt/data/dec5_gap_texture_admission`.

The findings motivate two controls on the **identical mesh**:

1. `study_target_backface_culling.py`: standard target-only backface culling.
   Source depth, labels, masks and colors remain unchanged. Original triangle
   IDs and barycentrics are restored after intersecting the front-face subset.
2. `recover_supported_front_surface.py`: a narrow ray fallback. The original
   pixel must be black with no selected source, hit a backface, and have zero
   measured near-depth observations. A farther front-facing intersection must
   have an admitted train RGB source and at least three measured near views.
   Only then are its exact RGB, source ID and depth taken from the verified
   culled render. All other outputs are bit-identical to baseline. No ROI or
   manually listed pixel selects the fallback; no averaging or inpainting.

The near test uses native nearest depth, tolerance .0015 normalized units,
with the existing raw-depth integer camera convention. Three close views are
an empirical confidence gate, not independent physical truth.

## Results

### Cause of the five black pixels

| Moving-view pixel group | Mesh-footprint-visible source views before / after masks | Measured near views | Continuous mesh-ray-visible views |
|---|---|---|---|
| Four-pixel cluster | 0,1,0,1 / all zero | all zero | all zero |
| Remaining single pixel | 2 / zero | 24 | 39 |

The two raw footprint admissions in the cluster use A/B. Continuous rays hit
nearer geometry at fractions .9994..9996 rather than the queried point. Thus
the loose relative depth test is not proof of visibility. In the inspected
central-camera photos the queried positions project onto occluding finger/tube
regions; the A/B projection is outside the retained foreground. Masks must not
be globally bypassed to recover these pixels.

The fifth point is slightly outside the visible tube in the inspected H/C,
I/C and K/B image/mask coordinates; the eligible I/C and N/A texture views
project onto background beside the tube. Close-depth counts alone again cannot
establish correct object identity. This point is not recolored by the fallback.

Crucially, the four cluster hits face **away** from the target camera: signed
incidences are about −.196, −.479, −.196, −.478. The fifth is front-facing
(+.337). The existing double-sided renderer sees an internal/back surface
before the exterior surface behind it. This is a mesh/ray explanation, not
proof that every current normal or occluder is physically correct.

### Global culling: reject as production replacement

Root: `/mnt/data/dec5_target_backface_culling/000995`.

| View | Original backface hits | Changed RGB | New black | Newly colored |
|---|---:|---:|---:|---:|
| Moving | 233 | 210 | 93 | 4 |
| H/C | 448 | 393 | 163 | 0 |
| K/B | 2482 | 2062 | 364 | 7 |

Global culling recovers the hand-side cluster but visibly worsens existing
crown holes. Source face labels remain exactly equal, and mesh bytes do not
change. RGB changes outside original backface hits are 0,0,1 pixels; the one
K/B difference is recorded rather than claiming global bit identity. Head and
lipstick panels were actually viewed. All new-black component sheets are saved
but were **not** exhaustively inspected because the broad control already fails.

### Confidence-qualified fallback: narrow recovery without global regression

Root: `/mnt/data/dec5_supported_front_surface_fallback/000995`.

| View | Candidate black pixels with a colored farther front hit | Accepted | Original near counts | Accepted farther-surface near counts |
|---|---:|---:|---|---|
| Moving | 4 | 4 | all 0 | 52,52,53,52 |
| H/C | 0 | 0 | — | — |
| K/B | 7 | 5 | all 0 | 8,7,6,11,11 |

The four moving pixels now reveal a coherent skin-colored exterior surface,
using normal train reprojection at depth approximately .602 rather than the
unsupported inward hits at .565. This is not reuse of colors from the removed
foreground, nor invented color on an unseen surface. The two rejected K/B
candidates have zero measured support on the farther surface.

All originally colored pixels, and all RGB/depth/source values outside the nine
accepted pixels, remain exactly unchanged. There are **zero new black pixels**
in the guarded result. The five K/B changes are isolated pixels on the ragged
hair edge; they do not repair that geometry or support an artifact-free claim.

[Moving recovery patch](/mnt/data/dec5_supported_front_surface_fallback/000995/review/moving/recovered_01.png).
The root agent viewed five source witnesses, six global head/lipstick panels,
and all six guarded recovery patches, including train GT for the five K/B
patches. Independent arithmetic replays the native near-depth counts and checks
the exact RGB/depth/source splice. The [review receipt](/mnt/data/dec5_supported_front_surface_fallback/000995/visual_review.json)
records scope and limitations. These are 1080x1920 diagnostic stills, not a
new 6K video. No PSNR/SSIM/LPIPS or full-frame quality claim is made.

Finalization verified 234 artifact bindings. All 41 focused tests pass, including
face orientation/ID mapping and each mandatory fallback gate. All three render
workers and subsequent audits finished; GPU memory returned to its idle level.
Compact logs and test output are retained under the guarded fallback root.

## Insights

An untexturable first intersection is not necessarily the exterior surface
that should be rendered. For this cluster, geometry confidence and orientation
identify an unsupported inward hit; a measured, textured exterior lies behind
it. The constrained ray fallback resolves that specific defect without copying
background onto the object or cutting all back-facing hair patches.

Global backface culling is unsafe on this incomplete/repaired mesh, whose
orientation and missing exterior patches are not globally validated. The
qualified fallback is a renderer visibility policy, **not** an improved saved
mesh. Actual mesh improvement remains the preceding train-gap carving work.

Next: test the qualified visibility policy on other actor times and return to
temporal transfer of the semantic geometry constraints. Do not keep tuning a
single isolated pixel or call the cheek-hole objective solved. The existing
native-6K delivery is unchanged; the dynamic video and general mesh goal remain
incomplete.
