# DEC5 001055: opacity validity gates and disappearing hair

## What was tested

Frozen background-separated LookCloser field at step4999, identical GT RGB,
fixed midpoint-selected source cameras, identical fallback and ray marcher.
Changed both the exported volume-opacity threshold and the packed front-q10
minimum-mass threshold. These are validity gates, not depth of field or semantic
output masks. Empty rays remain invalid even at a zero threshold. FP64 packed
mass is retained for offline thresholding.

Seven 16-view HD arms: both gates .001/.01/.03/.1/.2, source-only .01, and
target-only .01. The leading both-.01 arm then received an independent UHD
comparison on the three existing validation cameras; reserved test RGB was not
used. A 96-view HD confirmation is complete locally, outside silver-sable.

## Results

Three UHD held-out views, fixed five-pixel reference boundary band:

| Source / target floor | Hair PSNR | Hair SSIM | Hair LPIPS | Boundary LPIPS | Face/neck LPIPS | Holes | Halo |
|---|---:|---:|---:|---:|---:|---:|---:|
| .1 / .1 | 22.1433 | .40065 | .176732 | .028375 | .112947 | .3507% | .1757% |
| .01 / .01 | 22.1766 | .40312 | .174436 | .026129 | .112542 | .0196% | .1898% |
| .1 / .01 | 22.1002 | .40042 | .176884 | .029308 | .112880 | .0323% | .2211% |

Both-.01 improves boundary LPIPS **7.91%**, reduces holes **94.41% relative**
(a small absolute fraction of hair pixels), and improves whole-hair LPIPS1.30%.
Face/neck LPIPS improves0.36%. Halo increases0.0141 percentage points.
Full-frame LPIPS worsens.53452→.54328: lower thresholds also admit unwanted
background fragments. Target-only lowering is inferior at the contour.
“Holes” here means reference-hair pixels with no admitted GT source
(`source_id < 0`); fallback can still provide RGB. It is not a count of black
pixels or a direct measurement of absent physical strands.

[GT / .1 / .01 crown comparison](assets/dec5_silver_sable/opacity_4k_crown_gt_comparison.jpg).
Reference semantics are fallible; RGB/geometry predictions were sealed first.
The same fixed neutral-exterior crop protocol is used for all UHD LPIPS values.

The 16-view pilot reduced common-anchor visibility toggles12293→9141 (25.64%).
Unconditional present-hair proxy increased2930861→2975974 (1.54%); the apparent
toggle reduction did not come from deleting the whole hair region. Common
baseline-depth transported hair RGB MAE improved.0089086→.0086580 (2.81%).
These are viewpoint-path proxies, **not physical-time flicker or 3D accuracy**.
The higher floor.2 increased toggles and lost coverage; rejected.

[Pilot coverage metrics](assets/dec5_silver_sable/opacity_floor/both010_review/metrics.json),
[pilot RGB proxy](assets/dec5_silver_sable/opacity_floor/both010_rgb.json).

## Insights and next steps

- The original .1 gate can erase real low-opacity boundary evidence. Wrong depth
  also changes source visibility and source selection; no depth-of-field effect
  is needed for a discontinuity. Lowering both gates addresses a concrete cause.
- Both-.01 passes the proposed static holes criterion, but the additional bright
  background fragments remain at the full96 visual gate. The default remains.1;
  this is a candidate export/render setting, not a claimed training improvement.
- Replacing unsupported RGB with this field's own NeRF foreground RGB left
  geometry/source IDs unchanged and produced essentially no RGB-continuity gain
  (under.01% on anchored hair). Closed rather than extending the fallback model.
- Validity-normalized source-depth interpolation fixes zero/valid mixing in an
  analytic plane case, but yielded only0.46% hair-LPIPS improvement and unchanged
  holes here. Nearest-neighbor depth was similarly marginal. Neither is promoted
  as a substantial hair fix.
- Artifacts: `/fsx/tmp/lookcloser_hair_001083_20261027_1330/local_20261028/opacity_static`
  and `opacity_full96`; original seven-arm run is under
  `lookcloser16/opacity_sweep_20261028` in the same campaign root.

### Full96 confirmation

| Proxy | Both .1 | Both .01 | Change |
|---|---:|---:|---:|
| Visibility toggles | 94445 | 72742 | −22.98% |
| Source switches | 42378 | 12773 | −69.86% |
| Depth-exposure dropouts | 865070 | 843493 | −2.49% |
| Common-anchor present pixels | 16612982 | 16576635 | −0.22% |
| Unconditional present-hair proxy | 16799888 | 16835713 | +0.21% |
| Transported hair RGB MAE | .00916566 | .00890921 | −2.80% |

The reduction does not come from a large deletion of the visible-hair proxy.
At14750 baseline fallback-transition pixels, transported RGB MAE improves
.077847→.014449. All96 decoded candidate frames and three worst adjacent-pair
crops were inspected. Gross head mass is similar; false wall fragments above
the crown and around the ear remain obvious. No continuous playback or physical
time-stability claim is made. **Retain .01 as a component candidate, not as a
complete hair-geometry solution**; do not expand the threshold sweep.

[Metrics](assets/dec5_silver_sable/opacity_floor/full96/metrics.json),
[RGB proxy](assets/dec5_silver_sable/opacity_floor/full96/rgb_proxy.json),
[baseline video](assets/dec5_silver_sable/opacity_floor/full96/baseline.mp4),
[candidate video](assets/dec5_silver_sable/opacity_floor/full96/candidate.mp4),
[visual decision](assets/dec5_silver_sable/opacity_floor/full96/visual_decision.json).

### Smooth depth-visibility compositor: small, unpromoted gain

The original RGB transfer rejects a source when depth residual exceeds
`.01*z + .002`. A separate appearance-only prototype retains the same field,
depths, fixed source order and RGB, replacing this hard decision with smooth
ordered confidence transitions. Good correspondences preserve exact first-source
RGB. Uncertain colors and the same fallback mix in **bounded linear-sRGB**,
avoiding the saturated-white blowup of inverse-Reinhard pseudo-HDR mixtures.
Semantic masks do not decide output opacity.

On top of both-.01, three-view UHD hair PSNR/SSIM/LPIPS become
**22.2039/.40367/.173902**, boundary LPIPS.025361 (2.94% better), face LPIPS.111813,
halo.1725%. On the matched16-view path, transported hair RGB MAE changes
.00871177→.00867840 (**0.38%**); fallback-transition MAE improves8.69%.
This does not meet the substantial-improvement gate: **do not promote or extend**.
The confidence-weighted source labels make binary coverage/holes incomparable
to the hard-gate protocol; lower weighted reprojection scores are not better
physical geometry. No density or depth was changed.

[UHD metrics](assets/dec5_silver_sable/soft_visibility_uhd010_metrics.json),
[matched RGB proxy](assets/dec5_silver_sable/soft_visibility_path16_vs_both010_rgb.json),
[GT / both-.01 / soft comparison](assets/dec5_silver_sable/soft_visibility_crown_comparison.jpg).
The enlarged crop still shows substantial inherited false background patches
and incorrect curls. Neither opacity lowering nor smooth confidence solves that
geometry limitation. Full96 confirms only the leading floor's partial component
gain; no default changes have been made.
