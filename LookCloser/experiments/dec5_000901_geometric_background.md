# DEC5 000901: geometric background diagnostic

## What was tested

The corrected field still produces wall-colored density around the hair.
The larger ray-background MLP failed its earlier held-out gate. This separate
track changes the background representation instead of adjusting the hair quota:
fresh calibrated background-only SIFT correspondences from16 training cameras,
fixed-pose triangulation, robust planes, and source-background RGB lookup.
The excluded camera pose selects the source neighborhood; its RGB and semantic
reference are opened only after candidate prediction sealing. No old subject
geometry, camera refinement, Gaussian splats, or hard target alpha cutout.

Source masks use the existing background-confidence>.98 and16-HD-pixel erosion.
Plane queries require convex-hull support, at least3 trusted source samples,
and median RGB dispersion<=.05. Unsupported rays retain the existing frozen
ray background. These are background hypotheses, not independent depth GT.

## Results

The sparse-plane pilot is **closed before held-out rendering**. Across39 pairs
of16 train views it recovers only3 reliable background points; individual views
have16–188 default SIFT features. A single low-contrast check increases the
nearest pair to395/174 features but produces only2 valid 3D points. No planes
are fitted and no sparse or RGB quality gate is weakened to force a result.

A separate whole-image check on the same two train cameras finds106 mutual
matches,84 epipolar/triangulation inliers, median Sampson error.6600px. This
supports local calibration consistency and isolates the sparse-background
failure from a general pose-convention failure. The native50 recipe is unchanged.
Three checks pass: known-plane triangulation with epipolar
outlier rejection, plane recovery with outliers, and occluded-source exclusion
with unsupported-extent fallback. The held-out component gate compares PSNR,
SSIM and LPIPS on trusted background and background adjacent to hair, but no
actual plane-background score exists because source geometry did not qualify.

## Insights and next steps

An accurate background might reduce the incentive to fit false foreground
density where the current ray-color network is inaccurate, especially in the
band excluded from conservative background correction. This is a hypothesis.
Require a material held-out background improvement, inspect the output, then
test a matched foreground continuation before changing native production.
Hidden background behind the person has no directly observed target GT; it
cannot be scored as if it were visible.

The next distinct diagnostic uses a separate bounded Instant-NGP background
volume:64 train views, only background-confidence>.98 eroded16 HD pixels,
scene box1.5,12 hash levels/max1024, no per-camera appearance embeddings,
fixed1024 rays,20000 updates. An exact empty inner box of half-width.18 excludes
the foreground volume; both occupancy and rendering call the gated density.
The foreground field is unchanged. A two-update structural run precedes the
full trial. This method can fit continuous multi-view color without SIFT
correspondences; it remains unpromoted until component and foreground tests.

### Separate background volume: component result

The full20000-update fit completes in379 seconds locally. The two-update smoke
and full-state reload verify exact zero inner density, finite positive outer
density, and restoration of optimizer/scheduler/scaler/RNG/buffers. The serving
adapter agrees with the independent sealed prediction to6.56e-7 maximum RGB
error over2604 rays; initialization and prediction preserve RNG, and all
background parameters/buffers remain frozen. Five existing background/thickness
checks also pass. No foreground-density correction is claimed by these checks.

One excluded HD camera; fixed GT-composited masked ROI protocol:

| Region / component | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Trusted background, original ray model | 14.7310 | .85110 | .387720 |
| Trusted background, volume + ray residual | 16.0503 | .87798 | .378163 |
| Background adjacent to hair, original | 16.7258 | .95846 | .150469 |
| Background adjacent to hair, volume + residual | 23.5194 | .97385 | .101184 |

The hair-adjacent background improves6.79dB and32.75% LPIPS; trusted background
improves1.32dB and2.46% LPIPS. At1999 steps the adjacent region already improves
to23.4193dB/.105400LPIPS, so the full fit preserves the material effect.
Visual inspection shows a dark person-shaped pattern in the old background
component and wall-colored completion in the volume component. Hidden background
has no target GT, so this does not prove reconstruction behind the person.

An incorrect dark background could explain dark hair with insufficient
foreground opacity, while forcing false light density where GT shows wall.
This is a testable causal hypothesis. Matched20189–25188 continuations from the
same corrected20188 checkpoint now compare the old and new frozen background;
sampling, frequency treatment, all other terms and source RGB remain unchanged.
Production staysv6 until foreground quality and geometry pass their gates.

[Component metrics](assets/dec5_silver_sable/neural_background/final_component_metrics.json),
[GT / original background / volume background](assets/dec5_silver_sable/neural_background/final_component_preview.jpg),
[serving invariants](assets/dec5_silver_sable/neural_background/provider_probe.json),
[whole-image calibration check](assets/dec5_silver_sable/neural_background/whole_image_pose_audit.json),
[failed low-contrast stereo check](assets/dec5_silver_sable/neural_background/lowcontrast_pair_audit.json).

### Matched foreground continuation: material RGB improvement

Both20189–25188 runs complete from the same full20188 checkpoint. Pixel-allocation,
camera/source/class policy audits and known-background counts agree over5000
batches and20480000 rays. Geometry-dependent adaptive field-query counts differ
and are recorded separately: these are update-matched, not query-count-matched
runs. The frozen background is the changed component.

| Region | Original PSNR / SSIM / LPIPS | 3D background PSNR / SSIM / LPIPS |
|---|---|---|
| Hair | 24.2026 / .77880 / .243987 | 24.9980 / .78374 / .224547 |
| Hair boundary | 18.8973 / .94773 / .100904 | 21.5405 / .95511 / .084498 |
| Face/neck | 30.1713 / .92520 / .121104 | 32.2753 / .92782 / .110133 |
| Clothing/hands | 28.5399 / .86611 / .155125 | 29.0425 / .87251 / .149974 |
| Trusted background | 14.7732 / .84898 / .406717 | 15.9849 / .86565 / .400138 |

LPIPS improves7.97% hair,16.26% boundary,9.06% face and3.32% clothing/hands.
Hair opacity<.1 fraction falls .14661%→.02916% (80.11% fewer), boundary
4.27487%→1.02877% (75.93% fewer). However, mean trusted-background foreground
opacity **increases .024384→.030383**, and face opacity<.1 grows .28606%→.44810%.
RGB improvements therefore do not establish universally better geometry.
Inspection shows a cleaner surrounding background but still-soft hair and
residual contour defects. Matching4K GT-transfer tests are running with the
same native source RGB and original common fallback to isolate geometry.

A full HD continuation preserves the measured ancestry: old background through
20188, new background from20189, continuing the25188 full state to106316 with
the unchanged FR1→FR.3 schedule. Its pilot selection covers later canonical
gates; it does not claim the native production's complete seven-gate selection.
Native production remainsv6 pending geometry/full-schedule validation.

Eight train cameras compared across901/917/932 have identical calibrated poses
and intrinsics. Mean fixed-camera common-background PSNR is38.09dB (minimum31.77),
hair-adjacent background39.68dB (minimum33.71), after excluding changing foreground.
This supports investigating a shared static background; it does not validate
every frame or promote sharing automatically.

[Matched foreground metrics](assets/dec5_silver_sable/neural_background/foreground_matched.json),
[GT / original / 3D-background head comparison](assets/dec5_silver_sable/neural_background/foreground_comparison.jpg),
[sampling audit equality](assets/dec5_silver_sable/neural_background/sampling_equivalence.json),
[train-only static-background audit](assets/dec5_silver_sable/neural_background/static_background_audit.json).


### Native6K cold start and isolated transfer gates (2026-10-29)

Current production is **v7_seed42**, preserving the corrected v6 foreground
recipe and adding explicit background initialization seed42. The prior HD warm
comparison used the same legacy frozen background in both arms. Two independent
seed42 background fits from different incoming RNG states give identical
weights; the deployed fitting function matches the tested source exactly.
This is a reproducibility correction, not evidence that random background
initialization caused historical flicker.

Both native cold arms completed 0–15188 using genuine6144×3072 RGB and all64
native1000-step-per-level frequency maps. Original ray background weights are
identical. The volume arm starts its separate3D background from update0.
The metrics-only HD semantic reference is bilinearly resized to native6K;
boundary radius is16 native pixels, corresponding to5 HD pixels. These numbers
must not be compared directly with HD LPIPS. No new fitting uses the held-out
image or evaluation reference.

| Region | Original PSNR / SSIM / LPIPS | Volume PSNR / SSIM / LPIPS |
|---|---|---|
| Hair | 22.5087 / .77360 / .335967 | 23.6047 / .77930 / .337109 |
| Boundary | 17.5375 / .95947 / .116785 | 20.8618 / .96535 / .106770 |
| Face/neck | 30.3776 / .89794 / .287908 | 29.0594 / .89409 / .266821 |
| Clothing/hands | 25.4819 / .83897 / .284727 | 28.5532 / .84417 / .278178 |
| Trusted background | 12.8927 / .81789 / .353202 | 15.7036 / .85576 / .375024 |

Hair LPIPS is essentially unchanged (+.34% worse), boundary improves8.58%, but
boundary opacity<.1 grows **3.33%→6.62%** and mean trusted-background foreground
opacity .03169→.05185. Visually the volume arm has a less prominent dark halo
but still has noisy/floating wall fragments and soft hair. **Do not promote**
this cold-start variant on the basis of stronger HD continuation results.

The matched25188 fields also finished the identical4K GT-transfer gate with
native source RGB, q10 geometry, opacity floor.1 and the same old fallback:
hair LPIPS .202126→.204191 (1.02% worse), boundary .045268→.045832,
face .118311→.118551. Source-fallback holes .17999%→.08827% and halo proxy
.011536%→.004944% improve, but recovered texture does not. A common new-background
fallback leaves the conclusion unchanged (hair .201840→.204560); its HD background
is explicitly upsampled, not native4K detail. Overall PSNR/SSIM improve while
whole-frame LPIPS worsens, so background texture quality is a separate concern.

A static-time16-view small camera slide is running on clever-shadow using these
same frozen25188 fields, eight train-only native RGB sources, fixed source order
and a shared neutral-gray fallback. It is a missing-foreground diagnostic, not
a complete-scene render or a physical-time flicker claim. The native50 recipe
remains the original background pending stronger evidence.

[Native control](assets/dec5_silver_sable/neural_background/native15k_control.json),
[native volume](assets/dec5_silver_sable/neural_background/native15k_volume.json),
[visually inspected GT/control/volume](assets/dec5_silver_sable/neural_background/native15k_comparison.jpg),
[geometry control](assets/dec5_silver_sable/neural_background/geometry25k_control.json),
[geometry volume](assets/dec5_silver_sable/neural_background/geometry25k_volume.json),
[common fallback ablation](assets/dec5_silver_sable/neural_background/common_volume_fallback_comparison.json),
[deployed function equivalence](assets/dec5_silver_sable/neural_background/seed42_deployment_function_equivalence.json).


### Fresh hair feature geometry: bounded feasibility audit

A separate CPU-only audit used16 nearest training cameras, fixed calibration,
semantic hair>.8 eroded2HD pixels, SIFT mutual ratio.7, epipolar/triangulation
error≤1.5px, and minimum3-view tracks with≥3-degree parallax. Each view has
700–980 hair features, but58 candidate camera pairs give only413 accepted
pair points and **32** consistent3+view points. Median/max-track reprojection
error .860/1.200px (median/p90). Inspection shows very sparse support, insufficient
for comprehensive hair supervision. No sparse prior or additional training was
launched. These points are image-derived hypotheses, not independent depth GT.
Do not weaken thresholds to manufacture hair geometry. The qualitative route
is closed at this feasibility gate; denser learned correspondences would be a
different method requiring separate verification.
[Audit and decision](assets/dec5_silver_sable/fresh_sparse_hair/decision.json),
[train-only point overlay](assets/dec5_silver_sable/fresh_sparse_hair/E004_A005_1210QB.jpg).


### Completed16-view static path and full first-stage RGB checks

The matched HD25188 fields complete the16-view camera slide, with all32 encoded
frames decoded and both contact sheets plus the worst adjacent-view patch
inspected. Visibility-toggle proxy improves **14233→10548 (25.89%)**;
depth-exposure dropouts170608→159026 (6.79%). Total selected-hair pixels grow
2038615→2073896 (1.73%), while present coverage within baseline-voted hair
falls .98%. Thus the toggle reduction is not simply a large global deletion,
but reference-hair preservation is imperfect. Source switches improve only1.72%.
Common-warp RGB MAE is **.017404→.017868 (2.66% worse)** overall; specifically
at baseline-fallback transitions it improves .126123→.019751. Visible floating
wall fragments remain around the head in both arms. This is a partial coverage
improvement, not solved flicker or a completed96-view/physical-time result.
[Path metrics including unconditional coverage](assets/dec5_silver_sable/neural_background/path16/metrics.json),
[RGB-change proxy](assets/dec5_silver_sable/neural_background/path16/rgb_proxy.json),
[baseline decoded frames](assets/dec5_silver_sable/neural_background/path16/baseline_decoded.jpg),
[candidate decoded frames](assets/dec5_silver_sable/neural_background/path16/candidate_decoded.jpg),
[worst adjacent-view patch](assets/dec5_silver_sable/neural_background/path16/problem_patch_0.jpg).

The warm-background full HD run preserves its advantage through75940: relative
to the same-step corrected baseline, hair LPIPS improves8.19%, face11.41%,
contour17.90%, clothing4.35%. Gates30376/45564/60752 also improve all four regions.
The final FR.3 stage remains running. Because cold native6K did not reproduce the
HD gain, a **single matched native warm trial** now tests activating the same
frozen3D background at15189 from the common native15188 parent, through20188.
Its native sampling and all other terms match the contour-series control.
This is a phase/ancestry check, not promotion or a new threshold sweep.
[Full HD ROI checks through75940](assets/dec5_silver_sable/neural_background/full_through75k.json).


### Native warm-background20188: measured contour benefit

The common native15188 parent followed by5000 updates with volume activation at
15189 completes. Compared with the identical native control20188:

| Region | Control PSNR / SSIM / LPIPS | Warm volume PSNR / SSIM / LPIPS |
|---|---|---|
| Hair | 22.4654 / .77254 / .325391 | 23.9280 / .77857 / .314222 |
| Boundary | 17.4459 / .95887 / .115455 | 21.0799 / .96534 / .102304 |
| Face/neck | 29.5521 / .89622 / .276385 | 30.4638 / .89723 / .270454 |
| Clothing/hands | 26.7867 / .83798 / .279367 | 28.0616 / .84473 / .271213 |
| Trusted background | 12.9071 / .81834 / .353547 | 15.4466 / .85691 / .375151 |

Hair LPIPS improves3.43%, contour11.39%, face2.15%, clothing2.92%. Hair
opacity<.1 drops **.11151%→.01123%** (1410→142 pixels); contour2.507%→.593%.
Mean trusted-background foreground opacity falls .031441→.030052 rather than
increasing. The early cold-start regression is therefore phase-dependent: this
warm activation shows a better tradeoff. Nonetheless background LPIPS worsens
6.11%, and the full native schedule/temporal result is still unverified.

The same native GT hair domain contains1276 restored-threshold pixels and8 newly
missing ones. On the restored subset, RGB MAE improves .172776→.079957. Three
largest restoration clusters were inspected alongside GT, control/warm RGB and
both opacity maps. This supports an actual local appearance improvement, while
the opacity threshold is still not a semantic alpha or depth accuracy metric.
Matched4K source-transfer geometry is running; production remainsv7_seed42.
[Native warm metrics](assets/dec5_silver_sable/neural_background/native_warm20k.json),
[visually inspected native GT/control/warm](assets/dec5_silver_sable/neural_background/native_warm20k_comparison.jpg),
[threshold-crossing audit](assets/dec5_silver_sable/neural_background/native_warm_opacity_changes.json),
[inspected restoration clusters](assets/dec5_silver_sable/neural_background/native_warm_opacity_patches.jpg).


### Native warm geometry qualifies for full validation

The native20k same-source4K geometry gate completes: source fallback holes
.12714%→.01782% (85.99% fewer), contour LPIPS .043054→.040012 (7.06% better),
hair .195075→.191691 (1.74% better), face .120801→.119319 (1.23% better).
Halo proxy improves30.56%; reprojection3.64353→3.58004. This isolates geometry
from the changed background RGB: the final fallback is identical in both arms.
Clothing LPIPS .199941→.201531 (.80% worse) is retained as a regression.

A full native matched pair is now running on clever-shadow at
`/home/ubuntu/lookcloser_transfer/native_full_pair_20261029`. Both share the exact
cold0..15188 full-state prefix and sealed first eval render, then train through
106316 with all seven canonical gates retained. Adam, scheduler, scaler, RNG,
frequency buffers and occupancy restore from the common checkpoint. The warm
arm activates the frozen3D background at15189; the control keeps the original
seed42 background. A code-file hash manifest records the snapshot. Native data,
frequency fits and held-out camera stay identical. This is one-frame research,
not a completed production50 run. A96-view native-field static path is prepared
using the same translation extent as the previous16-view test; its RGB sources,
order, ROI and neutral-gray fallback are fixed for both arms. Physical-time
stability remains a separate50-frame evaluation.

The32 sparse image-derived hair points also lie well inside the existing±.15
scene box: smallest boundary margin .11954 normalized units. No clipping is
observed for these sparse hypotheses; they do not cover every thin strand.
[Sparse AABB audit](assets/dec5_silver_sable/fresh_sparse_hair/aabb_audit.json).


### Completed full HD schedule

The warm-background HD run completes106316, preserving its advantage in FR.3.
At the final matched gate LPIPS improves **5.73% hair,15.55% contour,10.32% face,
5.53% clothing/hands**, and1.24% trusted background relative to the corrected
full-schedule control. At91128 improvements are8.95%,17.20%,11.63%,4.84% respectively.
This supports continuing the independently running full native matched pair;
it does not substitute for its native6K validation or physical-time review.
[All completed HD gates](assets/dec5_silver_sable/neural_background/full_hd_final.json).
