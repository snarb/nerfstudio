# DEC5 000901: hair-focused sampling after numerical fixes

## What was tested

Matched5000-update continuations15189–20188 from the **same corrected cold
15188 checkpoint**, including Adam/scheduler/scaler/RNG. HD geometry training,
64 train cameras, excluded D004_A005_12103C. Existing body80/FAS50 control versus
hair-focused half of core draws, keeping20% whole-image draws and the protective
band. FAS selects nonempty frequency levels with1→3 weights, then pixels within
the selected level. Hair FAS uses reliable hair cells, not person cells relabeled
as hair. Source RGB and rendered output are not semantically masked.

Actual pixel allocation changes **11.3%→37.9% hair**, while background remains
25.7%→26.1%. A separate arm relaxes the frequency-grid update gate.9→.1; it does
not change normal rendering opacity or the source/target export validity floor.

## Results

One excluded HD view, fixed GT-composited tight ROI protocol:

| Arm | Hair PSNR | Hair SSIM | Hair LPIPS | Boundary LPIPS | Face LPIPS | Mean trusted-BG opacity |
|---|---:|---:|---:|---:|---:|---:|
| Corrected body80 control | 24.3671 | .77851 | .257754 | .101512 | .128886 | .024606 |
| Hair core/FAS mixture | 24.1409 | .77827 | .219964 | .098629 | .123920 | .023669 |
| Relaxed frequency-update gate | 24.3317 | .77886 | .253576 | .100791 | .128178 | .025778 |

Hair mixture improves **hair LPIPS14.66%, face3.85%, boundary2.84%**. Hair PSNR
decreases.23dB, face PSNR.42dB; perceptual gain is not a universal metric win.
The inspected head crop shows finer internal curl texture, while remaining softer
than GT. Mean background opacity improves only3.8%: this is not yet a demonstrated
large foreground/background depth separation improvement.

Clothing/hands proxy: control PSNR/SSIM/LPIPS**28.3472/.86392/.162471** versus
hair mixture**28.1465/.86217/.165352**; LPIPS is1.77% worse. **Do not promote to
the native queue yet.** The full schedule and downstream geometry gate remain.

[Head comparison: GT / control / hair mixture](assets/dec5_silver_sable/hair_mix_corrected_head_comparison.jpg),
[all three arms](assets/dec5_silver_sable/corrected_hair_ablations_summary.json),
[control with body](assets/dec5_silver_sable/body_control_with_body_hd.json),
[hair with body](assets/dec5_silver_sable/hair_mix_with_body_hd.json).

## Insights and next steps

- Hair importance has a substantial measured benefit, unlike the earlier1055
  tests that changed allocation *inside* a fixed60% hair quota. Develop this
  direction rather than extending small threshold/visibility sweeps.
- Full candidate preserves the measured ancestry: corrected body80 through15188,
  hair mixture.5 from15189, FR1 through75940, FR.3 through106316. The HD pilot
  restores that exact same-frame parent and continues all optimizer/RNG state;
  native frames would each reproduce the first stage independently. Default
  sampling and pre-switch indices remain bit-identical to the control.
- The relaxed frequency-update gate yields only1.62% hair-LPIPS gain and increases
  trusted-background opacity4.8%; close it, retain the opaque-update guard.
- A common4K GT-transfer test is complete with **real native6144×3072 RGB** from
  the same eight training cameras and identical fallback. Its3840×1920 target is
  excluded from all fitting. This separates geometry effects from NeRF color
  improvements. It still has no independent3D depth or physical-time ground truth.
- Full candidate runs at `recipe_pilot_20261028/hair_full_v7` on clever-shadow;
  production silver-sable remainsv6 while clothing/geometry validation is pending.

### Matched 4K geometry gate, 2026-10-28 22:23 UTC

The same eight native sources, q10 depth, opacity floor .1, renderer, and frozen
background were used for both arms; the target raster is 3840×1920. Geometry was
trained at 1920×960. Held-out RGB is used for scoring only.

| Metric | Corrected control | Hair mixture |
|---|---:|---:|
| Hair PSNR / SSIM / LPIPS | 19.3233 / .28024 / .202603 | 19.7047 / .27999 / .201816 |
| Boundary LPIPS | .045682 | .045985 |
| Face LPIPS | .119190 | .117686 |
| Clothing/hands LPIPS | .200783 | .200779 |
| Hair source-fallback fraction | .16399% | .35370% |
| Halo fraction | .01978% | .03461% |
| Hair reprojection-error proxy | 3.9202 | 3.8082 |

Hair LPIPS improves only .39%, while boundary LPIPS and missing-source coverage
worsen. Visual inspection of both head comparisons shows inherited fragments
of wall above the crown and beside the hair, with limited change between arms.
This **does not pass the geometry gate**. The 14.66% direct NeRF appearance gain
cannot be described as a depth or flicker fix. Keep production v6 while the full
schedule runs; do not promote the short-run hair recipe based on texture alone.

[Metrics](assets/dec5_silver_sable/sequence901_geometry/metrics.json),
[control GT/prediction](assets/dec5_silver_sable/sequence901_geometry/control.jpg),
[hair mixture GT/prediction](assets/dec5_silver_sable/sequence901_geometry/hair_mix.jpg).

### Depth statistic diagnostic

Using the same corrected-control cached weights, expected depth instead of q10
improves hair PSNR/SSIM/LPIPS to **21.2510/.28103/.188416** (LPIPS7.00% better),
boundary LPIPS **.039981** (12.48% better), face LPIPS **.119460** (.23% worse).
Clothing/hands LPIPS worsens5.32% to **.211458**; missing-source fraction grows
to .41120%, halo to .05109%. Both inspected renderings retain wall fragments
outside the hair. This is a useful geometry-statistic diagnostic, not a global
renderer promotion or a temporal-stability result. Median was not cached in
these exports and was explicitly skipped; no new field queries were run.

[Statistic metrics](assets/dec5_silver_sable/sequence901_geometry/depth_statistics.json),
[expected-depth GT/prediction](assets/dec5_silver_sable/sequence901_geometry/expected.jpg).

A single confidence-gated expert combination preserves q10 outside mutually
train-voted hair (both hair>=.8, skin<.2, expected support>=max(3,parent support),
both have a source). It changes412683 pixels and preserves all other arrays/RGB
bit-for-bit. Hair PSNR/SSIM/LPIPS becomes **19.4341/.27030/.200371**, only1.10%
LPIPS better, boundary **.044340** (2.94% better), face **.119170**, clothing/hands
**.200786**. Missing-source and halo fractions are exactly unchanged by design.
Visual inspection retains the crown/wall fragments. Close this short-run
combination without a confidence-threshold sweep: it does not meet the quality
gate. The larger global expected-depth gain depends on disputed regions that
the reliable-hair gate excludes; texture improvement alone remains insufficient.
[Gated metrics](assets/dec5_silver_sable/sequence901_geometry/hair_gated_metrics.json),
[gated GT/prediction](assets/dec5_silver_sable/sequence901_geometry/hair_gated.jpg).

### Full-schedule intermediate gate: do not promote

At the same45564 step, hair-mixture LPIPS improves16.15% relative to corrected
control, but face worsens10.34%, boundary7.66%, clothing/hands5.18%. At30376,
hair improves13.09% while clothing worsens3.22%. The face/body tradeoff grows
with training, rather than vanishing. Visual inspection of the GT/control/hair
head panel confirms sharper hair texture with a still-contaminated silhouette;
face detail is softer. **This sampling variant fails promotion.** Keep nativev6;
do not tune more mixture thresholds on the basis of the short20188 result.
The already-running full pilot remains a bounded comparison, with no extension
or native production promotion authorized by these results.

[Matched30376/45564 region metrics](assets/dec5_silver_sable/hair_full45_matched.json),
[45564 GT / corrected control / hair mixture](assets/dec5_silver_sable/hair_full45_comparison.jpg).

### Corrected first-stage75940 geometry gate

At75940, the same4K q10 transfer gives hair PSNR/SSIM/LPIPS
**19.1174/.27874/.211276**, versus20188 **19.3233/.28024/.202603**. Hair LPIPS
is4.28% worse despite improved direct NeRF appearance during training. Missing
source fraction improves .16399%→.09374%, while silhouette recall falls
.95202→.94575. Boundary LPIPS .046136 and face .119796 do not improve.
Inspection still shows wall fragments around the crown. Thus a longer first
stage alone has not demonstrated better hair geometry. The final FR.3 stage is
still running; this is not a claim about its completed result. A separate
[geometric background track](dec5_000901_geometric_background.md) now tests a
different explanation for false foreground density.

[75940 metrics](assets/dec5_silver_sable/sequence901_geometry/control75k.json),
[75940 GT/prediction](assets/dec5_silver_sable/sequence901_geometry/control75k.jpg).


### Completed full schedule and new contour-budget series

Both HD arms now complete106316. Strong hair mixture gives hair LPIPS
.209757→.191667 (8.62% better), but face .102883→.115951 (12.70% worse),
boundary .095665→.102703 (7.36% worse), clothing .150345→.151355 (.67% worse).
Boundary opacity<.1 grows2.72%→7.15%; improved internal texture does not mean
preserved contour. This arm remains rejected for production.
[Control all gates](assets/dec5_silver_sable/neural_background/control_full106k.json),
[hair mixture all gates](assets/dec5_silver_sable/neural_background/hair_full106k.json).

A new native6K series continues the identical seed42 cold15188 checkpoint for
5000 updates: unchanged control,10% contour-uniform,10% contour-frequency.
The new budget comes only from the20% whole-image quota. Original body-uniform,
protective-band and body-FAS pixel indices and camera identities stay identical.
This differs from taking draws away from the face within the body core.
The contour is a ±5HD-semantic-pixel band, mapped into native coordinates;
semantics localize sampling and never mask output RGB. Frequency mode weights
boundary pixels1→3 by their source-map level. A deterministic test verifies body
indices, RGB/index alignment and actual sampled class accounting. These are
isolated short native trials, not completed50-frame training or a temporal
consistency training objective. Current production v7_seed42 does not enable them.

[Visually inspected completed106316 GT/control/hair comparison](assets/dec5_silver_sable/hair_full106k_comparison.jpg).


### Native contour-budget trial: completed20188 review

All three continuations complete5000 updates from the identical native seed42
15188 full-state parent. The metrics-only HD reference is resampled to native6K;
contour width16 native pixels corresponds to5HD pixels. These are update-matched,
not identical adaptive-field-query-count comparisons.

| Arm | Hair PSNR / SSIM / LPIPS | Boundary LPIPS | Face LPIPS | Clothing LPIPS | Boundary opacity<.1 |
|---|---|---:|---:|---:|---:|
| Control | 22.4654 / .77254 / .325391 | .115455 | .276385 | .279367 | 2.507% |
| Contour uniform | 22.4782 / .77213 / .323359 | .113517 | .267197 | .277847 | 1.598% |
| Contour frequency | 22.3470 / .77223 / .325124 | .113442 | .271627 | .279650 | 1.703% |

Hair LPIPS improves only.62%/.08%; contour improves1.68%/1.74%. Face improves
3.32%/1.72%, so the old face-budget regression is avoided. Contour low-opacity
fraction falls36.27%/32.06%, and internal-hair low-opacity pixels41.70%/48.43%.
However, uniform mode raises mean trusted-background foreground opacity
.031441→.032763, while frequency mode lowers it to.030725. A larger opacity
is not necessarily recovered hair. Visual inspection shows little overall change
and persistent wall fragments around the crown. Neither variant is promoted
from these proxies. Matched4K GT-transfer geometry gates are running with the
same8 native source RGB views, q10, floor.1, target ROI and frozen fallback.

Four deterministic sampler tests pass for uniform/frequency modes at two raster
scales with permuted cache camera order, preserving body/FAS indices and RGB/index
alignment. The production v7_seed42 snapshot does not contain this experimental
sampling feature.
[Full results](assets/dec5_silver_sable/boundary_budget_native/complete.json),
[visually inspected GT/control/uniform/frequency](assets/dec5_silver_sable/boundary_budget_native/comparison.jpg).


### Completed native-field4K GT-transfer gate

All geometry arms use the identical8 native RGB sources, original frozen fallback,
q10, opacity floor.1 and fixed held-out4K reference. Field training is native6K.
The new `--field-data` argument binds source/target exports to the actual native
training metadata SHA; the original HD default remains unchanged.

| Geometry | Hair LPIPS | Boundary LPIPS | Face LPIPS | Source fallback holes | Halo proxy |
|---|---:|---:|---:|---:|---:|
| Native control20188 | .195075 | .043054 | .120801 | .12714% | .05933% |
| Contour uniform | .194612 | .042298 | .119408 | .09171% | .01813% |
| Contour frequency | .194871 | .042406 | .117879 | .07916% | .01978% |
| Warm3D background | .191691 | .040012 | .119319 | .01782% | .04120% |

Contour sampling reduces holes27.87%/37.74% and improves face LPIPS1.15%/2.42%,
but hair LPIPS improves only.24%/.10%, with multiview reprojection4.37%/4.91%
worse. Its coverage benefit is real in this protocol, but hair detail and geometry
are not broadly improved. Do not expand quota tuning before stronger evidence.
Warm background gives the better combined tradeoff: holes **85.99% fewer**,
hair LPIPS1.74% better, contour7.06%, face1.23%, and reprojection1.74% better.
Clothing LPIPS worsens.80%. These are image/visibility proxies, not measured
metric depth or physical-time flicker. The primary next gate is the warm method's
native full recipe and long static camera path.
[Control geometry](assets/dec5_silver_sable/boundary_budget_native/native_boundary_control20k_geometry.json),
[uniform](assets/dec5_silver_sable/boundary_budget_native/native_boundary_uniform20k_geometry.json),
[frequency](assets/dec5_silver_sable/boundary_budget_native/native_boundary_frequency20k_geometry.json),
[warm background](assets/dec5_silver_sable/boundary_budget_native/native_warm_background20k_geometry.json).
