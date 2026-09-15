# The three 001083 residual counts do not measure the under-cheek hole

## What was tested

Read-only ray attribution after the [fresh 001083 transfer](dec5_mhr_transfer_001083.md). The three enclosed moving-view misses were traced through the exact original mesh, full fresh MHR prior, neutral anatomical band, topology-safe band, local subdivision, raw proposals, semantic proposals, strict/interpolated initial admission and both native-guarded final branches. No fit, sweep, parameter change, production edit or new video render was performed. Earlier seals remain immutable.

**001083 in this current moving view is not a positive example of the original under-cheek hole. Its three residual counts are unsuitable for evaluating repair of that target.** They are isolated hair/shoulder/collar boundary pixels. This corrects the interpretation of the earlier negative transfer: its visual review still found no improvement, but the 3→3 count is not evidence that an existing under-cheek puncture resisted repair.

## Results

| Portrait pixel (x,y) | Actual landscape pixel (u,v) | Visible location | Full prior | Anatomical / safe / raw / semantic | Strict / interpolated final | Earliest absence |
|---|---|---|---|---|---|---|
| (145,825) | (1094,145) | Left hair rim | Miss | All miss | Both miss | Full prior |
| (176,1343) | (576,176) | Left shoulder/neck fringe | Miss | All miss | Both miss | Full prior |
| (1032,1428) | (491,1032) | Right collar fringe | Miss | All miss | Both miss | Full prior |

All 33 ray/stage records are misses, including local-before-centroid and both initial branches. Thus none of these rays loses coverage at masks, confidence, certificates, centroid exclusion or final native veto: the full fitted prior never intersects it. Every intersection triangle ID and depth is explicitly null in JSON; NPZ retains Open3D's no-hit ID `4294967295` and infinite depth. No nearby triangle is falsely called an intersected facet.

The exact calibrated production origin is `(0.1983491033,-0.6618776321,-0.1343351305)`; directions in table order are:

```
(-0.2772921622, 0.9364266396, 0.2162336856)
(-0.3067430258, 0.9277414680, 0.2157981098)
(-0.3165219128, 0.9349718094, 0.1664891839)
```

These are the actual non-unit float32 rays generated from the frozen production intrinsics/extrinsics, not rays inferred from an approximate pixel-to-camera map. Original full-frame depth replays exactly. An independent double-precision, two-sided ray/triangle calculation also finds zero intersections with the 36,874-face full prior for each ray. Stage triangle-ID namespaces, calibration, exact rays and input/output hashes are retained in [result.json](/mnt/data/dec5_mhr_residual_rays_001083/result.json) and [rays.npz](/mnt/data/dec5_mhr_residual_rays_001083/rays.npz).

### Actual native visual localization

The [annotated overview](/mnt/data/dec5_mhr_residual_rays_001083/localized_overview.png), [hair residual](/mnt/data/dec5_mhr_residual_rays_001083/residual_1.png), [left shoulder residual](/mnt/data/dec5_mhr_residual_rays_001083/residual_2.png), [right collar residual](/mnt/data/dec5_mhr_residual_rays_001083/residual_3.png) and [underchin context](/mnt/data/dec5_mhr_residual_rays_001083/underchin_context.png) were actually inspected. Panels preserve native pixels: three original rendered RGB branches, stage-specific clay raycasts, and original depth. Magenta marks each counted miss. Clay for prior/proposal stages shows only that stage, not a replacement of the actor.

The cyan posthoc box `[530,1160,845,1360]` localizes the current underchin context separately from the count. It shows continuous cheek/chin and shadowed neck skin, not a conspicuous interior puncture. The continuous dark band is a shadow; this diagnostic supplies no evidence to call it an artifact or fill it. Jagged skin-colored silhouette fringe remains along the neck/shoulder, and the hair boundary is visibly rough elsewhere. Boundary appearance alone does not establish that real skin should occupy every black pixel. The parent independently inspected all five diagnostic PNGs and confirmed this distinction.

Crucially, this completion is **append-only**: exact preservation of original triangles means it cannot remove an original outward flange. Adding a surface to a genuine puncture and correcting an over-extended contour are different operations. The current fringe therefore cannot be used as evidence that another anatomical fit is needed. Accurate native foreground/source ownership may be relevant to a separate contour investigation, but neither that hypothesis nor a mask change was tested here.

### Existing appropriate target, without a new fit

The already tested **001193 old-moving camera** has the visibly separate black puncture under the cheek: [native original / strict / interpolated comparison](/mnt/data/dec5_mhr_production_patch_001193/admission/rgb_review/old_moving_native.png), re-inspected during this diagnosis. Its actual-production fixed ROI was 45 misses originally, 1 strict and 0 interpolated, with train RGB on the fills and the surrounding natural shadow retained. The [001193 report](dec5_mhr_production_patch_control.md) and [001195 transfer](dec5_mhr_production_transfer_001195.md) remain the relevant local-hole evidence; this diagnosis does not rerun their measurements.

For an unresolved, already localized failure, the same 001193 **F/E train view** has a separate 30-pixel underchin residual at portrait box `[682,1148,696,1150]`. Existing attribution establishes 30 safe-prior hits, 29 raw proposals, and zero semantic admissions; projected samples cross real train outlines. Its [existing measured margin audit](/mnt/data/dec5_mhr_production_patch_001193/residual_hole/margin_audit.json) is a concrete starting point, unlike the unrelated 001083 counts. A previous zero-margin fit and static unsafe-vertex freeze already failed topology gates; repeating either blindly is not justified.

## Insights

The next useful algorithmic question is constrained silhouette fitting at the **known 001193 F/E residual**, retaining measured anchors and topology constraints—not loosening admission or expanding the 001083 face prior to cover hair/clothing rim pixels. This is a recommendation grounded in existing attribution, not an authorized new fitting run. The current diagnosis stops here.

Three tests cover portrait indexing, independent two-sided intersections and earliest-stage attribution. [Final seal](/mnt/data/dec5_mhr_residual_rays_001083/final_seal.json) binds the preserved transfer, diagnostic outputs, actual visual review, scripts and report. Independent PSNR/SSIM/LPIPS are N/A: this is ray attribution, not held-out image-fidelity evaluation. The reporting workflow explicitly separates the convenient count, observed visual defect and causal interpretation.

Reproduce with CPU two-thread settings and `/home/brans/repos/nerfstudio/.venv/bin/python scripts/diagnose_local_mhr_residual_rays.py --completion /mnt/data/dec5_mhr_completion_001083 --output NEW_ROOT`. Then inspect outputs and create its visual-review receipt before running `audit_local_mhr_residual_rays.py --output NEW_ROOT --report experiments/dec5_mhr_residual_rays_001083.md --tests tests/test_local_mhr_residual_rays.py`. Run tests with `python -m pytest -o addopts='' -q tests/test_local_mhr_residual_rays.py`.
