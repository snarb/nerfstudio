# Marshawn frame1660: native Splatfacto floor diagnosis

## What was tested

The corrected gain4 SDR control uses63 train cameras,3 held-out views (DC/HC/KA),9022 current-frame train-only triangulated XYZ, median colors from44938 actual track-camera observations, native Splatfacto, identity prepared-world parser normalization, and disabled camera optimization. The cap is22000 updates. Canonical images, calibration, and frozen native model source remain unchanged.

A matched cold-seed42 control changes only native `use_scale_regularization=True` (ratio10, native coefficient0.1 every10 steps). No geometric priors, new points, image compositing, or source-code changes are introduced.

## Results

| Control/checkpoint | All3 PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Corrected SDR, selected10000 |27.890005|0.832993|0.273440|
| Scale regularization,8000 |27.845339|0.832951|0.286241|
| Scale regularization, selected20000 |28.554022|0.840547|0.268282|

The first control completed22000 updates. Strict selection uses highest all-view PSNR, then LPIPS among checkpoints within0.07dB. Reloaded LPIPS may differ by about0.000052; PSNR/SSIM match. Its exact3 full-HD predictions and checkpoint receipt are in [download](/fsx/tmp/marshawn_001660_hd/splatfacto_gain4/download). The second control also completed exactly22000 updates without OOM; its strict-selected20000 checkpoint SHA256 is `96705ad870444529005e6e71b3b54e55f1ee93e03a5427c4a0ff5869fb00e32b`. Native full-HD outputs and hash receipts are in its [download folder](/fsx/tmp/marshawn_001660_hd/splatfacto_gain4_scale_reg/download). No training jobs remain active under this subtask.

[Baseline problematic crops](/fsx/tmp/marshawn_001660_hd/splatfacto_gain4/problematic_crops/review.json) compare native8000 and selected10000 pixels against GT. Central people improve, but colored mat/floor clouds and translucent KA foreground remain. [Regularized8000 contact](/fsx/tmp/marshawn_001660_hd/splatfacto_gain4_scale_reg/gate8000_renders/all_GT_prediction.jpg) still shows these defects. These results are not accepted as final quality.

## Geometry and contribution evidence

Automatic lower-Z plane RANSAC is ambiguous around the raised mat and is not accepted as physical-floor ground truth. Explicit AA/GA/OA TRAIN floor ROIs yield25 actual observed track candidates,15 consensus points, normal `[0.0297973,-0.13036318,0.99101845]`, offset0.1682054, residual threshold0.003 prepared-world units. [Manual review contact](/fsx/tmp/marshawn_001660_hd/splatfacto_gain4/floor_geometry_diagnosis/manual_floor_review_contact.jpg) and [reference plane](/fsx/tmp/marshawn_001660_hd/splatfacto_gain4/floor_geometry_diagnosis/manual_floor_fit.npz) preserve the sparse evidence. Some candidates lie on props or matching errors; this is an approximate floor reference, not dense verified geometry.

`scripts/trace_marshawn_gs_floor_cpu.py` independently evaluates installed gsplat1.4 pure-PyTorch covariance, projection, spherical harmonics, and depth-sorted per-pixel alpha contributions from frozen checkpoints. Nine diagnostic pixel RGB reconstructions match native saved PNG within0.00189, below0.5/255 quantization. The recorded contributor IDs and weights therefore explain the actual native render, rather than only projected Gaussian centers.

- Baseline selected10000: DC mat three samples have72–88% opacity weight from Gaussian means more than0.1 units below the approximate floor. HC floor samples have92–99.95%; aspect ratios above10000 account for18–53% weight at the HC samples.
- Regularized8000: aspect ratios above10000 account for zero weight at all nine samples, but DC mat75–89% and HC floor64–95% still come from means more than0.1 below the reference floor.
- Both checkpoints also use substantial view-dependent spherical-harmonic colors. A below-floor mean can represent a virtual reflection and does not by itself justify deleting that Gaussian. The raised mat has real thickness. Projected centers without contribution tracing are insufficient evidence of visible artifacts.

Exact traces: [baseline](/fsx/tmp/marshawn_001660_hd/splatfacto_gain4/floor_contribution_diagnosis/audit.json), [regularized8000](/fsx/tmp/marshawn_001660_hd/splatfacto_gain4_scale_reg/floor_contribution_8000/audit.json).

## Insights

Native scale regularization removes extreme aspect degeneracy at the measured pixels. It does not by itself constrain Gaussian means to physical surfaces or eliminate the colored floor clouds. Pixel contributions show a combination of virtual subfloor geometry and view-dependent colors; the defect cannot be attributed solely to surface specular color. Blindly removing below-floor Gaussians would also discard learned reflections and needs an independently validated policy. No deletion, image smoothing, or GT compositing was applied.

## Final matched visual gate

At selected20000, all438613 regularized Gaussians have aspect ratio below25.21 (median8.39, p999.97), versus12403 baseline Gaussians above10000. Native scale regularization therefore removes the measured scale degeneracy globally.

Nevertheless, exact [DC mat](/fsx/tmp/marshawn_001660_hd/splatfacto_gain4_scale_reg/matched_problematic_crops/DC_mat_comparison.jpg), [HC floor](/fsx/tmp/marshawn_001660_hd/splatfacto_gain4_scale_reg/matched_problematic_crops/HC_center_floor_comparison.jpg), and [KA feet/rod](/fsx/tmp/marshawn_001660_hd/splatfacto_gain4_scale_reg/matched_problematic_crops/KA_feet_comparison.jpg) GT|baseline|regularized comparisons fail the spatial-quality goal. DC mat remains cloudy, HC floor remains hazy, and KA has a stronger broad greenish translucent region obscuring the rod. The improved global metrics do not override these defects.

The [selected20000 contribution audit](/fsx/tmp/marshawn_001660_hd/splatfacto_gain4_scale_reg/floor_contribution_best/audit.json) again matches all nine native pixel RGB values within0.00189. DC samples still receive54–76% opacity weight and HC76–96% from means more than0.1 below the approximate floor. Aspect-over10000 contribution remains zero. Native scale regularization is useful, but insufficient as the final correction. No new run, deletion policy, or additional prior is launched from this diagnosis.
