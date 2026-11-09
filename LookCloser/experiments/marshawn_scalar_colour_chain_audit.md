# Scalar colour-chain audit and independent calibration feasibility

What was tested: replay the existing scalar colour dataset numerically, then
inspect independent frames1700/local448 and1740/local488 for static calibration
regions. No new corrections, image replacement or neural training were applied.

Results: all66 actual scalar EXRs are bitwise equal to the gain4 linear Rec709
source multiplied once by `2**camera_EV`. All66 PNGs exactly match the common
Reinhard+sRGB encoder applied to those EXRs. Source/output hashes and camera
geometry/splits pass. Train and eval use the same conversion domain and order;
there is no double EV application. Inverse Reinhard appears only in a separate
legacy comparison helper, outside dataset creation and native evaluation.

| Eval camera | Fixed EV | Linear multiplier | Source |
|---|---:|---:|---|
|DC|−0.183363|0.880648|Pose-only six-camera interpolation|
|HC|−0.218866|0.859241|Pose-only six-camera interpolation|
|KA|+0.187519|1.138803|Pose-only six-camera interpolation|

The interpolation uses inverse-distance power3 and has train-camera
leave-one-out error0.11945EV RMS,0.19247EV p90. These values are uncertainty
diagnostics, not measured calibration of the held-out cameras. The train fit
also mixes material/view-dependent reflectance with camera response; scalar
gain does not correct WB or vignetting.

All132 raw files for the two independent times exist. Only six sparse,
unrectified previews from KA/KB/LB were decoded for visual selection; they are
not photometric measurements. Their static printed cards/boards are visible,
while actors and floor reflections change. The separate acquisition audit finds
the same declared ISO800,5600K,tint0,shutter1/240,f8 and unity RGB gains across
the nine checked R3D segment headers. Those metadata do not explain the KA gap.

Insights: a legitimate next precheck is a connected graph of the same static
matte cards in independent1700 images, fitting relative camera EV in decoded
linear RGB and validating frozen parameters on1740 plus disjoint cards/materials.
Measure original-resolution approved ROIs; exclude floor, actors, metallic
hardware, clipping and specular highlights. Begin with KA and nearby train
cameras. Chromatic or radial corrections require repeatable independent evidence.
Even repeatability across time cannot by itself separate camera response from
static directional reflectance. No parameters should be fitted from target1660
eval RGB or applied to saved renders.

[Exact66-camera audit](/fsx/tmp/marshawn_001660_hd/report/scalar_chain_audit/audit.json) ·
[Independent-frame previews](/fsx/tmp/marshawn_001660_hd/report/scalar_chain_audit/static_precheck/KA_KB_LB_independent_frames_contact.jpg) ·
[Feasibility and source bindings](/fsx/tmp/marshawn_001660_hd/report/scalar_chain_audit/feasibility.json).

The authorized independent subset precheck then measured native raw ROIs for
KC(reference),KA,KB,LB,JA,LA. Unknown printed tags were matched by actual image
features and local homographies, with at least six inliers and pattern
correlation above0.90; no camera pose projection selected the matches. All43
accepted pattern pairs were inspected. Even physical cards in1700 fit relative
EV; odd cards, nearby cardstock and1740 validate frozen mappings/sample IDs.
Gradient, clipping and visibility checks reject unreliable patches.

The result is **inconclusive for camera-wide calibration; apply no new gains**.
KA has only one partly visible card near raw(7600,60), with no independent
card or radius validation. Its local apparent correction relative to KC is
+0.959EV in1700 and+0.949EV in1740; this repeatable local difference is not a
global camera response measurement. All observed cameras connect through the
reference, but the even-card fit graph excludes KA because its only card is
held out. Calibration support therefore fails the planned validation split.

| Camera | Fitted EV relative to KC | Fit cards |1740 paper median absolute residual, EV|
|---|---:|---:|---:|
|KA|Not accepted|0|Not independently validated|
|KB|+0.3947|9|0.0626|
|LB|+0.6426|7|0.1294|
|JA|+0.5386|2|0.0989|
|LA|+0.5616|3|0.2172|

JA's separate1700 held-out cards retain about0.63EV median error. Card-to-card
variation and limited radial coverage prevent treating these diagnostics as
universal camera gains. The earlier full-card subset shows relative temporal
stability around0.04–0.05EV p90 on identical points, despite shared illumination
changes on individual cards. Vignetting, curved paper and directional
reflectance remain confounded. The target floor has opposite-sign residuals
from some background regions, so one global exposure cannot explain it all.

[Final measurements and frozen correspondence bindings](/fsx/tmp/marshawn_001660_hd/report/independent_card_calibration_PRECHECK/actual_subset_v3_final/audit.json) ·
[All matched patterns](/fsx/tmp/marshawn_001660_hd/report/independent_card_calibration_PRECHECK/actual_subset_v3_final/matched_card_patterns_contact.jpg) ·
[Conclusion](/fsx/tmp/marshawn_001660_hd/report/independent_card_calibration_PRECHECK/actual_subset_v3_final/conclusion.json).
No current image, metric, checkpoint, neural model or download ZIP was changed.
