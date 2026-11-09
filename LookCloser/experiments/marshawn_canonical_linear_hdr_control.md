# Canonical physical linear EXR control

What was tested: prepare one cold control on the signed canonical unbaked
AP1-D60 EXRs. The existing frozen LookCloser positive linear-softplus field
integrates radiance before encoding `PQ(clip(52.13360318135631*RGB,.005,4000))`.
The ST2084 normalization remains10000nits. Physical black is linear0;0.005nits
is an encoding floor, not an additive radiance offset.

The chosen existing objective is PQ MSE with independent uniform sampling and
dynamic target1,048,576 samples. Scene_loose bounds, grid128/core0, step0.003,
hash21/max8192, appearance0 and seed42 are explicit. Frequency features and
DSSIM are disabled. The same observed-train SfM LOS/surface terms and bounded
linear packed distortion use weight0.01; there is no floor-plane scene prior.

Results: actual DEV3 Torch2.4/L40S zero-update checks pass at AMP65536. The first
64 rays contain36,867 samples; the next1,820 rays contain1,081,791 samples.
Actual batch RGB exactly matches source float32 EXR, including values above34.
The renderer preserves synthetic linear RGB[2,3,4] in evaluation. RGB/color and
density gradients are nonzero; the complete combined gradients are finite.
Cold predictions have no signed-floor clipping. No optimizer steps occurred.
Peak allocated GPU memory is2.13GiB; reserved memory is2.85GiB.

Train-only statistics use all122,411,520 pixels from63 images and vendor AP1
luminance coefficients: median0.02011050656, q99930.89463468. The inherited
geometry priors are relatively strong early in training: on the adapted packet,
LOS/surface/distortion gradient norms are1.705/0.446/0.0183 times RGB's total
gradient norm. These measurements were reported rather than automatically
changing weights.

The physical field/model source bytes remain unchanged. An external density
proxy exposes the existing `exp(h0+1)` raw path only for the stable SfM surface
likelihood. Canonical/original camera geometry and63/3 split agree after image
extension normalization; the six geometry NPZ files are byte-identical copies.

The actual DEV3 dry-run passes with the quiet runner's path bound to the real
LookCloser config directory. The cold budget is steps0…21999, with a mandatory
same-worker pause before2001 after checkpoint2000:2,001 attempts. Default
full-three-view PQ-MSE plateau stopping remains active. If terminal21999 exists,
fresh all-view metrics are required before PSNR/LPIPS selection.

Insights: this is the requested physical linear compositing control, distinct
from earlier encoded-PQ compositing and from the old noncanonical812-nit scale.
The paper motivates radiance rendering and regularization; it does not specify
this dataset's PQ transfer, which comes from the signed vendor contract.
Those checks preceded the separately authorized cold training described below.

Common SDR comparison applies the fixed AP1-D60→Rec709-D65 matrix, gain4 and
the old scalar metadata identically to GT/prediction, followed by Reinhard+sRGB.
Unscalar canonical previews are retained separately. Canonical common GT is
slightly different from historical PNG conversion because of its corrected
negative-channel clipping order; fair comparisons must use the same canonical
GT for both saved models, without fitting eval RGB or retouching renders.

[Actual zero-update proof](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/actual_zeroopt_DEV3/probe.json) ·
[Config](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/config.yml) ·
[Launch plan](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/launch_plan.json) ·
[Dry-run](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/dry_run_DEV3_retry1.log).

The authorized DEV3 worker completed the2k gate, resumed once after actual root
visual review, and stopped normally at the4k all-view PQ plateau. No restart or
new model was launched. The last logged step4030 plus a conservative10-step
allowance gives a cold-attempt upper bound4,041. Checkpoint4000 records3,998
Adam updates from4,001 attempts, with three native AMP skips and finite weights.

| Actual primary canonical-PQ evaluation | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
|Selected2k, fresh final evaluation|34.806004|0.934753|0.218554|
|4k all-view training evaluation|34.3828|0.936435|0.201894|

The0.423dB gap exceeds the0.07dB tie window, so the declared PQ rule selects2k.
These PQ scores must not be compared directly with sRGB metrics.

The following separate comparison scores all saved native1920×1012 PNG
predictions against **the exact same canonical common-SDR GT**. It applies no
mask, resizing, fitted gain or compositing; every branch uses the same AlexNet
LPIPS and SSIM implementation.

| Same canonical common-SDR GT | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
|Physical HDR2k|26.444376|0.752309|0.382875|
|Physical HDR4k|26.055322|0.762028|0.350814|
|Saved safe PNG13k|28.967771|0.804409|0.302255|

Root's actual HD review found a brown DC veil, HC background holes and remaining
floor colour errors. This control is not accepted as clean. Saved-buffer
diagnostics at the old KA dark pixel(242,747) show opacity0.99797 at2k and
0.999998 at4k, while conditional model AP1 luminance is about0.799/0.835 of GT;
its remaining error is therefore not chiefly an opacity hole. No GT alpha is
available or fabricated.

A separate read-only causal audit locates most mass on three problematic DC
rays before0.05 world units. Root is testing a phase-aligned near0.052 rendering
policy on the unchanged4k field, supported by independent camera/scene clearance
checks. That derived evaluation is separate from the default-near0.01 results
above; it has not been folded into the checkpoint selection or original metrics.

[Formal terminal/budget receipt](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/finished_receipt.json) ·
[Fair common-domain metrics and fixed crops](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/common_canonical_GT_comparison/report.json) ·
[Whole-view comparison](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/common_canonical_GT_comparison/all3_same_GT_HDR2k_HDR4k_safe13k.jpg) ·
[Nine saved-pixel opacity/radiance diagnostics](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_PQMSE_STAGED/floor_9pixel_opacity_radiance_gauge.json).

The source-photo/causal review associated the DC near density with a black
camera/rig assembly coincident with that camera centre. The root approved a
same-4k-state continuation using near0.052 in both training and evaluation,
without changing losses, radiance units, data, AABB, grid policy or field code.
Actual zero-update setup restored model/grid, Adam3998, scheduler, scaler16384
and RNG exactly. The native dynamic-ray counter is not checkpointed; its64-ray
constructor reset is explicitly recorded before returning to the1M target.

The continuation paused before6001 after2,000 new attempts and2,000 Adam updates,
with no new AMP skips. The charged family bound at this gate is6,041. Its
maximum end is21960 exclusive:4,041 prior charge+17,959 new attempts=22,000.

| Same near0.052 canonical-PQ domain | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
|Fresh parent4k, unchanged weights|36.108551|0.944219|0.162612|
|Resume6k|36.7007|0.948552|0.149956|

| Exact same canonical GT, saved uint8 common SDR | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
|Parent4k near0.052|27.706686|0.774309|0.320471|
|Resume6k near0.052|28.237720|0.783955|0.303250|
|Saved safe PNG13k|28.967771|0.804409|0.302255|

All three6k HD views and fixed floor/background crops were manually inspected.
The DC brown veil is absent, but black background tiles in HC/DC and floor
colour mottling remain. The separate6k audit finds36/9022 known point-cell
misses including4 body cells,0/128 body free-prefix mass above10%,126/128 body
bands containing at least50% mass, and zero mass before0.1 on the three queried
DC rays. These limited checks do not certify the complete grid or clean quality.

The current physical training targets are unscalar canonical AP1, whereas the
saved PNG13k used the old camera scalar correction. Identical common-display
GT makes the reported output comparison fair, but does not isolate training
compositing from that data correction. A scalar-matched canonical EXR derivative
would need to bake only the existing metadata EV in linear RGB, declare itself
a calibrated derivative, and omit the EV from its common display to avoid
double application. No such new neural run is claimed here.

[6k HD views](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_NEAR052_RESUME4k_STAGED/gate6000_HD/all3_common_GT_HDR_safe13k.jpg) ·
[Matched4k/6k/13k comparison](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_NEAR052_RESUME4k_STAGED/gate6000_common_comparison/report.json) ·
[Actual full-state proof](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_NEAR052_RESUME4k_STAGED/actual_fullstate_zeroopt_DEV3/proof.json) ·
[6k paused-gate receipt](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_NEAR052_RESUME4k_STAGED/gate6000.json).

After the root reviewed6k and resumed the same worker, training failed before8k
with `FloatingPointError: Non-finite HDR RGB prediction`. The last CSV step7260
still has finite training metrics; the last recorded AMP trace at7000 has no
phase skips. The failed batch, raw logits and current7k model/RNG were not
dumped, so the cause cannot be attributed to RGB activation, density or AMP.
There was no restart. Final evaluation selects the preserved finite6k checkpoint:
PQ36.700699/0.948552/0.149956. Both its NN and Adam tensors are finite.

The conservative total family charge is7,311 attempts, including the failed
post6k tail. A future continuation from6001 would therefore have at most14,689
new attempts, ending20690 exclusive. The original requested21959 terminal model
does not exist and requires no posthoc evaluation.

Actual DEV3 operator tests distinguish a separate evaluation fragility: with
autocast enabled the existing positive head returns finite FP32 outputs for
the tested finite logits through3000; with autocast disabled its FP16 output
overflows at2200+. Explicit FP32 activation stays finite. This does **not**
prove the training failure's cause. FP32 density exponential can also overflow,
and zero-weight infinite colour produces NaN in the actual renderer. Any future
production retry needs exception-time batch/RNG/model/optimizer and activation/
weight dumps before exit; replacing NaNs or silently restarting is not a repair.

[Numerical termination and final budget](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_NEAR052_RESUME4k_STAGED/finished_receipt.json) ·
[Compact forensics](/fsx/tmp/marshawn_001660_hd/canonical_linear_HDR_NEAR052_RESUME4k_STAGED/forensics/summary.json).
