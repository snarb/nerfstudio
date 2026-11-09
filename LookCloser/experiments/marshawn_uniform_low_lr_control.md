## What was tested

The immutable distortion8k checkpoint resumes with its full Adam moments/steps,
scaler and RNG. A task-local postload policy changes the current LR from
.0083178292 to exactly .0005 by scaling optimizer/scheduler LR metadata together.
It preserves the original LambdaLR function and actual successful-update epoch
7999; the next successful scheduler update is .0004999885.

This control uses the original independent PixelSampler, patch_size1 and dynamic
initial estimate16384 (initial16 rays). RGB MSE, bounded-linear distortion .01,
LOS .01, surface .01, native NN, scalar PNG data and scene.loose are unchanged.
There is no DSSIM or patch sampler. Evaluation/save cadence is1000 and the first
in-place review gate is9000, before the next update9001. The absolute end16000
keeps the declared accumulated attempt bound21609, including earlier trials.

## Results

The zero-update DEV3 proof verified the full restored state, intended LR-only
metadata change and 64 native CPU scheduler transitions without LR rebound.
The first64 actual training attempts produced64 Adam updates with no AMP skips.
The root reviewed all three HD views and identical HC/KA floor crops at9000,
then continued the same worker. This is continuation approval, not final quality
acceptance. The run subsequently stopped normally on the default plateau at13k;
fresh final evaluation selected13k by all-view PSNR/LPIPS. Both controller and
worker exited. Latest CSV13490 gives a conservative phase bound5500 attempts,
or accumulated bound19110 including the deliberately conservative earlier
trial accounting. No weights were restarted or trained beyond the cap.

| Three-view HD checkpoint | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Immutable parent8k |28.709667|.796946|.317703|
| Low-LR patch+DSSIM9k |28.930609|.800138|.315825|
| Low-LR uniform9k, online |28.9645|.799084|.309530|
| Low-LR uniform13k, fresh selected |29.088770|.803191|.300929|

| Fixed native600×330 floor ROI | Parent8k PSNR/SSIM/LPIPS | Patch9k | Uniform9k | Uniform13k |
|---|---|---|---|---|
| HC |31.7151/.851887/.260921|31.6630/.856891/.267971|32.2301/.855242/.256897|32.3445/.857365/.250919|
| KA |30.4239/.822590/.327458|31.3070/.828049/.328810|30.8044/.825288/.319748|31.2023/.827310/.317927|

These rectangles use exactly the same float GT, no semantic masks/compositing,
and each checkpoint's stored native core0. Uniform9k improves LPIPS in both
rectangles; patch9k retains higher KA color PSNR. Gray mottling is still visible.

- [Actual state/LR proof](/fsx/tmp/marshawn_001660_hd/native_png_uniform_LR0005_STAGED/prelaunch_proof_receipt.json)
- [Three HD views: GT, parent8k, patch9k, uniform9k](/fsx/tmp/marshawn_001660_hd/native_png_uniform_LR0005_STAGED/gate9k_matched_all3_GT_parent_patch_uniform.jpg)
- [HC floor crop](/fsx/tmp/marshawn_001660_hd/native_png_uniform_LR0005_STAGED/gate9k_matched_floor_HC.jpg), [KA floor crop](/fsx/tmp/marshawn_001660_hd/native_png_uniform_LR0005_STAGED/gate9k_matched_floor_KA.jpg)
- [Float ROI metrics and source hashes](/fsx/tmp/marshawn_001660_hd/native_png_uniform_LR0005_STAGED/gate9k_floor_metrics.json)
- [Root review at9000](/fsx/tmp/marshawn_001660_hd/native_png_uniform_LR0005_STAGED/root_gate9000_review.json)
- [Final stop/checkpoint/metric receipt](/fsx/tmp/marshawn_001660_hd/native_png_uniform_LR0005_STAGED/finished_receipt.json)
- [Selected13k all three HD comparisons](/fsx/tmp/marshawn_001660_hd/native_png_uniform_LR0005_STAGED/best13k_matched_all3_GT_parent_patch_uniform.jpg)
- [Selected13k HC floor](/fsx/tmp/marshawn_001660_hd/native_png_uniform_LR0005_STAGED/best13k_matched_floor_HC.jpg), [KA floor](/fsx/tmp/marshawn_001660_hd/native_png_uniform_LR0005_STAGED/best13k_matched_floor_KA.jpg)
- [Selected13k float ROI metrics](/fsx/tmp/marshawn_001660_hd/native_png_uniform_LR0005_STAGED/best13k_floor_metrics.json)

## Insights and next steps

Reducing LR avoids the severe quality drift of the high-LR patch phase. The
uniform control preserves detail and improves perceptual metrics without DSSIM,
but neither this gate nor the earlier patch gate achieves clean floor renders.
The fixed recipe completed default stopping and actual best-checkpoint evaluation
with LPIPS tie-break within .07dB. New weights require their own
geometry/grid audit; earlier checkpoint occupancy certificates do not transfer.

## Rejected variance-polish phase

A separate warm phase loaded the immutable selected13k state, retained uniform
sampling/native NN/RGB MSE/distortion/LOS/surface, and changed the sample target
from262144 to1048576 plus current LR to exactly .0001. The proof used the actual
parent Adam step12998, LambdaLR epoch12998 and scaler524288, preserving moments,
RNG and model values. Initial independent rays increased16→64. Nonserialized
optimizer default LR was reconstructed from the saved group initial LR.

At14000 the three-view result was29.0713/.803048/.300989, against
parent13k29.088770/.803191/.300929. Root inspected all three native HD views and
the same HC/KA native crops: floor mottling was nearly unchanged. The phase was
rejected at the mandatory gate. Root terminated the paused worker before14001;
the phase consumed exactly1000 attempts, bringing the declared family bound to
20110. There was no terminal14999 checkpoint or further optimizer update.

The controller correctly chose the external parent13k and freshly reproduced
29.088770/.803191/.300929 on all three views. Both processes exited. This is a
rejected polish test, not evidence that clean floor rendering has been achieved.

- [GT / parent13k / polish14k all three views](/fsx/tmp/marshawn_001660_hd/native_png_uniform13k_batch1M_LR0001_STAGED/gate14k_matched_all3_GT_parent_polish.jpg)
- [HC crop](/fsx/tmp/marshawn_001660_hd/native_png_uniform13k_batch1M_LR0001_STAGED/gate14k_matched_floor_HC.jpg), [KA crop](/fsx/tmp/marshawn_001660_hd/native_png_uniform13k_batch1M_LR0001_STAGED/gate14k_matched_floor_KA.jpg)
- [Root rejection and final receipt](/fsx/tmp/marshawn_001660_hd/native_png_uniform13k_batch1M_LR0001_STAGED/finished_receipt.json)
