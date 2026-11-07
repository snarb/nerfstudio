# Native LookCloser packed capture: mixed precision proof correction

## What was tested

The released v3 source loader passed complete DC/HC/KA RGB and alpha model replay,
then the first TRAIN capture failed in the independent foreground proof. Native
TCNN sample RGB was FP16 and native weights FP32; their product promoted to FP32,
but the observer had allocated its accumulator from FP16 RGB. `index_add` correctly
rejected the mixed accumulator/source types. The radiance field and native
renderer were not at fault.

V3 remains unchanged, including its manifest, root release, failed job evidence
and rendered outputs. New CPU-only namespace:
`/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_capture_plan_v4`.
Its adapter forms `product=w[:,None]*rgb` and accumulates into
`product.new_zeros(...)`. Native RGB, weights, model precision, renderer and all
tolerances remain unchanged. Worker, launcher, preflight and standard controller
reuse the exact proven v3 filenames and bytes under the fresh v4 root.

## Results

| Check | Result |
|---|---|
| Inherited v3 source RGB replay | All3 views passed; maximum saved-code error1; changed fraction<=1.37242e-6 versus1e-5 limit |
| Inherited v3 source alpha replay | All3 views passed; maximum saved-code error1; changed fraction<=.000566124 versus.001 limit |
| Inherited initial live pipeline | `c83bc7dbb1504063c35f0a2b14ae3fb329ff27ed2eb6b3661eb89b4918d46636` |
| Inherited initial source RNG | `7de999966deb5bc61d81ce39a81891b52df244e05881da2b15cfabe0235e5811` |
| Inherited primed LUT | `7efedc2b32dffe6b6c3ac2faf7f692941eb6c68ac1eb6b2495f83d91da1bab57` |
| Failed v3 resources | Peak observed RSS4,117,848,064bytes; GPU2594MiB; no OOM; worker/controller terminal and GPU idle |
| V3 TRAIN cache | Zero NPZ files; final immutability receipt/cap-hit totals unavailable |
| CPU tests | All36 prior plus7 new tests passed:43 total |
| Real CPU worker import replay | 169 actual project/helper bindings passed from v4 package cwd; CUDA not initialized; no model |

The new tests retain original field RGB and FP32 weights, exercise FP16/BF16/FP32/
FP64 product promotion, compare native composition to an independent serial
emission/absorption reference, include multiple rays and empty rays/batches, and
reproduce the old FP16/FP32 failure explicitly. The fake native CPU renderer also
accumulates in product dtype. No runtime tolerance is increased.

Full seven-state and initial live pipeline/RNG/LUT fingerprints from the existing
v3 receipt are preserved in `inherited_v3_initial_fingerprints.json`. V3 control
flow passed source core/EMA/admission/FR restoration and post-FHD pipeline/RNG/LUT
invariance before reaching the dtype failure. It did not produce the final
capture invariance receipt; no such final proof is claimed.

The existing v3 model-render thumbnails were visually inspected: scattered
floor/tail density remains. This observer fix makes no geometry or image-quality
improvement claim. Original69,888 query rows,63 declared/60 usable cameras, the two
approved IA UNKNOWN weights, identity floor transform and source sampling remain
unchanged. The separate MD shoe label review does not alter this experiment.

## Insights and next steps

The CPU fixture originally used FP32 RGB and missed the native TCNN output dtype.
The observer now follows the actual multiplication's promotion semantics while
preserving the native samples and renderer. V4 must repeat all mandatory original
source FHD and state gates, then complete native packed capture under a new exact
root release. No v4 GPU work, head fitting or physical export follows from CPU
preparation. Legacy cap1024/occupancy omissions still prevent a full-visibility or
unseen-free-space claim.
