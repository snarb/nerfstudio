# Native supplemental 4K visual rendering: resource proposal

This is a preparation document, not a rendering release. The primary FHD V1 score, stopping rule and checkpoint selection remain unchanged. The existing actual 4K EVAL images are excluded from training.

**What would be tested.** Compare actual 4K source images against native100k and the HD/TRAIN4K continuations at the same additional attempted step. Freeze the latest common completed gate after the paired controller finishes; record each checkpoint and hash. Render DC, HC and KA for each of those three checkpoints: nine full 3840×2024 outputs. Use fixed actor, hair, hand and foot regions derived by doubling the existing FHD crop coordinates. Do not upsample predicted FHD renders or select a favorable gate using 4K appearance.

The frozen source descriptor is `C/resolution_4k_eval_diagnostic_v1/render_views_4k.json`, SHA256 `f83aa03d9c148e05014907034d94d0f77f4e3d3282fa8169d9bf562a3cadda29`. The CPU feasibility audit in `B/native_resolution_pair_plan_v1/diagnostic_4k_feasibility.json` verifies all three source camera poses: inverse floor transform restores the native-world poses within `1.11e-16`, intrinsics and dimensions are exactly doubled, and distortion remains zero.

**Implementation proposed for review.** Make a separate version of the frozen AABB partition renderer. Copy the existing native EVAL camera, double its focal/principal-point values and integer image dimensions, and supply the corresponding independently prepared 4K source image only to the comparison panels. Preserve field sampling, compositing, accepted AABB, original-world camera pose and native R1 state. Add exact source-image/descriptor hashes, scale and camera metadata to its receipt. Require the unchanged model digest and checkpoint hash before/after rendering, plus the renderer's existing native-compositor and partition-sum parity assertions. Keep unmodified frozen renderer generations available.

Use the approved **HD identity configuration for all three read-only checkpoints**, so each renderer loads the smaller HD TRAIN cache. This changes neither the checkpoint weights nor the EVAL camera or private geometry; the prior technical gate already established constructor/FHD parity between the HD and 4K adapters. The diagnostic camera override is separate from the dataset descriptor: do not change the adapter's required FHD EVAL split, and never feed its external floor-frame poses directly to the native field. Before full rendering, the new code still needs review and a bounded resource/parity probe.

| Resource | Concrete proposal |
|---|---|
| Scheduling | Wait for paired production `complete.json`, controller exit and empty GPU; never interrupt a live training chunk or bypass its global lock. |
| Exclusivity | Acquire `/tmp/lookcloser_tuning_gpu_1000_0.lock`; run one renderer at a time on dev3 GPU0. |
| Admission | At least 20 GiB available RAM and 20 GiB free output storage; no other GPU compute process. |
| Initial probe | Native100k, HC only, full actual 4K camera, unchanged query chunk 512; record process RSS/VM high-water and GPU use at one-second cadence. |
| Abort limits | Stop only the diagnostic process if available RAM falls below 4 GiB, observed GPU allocation exceeds 12 GiB, or the initial camera exceeds 20 minutes. Preserve its partial output and logs. |
| Acceptance | Completed full frame with finite RGB/alpha, exact shape/source-camera binding, unchanged model/checkpoint, native compositor parity and contribution-sum check; inspect source/hair/hands/feet before releasing the remaining eight renders. |
| Cache lifecycle | Worker must exit and return resources before the next camera; no renderer shares a TRAIN cache with another process. |

**Resource estimates, not measurements.** There are 7,772,160 pixels per 4K frame. The five existing float32 RGBA partition buffers require 621,772,800 bytes (0.579 GiB) on CPU, plus 29.6 MiB depth, source/panel images, and compression temporaries. Origins, directions, pixel area and int64 camera indices alone require about 267 MiB for full rays, with additional metadata, ray-generation intermediates and query work beyond that lower bound. GPU query chunks stay at 512; full rays and output buffers still grow fourfold. Actual peak must come from the proposed HC probe.

The saved native100k FHD renderer measured 37.11 seconds for DC, 37.52 for HC and 30.47 for KA, excluding model/cache setup. A fourfold pixel-work extrapolation gives roughly 2.0–2.5 minutes per 4K camera and 21 minutes for the nine-camera matrix, plus nine process/cache startups and image compression. This is not a runtime guarantee. Reserve a 45-minute idle-GPU window initially and extend only from observed progress without competing training. Nine uncompressed partition arrays total about 5.2 GiB before depth/images/metadata; the 20 GiB storage admission leaves headroom rather than assuming a compression ratio.

**Expected interpretation.** Actual 4K can reveal differences hidden by FHD scoring, but it also exposes the source's fine noise and preprocessing characteristics. A better-looking native4K crop does not establish cleaner 3D geometry or remove the known floor/mat/bag residual. Human review should compare all three cameras and all declared regions. No supplemental metric or visual preference may silently replace the primary V1 selector.

`C=/fsx/tmp/marshawn_001660_hd/tuning_20261106`; `B=C/native_semantic_grid_20000`.
