# Sixteen extra TRAIN source rays: CPU capture proposal

## What was tested

Prepared a separate16-ray capture packet from the reviewed [actor proposals](marshawn_native_actor_label_review.md) and [near-floor proposals](marshawn_native_actor_near_floor_review.md). Original69,888 queries, masks, two IA overrides and every fitted head remain unchanged. No model, GPU, source sampling or optimizer was launched.

## Results

| Source class | Rays | Cameras | Current fitting weight | Proposed weight after source qualification |
|---|---:|---|---|---|
| Visible actor shoe | 3 | MD | 0 | 1 each |
| Hair native-center diagnostic | 1 | FD | 0 | 0; excluded |
| Visible floor class | 12 | MC/MD/FD | 0 | 1 each |

Standalone archives contain MD7/FD5/MC4 rays, with unique query IDs, continuous HD coordinates, mapped native footprints, exact floor poses, origins/directions, native camera pixel areas/direction norms, class targets, review confidence and source hashes. No global base-query IDs were reassigned or appended. Class targets express source material membership; they supply no alpha, depth, free-space or whole-ray-empty targets.

FD's original HD pixel has .230247 nominal AREA coverage. The diagnostic center instead uses RAW[3637,1629]→HD array[851.9152850391,381.1067675995], with both its one-native-pixel footprint and the actual HD camera footprint saved. **The unchanged source HD camera pixel area is retained.** Moving the ray center does not turn that footprint into a pure native-pixel observation; FD remains excluded from fitting unless a separately reviewed native-footprint/observability contract resolves this issue.

CPU `Cameras.generate_rays` matched independent floor pinhole directions within1.064e−7; rays are finite/unit-length with positive native pixel area. All63 original query archives were rehashed unchanged. Six original source DC/HC/KA rendered RGB/alpha references were rehashed for the required future source-parity gate; EVAL targets were not read.

[Capture manifest](assets/marshawn_native_extra16_capture_plan_20261107/manifest.json), [CPU validation](assets/marshawn_native_extra16_capture_plan_20261107/CPU_validation.json), [false release template](assets/marshawn_native_extra16_capture_plan_20261107/release_TEMPLATE.json). Manifest SHA`9205c0953dab3bc7b3710e1b750dc95b32f26f540df26daa1cd42061b20c721a`.

## Insights

Future capture must preserve corrected source126316 checkpoint`340e794…cd641`, full pipeline`c83bc7…d46636`, floor identity transform, original sigma, disabled additional semantic sigma selectors, and the source native1024 allocator. The reviewed stable-FP64 observer captures the actual source packed samples; no alternative sampler is proposed. All three full source RGB/alpha parity gates, source-state/RNG/LUT preservation, native unwrapped16-ray parity and per-row weight/depth/color observability remain pending.

Portable SG execution is preferred **after actual source parity qualifies it**. The original49e1 manifest bound in the immutable CPU packet is provenance only: its first alias failed before model load. The latest candidate is pathfix_v2`eb288…e4ffc`, still awaiting root/parity qualification. A [separate fail-closed dependency notice](assets/marshawn_native_extra16_capture_plan_20261107/execution_dependency_notice.json) requires a later root release to bind the final qualified source alias and parity receipt; no ray or label array was changed for alias preparation.

Actual source weights/alpha/cap/depth/color must be inspected before any class fitting. Even a clear floor photograph does not authorize hidden-actor negatives. Current fitting weights and every authorization/parity/observability gate remain zero/false; this packet authorizes no capture or fit.

Replay helper: [prepare_marshawn_native_extra16_capture_cpu.py](../scripts/prepare_marshawn_native_extra16_capture_cpu.py). Full artifacts: `C/native_actor_extra16_capture_plan_CPU_v1`, where `C=/fsx/tmp/marshawn_001660_hd/tuning_20261106`.
