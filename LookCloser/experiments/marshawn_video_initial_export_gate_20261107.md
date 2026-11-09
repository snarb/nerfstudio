# Marshawn video: initial GPU export gate and framing v2

The v1/v2 outputs below are preserved diagnostics. A later actual-GT review
found that HC's endpoint crop clipped the grey actor's head. The approved
neural trajectory is now IC→JC→KC→LC→MC, ending at native MC crop
`[3500,650,3840,2024]`. Delivery v4 preserves these72 v3 neural frames and
uses the user-authorized paused2020 native photograph pan for its final48
frames. See
[current path and seven-time framing evidence](marshawn_video_path_1660_24fps.md).
The final selected-model watcher was replaced before a worker started and now
waits for v3's IC native sources, then exports new IC rays after canonical HC
loader parity. Existing DC v2 exports cannot stand in for this new validation.

## What was tested

A technical read-only export of original frame1660 was completed before all72
video models finish. Checkpoint15188 had already been pruned when the task began;
the campaign owner approved checkpoint30376 and its matching archived HC monitor.
A same-filesystem hardlink preserves this checkpoint under silver's
`video/qualification/step-000030376.ckpt`, independently of training retention.

The gate loaded this exact checkpoint/config through a temporary operational
adapter, without creating a production `frame_completion.json`. All source code
remained the sealed campaign implementation. Nonpersistent runtime was restored
explicitly: feature reweighting1.0 and `current_train_step=30376`.

The model's full pipeline tensor tree matched the checkpoint exactly before
rendering and remained unchanged after both full4K exports. The checkpoint hash
also remained unchanged. No optimizer updates, occupancy edits, source-pose
changes or cosmetic geometry cleanup were performed.

## Results

| Technical check | Result |
|---|---|
| Checkpoint |30376, original1660, read-only preserved hardlink |
| Checkpoint SHA256 |`8dc0f49c6252bec7701d3eea9a7c0363160320455d1fbd7670ea58dad4ab889e` |
| Pipeline tree SHA256 |`da76e8e6a7e85912bf936aa3e0bca6b6f7a319a2a99eab5df5b57f860a1c3a60` |
| Restored native HC vs archived30376 HC |Max1/255 RGB code; changed fraction5.15e-7 |
| Custom packed HC rays vs native HC |Max1/255 RGB code; changed fraction1.03e-6 |
| Actual4K exports |LookCloser and native-pixel hybrid,3840×2024 |
| Native sources |16 nearest TRAIN cameras at the same1660 instant |
| GPU isolation |Silver GPU4, exact physical UUID, shared global lease inherited by worker |
| Gathered PNG checksums |Passed |
| Model/grid/checkpoint mutation |None |

These are pipeline correctness checks, **not a quality-leader selection or a
completed full-recipe result**.30376 is an intermediate checkpoint; the original
production training continued independently on GPU0. The known raw-recipe floor
fragments remain in the exports.

First16-camera input staging and all optical/color references finished before
the GPU gate. The Marshawn Log3G10/RWG decoder and frozen photometry were used;
historical Misha PQ decoding was not used on these RAWs. Hybrid eligibility
enforced W64, valid pixels and explicit crew/UNKNOWN vetoes.

## Framing comparison

The original path's first actual4K render placed both actors in the upper center,
occupying roughly53% of output height. The campaign owner requested a second
framing candidate. It is saved separately; v1 remains unchanged.

v2 uses the actual DC `black|grey` label bbox `[774,93,1086,598]` in the HD
source, not rendered alpha that includes floor fragments. The first frame has
zoom1.45 and recenters on the actor bbox. Its4K intrinsics are:

```text
fx3930.5832762375862 fy3925.9213406546705 cx2007 cy1477.45
```

A quintic C2 interpolation brings these intrinsics to the **exact same** native
HC crop intrinsics at t=3.000. Every physical camera pose, source frame ID, GT
crop and all48 GT records remain identical to v1. No photographic resize is
introduced.

The v2 first frame was rendered again with actual new4K rays, using the same
30376 checkpoint. Same-checkpoint source geometry caches were reused; no model
or learned grid was transferred. The second complete LookCloser/hybrid GPU gate
and its post-render immutability checks passed. Both actors occupy roughly77%
of output height. Manual inspection confirms that their hair and feet remain
inside the image with margin. Known floor fragments remain; framing is the
only change.

- [v1 LookCloser/hybrid preview](</fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/render_gate/output/LC_hybrid_30376_pair_preview.jpg>)
- [v2 LookCloser/hybrid preview](</fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/render_gate_v2/LC_hybrid_v2_30376_pair_preview.jpg>)
- [v2 full LookCloser PNG](</fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/render_gate_v2/gathered/frames/0000/lookcloser.png>)
- [v2 full hybrid PNG](</fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/render_gate_v2/gathered/frames/0000/hybrid.png>)

The immutable v2 path is
`/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/video/path_4K_v2/path.json`.
The campaign owner manually inspected both actual v2 renders, approved this as
the final path and authorized adoption in `video/export_pipeline.json`.

The frozen CPU timeline/native-crop checks and an additional v2-specific check
passed after adoption. The latter verifies every emitted intrinsic against its
C2 interpolation, confirms that all physical poses and48 GT records remain
unchanged, and projects the actor-label bbox to4K corners
`[1467.6,279.75]` and`[2372.4,1744.25]`, centered exactly at`[1920,1012]`.
Nominal actor top/bottom margins exceed10%. Proof:
`video/path_4K_v2/adopted_validation.json`.

The earlier local v1 GT2020 CPU unit was archived to
`video/diagnostics/GT2020_v1_cpu_preview`, preserving its original receipt and
PNG hashes. It cannot be mistaken for a production v2 unit. No movie had been
published at the time of adoption.

## Later first-model visual limitation and final gate

The campaign owner inspected HC at216316 and DC/HC/KA from1670 at15188.
Bodies and feet remain present and colors are correct, but substantial floor
fragments and a background shell around silhouettes remain. Longer training
has not established clean actor isolation. Masked actor metrics do not measure
these remnants, and the grid must not be described as physically perfect.
[Inspected216316 HC](</fsx/tmp/marshawn_video_1660_2015_24fps_20261107/initial_review/001660_216316/HC.png>).

A separate detached operational watcher now waits for the genuine1660
`frame_completion.json`. It will load the selected actor-LPIPS checkpoint with
the canonical frozen loader, verify restored post-switch feature reweighting,
compare native HC to that checkpoint's archived monitor, then render the actual
approved4K v2 LookCloser and hybrid frame. It preserves the selected checkpoint
read-only and verifies the full model/grid tensor tree before and after export.
Its private output is `video/final_first_model_qualification`; it does not create
production delivery frames or fake a completion marker. The resulting framing
and visible fragment footprint still require manual review.

The versioned watcher release can use any of silver's eight physical GPUs only
after genuine completion, preferring the just-released GPU0. Each attempt still
requires the fleet's global device lock, an idle-device check, physical UUID
binding and an inherited worker lock descriptor. The earlier watcher was
retired only after proving that no render worker had started; its source was
not edited while live. Release:
`operations/watcher_v2_release.json`, source SHA256
`345730f5531bb0156e299cf8cbbdbbaf0f44cdc89af25d8f2b45c1412b65bc43`.

A read-only audit found no existing qualified arbitrary-view, temporally safe
actor-depth cleanup suitable for silently applying to this video. The current
hybrid transfers qualified source detail and retains neural fallback; it does
not remove learned fragments. Historical floor-point deletion changed real
shoe and lower-leg boundary opacity, so it is not actor-invariant cleanup.
No clipping, erosion, density change or semantic-head experiment was applied.

## Native input prefetch

Native inputs for all72 frames are prefetched while training runs, rather than
waiting until the last model. The one-shot operational adapter calls the frozen
`marshawn_video_render_stage.stage()`; it never starts a production renderer.

To preserve network priority for native masks, prepared RGB and frequency-map
dispatch, only this prefetch is limited to4096KiB/s through a dedicated `rsync`
PATH wrapper. Each prior prefetch process group was checked for its exact owned
command, terminated and proved stopped before resumable `--partial` restart.

The latest wrapper is
`V/operations/render_native_prefetch_inputs_only.py`, with restart receipt at
`V/operations/native_prefetch_inputs_only/restart.json`. It transfers all RAW and
reference bytes unchanged, but suppresses the two control-plane uploads:
`SV/video/path.json` and `SV/video/render_spec.json`. This prevents a long-running
prefetch from later publishing an obsolete cached path over an approved v2.
The existing final after-training workflow remains responsible for publishing
the latest approved render request.

The47 actual GT continuation frames2025–2255 are still unavailable. Prefetching
does not synthesize them or authorize a frozen-image substitute.

## Insights and next steps

Use actor-label framing to avoid having floor artifacts drive composition.
The actual v2 first-frame exports support the requested1.45 framing adjustment.
Validate the complete dynamic camera flight once the corresponding per-frame
models are ready, then render both final videos from qualified checkpoints.
