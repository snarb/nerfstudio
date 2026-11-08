# Marshawn source CPU throughput

## What was tested

Read-only timing of the qualified frame1690 source factory, followed by private replay of its actual semantic qualification using eight camera-grouped CPU workers. Each worker calls the unchanged `marshawn_current_parts_v105.qualify_tile`, with one native thread and a read-only cache of repeated current/prior camera NPZ inputs. The parent merges ROI results in the original global binding order.

The test used an immutable preapplication snapshot, reconstructed from the exact recorded input hashes and saved per-camera preapplication archives. No canonical source masks, DATA, occupancy grids, model states, active producers, or thresholds were modified by this replay. Source1690 had already passed its ordinary grid/dense checks and started cold training on SG1 before the replay ran.

## Results

| Actual1690 source phase | Observed time |
|---|---:|
| Serial semantic qualification, 64 ROIs | 177 s |
| Four-worker component closures, 66 cameras | 242 s |
| Ordered source application plus 66 union checks | 47 s |
| Geometry UNKNOWN rebuild after application | 34 s |
| Four-worker final source guard | 192 s |
| Private eight-worker semantic replay, same64 ROIs | 65.35 s |

The ordered application stage had only seven nonzero camera scopes. Seven application receipts therefore represented completion, rather than a stalled7/66 queue. Historical per-camera frozen DATA semantic outputs spanned31–32 seconds; frozen DATA should not be rewritten based on the apparent serial-application backlog.

Semantic replay passed exact parity on16 camera contexts and49 qualified records:

- All qualified bool masks and NPZ archive bytes match the original serial result.
- All decisions, values, record order, native input bindings, and normalized qualification metadata match.
- Only the creation timestamp and private output directory differ.
- A separate actual CE-camera canary passed through the normal import/public API and spawned worker path.

The observed semantic-stage improvement is approximately2.71×. This is a single cached-NN same-input replay, not a controlled repeated performance benchmark. Fresh NN inference, transport, and dispatch costs are separate.

Evidence is under [semantic_CPU8_v1](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/semantic_CPU8_v1/release.json). The private replay [parity report](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/semantic_CPU8_v1/parity.json) has SHA256 `aad3337920ef0cb21858c0194b07da5c06dc3fa7c5f011c3e970f9bff05e6ddc`. Helper SHA256: `27b7207267c0fd0719af8553fbbaed7ea217432431019a590c7e646c4b97416c`. The exact qualified core function source remains `838e6b701e152d6591a1dbe7db33fd0196238ea0d0f2a80df56f2ef2cc4a5831`.

## Insights and next steps

The narrow candidate is grouped semantic qualification with eight bounded workers, followed separately by changing the existing independent closure and final-guard pools from four workers to eight. Preserve ordered source application, frozen DATA, field math, masks, and thresholds. The sole source-factory owner should integrate these only in a new reviewed operational release; the activeV107 files remain unchanged by this work.

Use a normal namespace import so spawned workers can import the helper:

```python
from semantic_CPU8_v1.marshawn_semantic_qualification_cpu8 import qualify
qualify(frame, bindings, root=V, workers=8)
```

The calling factory retains ownership of the canonical source-writer lock. Measured future-frame cadence should confirm the throughput benefit after integration.

## Native BODY input packing and collection

Frame1695's actual BODY preparation took27m34s from factory start to all66 verified results collected. All eight silver GPUs were eligible and used; it was not restricted to four GPUs. Per-GPU task counts were5/1/8/7/6/13/13/13. Olive was training1685 and SG1 subsequently became the1690 field slot. The measured serial task-packing span was416s, full portable-source staging completed about8m24s after staging started, and the first-to-last NN execution window was18m03s. Same-GPU gaps had a47.86s median; optional frequency fits occupied SG0/2/3/4 in the early recorded snapshot. These gaps therefore include multiple causes, rather than just inference time.

An inactive eight-process per-camera pack helper calls the original sealed V5 adapter unchanged. Each camera packs silver then olive in the original order; the parent preserves camera order. Separate processes isolate the adapter's temporary global seed override. Original source hashing, ownership-clean identity tensors, symlinks, atomic JSON writes, and fsync behavior are preserved.

Private replay used immutable1695 source receipts and the exact1690 native-reference archives recorded by the actually executed1695 tasks. All132 host variants passed:

- Complete task JSON, reference NPZ, original/portable source receipts, seed-policy JSON, and resolver bytes are identical.
- Native PNG symlink targets match exactly; no full-native PNG copy was made.
- Factory record values and camera/variant order match, apart from private output locators.
- All snapshot source bindings remained unchanged.

The replay completed in235.40s, approximately1.77× faster than the historical416s serial task span. This single replay is not a controlled repeated benchmark. Evidence: [inactive pack release](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/native_pack_cpu8_v1/release.json), SHA256 `be8162738a30878701456636f5815252c9e31ca9a7468c82edc3a08a7bc5bc51`; [full byte parity](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/native_pack_cpu8_v1/parity.json), SHA256 `90c2b283ea495b501c2ef4aa8ea3e60436f1417fb4e03e84b46bfb869e7d1e14`.

Three private collection replays of already completed NN outputs retained every original transport/provenance/array check:

| Camera | Total collection | Five serial rsync calls | Array collection/validation | Remaining receipts/checks |
|---|---:|---:|---:|---:|
| AA |13.31s|4.19s|5.40s|3.72s|
| BC |23.79s|4.26s|11.12s|8.41s|
| OD |20.64s|4.24s|9.82s|6.57s|

The current controller already calls `fleet.tick` before collection. Its next refill tick is blocked while all newly completed cameras are gathered serially. A single next operational candidate is bounded two-camera background collection, with one owner and crash adoption per camera, unchanged verification, and parent-only state merging in original order. Initialize shared source/code aliases once before collectors start. This proposal has not been activated. Private replay timings reflect current FSX/network contention; they are not exact historical per-call timings. [Collection profile](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/native_BODY_1695_profile/controller_collection_profile.json).
