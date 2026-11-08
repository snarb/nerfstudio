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
