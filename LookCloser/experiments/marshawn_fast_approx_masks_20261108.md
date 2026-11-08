# Fast approximate video masks

What was tested: the user accepted approximate two-actor masks and explicitly superseded the future HEAD/anatomy/CREW/PRE perfection workflow. SOURCE143 CPU parents were held by exact PID/start/boot checks; running BODY1720, fields, frequency fits and caches continued. All pending performance/prefetch activation was cancelled. Existing published sources through1715 remain unchanged.

The new G2 source root is `/fsx/tmp/marshawn_video_fast_masks_20261108`, frames1720..2015 at step5. It aliases the original current-frame RGB sources, calibration/code and common AABB. A simple four-frame CPU queue preserves original native W32/W64 falloff, RAW AREA prefilter, one calibrated remap, core/band and66 TRAIN membership. Missing approximate actor masks are labelled `approx_unavailable` with null crop; they are not certified physical absence. The G2 training I/O adapter skips that unavailable actor metric. No neural model parameters change.

| First source1720 | Actual result |
|---|---:|
| Existing66 BODY masks adopted, no new NN |9.19s|
| DATA and cheap shape/RGB/alignment QA |122.66s|
| Cameras |66|
| Approximate-unavailable targets |JE black only|
| New source marker |`1716cd176d7cd3e643c76aa6e967948d1101c00f6f0b56215d1cc47c902b9b11`|

HC/FD/KA previews were quickly viewed by ROOT and accepted as approximate: both bodies/feet are sufficiently covered and crew/background are suppressed. No refinement or extra manual gate follows. The simple source marker is `FAST_APPROX_MASK_SOURCE_READY_V1`; QA checks source and shape alignment only and makes no old HEAD/CREW/PRE/dense perfection claim. Same-frame RGB-hash-qualified frequency maps can be reused across this source generation; maps from other video frames remain prohibited. The ordinary frozen whole-cell admission builder and one-cell pad remain available to the new minimal dispatch route. Every NeRF remains cold with the140,000-iteration cap.

Insights: persistence matters more than repeated per-view jobs. The installed local SAM3 package includes its video predictor and the existing checkpoint contains1,156 detector plus309 tracker keys, so no new download is needed. The existing image helper also constructs its model outside its camera loop; one-view wrappers had repeatedly rebuilt it. Old recorded timings include model construction, inference and native refinement together, so they do not isolate pure NN compute from startup. The new mask worker owns persistent approximate inference; the DATA queue streams each completed66-camera frame without waiting for the full video.

Code/policy and first-source proof are snapshotted in `scripts/marshawn_video_ops/fast_approx_g2`. Training/grid/router integration is a separate honest G2 adapter; source readiness alone is not a claim that a field has started or finished.
