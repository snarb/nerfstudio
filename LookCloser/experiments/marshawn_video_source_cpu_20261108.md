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

The first packing candidate above was **never activated**. Its approval was revoked after identifying that pool children can survive a killed factory parent. The replacement [transactional V2 release](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/native_pack_cpu8_v2/release.json), SHA256 `e08507ee2a739df051bc496f54301b9ec770a353ed2f6fda1740b7dfb33902e1`, adds a per-camera flock and packs each unchanged input bundle into a private directory before source validation and atomic publication. A restarted factory waits for a surviving camera owner, validates the exact source/code/paths/current identity reference, and adopts completed bundles. Interrupted unsealed bundles are retained separately.

Fresh V2 replay again passed all132 task/reference/receipt/policy/resolver byte comparisons, symlink targets, and camera/host order. It took191.79s; the difference from the earlier replay can include filesystem cache and contention. Actual regressions passed for factory-parent death, camera-child death after the canonical base manifest write, normal namespace/spawn API restart, rejection of wrong frame/current-reference adoption without overwrite, and recovery of an interrupted legacy bundle. No partial bundle became visible at the final task locator. The original canonical adapter and NN functions remain unchanged. Helper SHA256: `917728fda0d2e1485005c6ef0ddb8c59b05752cd65432ec1c7ddf017e060872a`; [fresh full parity](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/native_pack_cpu8_v2/parity.json) SHA256 `457c8206fb44cb1f7aa45e1c820ffd837fcfc69223a94551dd4b9389d3afe1dc`. The active factory is not modified by this qualification; integration remains with its sole owner under a new operational release.

## Bounded asynchronous result collection

The next inactive candidate runs the unchanged sealed V62 `gather` in two detached CPU collectors. This retains the original five rsync calls, argument values, `assist.collect`, array/provenance/source checks, and union validation. The parent continues the original source-ready/GPU-pool/fleet ticks and merges completed proofs in its original camera order. A reverse-AST check confirms that NN job commands, pool policy, source readiness, retry behavior, and watchdog logic are unchanged except for enqueue/result handling. Native GPU-priority policy is unchanged in this candidate.

Each collector holds a camera lock and one of two pool-slot locks. Both descriptors pass through to the actual rsync children. Completed results have an atomic, source/config/helper-bound receipt and output inventory. A restarted controller adopts surviving collectors or verifies their finished result. A killed collector's incomplete isolated per-camera outputs are retained separately before the unchanged gather retries; canonical SOURCE masks are never edited.

Actual completed-result replay of all66 cameras took329.75s with at most two simultaneous collectors. Every native, person-candidate, and pre-clear archive is byte-identical to the original1695 collection. Metadata and order also match after only replacing the private isolated-output locator and the SHA of the application receipt containing that locator. Parent polling work had a0.195s median and6.69s maximum, rather than blocking for each13–24s collection. This is a completed-NN CPU replay, not a full BODY end-to-end timing claim.

Actual crash regressions passed:

- Kill the controller parent while two collectors run; a new controller adopts the exact surviving PID/start/boot identities.
- Adopt atomic collector completion after the parent died before merging camera state, without rerunning collection.
- Kill a collector after unchanged `assist.collect` completed but before the final application receipt; retain the interrupted artifacts and recover identical native/person archives.
- Kill a collector while its real rsync child is stopped; that child retains both inherited locks, preventing a duplicate writer until its natural completion, after which collection recovers exactly.

Evidence: [inactive async2 release](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/sam3_async_collect_v1/release.json), SHA256 `6e2e7634ec8c84eb2775a7d2d9d19d6ee580c7561f543ce48cf9bd258e8bfba3`; [all66 parity](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/sam3_async_collect_v1/parity.json), SHA256 `f58a7d4f1022cec259b8711e1c8a7e046b78ada1fe27dc9659161f1c1b6a2179`; [crash regressions](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/sam3_async_collect_v1/crash_regressions/report.json), SHA256 `b5aa7ee179fc8bdb48a8ec293bd47f3e6d9530c17d3c73c9a5a2ab2d7b97b56d`. No GPU, NN job, field model, native priority, or canonical source data was changed by these tests. Production integration remains a separate reviewed release owned by the source producer.

## Optional native PNG prefetch

Preserved old frame1700 BODY preparation completed in33m07s: serial task packing14m, first-to-last NN17m07s, and full source staging8m04s overlapping NN. Canonical NN execution summed1731s across66 cameras, averaging26.23s. All six non-field SG slots and olive participated. At01:57 four SG slots appeared empty, but the actual parts queue had64 jobs and refilled all seven non-field SG slots by01:58; that snapshot did not establish a scheduler deadlock. All64 parts input PNG hashes match the already verified BODY stream.

Frame1700 staging contained4,719,621,152 native PNG bytes and723,010 small task bytes. The frozen rsync command has no bandwidth limit, and the surviving factory's actual PATH resolves native `/usr/bin/rsync`, not a wrapper. Observed payload divided by the484s stage interval is9.30MiB/s; this is not a wire-throughput benchmark. Uncached SHA timing was not logged. Once all66 inputs were cached, combined physical GPU probing plus stream-ready work took0.772s. [Input stage evidence](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/native_BODY_1700_profile/input_stage_cap_and_probe.json).

The optional PNG-only prefetch copies already decoded, immutable source-ready native PNGs ahead to SG NVMe, outside native task/route directories. Per-file receipts bind original frame/camera, source-ready and source-receipt SHA, original TIFF SHA, native PNG SHA, byte size, and exact mtime. One chronological payload transfer is bounded at16MiB/s, with a384GiB cache budget and256GiB free-space reserve. Source/mask/model state is not transferred. No original, checkpoint, render, or proof is deleted.

The sole source owner can create a hardlink from an already verified cache file into its normal task input after claiming that frame's route. Missing/unready cache returns immediately; transport/helper failure falls through to unchanged original staging. Foreign existing targets are rejected without overwrite. Original `rsync -rLt` and actual streamed-source SHA verification remain mandatory. Exact size/mtime lets the normal transfer skip an already cached payload. Consumer markers carry exact PID/start/boot identities; the dispatcher yields payload copies for an active consuming frame.

Qualification used actual1705/AA and1710/AB PNGs of73,682,789 and72,658,671 bytes. An interrupted1,605,632-byte partial file was rejected before cache READY, then resumed to the exact source SHA/size/mtime. The isolated consumer produced a same-inode hardlink; foreign cache and normal files remained unchanged after rejection; an unready consumer returned without creating a target. Qualification did not create any real future native-task directory or touch frame1700. [Cache release](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/native_png_prefetch_v1/release.json), SHA256 `74245e7657069eab1e1abef950eb1a37664653d50021da45db152623b1cd7e49`; [qualification](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/native_png_prefetch_v1/qualification.json), SHA256 `2e4b62c74279b8ffa415b284a15104a908d81f4b107550d29bb0bb18b04d1e26`.

ROOT approved the cache under authorization `6dd8eefe…`; producer-owned V114 adds the optional consumer for unclaimed1705+ routes. Existing frame1700 retains its old owner. The source and frequency/model pipelines continue independently if prefetch fails. Actual1705 packing, collection, cache-hit counts, and full source cadence still need observation; no end-to-end speedup is claimed yet.

## First production use: frame 1705

The first complete BODY pass using transactional CPU8 packing, optional PNG cache, and bounded async2 collection finished successfully for all66 cameras. Every normal streamed-source SHA check remained enabled; all66 cached PNG inputs were create-only same-inode links. Original NN models, native crops, seed/reference math, camera order, and native-priority policy were unchanged.

| Observed stage | Old frame1700 | New frame1705 |
|---|---:|---:|
| Task packing,132 host variants |845s serial file span|39.25s actual CPU8 execution|
| Normal source staging and all66 SHA checks |484s|16.35s identity-mtime interval; producer timer16.06s|
| First NN start to last NN supervisor completion |1027.3s|475.80s|
| Last NN completion to final verified collection |about10s|15.32s|
| Factory/pack start to all66 collected |1987s|611.01s|

These are single production observations on different frames and concurrent loads, not a matched benchmark. The new cached PNG payload was transferred before the critical path. The1705 aggregate collection work was427.99s, averaging6.48s per camera and9.48s maximum, while two collectors overlapped with continued NN refill. Task counts were SG1/5/7:11 each, SG0:6, SG2:9, SG3:3, SG4:7, and olive:8. SG6 remained an active field slot. All66 jobs completed without failure and all66 results were collected.

The measured BODY pass decreased from33m07s on1700 to10m11s on1705. Full source cadence still includes HEAD/part inference, qualification, source guard, DATA, and final grid audits; this does not establish a ten-minute end-to-end frame pipeline. This observation does not justify changing native priority or scientific parameters. [Source-bound final timing receipt](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/native_BODY_1705_profile/BODY_complete.json).

## Frame1705 HEAD timing attribution

A bounded read-only review of existing timestamps and the actual source code found no border-expansion NN round for1705. The66 main HEAD jobs took294.59s from first start to last supervisor completion. All64 subsequent semantic-part jobs took290.42s and used the six non-field SG slots evenly,10–11 jobs per GPU.

| HEAD interval | Observed time | Actual work |
|---|---:|---|
| BODY collected → original HEAD worker start |61.59s|Producer/install/launch gap; not attributed to one function|
| HEAD worker → first NN start |238.39s|Includes serial ROI/spec generation and launch|
| First → last of66 specs |201s|Previous/current native PNG SHA checks and grayscale decoding, bounded ROI flow, small spec/proof writes|
| Last spec → controller launch |2.19s|Small-spec staging and launch|
| Launch → first NN |30.83s|GPU probing, lease/startup path; individual costs not instrumented|
| Last NN → first qualification record |29.39s|Terminal polling, unchanged result rsync, first qualification|
| First → last of66 qualification records |30s|Read-only qualification plus private proposal/proof writes|
| Last qualification → all66 source applications finished |154s|Serial requalification, current native/H/union archival and writes, union validation and SHA receipts|

The post-NN total was213.39s, from02:57:22 to03:00:56. The application receipt span was151s;56 cameras changed and10 weak-camera applications left foreground unchanged. This is combined CPU/decompression/hash/compression and filesystem work, not an instrumented CPU-versus-I/O split. `qualify` itself reads current source and NN outputs; `apply` repeats qualification and mutates canonical per-camera source artifacts. No new tests, inference, parallelization, scheduler policy, or source edits were performed by this profiling. Filesystem timestamps are rounded to seconds; these are observational bounds. [Bound timings and code hashes](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/native_BODY_1705_profile/HEAD_CPU_IO_attribution.json).
