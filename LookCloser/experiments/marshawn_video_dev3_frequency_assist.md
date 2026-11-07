# Marshawn video: DEV3 frequency assistance before silver dispatch

## What was tested

An operational helper outside the immutable production code seal uses DEV3's
physical GPU0 (`GPU-ab838312-8799-795c-7df2-b085cb3db092`) to prepare six frequency
maps per newly prepared frame: AA,AB,AC,AD,BA,BB. Every image runs in a separate
process through the already sealed `marshawn_video_frequency.py` and canonical
preprocessor. No scientific/model implementation changes or learned state move
between frames. All66 model training views remain on silver-sable.

The centralized preparation producer's `--defer-dispatch` flag publishes
`central_prepared.json` with the `prepared_ready.json` hash and advances to the
next original frame. The assist daemon verifies the six maps, then invokes the
existing sealed `marshawn_video_stage.py push` from olive. The dispatch marker is
published only after the existing full remote input verification succeeds.
Silver then reuses these six same-image maps and fits its remaining60 maps.

## Results

| CPU check | Result |
|---|---|
| Six1660 source-bound maps validate against sealed fitter | Passed |
| Attempted neighboring-frame reuse of1660 map | Rejected |
| Exclusive assist lock | Passed |
| PID/start-time/boot identity proof | Passed |
| Operational helper syntax | Passed |
| GPU or SSH fitting launched by preparation | No |

The helper and configuration live at:

- `/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/marshawn_video_frequency_assist.py`
- `/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/frequency_assist/config.json`

The configuration initially has `authorized:false`. Its source hashes bind the
operational helper, frozen frequency worker, canonical preprocessor and existing
production code seal. The campaign owner releases and launches it; no production
code seal is edited to add this operational capability.

## Durability and isolation

An olive watchdog restarts its dispatcher and adopts existing work. Remote
controllers and current image workers are identified by PID, process start time,
boot ID and host. An SSH timeout is UNKNOWN and never authorizes a duplicate
fit; a successful remote probe must prove that both controller and worker have
stopped before another launch attempt.

A per-frame remote assist lock and the existing per-image fit locks prevent
duplicate publication. DEV3 uses the same global GPU lease as the fleet,
`/tmp/lookcloser_tuning_gpu_UID_0.lock`; both the global lease and frame lock file
descriptors are inherited by the current image worker. Surviving workers retain
their leases if their controller crashes. UUID/index and occupied-device checks
precede launch; the frozen worker also verifies its actual CUDA UUID. Progress,
GPU memory and OOM evidence are recorded for supervision.

Frame1660 already has all66 validated maps and is dispatched directly by the
campaign owner; the assist skips already dispatched frames. No full-video
barrier is added. A six-map receipt contains original frame/camera identifiers,
RGB hashes, map/metadata hashes and the canonical preprocessor hash.

To launch after operational release, use the qualified olive Python:

```text
python /fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/marshawn_video_frequency_assist.py launch --config /fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/frequency_assist/config.json
```

## Insights and next steps

The six-camera batch is deliberately smaller than the roughly30-minute frame
preparation cadence, allowing DEV3 to shorten silver's frequency phase without
turning assistance into a preparation barrier. Actual DEV3 fitting/adoption and
silver reuse remain operational checks after release; this report claims only
the completed CPU checks above.

Root authorized and launched the final operational watchdog at12:52UTC after the UID/ACL startup correction; actual GPU fits wait for frame1665 preparation. The production1530-file model code seal was unchanged.
