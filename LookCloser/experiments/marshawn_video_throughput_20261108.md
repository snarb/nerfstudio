# Marshawn video throughput, 2026-11-08 08:00 UTC

## What was measured

Native accepted-generation completion/runtime/log records for ten140000-update
models; original frequency-fit progress/completion receipts for nine complete
66-camera groups; comparable original RGB-source readiness to accepted-model
completion for seven frames1675–1705. Retired1695/1700 models are excluded.
The1670 same-frame continuation is combined with its preserved earlier active
segment; its final956-second runtime alone is not a cold-model duration.

## Results

| Measurement | Observed time |
|---|---:|
| Training, allten reconstructed active runs, mean | About2h03m32s |
| Training, eight RTX silver/olive cold runs, mean |1h54m40s |
| Training, DEV3 observed run |3h13m45s |
|66 frequency maps, aggregate GPU work, mean |1h32m30s |
|66 frequency maps, complete observed wall span, mean |80.3min |
|66 frequency maps, observed wall span range |63–91min |
| RGB ready to accepted model, seven-frame mean |12.107h |
| RGB ready to accepted model, range |7.505–16.384h |

The end-to-end span includes asynchronous frequency fitting, mask preparation,
waiting, source-quality repairs and cold training. Initial RAW decoding is
excluded because this interval starts at ALL66RGBsource_ready. Stage durations
overlap and cannot be added as a measured wall-time total. Earlier15–30minute
frequency figures described a main60-camera batch/parallel estimate; the user
was explicitly corrected after full66-camera records became available.

At the cutoff ten models were complete,1710 had37000/140000 updates and61
models remained pending.1715source QA had completed; fresh grid/dense admission
and ordinary cold dispatch remained. The ETA communicated was50–80additional
hours for both full movies, conditional on source-quality repair variability;
unmeasured cache/prefetch/scheduling speedups were excluded. Roughly3hours per
frame without queues is a stage-based estimate, not the measured end-to-end mean.

## Insights

Sequential mask/source qualification is currently the throughput bottleneck.
Frequency fitting runs ahead and training uses independent physicalGPUleases.
Ten slots do not imply ten simultaneous fields when source preparation has not
yet released ten frames. Source actor ownership repairs and operational wiring
retries explain substantial historical waiting; hybrid quality experiments
remain separate from the production recipe. No training, source or renderer
parameter was changed to produce this timing report.

[Compact ROOT accounting](/fsx/tmp/marshawn_video_source_gcrole_g1_20261108/operations/ROOT_status_timing_20261108_0800.json)
and [native timing receipts v3](/fsx/tmp/marshawn_video_source_gcrole_g1_20261108/operations/actual_completed_training_times_20261108_0800_v3.json).
