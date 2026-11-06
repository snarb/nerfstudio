# Marshawn DAG transport recovery, 2026-11-06

## What was tested

After eight initial arms completed, `actor_width32` and `repeat_control_seed43`
could not dispatch: `[Errno 7] Argument list too long: 'ssh'`. Their inherited
dependency JSON was 128,121 bytes; base64 encoding exceeded Linux's per-argument
limit. Larger later histories would also exceed the per-environment-string limit.
Neither job had a remote directory, receipt or worker before recovery.

The transport now sends specifications through SSH stdin, checks their digest,
and starts detached supervisors using an immutable specification file. Dependency
results travel in a separate JSON file with SHA256 verification. The task reader
still accepts the legacy environment format. Winner selection, map fitting and
all training code remain unchanged.

## Results

- Fourteen CPU tests passed, including real subprocess crash/resume, duplicate
  prevention, large payloads, quotes/newlines/Unicode and tamper rejection.
- A real silver-sable SSH test passed with a 775,585-byte specification.
- All 1,485 frozen source files were verified on silver. Only the controller and
  dependency reader differ from the original snapshot. Training worker SHA256:
  `7fd63b3f00859053de16e7b7eef8fde734b0407d0e90a5206ff36dd1c1bfe417`.
- Dispatcher PID2054880 resumed both affected jobs on silver GPUs3 and4.
  Healthy trainers were not interrupted; no recipe or stopping rule changed.

Campaign root: `/fsx/tmp/marshawn_001660_hd/tuning_20261106`.
The original controller is retained at `controller_before_transport_v2`.
`controller` links to the new `transport_v2_controller` generation, preserving
existing reporting paths. Full diff, source inventory, manifests, old state,
tests and SSH receipts are in `transport_recovery_v2`.

## Insights

Large DAG results must stay out of both command arguments and process
environments. A transport failure does not prove a detached worker never started;
receipt and process checks remain mandatory before recovery.

Existing SH, prefix, early-distortion and two-arm structural controllers have no
dependency payloads and need no live migration. Future conditional DAGs must use
both the new controller transport and the dependency-file-aware task reader.
