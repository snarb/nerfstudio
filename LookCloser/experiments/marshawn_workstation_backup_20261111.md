# Workstation evacuation

The user requested urgent preservation before Workstation07 and Workstation08 are deleted. The accepted scope is the latest full checkpoint per frame where available, the selected inference export, and restoration files. Completed production frames had already released optimizer checkpoints after durable inference export.

The destination on silver-sable is `/cache/ubuntu/lookcloser/emergency_workstations_20261111/`. Each machine's files retain their source path below `ws07/filesystem/` or `ws08/filesystem/`.

## Verified checkpoint snapshot

Both inventories contain eight files, comprising seven selected inference exports and one full checkpoint from active training. All sixteen files passed SHA-256 comparison with the source snapshot. Each machine contributes about 24.6 GB. Files were pinned on the sources before transferring so normal training cleanup could not remove them during transfer.

The snapshot precedes ongoing training progress. A final delta inventory is required before declaring workstation evacuation complete.

[WS07 verification](marshawn_workstation_backup_20261111/ws07/checkpoint_verification.json) and [WS08 verification](marshawn_workstation_backup_20261111/ws08/checkpoint_verification.json) record copied counts and legacy optimizer files that were already absent.

## Dataset evacuation in progress

The user additionally requested training data for all 72 frames in the 24 fps grid, 908 through 1263 with step 5. Service `marshawn-emergency-data24.service` starts after checkpoint checksum verification. It reuses existing data on silver-sable and prepares/transfers missing native datasets through the existing source-owner pipeline.

Canonical datasets are under `campaigns/marshawn908_20261109/frames/<frame>/native/`. Each dataset must contain 72 cameras. RGB crops retain source pixel resolution. RGB and semantic files are checked against their native manifest. Cameras, sampling masks and seed points are checked for availability. These datasets support the selected PNG recipe and accepted AABB.

Dataset evacuation completed. The receipt contains exactly the 72 requested frames and 5,184 camera records, with verified RGB and semantic checksums. These native datasets are available through the normal training entry point on silver-sable. [Completion receipt](marshawn_workstation_backup_20261111/data24_complete.json).

A final checkpoint delta scan found eight current model/checkpoint files on WS07 and nine on WS08. Transfer and verification of this refreshed inventory are in progress. The previous sixteen-file snapshot remains preserved.

## Transfer operation

Service `marshawn-emergency-backup.service` performs restartable transfers. Eight SSH streams per workstation and compressed rsync are used. Large optimizer checkpoints are split into immutable source chunks, transferred in parallel, reassembled and verified by whole-file SHA-256. Configuration and supporting campaign directories follow the checkpoint snapshot.

The S3 alternative returned `403 AccessDenied` for PutObject on both accounts. Transfer therefore uses SSH to silver-sable.

[Backup script](marshawn_workstation_backup_20261111/backup.py), [model verifier](marshawn_workstation_backup_20261111/verify_models.py), and [dataset evacuation](marshawn_workstation_backup_20261111/evacuate_data24.py) are retained with the evidence.

### WS07 bulk transport diagnosis, 2026-11-12

Receiver `/proc/<tar>/io` counters increased across13seconds: approximately30–55KiB/s per existing WS07 stream. These are live transfers, not dead handles. Source TCP statistics showed substantial retransmission (about6% of sent bytes on sampled long connections) and small congestion windows. The WS08 stream continued at a higher rate.

Read-only binary probes on calm-otter measured the WS07 leg. Every completed probe returned the exact requested zero bytes; none changed training or source files.

| Probe | Payload | Seconds |
| --- | ---: | ---: |
| Default SSH | 1MiB | 12.55 |
| IPQoS=none | 1MiB | 20.80 |
| TCP MSS960 | 1MiB | 18.71 |
| TCP MSS512 | 1MiB | 27.53 |
| Raw PTY | 1MiB | 3.93 |
| Default SSH confirmation | 4MiB | 44.73 |
| Raw PTY confirmation | 4MiB | 52.01 |

The small raw-PTY result did not reproduce at4MiB; no transport variation was deployed. Existing SHA-verified transfer services continue. This experiment does not justify killing live transfers or changing system-wide network settings.
