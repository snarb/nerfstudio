# Recovery audit after reported silver-sable /cache loss

Live read-only checks found 20 exact selected model files outside Silver,
verified by full SHA256 against the committed 72-frame storage audit:

| Host | Selected models | Root |
| --- | ---: | --- |
| Workstation07 | 8 | `/home/pavlo/marshawn_campaign_20261109/frames` |
| Workstation08 | 7 | `/home/nvuser/lookcloser/campaigns/marshawn908_20261109/frames` |
| olive-lark / FSx | 5 | `/fsx/tmp/marshawn908_20261109/olive/frames` |

Each uses `<frame>/model/inference.ckpt`. WS07 also retains an older,
unselected version of frame988; its repaired selected hash has not been found.
52 selected checkpoints remain without an identified surviving copy.

Clever's original campaign has no completed inference exports in the scanned
root. `archive.relocate` verifies the destination model and then removes source
model/data/train directories. Thus the archive on Silver was the remaining
copy for those relocated artifacts. Dev3 has tuning models, not another verified
complete solve. The emergency workstation backup was on Silver's same `/cache`.
No independent complete archive has been identified. Infrastructure snapshots
have not been checked; their existence cannot be assumed.

Source PNG presence was checked for all 72 cameras across all 72 required
instants: 41 frames on WS07 and31 on WS08, no missing image files. This is a
presence audit, not a new checksum/decode of every source image. Git retains
recipe/code/calibration records; Drive retains delivered renders. Missing
weights can be retrained from surviving inputs, but renders do not restore them.

Exact surviving paths, hashes, frame IDs and missing selected IDs are recorded in
`marshawn_silver_cache_loss_recovery.json`. This audit did not start training or
copy model weights. Prioritize independent durable copies of surviving artifacts.
