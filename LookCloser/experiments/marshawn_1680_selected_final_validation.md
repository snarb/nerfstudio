# Marshawn1680 selected final validation

What was checked: genuine DEV3 training completion at140000 updates, followed by
CPU validation of the actual export-selected checkpoint and its own saved
runtime occupancy. No new rendering, GPU query, source modification, R1 grid
promotion or checkpoint-selection change was performed. Heavy CPU work began
only after1690's actual first gradient was independently confirmed.

Training ended at step139999. The `actor_lpips` export selector also chose
step139999; these were checked separately. Selected checkpoint SHA:
`11d4ff0c6e83e8aa5e0331f24fc7aebc9c552aca5ff1266a835d6912a3ff3e44`.

| Selected actor metrics, DC/HC/KA training views | Value |
|---|---:|
| PSNR |47.14889 dB|
| SSIM |0.993502|
| LPIPS |0.00062119|

The actual saved runtime snapshot has37,142 core/active cells and433,722 admission
cells. Its checkpoint/source bindings and unchanged hashes were verified.
The exact CPU audit previously used for1675 traversed all66 cameras and both
actors, covering163,588 nominal contour rays per grid:

| Grid | Missed contour rays |
|---|---:|
| Source base |0|
| Source pad1 |0|
| Source mapped |0|
| Saved runtime core |0|
| Saved runtime active |0|
| Saved runtime admission |0|

[HC/DC/KA actual GT and existing trainer PNG contact](/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/contour_runtime_001680/selected_actor_contact.jpg).
Manual child-agent inspection found coherent colors, clothing, hands and boots.
Substantial noisy floor/background shell remains visible. This is not a clean
cutout or a certificate of physically correct empty/occupied cells. Nominal mask
contour retention does not prove complete native-mask recall, correct hidden
surface depth or full pixel-cone coverage. No background cleanup was applied.

The source-bound summary and contact binding are at
`/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/operations/contour_runtime_001680/{selected_runtime_compact_report,contact_binding}.json`.
Snapshot SHA:
`e7e4cf94a59b954e17c45675648321c816e8de31efb59d26a0ee2df072928751`.
Dense report SHA:
`a59c2a9a7de6a19dfaf493fde83e6d81a87e08af4bdc9c8f4c0374988005f203`.
The contact was sent to ROOT for required personal visual review. The72-model
production recipe, source data, runtime R0 policy and final encoding workflow
remain unchanged.
