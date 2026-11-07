# Marshawn video storage and input sharing, 2026-11-07

## What was tested

Read-only scan of both source campaign `marshawn_video_1660_2015_24fps_20261107` and capped model campaign `marshawn_video_1660_2015_24fps_140k_20261107` on shared FSX. Counted allocated bytes once per `(device, inode)`, inspected native PNG symlinks, and SHA-checked possible copies against canonical source PNGs. Original TIFF storage was not traversed or changed.

Evidence: [inventory and all file metadata](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/capacity_20261107/inventory.json), [capacity forecast](/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/capacity_20261107/forecast.json). Inventory SHA256: `3f928607e30d0d1faa1479f9bb15dfcad97729aed41eb1ca5915cfb71c1e9e55`.

## Results

| Measurement | Result |
|---|---:|
| Unique allocated bytes across source and capped campaigns | 353.57 GiB |
| FSX free at completed scan | 208.93 GiB |
| Current frame 1690 native/task/input PNG symlink paths | 204 |
| Verified duplicate full native PNG bytes for 1690 | 0 |
| SHA-identical PNG copies across both campaigns, unique candidate inodes | 341 / 1.095 GiB |
| Current 1690 operation artifacts, unique allocated bytes | 0.132 GiB |

The future canonical native source PNGs were already allocated before this scan. They must not be added a second time to remaining-growth forecasts. Current native tasks point back to these canonical files; packing frame 1690 did not duplicate another full 66-view native raster set.

Most detected PNG copies are the intentionally materialized, owned training RGB snapshots in the capped campaign. Replacing them with links would reclaim little space and change ownership semantics of sealed training inputs. No files were deleted, moved, or relinked.

Two retained checkpoints occupy approximately 3.905 GiB apparent bytes per locally trained frame. If 20% of the 66 remaining frames land on the two shared-FSX GPU hosts, rounding up to 14 models gives a 54.7 GiB checkpoint scenario. This is a planning assumption, not a placement quota: placing all 66 on FSX would instead require about 257.7 GiB for checkpoints alone. Actual inode allocation can be lower than apparent checkpoint size.

The render owner measured the first complete compressed geometry package: 36 files, 92,773,290 logical bytes and 98,646,016 allocated bytes. Seventy-two equal packages would total 6.22 GiB logical / 6.61 GiB allocated. The earlier estimate of 10 GB **per frame** was incorrect. No geometry-cache eviction is warranted from that estimate. Shape-based upper bounds and PNG/intermediate output growth remain separate checks owned by the renderer.

## Insights and next steps

Continue storage supervision as models, source evidence, and renders accumulate. The immediate forecast fits the measured headroom under approximately balanced host placement. Keep the original TIFFs, selected checkpoints, source proofs, and renders. If later measurements show genuine pressure, qualify an owned-cache archival plan with SHA verification and hydration at original paths before any removal; do not silently alter sealed inputs.
