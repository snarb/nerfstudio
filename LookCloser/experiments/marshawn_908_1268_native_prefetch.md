# Marshawn receiving-side native prefetch

## What was tested

Actual Silver second-frame transfers took roughly15–16minutes for approximately2.4GB of native crops per queue, after the previous model finished. Existing overlap prepared only source-side CPU data. Extend it to transfer one never-started future frame to each remote receiver during current training, with the frozen numerical recipe and source pixels preserved.

## Results

- `native_transfer.py` now serves both foreground staging and optional prefetch. One receiver lease serializes extraction, including a surviving tar child after its parent crashes. Frame/manifest/seed checks precede transfer commitment and scoped derivative-cache release.
- Four temporary-data tests passed: real tar byte parity/corruption detection, cached-source independence, actual extractor FD inheritance plus killed-parent retry, started-model protection, and no source-cache regeneration for ready receivers. The existing production orphan-worker recovery test also passed.
- Actual future1111 transfer on Clever passed72RGB and72semantic file SHA comparisons. It created no training spec, model receipt or neural state. The source derivative cache was released after acknowledgment.
- Deployed to five campaign roots. All four sealed numerical files remained byte-identical on each host. Controller reload preserved12actual worker process identities, the selected recipe and the original six-hour tuning start/deadline. [Deployment/canary proof](marshawn_908_1268_diagnostics/native_transfer_prefetch_audit.json).
- Current native-ready receivers are checked before the source, so released source caches are not regenerated. Optional failures leave normal foreground preparation available.

## Insights

Transfers can now finish during current training instead of waiting for the next frame. Actual end-to-end cycle savings must be measured at the next model transition; no SGD speedup is claimed. RGB decoding, crop resolution, masks, admission geometry, training splits, step budgets and neural code were not changed. Original RGB16 images, masks and persistent seed clouds are untouched. Queues retain their unique frame assignments.
