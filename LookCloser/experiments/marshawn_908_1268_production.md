# Marshawn908–1268 production export checks

## What was tested

Independent export verification and visual review after the six-hour tuning campaign. Frozen recipe: [selected recipe](marshawn_908_1268_tuned_recipe.md). Each final model trains independently on all72cameras, using native crops and the accepted common AABB. Metrics below are in-sample actor metrics, not held-out generalization.

## Results

Verified snapshot: 2026-11-10T08:16:13.149748+00:00. Twelve verified exports; the361-frame campaign remains incomplete.

| Frame | Selected iterations | PSNR(dB) | SSIM | Actor LPIPS |
|---|---:|---:|---:|---:|
| 908 | 144000 | 32.535930 | 0.799172 | 0.108058 |
| 909 | 150000 | 32.572324 | 0.801283 | 0.112001 |
| 910 | 150000 | 32.611592 | 0.801385 | 0.115309 |
| 911 | 150000 | 32.592926 | 0.799519 | 0.116112 |
| 912 | 150000 | 32.609488 | 0.800391 | 0.116158 |
| 914 | 150000 | 32.669783 | 0.802720 | 0.112677 |
| 915 | 144000 | 32.631765 | 0.801563 | 0.113487 |
| 916 | 144000 | 32.661553 | 0.801708 | 0.115050 |
| 918 | 150000 | 32.712914 | 0.803937 | 0.112944 |
| 919 | 150000 | 32.629584 | 0.800610 | 0.111737 |
| 1102 | 150000 | 32.668340 | 0.798791 | 0.140155 |
| 1112 | 144000 | 32.774396 | 0.798674 | 0.137925 |

The eight Silver models completed150000iterations. Frames915/916 selected144000; the other six selected150000. All eight model files passed streamed SHA verification, cold/native/split checks, and input-frame identity checks. [Batch audit](marshawn_908_1268_diagnostics/silver_first8_completed_model_audit.json). Earlier export receipts: [908](marshawn_908_1268_diagnostics/frame908_completed_model_audit.json), [1112](marshawn_908_1268_diagnostics/first_completed_model_audit.json), [1102](marshawn_908_1268_diagnostics/frame1102_completed_model_audit.json).

[Eight-page raw Full HD and novel-view contact PDF on Drive](https://drive.google.com/file/d/1zyLbyRfg_rgaYcr5VwDCOXKA76T3Z8xT/view). All48 views were visually inspected; each frame has CB/GC/NB calibrated views and0.25SLERP neighbour views,1920x1080 letterboxed output. Both render receipts for every model matched the independently verified selected model SHA. No GT masking was applied to these full-FOV field renders.

## Insights

Four performers remain represented, with source-FOV occlusion/cutoff where applicable. Significant contour halos and unsupported background fragments persist; these results do not establish a clean matte. Native head/source-mask diagnostic patches remain linked in the selected-recipe notes. Continue the frozen sequence campaign and audit each new export; choosing144000 uniformly would miss improvements observed at150000 in six of these eight frames.

Silver artifacts: `/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/frames/<frame>/model/inference.ckpt`; full-resolution review assets: `final_reviews/<frame>/{fullhd,novel}/`. The relocated Clever1102 model is under `external_models/clever/001102/model/`. All eight Silver queues advanced automatically to their next frames; use live controller state for current progress rather than this historical snapshot.

Olive919 completed150000iterations and selected150000 cap10. [Independent export audit](marshawn_908_1268_diagnostics/frame919_completed_model_audit.json); [six reviewed Full HD/novel views on Drive](https://drive.google.com/file/d/1xR9crAA1KfGktum3D01fAnj5aMBwVw12/view). Four performers remain represented; contour halos persist. Olive advanced automatically to930. Receiving-side native transfer overlap is documented in [prefetch checks](marshawn_908_1268_native_prefetch.md).

### Second Silver batch initial review

Frames920,921,923,924,925,926,928,929 passed initial visual inspection at8k cap10. [GC comparisons1](marshawn_908_1268_diagnostics/second_batch_GC_initial8000_page1.jpg), [GC comparisons2](marshawn_908_1268_diagnostics/second_batch_GC_initial8000_page2.jpg); [metric snapshot](marshawn_908_1268_diagnostics/second_batch_initial_review.json). Four performers remain represented; early blur/contour halos persist. These sampled ROI views do not establish full-FOV/AABB coverage. All72TRAIN; metrics are in-sample.

| Frame | PSNR(dB) | SSIM | Actor LPIPS |
|---|---:|---:|---:|
| 920 | 30.241199 | 0.714428 | 0.205532 |
| 921 | 30.350388 | 0.716889 | 0.210289 |
| 923 | 30.929717 | 0.738227 | 0.200044 |
| 924 | 30.276780 | 0.710189 | 0.207718 |
| 925 | 30.966095 | 0.738526 | 0.202163 |
| 926 | 30.967167 | 0.738635 | 0.205449 |
| 928 | 30.908933 | 0.739744 | 0.195961 |
| 929 | 30.938008 | 0.737273 | 0.209951 |

The export audit accepts the nominal target or a documented exhausted150000 reserved-attempt ceiling after crash replay. Premature completion, zero steps and over-budget values are rejected. Six boundary canaries and a real completed export passed this protocol check; the twelve current models all completed150000 iterations.

### 24 fps preview scheduling — 2026-11-10 01:12 PST

User requested earlier coverage of the whole three-second interval. All twelve live queues now prioritize the 72 frames `908 + 5*k`, `k=0..71` (908–1263), corresponding to 24 fps playback from the 120 fps source. Frame1268 remains in the dense remainder; it is the exclusive three-second endpoint for this preview.

Only ordering changed: job descriptors, ownership, recipe, numerical runtime and active process identities remain unchanged. Backups were saved beside each queue. Readback verified exactly361 unique frames908–1268 and sparse-first order among unstarted jobs on every worker. Existing twelve jobs continue uninterrupted. Two priority models (908,918) are already verified complete; three (913,923,928) are currently training, with67 not yet started. Workers return to their dense remainder after their own sparse jobs, avoiding idle GPUs while other workers finish sparse coverage. This is per-worker priority rather than a global barrier; no completion-time improvement is claimed before measurement.

Receipt: [sparse24_queue_applied.json](marshawn_908_1268_diagnostics/sparse24_queue_applied.json). Receiver prefetch follows the updated queue order automatically; previously prefetched frames remain on their original owners.

### Frame930 intermediate mask inspection

The GC cap10 review at15999 steps shows foreground detail with broad contour halos and surrounding floor/background fragments. Inspection of the actual source soft mask confirms broad gray transition bands and rectangular seams around the actor silhouettes. This supports a mask contribution to contour artifacts, but does not establish that masks explain all unsupported opacity. The frozen production recipe and data remain unchanged during this diagnostic. Evidence and mask SHA: [inspection](marshawn_908_1268_diagnostics/frame930_mask_review/inspection.json), [source alpha preview](marshawn_908_1268_diagnostics/frame930_mask_review/GC_sourcealpha_preview.png), [intermediate render comparison](marshawn_908_1268_diagnostics/frame930_mask_review/GC_16k_review.jpg). This is an early in-sample review, not a final quality gate.

### Same-host priority balancing — 2026-11-10

Measured first-frame wall times were4.38–4.63h on WS07/08,5.92–6.00h per Silver GPU,5.31h on Clever and6.57h on Olive. Initial sparse-first queues left a roughly45h critical path. Five unstarted, unstaged priority jobs were exchanged with unstarted dense-tail jobs between Silver GPUs, retaining29 jobs per queue. Transfers already in progress and all active trainers were preserved. Readback verified361 unique frames, unchanged active process identities, and identical job descriptors except the assigned GPU environment. Silver queues now have5–6 unstarted priority jobs each. A rough critical-path projection falls to about39h (allow35–45h for variation), excluding final rendering/video assembly; this is a forecast, not a measured speedup. Other hosts retain source ownership and their existing assignments.

Receipt: [balanced queues](marshawn_908_1268_diagnostics/sparse24_queue_balanced.json). The live queue files are authoritative for allocation; the original controller allocation snapshot predates operational swaps.
