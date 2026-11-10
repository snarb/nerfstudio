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

### Thirteenth verified model:1116

Frame1116 completed150000 iterations and selected its150000-step cap10 export. Independent streamed SHA/recipe/AABB/all72-train/cold-start/budget checks passed. Export SHA:`2043f5af8d6f9d7a0449d7722effb42b5229d44415df1499524f6ebf9396ad79`. In-sample actor metrics: PSNR32.762860dB, SSIM0.802186, LPIPS0.131044. WS08 advanced automatically to priority frame1123.

Six FullHD RGB and RGBA views (CB/GC/NB calibrated and0.25interpolated) were rendered from this exact export and visually inspected. Prominent contour halos and background fragments remain. Pose and actor layout appear consistent, but these views do not prove a clean matte or independent AABB containment. [Export audit and visual evidence](marshawn_908_1268_diagnostics/frame1116_final_review/visual_inspection.json), [contact sheet](marshawn_908_1268_diagnostics/frame1116_final_review/final_contact.jpg), [Drive review](https://drive.google.com/file/d/1bsSNj1fIQEsX77pETCjn2wjxlIC1rG4c/view). Totals:13 verified and visually reviewed exports out of361;2/72 sparse24 frames complete (1116 is not on the sparse cadence).

### Fourteenth verified model:913

Frame913 completed150000 iterations and selected the150000-step cap10 export. Streamed SHA, recipe/AABB, all72train views, cold start and attempt-budget checks passed. Model SHA:`4afdca7e52fdc0088d91d5c1f6575fe31645bab3442ab23d058358f79f5c2731`. In-sample actor metrics: PSNR32.663527dB, SSIM0.802644, LPIPS0.115083. WS07 automatically advanced to priority frame988.

Six FullHD RGB/RGBA views (three calibrated and three0.25interpolated) were rendered from the verified export and visually inspected. Broad contour halos and background/floor fragments persist. [Audit and visual evidence](marshawn_908_1268_diagnostics/frame913_final_review/visual_inspection.json), [contact sheet](marshawn_908_1268_diagnostics/frame913_final_review/final_contact.jpg), [Drive review](https://drive.google.com/file/d/1_cWlfmGR-SavMvKNXW26GbuwHnTjOsgd/view). Totals:14 independently verified and visually reviewed models out of361, including3/72 priority frames908,913,918.

### Frame1123 initial visual/mask check

WS08's next sparse24 frame1123 reached8000 iterations. Cap10 in-sample actor metrics: PSNR30.848111dB, SSIM0.740405, LPIPS0.243612. The initial default-cap GC comparison was visually inspected: actor poses agree, with broad surrounding background bands and contour halos. Inspection of the actual GC source soft mask confirms wide gray transition bands and rectangular seams. This supports a mask contribution without proving all artifact causes. Frozen production inputs remain unchanged. [Inspection and mask SHA](marshawn_908_1268_diagnostics/frame1123_initial_review/inspection.json), [GC comparison](marshawn_908_1268_diagnostics/frame1123_initial_review/GC_8k_default_review.jpg), [source alpha preview](marshawn_908_1268_diagnostics/frame1123_initial_review/GC_sourcealpha_preview.png).

### Frame988 initial visual/mask check

The initial8000-step default-cap GC comparison was visually inspected. Poses agree with source RGB; early detail blur, contour halos and floor/background fragments remain. The actual source soft mask shows broad gray bands and rectangular seams, including around moving limbs. This plausibly contributes to artifacts without isolating every cause. Rectangular diagnostic query-domain edges are not evidence of AABB clipping. Inputs and frozen recipe were not changed. [Inspection and source-mask SHA](marshawn_908_1268_diagnostics/frame988_initial_review/inspection.json), [GC comparison](marshawn_908_1268_diagnostics/frame988_initial_review/GC_8k_default_review.jpg), [source alpha preview](marshawn_908_1268_diagnostics/frame988_initial_review/GC_sourcealpha_preview.png).

### Fifteenth verified model:1106

Clever frame1106 completed150000 iterations and selected its150000-step cap10 export. The full frame archive arrived onSilver and passed independent streamed SHA, recipe/AABB, all72TRAIN, cold-start and attempt-budget checks. ModelSHA:`bb24be7c52fa6916e020e5e49facafa2c0720bdd8f2304d5fd40611dd230695f`. In-sample actor metrics: PSNR32.706726dB, SSIM0.799378, LPIPS0.127070. Clever advanced automatically to sparse24 frame1248.

Six FullHD RGB/RGBA views (CB/GC/NB calibrated and0.25interpolated) were rendered from the exact verified archive usingSilver's qualified inference runtime; both render receipts match the modelSHA. Visually inspected actor layout is consistent, but prominent contour/hair halos and unsupported background/floor fragments persist. [Audit and visual evidence](marshawn_908_1268_diagnostics/frame1106_final_review/visual_inspection.json), [contact sheet](marshawn_908_1268_diagnostics/frame1106_final_review/final_contact.jpg), [Drive review](https://drive.google.com/file/d/1e-0V0XF6pSbbkLVOz8ExPIe08h9y3L3h/view). Totals:15 independently verified and visually reviewed models out of361, including3/72 sparse24 frames (1106 is outside that cadence).

### Sixteenth verified model: 925

Completed 150,000 iterations; selected 150,000, cap 10. All 72 cameras train; metrics are in-sample: PSNR 32.5823 dB, SSIM 0.799624, equal-person/equal-view LPIPS 0.118150. Inference SHA256 `68978d28760d83fda3a54d7cc59bd46f4789c903200456356cd4c709c117b33e`. Streamed export audit passed; six final FullHD camera/novel views inspected, render receipts match the exported model. Contour halos and background/floor fragments persist. Evidence: [review](marshawn_908_1268_diagnostics/frame925_final_review/final_contact.jpg), [audit](marshawn_908_1268_diagnostics/frame925_final_review/export_audit.json). Silver GPU4 automatically advanced to sparse24 frame948. Total verified and reviewed: 16/361; sparse24 remains 3/72 because925 is outside the sparse cadence.

Frame925 six-view final review uploaded via Google Drive connector: https://drive.google.com/file/d/1IVEuwlu2CG9AJfQ2ujDTJSVLIQVeFiK4/view .

### Seventeenth verified model: 929

Completed and selected 150,000 iterations, cap10. All72 cameras train; in-sample PSNR32.6221 dB, SSIM0.800991, actor LPIPS0.121128. Export SHA256 `42d96d05a090f8812deb173c8182fa795bda9669b479238ed5c31ce37eb11a57` verified. Six FullHD camera/novel views inspected; matching render receipts saved. Contour halos and background/floor fragments remain. Evidence: [review](marshawn_908_1268_diagnostics/frame929_final_review/final_contact.jpg), [audit](marshawn_908_1268_diagnostics/frame929_final_review/export_audit.json). GPU7 advanced automatically to sparse24 frame963. Verified and reviewed17/361, sparse24 remains3/72.

### Eighteenth verified model: 920

Completed150,000 iterations, selected144,000 at cap10. All72 cameras train; in-sample PSNR31.8691 dB, SSIM0.776448, actor LPIPS0.121228. Export SHA256 `4ed578e36e9709d0e103c8d6ce0ced8c3ea71163aec2d7ead9ff37a564aa5716` verified. Six FullHD camera/novel views inspected with matching receipts. Prominent contour halos and background/floor fragments persist. Evidence: [review](marshawn_908_1268_diagnostics/frame920_final_review/final_contact.jpg), [audit](marshawn_908_1268_diagnostics/frame920_final_review/export_audit.json). Verified and reviewed18/361.
