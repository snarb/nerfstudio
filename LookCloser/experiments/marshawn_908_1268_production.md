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

### Nineteenth verified model: 921

Completed150,000 iterations, selected150000, cap10. All72 cameras train; in-sample PSNR31.8378 dB, SSIM0.774087, actor LPIPS0.124900. Export SHA256 `a2cede0f24ee51c5689eab568edb2419376b0955947c7189bbc1b94b7b5a10cb` verified. Six FullHD camera/novel views inspected with matching receipts. Prominent contour halos and background/floor fragments remain. Evidence: [review](marshawn_908_1268_diagnostics/frame921_final_review/final_contact.jpg), [audit](marshawn_908_1268_diagnostics/frame921_final_review/export_audit.json).

### Twentieth verified model: 923

Completed150,000 iterations, selected150000, cap10. All72 cameras train; in-sample PSNR32.5700 dB, SSIM0.798978, actor LPIPS0.115894. Export SHA256 `8fb88faa19fb3bb6a7e98b0b5246540c7ef2a76f5cc25a051e4b4ed2c767aac0` verified. Six FullHD camera/novel views inspected with matching receipts. Prominent contour halos and background/floor fragments remain. Evidence: [review](marshawn_908_1268_diagnostics/frame923_final_review/final_contact.jpg), [audit](marshawn_908_1268_diagnostics/frame923_final_review/export_audit.json).

### Twenty-first verified model: 924

Completed150,000 iterations, selected150000, cap10. All72 cameras train; in-sample PSNR31.7869 dB, SSIM0.767708, actor LPIPS0.125324. Export SHA256 `e6eb793ac0f1b8b24b8a06e677f975fdfc121099a1745fd96ef7b27503b8d7e0` verified. Six FullHD camera/novel views inspected with matching receipts. Contour halos and background/floor fragments remain. Evidence: [review](marshawn_908_1268_diagnostics/frame924_final_review/final_contact.jpg), [audit](marshawn_908_1268_diagnostics/frame924_final_review/export_audit.json). Verified and reviewed21/361; sparse24 verified and reviewed4/72.

### Twenty-second verified model: 928

Completed150,000 iterations, selected150000, cap10. All72 cameras train; in-sample PSNR32.5065 dB, SSIM0.801264, actor LPIPS0.113716. Export SHA256 `8fc7e7c5f9eee38d01c055893faa5e10bae4710c66109f76e739bb7704f89c61` verified. Six FullHD camera/novel views inspected with matching receipts. Contour halos and background/floor fragments remain; bat-specific geometry investigation remains open, so this is not defect-free-prop certification. Evidence: [review](marshawn_908_1268_diagnostics/frame928_final_review/final_contact.jpg), [audit](marshawn_908_1268_diagnostics/frame928_final_review/export_audit.json). Verified and reviewed22/361; sparse24 reviewed5/72.

### Twenty-third verified model: 926

Completed150,000 iterations, selected150000, cap10. All72 cameras train; in-sample PSNR32.5937 dB, SSIM0.800304, actor LPIPS0.119486. Export SHA256 `df4c41ac43f99c5722090a362c0323bd39861c2c79144b456b98f8b289af8891` verified. Six FullHD camera/novel views inspected with matching receipts. Contour halos and background/floor fragments remain; bat-specific defect investigation is open. Evidence: [review](marshawn_908_1268_diagnostics/frame926_final_review/final_contact.jpg), [audit](marshawn_908_1268_diagnostics/frame926_final_review/export_audit.json). Reviewed23/361; sparse24 reviewed5/72.

### Twenty-fourth verified model: 930

Completed150,000 iterations, selected150000, cap10. All72 cameras train; in-sample PSNR32.5330 dB, SSIM0.800252, actor LPIPS0.116981. Export SHA256 `f0b68e05d5336900584bf9dcbf4f7426b62a21c7cffa4b4e362cafcec0966a45` verified. Six FullHD camera/novel views inspected with matching receipts. Contour halos and background/floor fragments remain; bat-specific defect investigation is open. Evidence: [review](marshawn_908_1268_diagnostics/frame930_final_review/final_contact.jpg), [audit](marshawn_908_1268_diagnostics/frame930_final_review/export_audit.json). Reviewed24/361; sparse24 reviewed5/72. Olive automatically advanced to953.

### Twenty-fifth verified model: 1123

Completed150,000 iterations, selected150000, cap10. All72 cameras train; in-sample PSNR32.4329 dB, SSIM0.803660, actor LPIPS0.142440. Export SHA256 `248f39a3fa67ec785219b33921659ceac1eef4f23c32eacf462530d788a75626` verified. Six FullHD camera/novel views inspected with matching receipts. Contour halos, floor and detached overhead/background fragments remain; hard-admission prop defect investigation is open. Evidence: [review](marshawn_908_1268_diagnostics/frame1123_final_review/final_contact.jpg), [audit](marshawn_908_1268_diagnostics/frame1123_final_review/export_audit.json). Reviewed25/361; sparse24 reviewed6/72. WS08 advanced1148.

### Twenty-sixth verified model: 988

Completed150,000 iterations, selected150000, cap10. All72 cameras train; in-sample PSNR32.5857 dB, SSIM0.796479, actor LPIPS0.145550. Export SHA256 `74698bf954785a3d5203928034e4f6a39b323716fc59ef055547c725e3763a08` verified. Six FullHD camera/novel views inspected with matching receipts. Halos, floor/background fragments and incomplete GC bat reconstruction remain; this is not defect-free geometry certification. Evidence: [review](marshawn_908_1268_diagnostics/frame988_final_review/final_contact.jpg), [audit](marshawn_908_1268_diagnostics/frame988_final_review/export_audit.json). Reviewed26/361; sparse24 reviewed7/72. WS07 advanced993.

### Frame 1248 final review — 2026-11-10

Independent 150,000-step model exported and audited; actor PSNR 32.45573, SSIM 0.794881, LPIPS 0.137011. Reviewed CB/GC/NB and three interpolated Full HD views. Four performers remain visible; contour halos and background fragments persist. Thin prop integrity is not certified by the contact sheet. Model SHA256 `4fd59441329694ab6d311a1e3781c67ea4905c75bdb95f263739856f5c2890a3`. [Evidence](marshawn_908_1268_diagnostics/frame1248_final_review/visual_inspection.json), [six views](marshawn_908_1268_diagnostics/frame1248_final_review/final_contact.jpg). Total reviewed exports: 27; sparse 24 fps frames: 8/72, including legacy 923/988 pending corrected versions. Clever-shadow advanced to priority frame 1258.

### Frame 948 final review — 2026-11-10

Export audited and six Full HD views reviewed. Metrics: {'psnr': 32.94952883371073, 'ssim': 0.8092843476268979, 'lpips': 0.13353320894141993}. Selected iterations 150000. Four performers visible; bat appears continuous at contact-sheet scale. Halos and floating background/floor fragments persist. [Evidence](marshawn_908_1268_diagnostics/frame948_final_review/visual_inspection.json), [views](marshawn_908_1268_diagnostics/frame948_final_review/final_contact.jpg). Reviewed total 28, sparse24 9/72; corrected923/988 pending.

### Frame 933 final review — 2026-11-10

Audited independent export, selected iterations 150000. Actor metrics: {'psnr': 32.86719777516944, 'ssim': 0.8096945716275109, 'lpips': 0.12718014832999971}. All six views reviewed; performers visible, contour halos and background/floor fragments persist. [Evidence](marshawn_908_1268_diagnostics/frame933_final_review/visual_inspection.json), [views](marshawn_908_1268_diagnostics/frame933_final_review/final_contact.jpg).

### Frame 943 final review — 2026-11-10

Audited independent export, selected iterations 150000. Actor metrics: {'psnr': 32.890083066041775, 'ssim': 0.8113675779766507, 'lpips': 0.13845400429434246}. All six views reviewed; performers visible, contour halos and background/floor fragments persist. [Evidence](marshawn_908_1268_diagnostics/frame943_final_review/visual_inspection.json), [views](marshawn_908_1268_diagnostics/frame943_final_review/final_contact.jpg).

### Frame 958 final review — 2026-11-10

Audited independent export, selected iterations 150000. Actor metrics: {'psnr': 32.87865943734685, 'ssim': 0.8071454879310397, 'lpips': 0.12992331820229688}. All six views reviewed; performers visible, contour halos and background/floor fragments persist. [Evidence](marshawn_908_1268_diagnostics/frame958_final_review/visual_inspection.json), [views](marshawn_908_1268_diagnostics/frame958_final_review/final_contact.jpg).

### Frame 978 final review — 2026-11-10

Audited independent export, selected iterations 150000. Actor metrics: {'psnr': 32.63079192765056, 'ssim': 0.7984894663095474, 'lpips': 0.12992210375765958}. All six views reviewed; performers visible, contour halos and background/floor fragments persist. [Evidence](marshawn_908_1268_diagnostics/frame978_final_review/visual_inspection.json), [views](marshawn_908_1268_diagnostics/frame978_final_review/final_contact.jpg).

After these reviews: 32 reviewed frame exports, 13/72 sparse24 frame IDs. Legacy923/988 corrected versions still pending. Scope remains361 independent frame models.

### Frame 963 final review — 2026-11-10

Audited independent export, selected iterations 150000. Actor metrics: {'psnr': 32.613140365448935, 'ssim': 0.8046275476614634, 'lpips': 0.1209171125665307}. CB/GC/NB and three interpolated views inspected. Performers visible; no evident box truncation. Halos and background/floor fragments persist. Thin props not certified from thumbnail. [Evidence](marshawn_908_1268_diagnostics/frame963_final_review/visual_inspection.json), [views](marshawn_908_1268_diagnostics/frame963_final_review/final_contact.jpg).

### Frame 983 final review — 2026-11-10

Audited independent export, selected iterations 150000. Actor metrics: {'psnr': 32.6591784506632, 'ssim': 0.7985668728748957, 'lpips': 0.13769685725371042}. CB/GC/NB and three interpolated views inspected. Performers visible; no evident box truncation. Halos and background/floor fragments persist. Thin props not certified from thumbnail. [Evidence](marshawn_908_1268_diagnostics/frame983_final_review/visual_inspection.json), [views](marshawn_908_1268_diagnostics/frame983_final_review/final_contact.jpg).

### Frame 1148 final review — 2026-11-10

Audited independent export, selected iterations 150000. Actor metrics: {'psnr': 32.89655164239553, 'ssim': 0.8125707258780798, 'lpips': 0.1284052688214514}. CB/GC/NB and three interpolated views inspected. Performers visible; no evident box truncation. Halos and background/floor fragments persist. Thin props not certified from thumbnail. Broad floating horizontal background fragments above actors are conspicuous in real and novel views; retained as visible quality limitation, not clean alpha. [Evidence](marshawn_908_1268_diagnostics/frame1148_final_review/visual_inspection.json), [views](marshawn_908_1268_diagnostics/frame1148_final_review/final_contact.jpg).

Reviewed total35, sparse24 16/72; corrected923/988 pending. Frame1258 actual immutable binding and training dataset confirmed votes3 (recipe bcaec09a…), first end-to-end main-fleet cold start using repaired admission.

### Frame 993 final review — 2026-11-10

Audited independent export, selected iterations 150000. Actor metrics: {'psnr': 32.58373661407237, 'ssim': 0.7948969006538391, 'lpips': 0.15934432173768678}. Six views inspected; four performers visible, halos and background/floor fragments persist. Thin-prop integrity not certified from thumbnail. [Evidence](marshawn_908_1268_diagnostics/frame993_final_review/visual_inspection.json), [views](marshawn_908_1268_diagnostics/frame993_final_review/final_contact.jpg). Total reviewed36, sparse24 17/72, repaired923/988 still pending.

### Frame 953 final review — 2026-11-10

Audited independent export, selected iterations 150000. Actor metrics: {'psnr': 33.02775761470562, 'ssim': 0.8119279791911443, 'lpips': 0.1329596479319864}. Six views inspected; four performers visible, halos and background/floor fragments persist. [Evidence](marshawn_908_1268_diagnostics/frame953_final_review/visual_inspection.json), [views](marshawn_908_1268_diagnostics/frame953_final_review/final_contact.jpg). Reviewed37 total, sparse24 18/72; repaired923/988 pending.

### Frame 1038 final review — 2026-11-10

Audited independent export, selected iterations 150000. Actor metrics: {'psnr': 32.597246172190154, 'ssim': 0.7923610111077627, 'lpips': 0.14869357893864313}. Six views inspected; four performers visible, bat broadly continuous in GC, halos and background/floor fragments persist. [Evidence](marshawn_908_1268_diagnostics/frame1038_final_review/visual_inspection.json), [views](marshawn_908_1268_diagnostics/frame1038_final_review/final_contact.jpg). Reviewed38 total, sparse24 19/72; repaired923/988 pending. Frame938 automatic retry confirmed running (step9608).

### Frame 1258 final review — 2026-11-10

First audited main-queue export trained cold with the three-vote admission recipe; selected iterations 144000. Actor metrics: {'psnr': 32.41546822508791, 'ssim': 0.7950127555264367, 'lpips': 0.1273352113655872}. Six Full HD views inspected; performers visible, contour halos and sparse background fragments persist. [Evidence](marshawn_908_1268_diagnostics/frame1258_final_review/visual_inspection.json), [views](marshawn_908_1268_diagnostics/frame1258_final_review/final_contact.jpg). Reviewed39 total, sparse24 20/72; repaired923/988 pending.

### Frame 1163 final review — 2026-11-10

Audited independent export. Actor PSNR 32.80689, SSIM 0.809141, LPIPS 0.123702. Four performers visible in six Full HD views. Contour halos and conspicuous floating background fragments persist. [Evidence](marshawn_908_1268_diagnostics/frame1163_final_review/visual_inspection.json), [views](marshawn_908_1268_diagnostics/frame1163_final_review/final_contact.jpg). Reviewed 40 total and 21/72 on the 24 fps grid.

### Frame 998 final review

Audited independent export at 144000 iterations. Actor metrics {'psnr': 32.539058220557756, 'ssim': 0.7949514587720236, 'lpips': 0.1605624184012413}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame998_final_review/final_contact.jpg).

### Frame 1033 final review

Audited independent export at 144000 iterations. Actor metrics {'psnr': 32.53308881242764, 'ssim': 0.7905687093734741, 'lpips': 0.14621592064698538}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1033_final_review/final_contact.jpg).

### Frame 1048 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.560969255465814, 'ssim': 0.7896863718827566, 'lpips': 0.14435973080495992}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1048_final_review/final_contact.jpg).

Reviewed 43 model exports, including 24 of 72 priority frames. Corrected versions of 923 and 988 are pending.

### Frame 1043 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.61882428359678, 'ssim': 0.7919205576181412, 'lpips': 0.14644072080651918}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1043_final_review/final_contact.jpg).

### Frame 1013 final review

Audited independent export at 144000 iterations. Actor metrics {'psnr': 32.466334541832964, 'ssim': 0.7917343278725942, 'lpips': 0.1448110161970059}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1013_final_review/final_contact.jpg).

### Frame 968 final review

Audited independent export at 144000 iterations. Actor metrics {'psnr': 32.67955525757292, 'ssim': 0.8025645464658737, 'lpips': 0.11468589895715316}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame968_final_review/final_contact.jpg).

### Frame 1018 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.48664466626032, 'ssim': 0.7922541201114655, 'lpips': 0.13859394068519273}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1018_final_review/final_contact.jpg).

### Frame 973 final review

Audited independent export at 144000 iterations. Actor metrics {'psnr': 32.63195596100858, 'ssim': 0.8010961016019186, 'lpips': 0.1132686057438453}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame973_final_review/final_contact.jpg).

### Frame 1053 final review

Audited independent export at 144000 iterations. Actor metrics {'psnr': 32.61182941101117, 'ssim': 0.790805384516716, 'lpips': 0.1396927752842506}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1053_final_review/final_contact.jpg).

### Frame 1263 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.37120870275098, 'ssim': 0.7938785403966904, 'lpips': 0.12772330755574834}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1263_final_review/final_contact.jpg).

### Frame 1188 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.71634331877562, 'ssim': 0.8063778926928838, 'lpips': 0.1260767701185412}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1188_final_review/final_contact.jpg).

### Frame 1073 final review

Audited independent export at 144000 iterations. Actor metrics {'psnr': 32.693170872427515, 'ssim': 0.795014942685763, 'lpips': 0.12852260284125805}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1073_final_review/final_contact.jpg).

### Frame 1113 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.808582697664896, 'ssim': 0.8013776292403539, 'lpips': 0.1389852627283997}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1113_final_review/final_contact.jpg).

### Frame 1093 final review

Audited independent export at 144000 iterations. Actor metrics {'psnr': 32.7151704563641, 'ssim': 0.7980637649695078, 'lpips': 0.12587839473659793}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1093_final_review/final_contact.jpg).

### Frame 1088 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.6995867270635, 'ssim': 0.7973837107419968, 'lpips': 0.12178522255271673}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1088_final_review/final_contact.jpg).

### Frame 1023 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.545721997992416, 'ssim': 0.7931173493464788, 'lpips': 0.1361683364957571}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1023_final_review/final_contact.jpg).

### Frame 1003 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.531059685671, 'ssim': 0.792578806479772, 'lpips': 0.15808398587008318}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1003_final_review/final_contact.jpg).

### Frame 1028 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.567322458874045, 'ssim': 0.7927231540282568, 'lpips': 0.13976100335518518}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1028_final_review/final_contact.jpg).

### Frame 1008 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.5299525928969, 'ssim': 0.7935556968053182, 'lpips': 0.15480650402605534}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1008_final_review/final_contact.jpg).

### Frame 938 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.896946437026735, 'ssim': 0.8098267631398307, 'lpips': 0.13171767443418503}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame938_final_review/final_contact.jpg).

### Frame 1098 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.72450275928244, 'ssim': 0.801527147491773, 'lpips': 0.12127740603561203}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1098_final_review/final_contact.jpg).

### Frame 1078 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.71407243338711, 'ssim': 0.7974197516838709, 'lpips': 0.12428967375308275}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1078_final_review/final_contact.jpg).

### Frame 1203 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.35843142353453, 'ssim': 0.799080568883154, 'lpips': 0.12843881630235246}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1203_final_review/final_contact.jpg).

### Frame 1253 final review

Audited independent export at 144000 iterations. Actor metrics {'psnr': 32.437714188130535, 'ssim': 0.7946743485000399, 'lpips': 0.1337661591048042}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1253_final_review/final_contact.jpg).

### Frame 1208 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.41903337102762, 'ssim': 0.7997766418589486, 'lpips': 0.12938686760349405}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1208_final_review/final_contact.jpg).

### Frame 1173 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.85323954944465, 'ssim': 0.8105226589573754, 'lpips': 0.12541806868587932}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1173_final_review/final_contact.jpg).

### Frame 1068 final review

Audited independent export at 144000 iterations. Actor metrics {'psnr': 32.65147884304883, 'ssim': 0.7942131658395132, 'lpips': 0.1310582741474112}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1068_final_review/final_contact.jpg).

### Frame 1058 final review

Audited independent export at 144000 iterations. Actor metrics {'psnr': 32.561937408184285, 'ssim': 0.79100068906943, 'lpips': 0.13666661766668162}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1058_final_review/final_contact.jpg).

### Frame 1153 final review

Audited independent export at 144000 iterations. Actor metrics {'psnr': 32.93148448397283, 'ssim': 0.8142245494657092, 'lpips': 0.12350408505234452}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1153_final_review/final_contact.jpg).

### Frame 1143 final review

Audited independent export at 150000 iterations. Actor metrics {'psnr': 32.77877265194895, 'ssim': 0.8117487049765058, 'lpips': 0.12381116725090476}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1143_final_review/final_contact.jpg).

### Frame 1063 final review

Audited independent export at 128000 iterations. Actor metrics {'psnr': 32.60621491938653, 'ssim': 0.7921509097019831, 'lpips': 0.13268741282323995}. Six Full HD views inspected. Four performers visible without obvious box truncation. Contour halos and floating background fragments persist. [Views](marshawn_908_1268_diagnostics/frame1063_final_review/final_contact.jpg).
