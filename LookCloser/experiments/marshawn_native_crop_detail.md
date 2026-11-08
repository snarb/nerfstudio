# Marshawn 1660: native actor crops and adaptive marching

## What was tested

User direction on 2026-11-08: stop the video campaign and improve first-frame LookCloser detail using original-resolution actor crops, fewer iterations when quality permits, and better adaptive marching. Use all ten available GPUs (olive-lark: one Blackwell, silver-sable: eight Blackwells, dev3: one L40S). Do not restart the old video jobs.

All old campaign controllers, workers and automatic continuations were stopped before starting this work. The stop receipt is `/fsx/tmp/marshawn_fleet10_140k_gcrole_g1_20261108/operations/user_stop_all_20261108_0954/complete.json` (SHA256 `32f74f35cb1fc07a8529c5ce72e4f5bd9a78df42fba39822ec8bcc50423b8787`). Existing models and renders remain available.

New isolated campaign: `/fsx/tmp/marshawn_native_crop_tuning_20261108`. Source preparation and overlays do not modify the frozen video implementation or historical data.

## Results

### Source preparation and verification

The actual originals are 8192×4320 RAW TIFFs. All 66 native actor-union crops were prepared in 69 seconds using eight CPU workers. Crop widths are 1208–1944 pixels and heights 1704–2536 pixels, with 128 original-pixel margin and outward rounding to eight pixels. They retain 232,338,624 pixels, 9.95% of the complete original rasters. There is no resizing of these crops.

The frozen RAW decode, gain, camera exposure, radial correction and display transform are preserved. HC, DC and KA RGB crops match the existing genuine full-native reference bytes exactly. All 330 sampled camera rays, including pixel area, match their original full-native counterparts bit for bit. Only crop dimensions and principal-point offsets change; focal lengths and poses do not. The fixed common floor-aligned AABB remains `[-0.91,-0.26,-0.08]` to `[0.15,0.50,0.63]`.

Reviewed original actor/hair masks provide the unchanged native W32/W64 confidence falloff. No new segmentation or part/crew annotation campaign is required. All nonsensor crop edges have zero W64 confidence. Root visually inspected the actual HC RGB, native detail patch and confidence overlay: both actors, hair and boots are retained. These are source checks, not learned-quality results.

Evidence: [preparation report](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/native_crop/REPORT.md), [verification](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/native_crop/verification.json), [HC source preview](/fsx/tmp/marshawn_native_crop_tuning_20261108/data_native/review/HC_RGB.jpg), [HC native detail](/fsx/tmp/marshawn_native_crop_tuning_20261108/data_native/review/HC_upper_native_1to1.png).

The variable-resolution sampling overlay fixes per-camera flat-index decoding and aligned raster gathers. CPU checks verify exact parity with the original tensor sampler, including random-number state, and actual variable frequency-map loading. Evidence: [CPU report](/fsx/tmp/marshawn_native_crop_tuning_20261108/code_overlay/cpu_evidence/report.json).

### Controlled experiment plan

All runs start from a new field, optimizer, learned occupancy and 3D frequency state. Only immutable source images, masks and their fitted 2D maps are shared. The nonlearned semantic admission prior stays bound to its original frame-1660 source manifest; it is not relabelled as a newly inferred crop grid. Fresh 2D maps must be fitted against the exact crop RGB and crop-normalized coordinates.

All 66 cameras remain in TRAIN, following the user's explicit instruction. DC/HC/KA measurements therefore monitor in-sample reconstruction; they do not establish held-out-view generalization. Fixed novel-view renders must also be visually reviewed before selecting a recipe.

| Arm | Change from parent | Purpose |
|---|---|---|
| A | Archived settings, native crops | Native-resolution reference |
| B | A + corrected interval allocation | Preserve all ray intervals under the sample cap |
| C | A + normalized projection Jacobian | Independently test consistent frequency projection units |
| D | B + minimum step 0.000025 | Test detail previously limited by the 0.0001 floor, retaining interval coverage |
| E | D + sample cap 2048 | Test whether the cap limits detail |
| F | B + coarse traversal step 0.003125 | Test traversal speed and coverage |
| G | B + dynamic target 524288 field points | Test point budgeting rather than fixed ray count |
| H | D + FR switch at 20000 and LR horizon 50000 | Explicit shorter-schedule bundle |
| I | HD reference, native evaluation | Measure the actual crop-resolution benefit |
| J | D + dynamic target 262144 field points | Paper-sized point budget; L40S capacity check |

The eight silver-sable GPUs provide identical-hardware contrasts A–H. Olive runs the HD control; dev3 runs the additional point-budget arm. Compare image metrics at equal completed iteration counts and record field samples, ray counts, memory and elapsed training/evaluation time. Do not interpret L40S versus Blackwell wall time as a recipe speedup.

Frequency projection is deliberately probed separately before combining it with detail settings. The smaller marching floor and higher cap use the coverage-preserving allocator so a legacy far-interval truncation cannot masquerade as an intrinsic failure of finer sampling. G versus B isolates point budgeting on identical hardware; J versus D compares quality at equal iterations, with hardware timing kept separate.

Initial evaluations are planned at 8000, 16000 and 32000 completed iterations, with a durable stage gate at 32000. Subsequent 50000/80000/140000 stages require evidence from the previous stage; 140000 is the absolute ceiling. The archived FR switch at 75941 remains unchanged in A–G, so early gates alone cannot declare those schedules saturated. H explicitly tests reaching the later phase sooner. Select by raw actor LPIPS, report PSNR and SSIM alongside it, and inspect native face, hair, cloth and boot patches. No final recipe has yet been established by these new runs.

## Insights

Cropping retains the original angular pixel resolution while concentrating training on the actors. Merely rendering an HD-trained model at 4K cannot recover the native texture supervision that was discarded during input reduction.

Two implementation concerns deserve controlled measurement: legacy frequency projection can saturate the grid through inconsistent normalized-coordinate units, and the legacy capped allocator can truncate later ray intervals. Corrected code paths already exist, but empirical quality decides whether to adopt them. A smaller minimum step and a higher sample cap can increase cost substantially; compare quality per unit time as well as per iteration.

The accepted rig is retained: independent static printed-target residuals are approximately 0.3–0.6 native pixels, and tested replacements failed validation. All 66 active cameras passed integer-frame synchronization checks; subframe timing and moving-subject effects remain unmeasured. Native source patches visibly contain grain and soft detail, so a fourfold increase in pixel sampling does not prove a fourfold increase in resolved texture. See [calibration measurements](marshawn_printed_target_calibration.md) and [acquisition metadata](marshawn_camera_metadata_audit.md).

### Launch verification

All 66 fresh maps were completed and source-verified before field training. Actual fit time averaged 77.42 seconds for 55 silver-sable maps, 79.25 seconds for seven olive maps and 170.22 seconds for four dev3 maps. Total fitting work was 5493.66 GPU seconds; source preparation took 69 CPU-wall seconds. Queue/staging overhead and the final slower-worker tail are additional wall time. [Timing receipts](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/frequency_timing.json).

The cold warmup/own-checkpoint-resume and separate cold adaptive canaries passed. Restored RNG digests match exactly. The adaptive canary used 160747 field samples for 4096 rays, maximum 328 samples per ray, no capped rays and no OOM. Root inspected actual source and technical render previews. Four-update canary images establish execution correctness only, not image quality. [Technical proof](/fsx/tmp/marshawn_native_crop_tuning_20261108/ops/smokes/ROOT_review.json).

The ten-arm matrix was released on 2026-11-08 around 10:26 UTC. Its immutable manifest is `/fsx/tmp/marshawn_native_crop_tuning_20261108/ops/arm_matrix.json` (SHA256 `9543b918eafc930da1fa61e8cd64a6bcbfc24dfaeaf090b950325cba80b5432b`). A detached controller and watchdog retain physical GPU leases and own-output crash resume; a separate artifact mirror publishes checkpoint renders to `/fsx/tmp/marshawn_native_crop_tuning_20261108/review/arms/`. The first quality gate is recorded below.

### First 8000-iteration gate

All rows use the identical native actor targets, masks and crops. These are provisional in-sample results before the later feature-reweighting phase. A–I completed; the slower dev3 J arm was still approaching this gate when this table was recorded.

| Arm | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| A native reference |29.147654|0.672829|0.217881|
| B corrected allocation |29.157454|0.673033|0.215897|
| C projection Jacobian |29.160343|0.671097|0.220065|
| D smaller marching floor |29.171431|0.671361|0.220549|
| E higher sample cap |29.163569|0.672665|0.218225|
| F coarser traversal |28.950140|0.661891|0.227976|
| G smaller point budget |28.685484|0.660518|0.238075|
| H shorter schedule |29.238961|0.674361|0.215638|
| I HD training, native evaluation |28.908160|0.657477|0.233475|

Native B/H reduce actor LPIPS by approximately 7.5–7.6% relative to HD at the same iteration count. They are close to each other at this early gate. Root inspected actual native face and cloth comparisons: native supervision retains sharper seams and facial contours; substantial blur and synthetic texture remain. This is an improvement, not a final quality result. [Unscaled patches](/fsx/tmp/marshawn_native_crop_tuning_20261108/review/8k_native_vs_HD/grey_head_comparison_1to1.png) and [cloth comparison](/fsx/tmp/marshawn_native_crop_tuning_20261108/review/8k_native_vs_HD/grey_cloth_comparison_1to1.png).

At the same 8000-iteration checkpoint, A contains 10767 maximum-level frequency cells versus I's 7335, and has queried 9.688 billion field points versus 6.185 billion. Native crops change source detail and frequency-update coverage; this is not a pure pixel-count timing comparison. G reduces field work per iteration and reaches gates sooner, but is worse at equal iteration count so far. Its quality-versus-wall-time tradeoff remains under review.

### CPU sampling optimization

An actual66-camera profile isolated approximately 9 ms/batch of list-gather overhead: the same HD data and random draws take 8.61/12.06 ms as a tensor versus 17.96/21.23 ms as lists (warmup/FAS). A separate packed-CPU gather candidate reduces native gathering from about 9.45 ms to 0.33 ms, and full sampling from 18.32/22.02 ms to 9.30/12.99 ms. Packing costs about 0.59 seconds once and 2.81 GiB additional RAM. Actual RGB, masks, indices, confidence, normalization and random-number states are bit-identical in CPU checks.

The candidate also handles CUDA camera-ID/index metadata with CPU image storage; actual GPU qualification remains pending. Running A–J are unchanged. A separate K arm will test the same H recipe from cold initialization with this gather optimization after a physical slot becomes free. No trained weights transfer between arms. [Profile evidence](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/cpu_sampling_profile_v2/summary.json).
