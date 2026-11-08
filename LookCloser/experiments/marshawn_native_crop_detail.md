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

The candidate also handles CUDA camera-ID/index metadata with CPU image storage. An actual GPU canary passed exact RGB/mask and CPU/CUDA RNG checks, and cold K tested the same H recipe with this gather optimization. K's 8k/16k LPIPS is 0.213139/0.190916 versus H's 0.215638/0.193874; this does not attribute small GPU-training variation to mathematically unchanged gathering. Running A–J were unchanged. No trained weights transfer between arms. [Profile evidence](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/cpu_sampling_profile_v2/summary.json), [actual CUDA proof](/fsx/tmp/marshawn_native_crop_tuning_20261108/ops/K_fast_gather/actual_GPU_canary.json).

### 32000-iteration gate

| Arm | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| A native reference |29.782408|0.709723|0.156527|
| B corrected allocation |29.785252|0.710604|0.155978|
| C projection Jacobian |29.781846|0.708910|0.157870|
| D smaller marching floor |29.759243|0.709165|0.156853|
| E higher sample cap |29.732521|0.709954|0.157258|
| F coarser traversal |29.614516|0.700949|0.167085|
| G smaller point budget |29.257211|0.680112|0.199335|
| H shorter schedule |29.705287|0.699527|0.171812|
| I HD training, native evaluation |29.067315|0.655165|0.194555|

B improves LPIPS by 19.83% relative to the matched HD control, with +0.718 dB PSNR and +0.0554 SSIM. Root inspected unscaled native face/cloth comparisons. The shorter LR50k/FR20k bundle loses its early advantage; it is not selected. Coarser traversal and the 524288-point budget also lose quality. Smaller marching steps and a higher cap do not improve masked LPIPS at this gate, although their opacity behavior differs. A/B continue to 80k to observe the original FR75941 phase; D continues to50k for geometry comparison. These remain intermediate results, not a saturation claim.

Native source versus HD comparison: [grey face](/fsx/tmp/marshawn_native_crop_tuning_20261108/review/32k_native_vs_HD/grey_head_comparison_1to1.png). Fixed novel views are also rendered from the same interpolated cameras. The HD control still contains substantial floor/background residue; actor metrics do not certify clean isolated geometry. [Novel control](/fsx/tmp/marshawn_native_crop_tuning_20261108/review/novel_032000/I_HD/contact.jpg).

### Larger ray batches and phase timing

Separate cold trials use the packed gather and parent B settings:

| Arm | Rays/update | LR horizon | FR switch | Early LPIPS |
|---|---:|---:|---:|---|
| L |8192|200000|75941|8k:0.188851;16k:0.165181|
| N |8192|100000|37971|8k:0.188911;16k:0.167777|
| M |16384|50000|18986|8k:0.168235|

L at8k approximately matches B at16k (0.190463), using the same cumulative ray count. M gains more per iteration but has a larger GPU footprint. Wall time, field-point counts and later quality gates decide the recipe. L/N/M retain iteration-based warmup, occupancy/FAS/frequency cadence and distortion ramp, so LR/FR scaling alone is not a completely exposure-matched schedule. A further explicitly scaled schedule tests these phases too, preserving actual optimizer-attempt counters and the140k ceiling.

### Fast frequency-map ablations

The learned native maps contain 10–15 eligible FAS levels per camera; level15 covers only19.46% of eligible cells. Replacing them changes the proposal. Legacy projection nevertheless saturates updated in-box cells: even scalar16 projects above42194, versus the8192 cap. [Measured histogram and cap certificate](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/frequency_cost_audit/report.json).

Two independent CPU proxy map sets were prepared in5.945 seconds: constant scalar16 and eligible-actor Sobel-energy quantile ranks. Both have honest zero-training metadata and distinct manifests. The original NN validator rejects them; an isolated strict proxy validator binds identical RGB, cameras, masks and AABB to the original ready SHA. P/Q train from cold using the B neural settings and the unchanged native evaluation source. These are controlled frequency-estimator ablations, not claimed equivalents of progressive SSIM. Their quality remains unproven; the original learned maps stay unchanged.

### Updated gates and quality/geometry review — 2026-11-08 12:39 UTC

Selected later evaluations use the same native in-sample actor protocol. Steps are completed trainer iterations, including any AMP-skipped optimizer attempts. These are intermediate checkpoints, not a selected final recipe or evidence of saturation.

| Arm | Iterations | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| A | 50000 | 30.159932 | 0.727258 | 0.137948 |
| A | 80000 | 30.485385 | 0.743627 | 0.123037 |
| B | 50000 | 30.137022 | 0.727685 | 0.137904 |
| B | 80000 | 30.459985 | 0.743281 | 0.124236 |
| D | 50000 | 30.130760 | 0.726478 | 0.137101 |
| L | 32000 | 30.279158 | 0.735042 | 0.129428 |
| L | 50000 | 30.695631 | 0.754812 | 0.113923 |
| M | 32000 | 30.404836 | 0.737376 | 0.125755 |
| N | 32000 | 30.271978 | 0.731802 | 0.135669 |
| N | 50000 | 30.580419 | 0.746282 | 0.122020 |
| P | 50000 | 30.228487 | 0.735091 | 0.134499 |
| Q | 50000 | 30.240323 | 0.735833 | 0.133613 |
| U | 32000 | 29.763338 | 0.710156 | 0.157856 |

The [quality-versus-iterations plot](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/learning_curves/quality_vs_steps.png), [active-loop time plot](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/learning_curves/quality_vs_active_loop.png) and [source-linked CSV](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/learning_curves/learning_curves.csv) were refreshed around13:00 UTC to68 exact PID-qualified gate times, including L/N/M50k and S/T16k. Later A/B140k and new density trials are outside that snapshot. At measured gates, L32k reached `30.279158 / .735042 / .129428` in 2357.15 active-loop seconds versus B50k `30.137022 / .727685 / .137904` in 3059.19 seconds: 22.95% less time with better aggregate metrics. This is an observed operational comparison, not a causal attribution to batch size. Setup is reported separately in [GPU-lease wall time](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/learning_curves/quality_vs_lease_wall.png); between-stage idle is excluded from both clocks. The shared 69-second source preparation and 5493.66 GPU-second map fitting are not added independently to every arm.

**The aggregate gain does not resolve clothing geometry.** In the reviewed native HC black-cloth interior `[500,650,756,906]`, the same mask intersection selects 65474 pixels. Mean rendered opacity is:

| Checkpoint | Mean rendered alpha |
|---|---:|
| B80k | .984972 |
| L50k | .865599 |
| M32k | .664907 |
| N50k | .852352 |
| T8k | .649499 |

The [unscaled RGB/opacity contact](/fsx/tmp/marshawn_native_crop_tuning_20261108/review/black_cloth_opacity_contact_1to1.png) shows the large deficit relative to B despite plausible dark-cloth RGB. The mask selects a diagnostic region; it is not alpha ground truth and has not become an opacity target. This observation alone does not separate occupancy culling from insufficient learned density. Lower whole-actor LPIPS and faster iteration progress therefore do not justify selecting the larger-batch recipe. [Patch measurements and source hashes](/fsx/tmp/marshawn_native_crop_tuning_20261108/review/detail_patch_metrics.json).

At 50k, whole-actor P/Q LPIPS improves over B (`.134499/.133613` versus `.137904`), but HC grey-face patch LPIPS worsens (`.233677/.236270` versus B `.222631`). Black-face P also worsens (`.375472` versus `.370797`), while Q improves (`.359551`). Thus the actor average does not establish a uniform facial-detail gain. These patch scores use saved native PNGs and a separate local domain, not the float-render whole-actor domain. [Frequency curves](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/learning_curves/frequency_proxy_lpips_vs_steps.png).

### Full exposure-phase compression and proxy dependency

S/T test whether scaling the other training phases improves on partial LR/FR compression in N/M. S uses 8192 rays and exposure scale 2: all three warmups 2048, occupancy interval 8, frequency interval 512, LR horizon 100000 and FR switch 37971. T uses 16384 rays and scale 4: warmups 1024, occupancy interval 4, frequency interval 256, LR horizon 50000 and FR switch 18986. Body80 FAS and distortion phases use `actual_step × exposure_scale`; the distortion ramp ends at actual iterations 6144/3072 respectively. Optimizer-attempt counters, evaluation steps and the 140k ceiling remain actual counts; fewer Adam updates are still an algorithmic change. S runs on L40S and T on Blackwell, so their [step comparison](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/learning_curves/S_T_quality_vs_steps.png) is not a causal wall-clock hardware comparison.

P/Q currently still require the original complete NN-map generation: the dedicated proxy validator first runs the unchanged strict progressive-SSIM reference verification against native READY `0d1e3015938ad5cf377196e417c29a80df551c0a314008fb14b24d72012aca70`. Training uses honest zero-fit proxy maps, but this reference dependency means the present workflow is not an NN-free production preprocessing pipeline. Original RGB, masks, cameras, AABB and learned maps remain unchanged.

A separate frozen-B50 inference check found that 2×2/3×3 subpixel averaging softened face/hair detail and worsened all four declared patch LPIPS scores; reducing the marching floor did not recover them. State remained unchanged, no AA variant was promoted, and this does not establish a general geometry-oracle result. [ROOT review](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/inference_AA_patch/ROOT_result_review.json) and [native AA comparison](/fsx/tmp/marshawn_native_crop_tuning_20261108/review/inference_B050000_AA_v3/grey_head_contact_1to1.png).

### Frozen-field opacity diagnosis and next controlled trial

Nine fixed native HC rays through the opaque-cloth diagnostic region were evaluated with the original occupancy, one-cell dilation, all statically admitted cells, and uniform 16384/32768/65536-point quadrature. All variants retain the same field and static density admission. The dense reference measures the existing field; it does not supply physical opacity ground truth.

| Frozen field | Original grid mean alpha | +1 grid | All admitted | Uniform 65536 |
|---|---:|---:|---:|---:|
| L at 50000 | .874062 | .882026 | .884075 | .884396 |
| M at 32000 | .650933 | .688909 | .702857 | .699499 |

Between 32768 and 65536 samples, the largest absolute alpha change is .000117 for L and .000059 for M. The opacity deficit therefore persists with converged dense integration on these rays; occupancy culling explains a smaller part. Both finite checks took approximately nine seconds and restored the exact field, persistent state, binary aliases, frequency grids, configuration and RNG. [Per-ray measurements](/fsx/tmp/marshawn_native_crop_tuning_20261108/ops/opacity_oracle/early_summary.json) and [complete evidence](/fsx/tmp/marshawn_native_crop_tuning_20261108/review/opacity_oracle).

Cold V tests only reducing final distortion weight from .1 to the paper's .01 against fully exposure-scaled T (16384 rays, scale4, LR50000, FR18986), using the original learned frequency maps. No opacity target or new mask interpretation is introduced. Its initial review budget is 32000 actual iterations. The trial is a hypothesis about learned density, not an established repair.

Actual NVIDIA process checks at 12:44:45 UTC confirm all ten field GPU contexts simultaneously: eight on silver-sable, one on olive and one on dev3. [Process/progress receipt](/fsx/tmp/marshawn_native_crop_tuning_20261108/ops/actual_arms_20261108_124445.json). P/Q/N now have own-output 80k gates; S/T have 50k gates. A/B retain the 140k ceiling. Earlier video training remains stopped.

M finished 50000 iterations at `30.532138 / .743817 / .123683`; its LPIPS gain over32k is only .002072. It was retired at that gate. Its GPU automatically started cold W, the corresponding S8192/scale2 control with only distortion changed to .01. No state moved from M to W. [W cold initialization](/fsx/tmp/marshawn_native_crop_tuning_20261108/ops/W_batch8k_scale2_dist001/actual_cold_first_gradients.json).

The reusable NN worker and packed gather are committed as `7eeeee70`, with Python3.8 compatibility follow-up `d0f9b7fd`. The public worker SHA is `69def4a01c5e92cc1e98411cec5266c5108204ef306352e358e9f22965036701`; active private workers retain their original hashes. Public imports explicitly bind NN helpers to the declared sealed release and the two crop samplers to their exact local files. CPU tests cover configuration/loop parity, source-checkout rejection, variable-raster gathers and RNG, and actual DEV3 Python3.8/Torch2.4 configuration/Adam/YAML compatibility. They do not claim a new public GPU qualification. [Local qualification](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/geometry_v7/public_template_cpu_checks.json), [DEV3 proof](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/public_template_py38/result.json).

### Terminal references and density screens — 13:37 UTC

| Arm | Iterations | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| A native reference |140000|30.687317|.755608|.107840|
| B corrected allocation |140000|30.652963|.754867|.108218|
| L8192, long schedule |80000|31.040898|.771832|.100177|
| N8192, compressed LR/FR |80000|30.766615|.755794|.114730|
| P constant-map proxy |80000|30.561121|.752345|.117438|
| Q gradient-rank proxy |80000|30.570177|.752349|.119251|
| S8192, all phases scaled |32000|30.376257|.739661|.125491|
| T16384, all phases scaled |32000|30.695378|.756164|.112290|
| V: T with distortion .01 |8000|30.461844|.732769|.145681|
| W: S with distortion .01 |16000|30.423760|.731943|.144080|
| Y: W with Adam beta2 .99 |8000|29.713392|.698219|.179383|

A/B ended at exactly140000 attempts and selected their139999 checkpoints. Root inspected fixed native novel-view contacts for B140 and L80: both actors and boots remain framed; substantial floor/background fragments persist. L80 has the best raw actor LPIPS in this table, but cloth mean alpha remains .873695 and60.42% of the diagnostic interior is below .9. A/B140 cloth means are .982653/.986770. Thus L80 is a detail reference, not a geometry-safe selected recipe. [B140 novel contact](/fsx/tmp/marshawn_native_crop_tuning_20261108/review/novel_140000/B_native/contact.jpg), [L80 contact](/fsx/tmp/marshawn_native_crop_tuning_20261108/review/novel_080000/L_native/contact.jpg).

Reducing distortion improves aggregate image metrics but has not repaired opacity: V8k cloth alpha is .665269; W16k is .763699 versus S16k .7874. Y8k beta2=.99 increases cloth alpha to .848613 versus W8k .724279, while LPIPS is .00405 worse at that gate. X tests the same beta change at16384 rays; Z isolates distortion .01 at4096 rays. Every new arm is cold. P/Q continue their own checkpoints to140k to test the later FR phase against matched A/B endpoints; their source/code/specs remain unchanged.

T16k takes2332.21 active-loop seconds versus L32k2357.15 seconds, at the same cumulative ray count. PSNR is slightly lower, SSIM slightly higher and LPIPS nearly equal; fewer iterations do not establish a proportional wall-time speedup. [Exact timing comparison](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/learning_curves/T16_vs_L32.json).

An optional deficit-only adaptation of the green-scene opacity recipe is now screened from cold. AA/AB preserve W and add coefficients .1/.01 only on exact32-native-pixel Euclidean erosions of current valid actor unions. All66 overlays and HC close view were inspected; direct band/exterior overlap is zero. Thick hair interiors can remain, and shared-field changes can affect other pixels indirectly. This is an explicit opaque-material prior, not measured alpha ground truth. Existing RGB/W64, sampling, field, occupancy and cameras are unchanged; there is no profile penalty or extra field query. Packed flags cost .093ms per8192 rays versus3.245ms with camera loops. [Definition, proofs and risks](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/opaque_core_prior_v9/REPORT.md). No quality improvement is claimed before native gates.

The read-only checkpoint evaluator reproduces original B50 float metrics within3e-7 and restores all state, source bytes and RNG. It ran in67.46 seconds including preflight/setup, with2.42GB peak allocated GPU memory. This enables separate intermediate-step scoring without altering ongoing training. [Actual canary](/fsx/tmp/marshawn_native_crop_tuning_20261108/review/readonly_checkpoint_evaluations/B_050000_olive_canary/complete.json).

A private, inactive v8 builder also removes the native NN-ready prerequisite from fast proxies: freshly generated maps match P/Q on all66 cameras with forbidden NN paths poisoned. Original HD geometry READY metadata is still required, so this is not yet a complete new-frame production replacement. [Scope and remaining dependency](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/pure_proxy_v8/REPORT.md). Existing P/Q retain v4 and their original dependencies.

The13:37 root supervision confirms ten actual field GPU contexts, live expected workers/controllers, current progress and no recent OOM/CUDA traceback evidence. [Hourly receipt](/fsx/tmp/marshawn_native_crop_tuning_20261108/operations/ROOT_supervision_1337.json). No old video job was restarted.
