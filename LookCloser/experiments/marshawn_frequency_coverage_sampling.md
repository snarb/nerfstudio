# Frequency coverage and native integration diagnostics

## What was tested

CPU-only inspection uses the actual selected semanticR0@111316 checkpoint and its saved103 TRAIN clothing plus12 DC diagnostic ray arrays. Native sample weights are preserved exactly. No field query, optimizer update or occupancy reconstruction occurs. Frequency lookup uses the exact stored float32 frequency AABB/size and `(resolution-1)` indexing from the frozen FrequencyGridManager, not occupancy-cell bounds. Independent Torch/NumPy float32 indices agree.

## Results

None of115 native rays saturates the1024-sample cap. Clothing counts have median317/max843; DC median270/max517. Thus these native traces do not demonstrate legacy allocator far-tail truncation. Fresh halo1 saturates one clothing ray and all-cells saturates two; those artificial profiles cannot be substituted for actual native saturation.

All native traces contain a maximum interval of approximately.00625. Native versus dense16384 RGB maximum-channel differences have median.01052/p90.02817/max.04659 for clothing, and median.00740/max.05057 for DC.89/103 clothing and9/12 DC rays exceed one8-bit code; dense8192/16384 converges in all cases. All-cells versus dense clothing median difference is still.00781. This supports testing the integration step, while occupancy omissions also contribute. Dense probability after the native last interval is small: max.00180 clothing/.00273 DC.

The stored frequency grid contains only levels0/15. More importantly, original native rendered mass is not confined to level15:

| Domain | Level0 share of termination | Level0 share of witness±.02 mass |
|---|---:|---:|
|All103 clothing rays|51.27%|44.95%|
|Old18|72.37%|71.11%|
|New66|49.65%|42.19%|
|Old8 reserves|58.35%|54.74%|
|New9 patch reserves|7.44%|6.54%|
|DC12|40.62%|No depth truth|

These are unrenormalized original-weight partitions, with relative shares reported separately. Frequency at a refined sample position need not equal the level used at its parent coarse-interval midpoint. Level0 is minimum stored frequency, not evidence of physical emptiness. Frequency-gridSHA `81a1c6ec7866c4d18609260801d5c357b1b1a74fc83248714a27f75ad3cbe4d5` matches the independent cell-center inventory.

[Sampler report](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/selected_R0_sampling_review/report.json), [original-weight partition](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/selected_R0_frequency_weight_mass/report.json), [query/source provenance](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/selected_R0_frequency_weight_mass/source_provenance.json).

## Controlled follow-up

A read-only selectedR0 comparison changes only scratch inference coarse_step.00625→.003125, preserving the trained field, actual occupancy, sources and all other settings. It uses the same103+12 rays, dense convergence and explicit actual-import bindings, under an idle-exclusive GPU lease. This is not a training or quality promotion.

A separate proposed training factor is grid_update_batch_size2048→8192, keeping interval1024, global uniform TRAIN camera/pixel sampling, original maps/projection and all other recipe settings. It is not launched. The frozen parent has independent_rng_streams=false: the CPU prefetch barrier and larger image/y/x draws necessarily change subsequent global RNG/pixel trajectories. The proposal must report that consequence rather than silently enable another RNG mode or claim matched training rays.

Required scratch proof: actual global camera IDs must map to all and only the63 TRAIN cache entries; record coordinates, rendered-depth positions and retrieved frequencies; measure CPU/CUDA RNG deltas, peak GPU memory, finite outputs, new15-level coverage and the original native-weight frequency coverage. Parameters and actual learned core/EMA remain unchanged; only a scratch frequency grid may update before restoration. Legacy integer-coordinate projection and lack of opacity filtering remain unchanged for attribution, with the explicit risk that extra rays reinforce background or biased depth. A larger batch is not a substitute for validating source geometry.

## Read-only half-step and actual update results

The selectedR0 field was replayed at `.00625` and `.003125`. Dense16384 RGB is bit-identical across all115 rays, proving the field stayed fixed. Native cap saturation remains0/115. All-cells versus dense median RGB error falls from.007813 to.004354 on clothing and.006742 to.002324 on DC; native median depth error also falls. However DC12 mean GT maximum-channel error rises.049227→.052005: more accurate integration does not guarantee improved images from a field trained with coarse integration. [Exact replay](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/selected_R0_halfstep/complete.json).

A separate cold half-step arm therefore starts from seed42, changing only `adaptive_coarse_step_size=.003125`; legacy allocator, original semanticR0/W64, body80, .01 distortion, maps/projection/LR and stopping remain unchanged. The actual config comparison has one field difference. Warmup and ARM32 pause/resume pass with successful Adam16→32 and15→30, exact RNG restoration, finite six-view technical outputs, peak7952MiB. It launched on silverGPU3 under global locking, controller1479253/trainer1479267. Minimum106316 and unchanged plateau apply; established selection, raw LPIPS minimum and terminal checkpoints are preserved separately. First15188 audit uses actual trained coarse values without inference overrides. [Source/config/technical release](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_cold_halfstep_candidate_20261107/release.json). Technical validity is not quality approval.

The single-update2048/8192 scratch proof also passed: exact original fullstate/RNG restored per arm, all model/core/EMA state unchanged, only scratch frequency grid updated. It took20.3s with2108MiB process peak; initial compilation makes per-arm time incomparable. New level15 cells440→1652, but only115→450 receive any admission-qualified projected position. Most additions are not actor-supported;36→127 new cells also receive an outside-AABB projected position via legacy clamping. Original clothing level0 termination share improves only51.269%→51.172% with8192, versus no change with2048. The larger update consumes different global CPU/CUDA RNG. No8192 cold training is launched. [Actual update proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/selected_R0_coverage_update_proof/results/complete.json), [new-cell attribution](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/selected_R0_coverage_update_proof/new_cells_attribution.json).

## CPU proposal: actor-focused frequency updates at the original budget

The existing `dec5_frequency_sampler(camera_indices)` hook runs **after** the original global Torch camera/y/x draws. It can replace only pixel coordinates while preserving all2048 global TRAIN camera IDs. Under the current legacy projection, `support_valid` is ignored; merely adding this hook does not introduce the normalized-Jacobian opacity threshold. No coordinate, map, depth or projection correction is proposed.

The isolated CPU prototype uses1229 existing core pixels,410 existing protective-band pixels and409 exact original uniform pixels per2048 update. Camera assignments are untouched. Existing upper-body32px/lower-body16px proposal bands preserve a dedicated contour budget; the20% original uniform stream can still discover support outside these masks. This does not guarantee missing hair coverage. Original core/band includes the known bag falsepositives; no simultaneous bag correction is hidden in this factor.

Because the hook does not receive original y/x, `prepare()` shadows the exact historical Torch draws using a cloned CPU generator state. The unchanged pipeline consumes its original draws normally; the hook asserts camera order and resulting CPU RNG bytes before substitution. Additional permutation/pool draws use a private, serialized NumPy generator. No changes to the main body80 sampler, RGB weights/normalizer, FAS, occupied cells or positive geometry targets occur. The private generator state must be saved/restored alongside the existing runtime state, including the75941 historical seed reset policy; it must not be reseeded implicitly on resume.

Three CPU tests pass: original global RNG and uniform pixels are exact, private/global state reproduces the next update after resume, and changed camera ordering or an extra global draw fails closed. Across32 simulated updates using all63 hash-verified original TRAIN masks, core hits rise2531→39827 and protective-band hits1550→13428 out of65536 rays; no empty-pool fallback occurs. Camera histograms are identical by construction. No field, grid or GPU was touched. [CPU proof/source hashes](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/actor_frequency_CPU_proposal/report.json).

Source inspection suggests CUDA RNG consumption can also remain identical: native nerfacc stratification draws one `rand_like(near_planes)` per input ray before traversal, while the fixed warmup marcher is deterministic. Keeping2048 rays therefore preserves the apparent draw shape, unlike8192. This is an inference from code, **not yet an actual GPU RNG proof**. Before integration, compare original versus focused single updates from identical selected-model/RNG state, assert CPU and CUDA post-update RNG byte equality, exact camera IDs and original-uniform coordinates, unchanged non-frequency tensors, finite depth/output and actor-admitted new-cell attribution. Record source `nerfacc` path/hash. No speculative manual RNG advancement or duplicate baseline render should hide a mismatch; if actual CUDA states differ, report it and revise the causal design explicitly.

A viable isolated cold arm would retain originalR0/W64/.01distortion/.00625coarse/global2048/interval1024 and change only this frequency pixel proposal. It would first require actual warmup/ARM and private-state Adam pause/resume checks, plus full106316→unchanged plateau and matched15188 visual/geometry gates. Neither this proposal nor8192 is launched. Sparse reserves are diagnostic only, never sampling targets.

## Actual focused2048 scratch proof

The proposed hook was then tested only in an isolated restored selectedR0 process. Original2048, focused2048 and focused2048 resumed from a serialized private-state fixture all ran the actual frozen model/update under AMP. All non-frequency tensors, learned occupancy/core/EMA and original source weights stayed unchanged; each scratch frequency grid was restored. No optimizer or producer changes occurred.

|Actual single-update result|Original2048|Focused2048|
|---|---:|---:|
|Projected depth inside frequency bounds|594|1739|
|Projected depth admitted by actual field support|256|1636|
|Admitted and rendered alpha>.1|231|1616|
|New level15 cells|440|737|
|Original native103 termination mass at level0|51.269%|47.093%|
|Original native103 witness-band mass at level0|44.945%|40.140%|

The projected admission/opacity counts are diagnostics, not newly added filters. Legacy depth/coordinates/map projection remains unchanged. The update makes meaningfully more use of the same ray budget than8192 uniform in this measured state.

Actual global CPU **and CUDA** post-update RNG bytes agree with original. The unchanged actual body80 datamanager then emits identical next main-training batch indices, origins and directions, with identical resulting RNG states. Private auxiliary state serialization/restoration reproduces the focused grid, samples and next main batch exactly. Runtime import paths/hashes include the proposal, trace helper and actual nerfacc estimator. The bounded idle-exclusiveGPU7 proof took30.5s, peak2118MiB; allocator≈1.23GB. The device was released. [Proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/actor_frequency_actual_proof/complete.json), [full result](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/actor_frequency_actual_proof/results/complete.json).

This validates the immediate update/RNG mechanism, not training quality or general future ray identity after learned fields diverge. No production hook is installed. An eventual cold arm still requires actual worker checkpoint binding and successful Adam pause/resume with the auxiliary generator state, source hash checks and unchanged body80/objective configuration.

## Cold half-step first15188 review

Matched first15188 PSNR/SSIM/LPIPS improves31.41439/.841056/.03117262→31.74428/.848688/.02513011 (LPIPS−19.4%). DC/HC/KA LPIPS is.02784/.02347/.02409. Personally reviewed DC black, HC grey and KA black crops show reduced integration striping and improved clothing detail; visible feet/hair remain, broad halos and floor residuals persist.

Geometry does not improve uniformly: native witness-band mass original→half is80.23→80.21% old18,83.95→84.12% new66,84.48→82.48% old8reserves,95.95→94.71% new9patch reserves. Old8 early mass rises9.76→13.37%, median10–90width.02668→.03323. All206clothing+24DC convergence and native saved-PNG parity checks pass. This supports continued training and numerical diagnosis, not final geometry promotion. [Matched geometry](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_halfstep_15188/clothing103/comparison.json), [DC crop](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_halfstep_15188/DC_black_gate_comparison.jpg). An inherited human report title saysR0/R1; [erratum](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_halfstep_15188/report_label_erratum.json) identifies the actual two R0 coarse-step arms without changing measurements.

A further **read-only** `.003125→.0015625` comparison uses this half-trained15188 checkpoint, original1024 cap and legacy allocator. Its policy explicitly prevents cap/allocator changes. It must quantify native saturation, far-tail mass and dense convergence before any quarter-step training proposal.

## Isolated focused-frequency cold launch

Actual worker integration preserves all1494 original non-worker files and adds only the proposal/runtime helpers. The model/pipeline/optimizer config is exactly identical to originalsemanticR0/W64; the only scientific spec addition is the focused2048 frequency distribution. Ordinary checkpoints store private generator/source/config state and verify exact restoration. The global75941 seed reset remains original; the private stream continues its saved state.

Technical32+32 paths passed with warmup successfulAdam32→64 and ARM30→62. Counterfactual technical warmup starts1023 and actually crosses frequency1024 before checkpointing; ARM crosses4096. Both restore an already advanced private generator from the ordinary checkpoint. Six64-update RGB views were inspected, finite coarse silhouettes only. Peak7828MiB,148.4s under one idle-exclusive lease. Five meaningful CPU proposal/runtime tests also pass.

The delegated production release binds1497 source files and32 artifacts; SHA`d1416d153d1b1e1a26bb8e7e847176f392825c2938e55890d67ef26c726867e7`. Controller1553823/trainer1553871 runs on silverGPU7 with full106316→unchanged plateau, first15188 geometry/visual watcher and separate established/raw-min/terminal retention. Original.00625/.01/R0/W64/body80/maps/projection remain fixed. At00:41UTC it reached1635 with controller/worker alive and no OOM. [Exact release](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_frequency_focus_candidate_20261107/release.json). It is not combined with half-step, distortion.1, bag correction or source-point density.

## Quarter-step inference on the half-trained15188 field

The next numerical comparison uses the immutable **half-step-trained15188** field, not the old.00625-trained field. It changes only scratch inference.003125→.0015625, explicitly keeping cap1024 and the legacy allocator. The idle-exclusive audit actually usedGPU5 (GPU6 was requested first but its factory lease was busy and correctly skipped). Both GPU profiles and CPU postprocessing are complete; the lease was released.

All115 native rays, and all halo/all-cells profiles, have zero cap saturation. Native clothing mean samples109.5→151.9(max345→363); DC75.6→114.6(max145→191). No evidence here warrants bundling a raised cap or corrected allocator. Clothing maximum dense mass after the native final interval falls.01080→.00349; DC stays.003814. This residual is not cap truncation.

All-cells versus dense median RGB error falls.007659→.004172 for clothing and.007280→.003187 for DC. Native-versus-dense median clothing error is mixed.018237→.018628, while DC improves.017524→.012569. Thus actual occupancy still explains much of the remaining discrepancy. Dense16384 RGB is exactly identical across all115 rays; all dense convergence tests pass. No field/occupancy/optimizer change occurs. A pure cold quarter-step remains a viable controlled proposal, not a proven quality improvement or launched training arm.

[Native sampling/cap reports](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/half15188_quarterstep/quarter_step_sampling/report.json), [same-field identity](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/half15188_quarterstep/same_field_dense_identity.json), [exact policy](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/half15188_quarterstep/policy.json).

## Focused first15188 geometry and cap qualification

Focused2048 first15188 PSNR/SSIM/LPIPS is31.61591/.847553/.02409099:22.7% lower LPIPS than originalR0, with per-view DC/HC/KA.02970/.02118/.02139. Compared with half-step, aggregate LPIPS improves while half-step retains the better DC result. Three source/focused/half/original actor contacts were inspected; both factors reduce visible integration artifacts, but broad halos and floor residuals persist.

The focused geometry result is mixed. Old18 native band80.23→66.66%, early mass8.21→19.90%; old8 reserves band84.48→70.94%, early9.76→24.18%. New66 band improves83.95→86.85%, while new9patch reserves95.95→90.56%. All206 clothing+24DC convergence/native-PNG checks pass. This is not a clean-geometry promotion or evidence to bundle focus with another factor immediately.

CPU inspection of the saved focused native arrays finds0/115 rays at the1024 cap: clothing max898, DCmax584; maximum dense tail mass.00419/.00121. Therefore the measured old-shirt early-mass regression is not explained by native cap truncation. [Focused geometry](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_frequency_focus_15188/clothing103/comparison.json), [cap/tail report](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_frequency_focus_15188/focused_native_sampling/report.json), [three-arm DC visual](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/focus_half_baseline_15188/DC_black_comparison.jpg).

## Cold quarter-step launch

After the negative-tau final geometry gate finished, the delegated pure quarter-step cold arm launched on idle-exclusive silverGPU0, controller1647009/trainer1647024. Source1495SHA`b001cda438bd539135546722b1b8c7ddfda522421fbe4127e5005a2e4dce60f2`, specSHA`494618453601afcd061ba971395614fd846ea2e71c4723ceb21f0618347264da`, releaseSHA`789a26e5c1fa6a3545b17d515ee192c59515429b82df3d2ca5c4de9feaa24fee`. Only coarse.0015625 changes; original1024cap/legacyallocator/.01distortion/body80/W64/R0/maps/projection remain fixed. Actual config parity explicitly asserts cap/allocator. Technical warmupAdam16→32 and ARM15→30, exact RNG restore, six finite32-update views reviewed, peak7964MiB/143.2s. These are integration checks only.

Production retains full106316→unchanged plateau, original selection and separate raw-min/terminal weights. A durable matched quarter/half15188 audit measures the actual trained native115-ray cap/tail behavior after tracing; those rays are not claimed to cover every training batch. No focus, distortion.1, bag or source-point factor is combined. [Exact release](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_cold_quarterstep_candidate_20261107/release.json).

## Matched15188 actual frequency contribution

CPU lookup of the actual saved native termination weights confirms a substantial frequency coverage change. No model query or CUDA context was used. The original checkpoint grid, float32 bounds/indexing, and trace checkpoint hashes are checked.

| Arm at15188 | Clothing termination at frequency0 | Clothing ±.02 band at frequency0 | DC12 termination at frequency0 |
|---|---:|---:|---:|
| Original R0 |89.97%|88.70%|90.20%|
| Half step |85.85%|83.85%|88.66%|
| Focused2048 |44.06%|36.19%|32.10%|

These are descriptive partitions of different trained fields and sample locations, not a matched-location causal proof. Frequency0 is a stored minimum, not a free-space label. Better coverage accompanies better image detail but does not prevent the focused old-shirt early-mass regression. [Exact per-ray evidence](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/frequency_mass_matched15188/focus/report.json).

## Quarter-step first15188 qualification

Quarter PSNR/SSIM/LPIPS is31.69965/.845859/.02189069,12.9% lower LPIPS than half at the same step. DC black, HC grey and KA black source/quarter/half contacts were inspected: clothing detail improves, feet and hair remain visible, but the broad halo and floor residual remain.

Native witness-band mass half→quarter is80.21→82.00% old18,84.12→84.35% new66,82.48→86.47% old8reserves,94.71→91.71% new9patchreserves. Old8 early mass improves13.37→9.83%, while new9 worsens1.35→2.93%. All206 clothing and24DC dense-convergence/native-PNG checks pass. This is mixed geometry, not a clean-asset promotion.

The **quarter-trained** native115 rays have zero cap saturation, as do halo/all-cells profiles. Clothing mean157/max366 samples, DCmean118/max220, all below the unchanged1024 cap. Dense mass beyond the native final interval reaches.01795 clothing/.00357DC; this is not cap truncation. [Actual trained cap/tail](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_quarterstep_15188/cold_quarterstep_sampling/report.json), [geometry](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_quarterstep_15188/clothing103/comparison.json), [DC visual](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_cold_quarterstep_15188/DC_black_gate_comparison.jpg).

## Predeclared focus × distortion interaction

The focused arm improves detail but worsens old-shirt geometry; the independent cold distortion.1 arm improves local geometry. A fourth corner now tests their interaction explicitly. It retains the focused1497-file source except the exact validator coefficient.01→.1. Original coarse.00625/R0/W64/body80/maps/projection/allocator/LR remain unchanged. The existing distortion ramp4096..12288 and full106316→plateau continue. This is not combined with half/quarter-step, bag correction or source-point lookup.

Actual technical32+32 tests pass with warmupAdam32→64 and ARM30→62. The actual1024/4096 frequency boundaries advance the private generator; ordinary checkpoints restore that nonempty state exactly. Six actual64-update views were inspected for finite coarse output only. Peak7828MiB,148.5s under one idle-exclusiveGPU5 lease. The CPU config proof forces CUDA invisible; an earlier import-only attempt failed closed when tcnn initialized a context, without constructing a model/optimizer. That attempt and the successful CPU rerun are recorded.

Delegated releaseSHA`e742428ad88729822a058df329c8b30e310fcc177cfc4c3134dde3cb1dc9b3d8` binds35 artifacts and sourceSHA`0c94264ea3a66b8a009815cf9d4ccd9e3e14e637192f29453bd7eb272c62e975`. SpecSHA`65e3a9f98f229d14a126ef75724b2d29d58186b911bf17d95067ecd19e405aea`. Controller1727773, final-choice observer1727774, first15188 auditor1727775 and CPU interaction comparator1727776 are durable. Only the new corner needs GPU replay; the three already sealed15188 corners are reused for the four-way CPU comparison. Local evaluation/archive and visual-fetch watchers remain active. [Release](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_frequency_focus_distortion_candidate_20261107/release.json).

The interaction first15188 gives31.83154/.848052/.02419866, versus focus-only31.61591/.847553/.02409099: LPIPS+0.45%, PSNR+0.216dB. DC/HC/KA LPIPS is.028675/.020532/.023389; KA worsens from focus-only.021389. Native bands focus→interaction recover66.66→89.44% old18,86.85→96.30% new66,70.94→93.45% old8reserves,90.56→99.94% new9patchreserves. Old18/old8 early mass falls19.90→3.39% and24.18→2.50%.

Median10–90 widths focus→interaction are.06137→.02240 old18,.01367→.00797 new66,.05812→.01482 old8,.01260→.00371 new9. All412 clothing and48DC convergence/PNG checks pass in the four-corner comparison, with only the new115 rays queried again. Three full source/four-corner actor contacts were manually inspected: strong clothing detail and visible shoes/hair, narrower torso haze, substantial floor/bag/hair-adjacent residual still present. This supports continuing the unchanged full trajectory, not early clean-asset promotion. [Four-corner comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/focus_distortion_2x2_15188/clothing103/comparison.json), [DC contact](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/focus_distortion_2x2_15188/DC_black_gate_comparison.jpg).

## Separate quarter × distortion interaction

The next declared four-corner comparison combines quarter-step.0015625 with distortion_final.1, retaining the original **uniform**2048 frequency updates. It contains no focused hook. Original maps/R0/W64/body80/LR/legacy allocator and1024cap remain unchanged. The1495 source files differ from the quarter parent only at the exact distortion validator. CPU proof checks the actual config, original callback ramp and explicit rejection of a bundled focus/width/radius/coarse variant.

Technical warmup16→32 and ARM15→30 successful Adam updates passed with exact full-state RNG restoration. Six actual32-update RGB outputs were inspected, finite coarse silhouettes only. Peak7964MiB,143.35s under idle-exclusiveGPU6 after the native-agent probe released its lease. This leaves GPUs1/2/4 available for the source-reviewed negative-tau diagnostics; no producer was preempted.

SourceSHA`ad32a034c492314162fac8d966be2215c834393a1bb984beb05f1eed8f9c43b3`, specSHA`54e55e212bb7c1d27c12c9428cef8cd6b99ccdefa9ca3fbef776ba01d828aaa8`, delegated releaseSHA`64a47315bc2d651bdc5925262a13f6decd5346a2f513edb351b32077860e2b66` binds30 artifacts. Controller1859266, final-choice observer1859267, first15188 auditor1859268, CPU four-corner comparator1859269 and trained cap/tail observer1859270 are durable. Only the new corner needs115-ray GPU replay; originalR0/cold.1/quarter15188 evidence is reused by CPU. Full106316→unchanged plateau and established/raw-min/terminal retention remain mandatory. [Release](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_quarter_distortion_candidate_20261107/release.json).

At15188 the interaction gives32.09120/.854067/.02219093, versus quarter-only31.69965/.845859/.02189069. PSNR+0.392dB and SSIM+0.00821 accompany LPIPS+1.37%. The aggregate hides an asymmetry: black-actor LPIPS.0236085→.0245238(+3.88%), grey.0201728→.0198580(−1.56%). DC-black LPIPS.023837→.026132 contributes the regression. A3% per-actor LPIPS guard would reject early promotion.

Native bands quarter→interaction improve82.00→86.57% old18,84.35→95.27% new66,86.47→89.68% old8 reserves,91.71→99.32% new9 reserves. However old-shirt median widths barely change: old18.03010→.03033 and old8.02773→.02804. New66.02174→.01152 and new9.01820→.00837 narrow substantially. This differs from the stronger old-shirt concentration under focused+.1.

All412 clothing/48DC convergence/native-PNG checks pass across the four corners; only the new115 rays were queried. Actual trained native cap hits0/115, clothingmean138/max369 and DCmean92/max158; halo/all-cells also have no cap hits. Maximum dense mass beyond the native final interval is.01067/.00495. Three full four-corner actor contacts were inspected: strong detail and visible feet/hair, narrower torso halo, substantial floor/hair residual. Continue the unchanged full trajectory; no early clean-geometry promotion. [Four-corner evidence](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/quarter_distortion_2x2_15188/clothing103/comparison.json), [actual native cap/tail](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_quarter_distortion_15188/quarter_distortion01_sampling/report.json), [DC contact](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/quarter_distortion_2x2_15188/DC_black_gate_comparison.jpg).
