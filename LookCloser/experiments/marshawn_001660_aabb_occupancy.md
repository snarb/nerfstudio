# Marshawn 001660: HD AABB and occupancy campaign

Current acceptance status (2026-11-06): **not accepted: visible gray/brown floor
mottling remains**. The latest native MVS mean-depth trial stopped at16k with
PSNR29.19285, SSIM.81343, LPIPS.26807; it improved measured depth consistency,
but failed the manual color review. No artifact is certified as an ideal grid
or a clean final render. A separately validated fixed radial-correction dataset
is ready for a new bounded cold trial; see
[radial photometry](marshawn_shared_radial_photometry.md).

The following paragraphs retain the earlier2026-11-05 initial review and
campaign history; their acceptance/in-progress statements are historical.
Both final matched native NGP controls finished below 8200 attempted iterations.
Selected step8000 radius0 scores PSNR27.5326, SSIM0.78046, LPIPS0.35916;
training radius1 scores26.2432,0.76203,0.39911. Corrected scalar SDR PNG is the
final-pair input; physical linear EXR/PQ was a separate completed control.

All three radius0 eval renders preserve opaque people, mat and floor, with
remaining blur. Actual ray/source projections localized the original path's
near-camera veil to real softboxes; all24 geometry-selected inner-standoff
views pass manual appearance review. Default grid misses36/9022 known point
cells and ordinary inference halo leaves2. The selected inference policy
unions observed train-witness cells before ONE halo: zero misses among9022
scene/495 body points, all-eval27.543148/.780200/.358109. Both remaining edge
rays now sample their actual witness cells with depth errors below0.1%,
without changing the learned field or EMA. Every final XYZ atlas page was
manually reviewed; peripheral and hidden volume remains physically unverified.
See the live [acceptance evidence](/fsx/tmp/marshawn_001660_hd/final_pair/release_status.md)
and [root visual review](/fsx/tmp/marshawn_001660_hd/final_pair/release_audit/root_visual_review_2009.json).

## What was tested

Campaign started 2026-11-05. Compare the central pair plus props against the full visible scene, three box sizes each, occupancy radius0/radius1. Maximum **22000 total training iterations per variant**, including continuations. No scene-quality outcome yet.

Source inventory read on this host: 66 cameras ×241 frames1660–1900 =15906 files; 66×120 frames1901–2020 =7920 files. All23826 expected paths exist with expected212336804-byte size, total5059136692104 bytes. This is inventory/size validation, not full-source checksum validation.

Sources: `/fsx/oregon/gta_marshawn_2_1/cedar/tiff16/by_local` and `/fsx/oregon/gta_marshawn_2_1_tiff16_f1901-2020_20261105/by_local`. Frame1660 is local000408. Provenance README identifies REDWideGamut/Log3G10 uint16 RGB8192×4320, lens distorted, not PQ. The supplier's join identity tests are documented; not independently re-decoded here.

Prepared artifacts: `/fsx/tmp/marshawn_001660_hd/data`. Direct TIFF decode, gain1/Reinhard/sRGB, aspect-preserving1920×1012, one distortion remap with valid-border masks and half-pixel convention conversion. Train-only pose normalization fixed. All66 source SHA256s and output/mask SHA256s recorded in receipts. Train63; eval HC/DC/KA. Contact sheet manually reviewed. Camera raster and image IDs validated against supplied rig.

Reference rig: `/fsx/oregon/gta_marshawn_2_1_prep_20261028/root/Marshawn_2_1/golden/rig`. Existing independent-frame rig report PASS is prior evidence; current-frame SIFT epipolar/triangulation audit is running. No recalibration claimed.

Frequency preprocessing: 1000 updates/level,16 levels,max8192,patch8,SSIM.95. Train-only shards32/31 on local GPU/dev3 share output directory. Not yet qualified until complete map inventory and overlays reviewed.

Background agents read the local paper and leader/later experiment recipes. Critical references: [leader](static_leader_reproduction_recipe.md), [Luster halo1](luster_000540_face_fix_occupancy.md), [geometry guard](exr_geometry_occupancy_guard.md), [DEC5 lessons](dec5_sequence50_training.md). Full historical75940/106316 schedule is excluded. Dilation must not accumulate; dense bypass can expose untrained fog and is diagnostic only.

## Results

Preparation complete; geometry audit and frequency fitting in progress. **No trained AABB comparison, PSNR/SSIM/LPIPS, or validated occupancy grid yet.**

CPU helper checks cover grid boundaries, all cascades, one-cell dilation, ray intervals and independent volume integration. Frozen scene checkpoint probing and complete visual atlas review remain pending.

## Insights and next steps

Scene includes central fighting pair, surrounding performers, mats, crew and rigging. Central pair localization reviewed on train AA/GA is provisional, not a full silhouette certificate. Sparse foreground bounds must be expanded/verified against limbs, props and floor contact before campaign release.

Local GPU and dev3 are available. Clever-shadow has unrelated active GPU jobs; do not interrupt them. Dev3 system disk is almost full; campaign outputs/caches belong on FSx. Use matching frozen code, log numerical environment differences and keep matched variants on the same host.

Unknown/occluded volume cannot be certified from RGB alone. Export unknown evidence separately; do not equate low learned density with proven empty space.

Current-frame calibration audit completed:137 train pairs, median pair epipolar median0.1253HDpx,9022 tracks with at least3 camera witnesses and<1px reprojection in every support view. Supplied calibration retained. Root reviewed all63 train projections and JA/JB detail; approved provisional candidates for comparative training in `geometry/boxes_approved.json`. Initial two-view scene outliers were rejected; remaining distant scene extrema have three-view support.

Frequency execution now uses a shared advisory-lock queue (4 local workers,2 dev3), retaining1000updates/level. Older sequential workers stopped; completed maps preserved. The queue calls the same progressive fitter and publishes each image atomically after completion.

Controller has passed real isolated GPU fixed→adaptive and ns-eval serialization smoke; this is not a scene result.19 focused CPU tests passed. The voxel bridge synthetic integration removed one known witness cell and detected that exact omission. Full scene experiments remain gated on complete frequency maps.

Dev3 Python3.8 requires a separate argparse.BooleanOptionalAction backport in `compat_py38/sitecustomize.py`; the frozen experiment source is unchanged. Original-scaler32-iteration structural test reached28 real Adam updates with4 AMP skips (iterations0/1/2/7; first Adam update at3); step31 checkpoint finite. The original2-iteration smoke only established forward/backward execution and serialization, not Adam updates. Actual trace is saved in `dev3_controller_smoke/run/optimizer_trace.json`.

### Frequency threshold correction

The replacement shared queue omitted the explicit SSIM threshold and inherited the upstream0.97 default.41 completed maps from that recipe are archived in `frequency_threshold097_archive`; they are excluded from the training inputs. Four original local0.95 maps (AA/AC/BA/BC), proven by the wrapper command log, are retained. The corrected queue explicitly pins0.95 and writes threshold, steps, chunk size, source SHA and fitter-script SHA per map. A regression test deliberately changes upstream defaults and verifies the pinned recipe. Training release remains closed until all63 maps pass the updated strict reviewer. Previous33-map visual audit establishes alignment only, not recipe conformance.

An isolated same-weight evaluator benchmark verified bitwise-identical TCNN predictions, SSIM scores and threshold assignments for64/256/512/1024 patch chunks, including scores within1e-7 of0.95/0.97.1024 patches is used for corrected local fits; evaluator-only speed improved3.54x in this benchmark, not a measured whole-fit speed claim. Evidence: `frequency_chunk_benchmark/results.json`.

Before any scene training, a second comparison confound was fixed: unset render_step_size was derived from each candidate AABB diagonal, and occupancy_update_step_size inherited it. The controller now pins both intervals to the commoncoarse0.001, keeping density-to-opacity admission comparable across boxes. An independent regression test checks different boxes preserve the same intervals. Unused initialspecs are preserved in `campaign_before_common_occupancy_interval`; the current campaign was regenerated with sourceSHA44ac1e6e76c41a472b621665ef123dc39fd0d9a1b7cd6fea4e30e078d97c840b. No actual scene training outputs existed when specs changed.

### Initial zero-frequency control (rejected)

Whole-scene tight radius0 ended by the standard early stopper near8k. All-view metrics:4k PSNR21.1518/SSIM0.654326/LPIPS0.773445;8k20.8172/0.674786/0.767863. BestbyPSNR is4k; fullHD renders saved under `campaign/runs/scene_tight_halo0/lookcloser/cold_seed42/renders_best_step-000004000`. Both4k and8k previews were manually rejected: largely opaque brown cloud with sparse learned patches, not recovered scene geometry. This is not a qualified box/grid result.

Sampling audit128actualtrainrays passed: RGB error<3e-8, origin exact, direction error<1.1e-7. At4k9021/9022SfMwitnesses andall501subjectwitnesses havefrequency0, versus requiredmedianlevels12/14. Only0.1033% frequencycells nonzero; at8k0.1507%. FR0 still leaves high-level weights0.8486, so zero-grid mismatch does not establish solecausation. Fixed256 trainingwarmup until4096 versus adaptiveeval is another integration difference.

At4k renderedcappedrays have residualtransmittance<6e-6 (DC13.18%,HC23.81%,KA1.48% capped; no cappedpixelT>.01), so their budgetlimit does not explain the gross visiblecolorfailure. Native occupancy is100% at4kwarmup and52.86% at8k. FreshpaperSfMfrequencyseeds are being evaluated separately; no occupancy admission is implied.

Dev3 subjectzero-init failed during first4keval in nerfacc traverse_grids; no checkpoint had yet been saved. Failedattempts preserved, isolatedsm89 cache validated on actualpreparedrays/traversal/compositing/backward; see `dev3_cuda89_preflight/result.json`. Futurecontroller savescheckpoint BEFORE eval to preserve trainedstate if evaluation fails.

### Native NGP32 and source color review (November 5)

At the user's request, a clean native `instant-ngp-bounded` control used the prepared HD PNGs, unchanged prepared poses and cube[-32,32]^3. No LookCloser frequency grid, occupancy priors or background compositor. It stopped under the normal early-stopping policy around10.9k; best checkpoint10k (all-view PSNR24.788898, SSIM0.738468, LPIPS0.676372). HC visual inspection shows recognizable but cloudy/blurry geometry; rejected as acceptable final output. Renders and review receipt: `/fsx/tmp/marshawn_001660_hd/native_ngp32/marshawn_001660_native_ngp32/instant-ngp-bounded/cold_seed42/`. This large cube also changes native automatic sample spacing and hash feature spacing; it is a sanity control, not a fair box-size quality comparison.

Two independent source audits agree: TIFF is RGB uint16 REDWideGamut/Log3G10; tifffile RGB equals OpenCV BGR reversed exactly. Independent vendor decode reproduces our PNG to rounding;4096 actual training rays across all63 cameras exactly equal PIL RGB/255. No channel swap, double division or accidental PQ/gamma path found. The existing vendor JPEG is visibly brighter than our declared gain1/Reinhard preview. This is a display/exposure issue still requiring an explicit comparison, not evidence that input loading is correct aesthetically. Evidence: `source_color_validation_HC1660/display_comparison.jpg` and `report/color_decoder_audit/audit.json` under the same root.

User requested corrected PNG and PQ alternatives. Separate immutable color variants are in preparation, with one train-derived global exposure, no per-view white balance and a declared PQ luminance scale. Canonical PNGs and existing frequency maps remain bound to their original data; frequency maps must not be reused as if fitted to the new color variant. Native Splatfacto with train-only triangulated initialization is being staged as an independent reconstruction check. Neither control qualifies an ideal occupancy grid.

The color comparison is now sealed at `color_variants/gain4/color_gate.json`: gain4(+2EV) chosen by manual HC/DC/KA comparison against gain1, gain10.5 and vendor JPEG. Automatic log-mean-to18% suggested gain10.5 but looked washed out and was rejected. All66 views share gain4 and retain as-shot white balance; no per-view grade. SDR PNG and physical linear Rec709 EXR share one remapped linear master, and all3 eval EXRs reproduce corresponding PNGs exactly after Reinhard+sRGB. Reference203nits/unit is explicit. Rec2020 ST2084 PNG16 archives are labelled separately; they are not fed through the normal PIL PNG loader. Physical EXR keeps superwhite data; supplementary PQ archives clip0.455% of channels on average at10000nits.

Large-box native defaults were also inadequate spatially: side64/maxres2048 gives .03125-unit cells (~13 across the central pair), and automatic dt .110851 gives ~4 samples across it. The local native model additionally creates32 appearance dimensions when both appearance flags are false; the corrected control explicitly sets raw_no_appearance_embedding=True. Dev3 control launched17:11UTC with gain4 PNG, same box32, maxres65536/hash21, fixed dt.003, explicit no appearance embedding and normal dynamic batching; max22000, eval/save every2000. These sampling/feature changes mean its improvement cannot be attributed solely to color. A physically linear EXR/PQ-objective control is staged separately with frequency/FAS disabled.

Native Splatfacto train-only9022-point control reached its first2k gate with unchanged original PNGs: PSNR26.12818, SSIM0.819531, LPIPS0.384339. Root reviewed all3 native held-out renders: central people/mats/background are recognizable and gross opaque NGP fog is absent; HC left background still has unsupported dark splats and floors remain blurry. It is an independent reconstruction sanity result, not acceptable final quality or an occupancy-grid certificate. Evidence: `splatfacto_baseline/gate2k_renders/review_contact.jpg`. Cold gain4 Splatfacto campaign is separate, capped22000 with native all-view checkpoint selection.

Corrected native NGP32 failed its2k visual gate: all-eval PSNR21.1554/SSIM0.674755/LPIPS0.712853; lockers/floor recognizable, people largely replaced by opaque cloud. Actual2k checkpoint confirms no appearance embedding; field values finite. The worker was terminated after3581 total attempts (checkpoint2000 preserved), not represented as a2001-update run. This rules out exposure correction as a sufficient fix. A separate native cube4 control uses maxres8192 to retain identical finest world spacing .0009766 and dt.003; all online/final eval chunks explicitly512.

Physical EXR/PQ control launched with vanilla pipeline, LookCloser HDR field only, frequency grid/reweighting/FAS/ARM/priors/compositors disabled. Linear compositing, softplus output and EAG PQ+DSSIM(.3) objective,203nits/unit,3993rays/11pxpatches, cube32, maxres65536/hash21, occupancy128/native256 warmup and dt.003. Full-batch no-optimizer preflight:4,512,090samples, finite gradients, parameter hash unchanged, max reserved10.786GB. Two native PatchPixelSampler configuration defects were isolated before training: constructor defaultpatch32 rounded121rays to0 before patch11 override (native nerfacc SIGFPE); masked rejection passed NCHW to an NHWC helper. Explicit patch11 config and ignore_mask=True avoid both; all66 source masks are asserted entirely valid. No source/model/CUDA edits were needed. PhysicalPQ2kgate is pending; this is not yet a quality result.

### Corrected-color reconstruction screen

All following rows use the sealed gain4 source master. PQ and SDR metrics are different domains and must not be ranked against each other.

| Run | Domain | Selected step | PSNR | SSIM | LPIPS | Visual result |
|---|---|---:|---:|---:|---:|---|
| Native NGP cube4 | SDR | 8000 |24.92330|.754104|.424170|People/props coherent; residual haze and blur|
| Native NGP exact scene.tight | SDR |6000|24.38259|.740420|.396618|People preserved; background/floor still imperfect|
| Native Splatfacto | SDR |10000|27.89001|.832993|.273492|Sharp people; mat/floor clouds and KA streak fail|
| Native Splatfacto + scale regularization | SDR |20000|28.55402|.840547|.268282|Extreme aspect ratios removed; floor artifacts persist|
| Physical EXR/PQ cube32 | PQ |2000|26.03660|.808624|.606547|Rejected opaque cloud/missing upper bodies|
| Physical EXR/PQ cube4 initial gate | PQ |2000|26.80580|.815803|.576567|People return; hazy bootstrap, continuation pending|

Each GS run completed exactly22000 updates, retaining all checkpoints; native NGP cube4 and tight stopped by standard plateau policy before22k. GS strict selection uses all-view PSNR with lower LPIPS inside0.07dB; fresh reload LPIPS differs by about5e-5 from online selection telemetry, documented in receipts. Actual outputs remain raw model renders; no GT compositing, denoising or color repainting.

Native cube4 eight-thousand-step geometry audit over the same128 train-observed SfM rays found median opacity.3303 before a conservative pre-surface margin (vs near1 for failed earlier large-scene control). Median depth relative error−.2290; this is not correct geometry. Default grid misses15 SfM point cells; noncumulative radius1 recovers14, including all7 high-density missed features (tripod/boot/locker/cloth edges). Remaining dim point6349 has very low learned sigma. Grid dilation improves traversal but cannot fix missing field density.

All24 cube4 XYZ slice-atlas pages were manually reviewed at full resolution, covering every128 slice on each axis.603606 cells admitted,1493546 skipped/unverified;61% of admitted centers lie outside all63 training frusta. Broad low/high-Z and outer-X/Y lobes are unverified field structure, not occupied geometry certificates. Evidence: `native_ngp4_corrected_sdr/grid8k_all_slices/manual_review.json` and `halo_8k/report.json`.

The optional LOS objective is independently audited against all44938 actual train observations and all6 geometric interval bundles: exact reconstruction of origins/directions/margins/AABB clips, no evaluation RGB. It penalizes differentiable optical thickness along observed pre-surface segments, not whole voxels or hidden volume. A no-optimizer CUDA probe verified finite gradients, unchanged parameter groups/state hashes/global RNG, and reconstruction after checkpoint load. Only then was a separate matched tight+LOS control authorized; pure baseline checkpoints remain unchanged.

### Relative camera radiometry candidate

Source decoding correctness does not establish inter-camera radiometric consistency. A new read-only screen samples the same44938 train-observed tracks in physical gain4 EXR, using5px linear patches. Independent scalar fitting uses actual Rec709 luminance, a connected63-camera graph, point nuisance values and robust camera log-gains.26030 lower-contrast observations/5287tracks yield relative corrections−.5396…+.5045EV; five held-out TRACK folds reduce median residual consistently, overall.15667→.11051EV. This is repeatable camera-correlated variation, not proof of camera ISO/WB: view-dependent material response, patch footprints and correspondence errors remain confounders. Hair contacts are not matte calibration targets. REDline metadata loading failed; no sensor-setting claim is made.

A separate scalar-only dataset keeps as-shot white balance. Training gains come only from63train tracks; DC/HC/KA gains are predicted from camera positions alone with leave-one-camera-outRMSE.11945EV. Both float32linearEXR and SDR PNG preserve exact numeric poses/K/splits. Contract: `color_variants/gain4_scalar_trainfit_poseeval/complete.json`. The matched scale-regularized GS run changes only dataset exposure normalization and recolored train-only PLY, coldseed42,max22000.

To avoid a target-brightness shortcut, comparisons undo the fixed candidate exposure in LINEAR space before the original gain4 display transform, then score against unchanged gain4 GT. GT roundtrip differs by at most1PNGcode. At matched8k, saved-PNG common-reference metrics improve27.8417→28.4530PSNR,.831779→.834316SSIM,.280984→.280212LPIPS. KAfeetROI PSNR improves21.7583→29.6612; its broad translucent pillar is visibly reduced. DCmat remains cloudy; HCfloor has improved LPIPS but worse PSNR. This candidate is not yet a final spatial-quality acceptance. Evidence: `splatfacto_gain4_scalar_scale_reg/common_gain4_step8000/report.json` and its all-view/ROI comparisons.
