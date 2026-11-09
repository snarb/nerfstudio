# Foreground semantics over a frozen full-scene radiance field

Current outcome: both2064-update prior experiments completed. The p=.01 initial removal prior greatly reduces physical floor fragments compared with p=.99, while residual rig/bag fragments and one ambiguous wrist/sleeve gap remain. Neither is a clean-asset promotion. Direct-LC independent semantics is the next CPU-only preparation; no direct-LC fitting/export is authorized.

## What was tested

CPU feasibility, completed native-visibility capture and a completed32+32-versus64 semantic-head technical probe for separating the two frame1660 actors without fitting background colors inside their bounded volume. The local paper motivates the radiance-field/frequency design but does not specify this semantic extension. Existing source-layer, RAW-coverage and bounded-background audits motivate a separate view-independent3D class field.

The chosen first source is native NGP HD selected190000, checkpoint SHA256 `dfa0305b25e5ab6fbda2977e1beb94b22bef132ed5842209e047e3d03f08d8ea`. It represents the full scene; its frozen RGB/density/traversal remain unchanged while an independent128³ trilinear grid can learn actor membership. Splatfacto scale-regularized20000 remains a possible later source, but its older pre-k4 raster, virtual subfloor Gaussian contributors and failed floor/feet visual gate require additional calibration and visibility work first.

## Results

| CPU check | Result |
|---|---|
| Native source PNG versus current common-domain PNG | All66 files byte-identical by SHA256; mapping **identity** |
| Native camera → frozen floor camera | `world_to_box @ native_pose` equals current pose exactly for all63 TRAIN cameras |
| Raster/intrinsics/distortion | Exact agreement for all63 TRAIN cameras |
| Selected checkpoint metadata |38 model tensors, no other pipeline-state keys;594,816,110 bytes; CPU memory-mapped inspection, CUDA uninitialized |
| Declared TRAIN / supervised TRAIN |63 /60; CE,JE,NE are entirely UNKNOWN and supply no labels |
| Semantic queries |69,888 rays:30,720 eroded actor interiors,30,720 near clear,7,680 far clear,768 reviewed bag interiors |
| CPU mathematical tests |8 passed; no optimizer or CUDA context |
| Historical dev3 runtime | Read-only SSH confirms Torch2.4.1+cu121, CUDA build12.1, CUDA uninitialized |

All query labels come from existing TRAIN-only nominal-coverage/source-support archives and reviewed bag interiors. Positives require nominal coverage≥.999 and an interior distance>2HD pixels. Negatives require valid/source-eligible pixels outside W64 support and UNKNOWN, or the already qualified bag interior. Protected hair, contacts, broad mat UNKNOWN and the three whole-raster UNKNOWN sources are not relabeled. Nominal coverage is a semantic filter, **not optical alpha ground truth**. The69,888-ray balanced sampling policy is a classifier experiment, not an unbiased full-image objective.

The first preparation generation exposed an empty-camera case at CE; its incomplete directory is preserved. V2 handles explicit empty records and completes all63 source records.

Source receipts and artifacts:

- [TRAIN-only query manifest](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_CPU_v2/manifest.json)
- [66-image byte identity audit](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_RGB_identity_CPU.json)
- [Checkpoint CPU metadata](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_CPU_v2/checkpoint_CPU_metadata.json)
- [Capture controller plan](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_capture_plan_v2/controller_plan.json)
- [CPU readiness and resource policy](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_capture_plan_v2/CPU_readiness.json)
- [Root review template; capture/fit both initially false](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_capture_plan_v2/root_review.template.json)

The initial visual review compared the existing LC130316 actor contact sheet with native200000 detail pages. Native hair outlines look substantially cleaner, while its clothing detail is weaker. Native feet retain modeled floor/mat/bag surfaces. This supports testing semantic separation on its physically available background representation; it does not establish a clean actor asset or superior detail.

A subsequent [exact selected190k actor review](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_capture_plan_v2/selected190k_actor_review.jpg) confirms that assessment in all3 views. Its [hash/crop receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_capture_plan_v2/selected190k_actor_review.json) is diagnostic-only; these source-view review panels never enter the fit cache.

## Semantic objective and its limits

For frozen native termination weights `w_i=T_i(1-exp(-sigma_i*delta_i))` and a3D actor probability `p_i`, render class masses:

`Q_actor=sum(w_i*p_i)`, `Q_nonactor=sum(w_i*(1-p_i))`, `Q_escape=1-A`, with `A=sum(w_i)`.

The objective is native-opacity-weighted conditional binary cross-entropy on `Q_actor/A`. This is equivalent, up to constant escape terms, to target masses `[A*y,A*(1-y),1-A]`; it never demands actor mass1 when frozen alpha is less than1. Native weights, positions and labels are detached. No gradient reaches RGB, density, occupancy, camera poses, or the native optimizer. The classifier is view independent, and samples outside the accepted output box have actor probability0 while their full-scene density still participates in native visibility during fitting.

This objective does **not** automatically solve partial foreground followed by background. A positive ray with foreground weight.60 and background weight.39 gives positive-class gradients to both layers, in ratio.60:.39. A CPU test explicitly retains this counterexample. Even full-scene alpha≈1 does not certify opaque foreground; eroded semantic positives reduce boundary ambiguity but cannot eliminate inaccurate geometry or actor-support mistakes. Cross-view negative observations must disambiguate those layers, and some hidden geometry will remain unidentifiable.

The physical export is `sigma_actor=sigma*p_actor`, followed by recomputation of transmittance. It differs from the diagnostic contribution partition `sum(native_w*p*RGB)`. Removing an occluder can expose hidden actor/uncertain geometry; both outputs need separate visual inspection. This operation can remove unwanted density but cannot recover detail absent from the frozen source.

Soft classification also does **not** guarantee removal of dense background: `sigma=1e9,p=.001,delta=.003` still gives physical actor alpha1, despite assigning99.9% probability to nonactor. The eighth CPU test explicitly reproduces this failure. CaptureV2 therefore saves actual sigma, interval endpoints and native sample RGB alongside weights/positions, and reports maximum sigma/interval optical depth. Any later semantic result must measure recomputed physical actor alpha and inspect physical renders. Predeclared read-only export alternatives are hard membership thresholds.1 and.5, or a separately reviewed smoothstep with exact zero below.1 and exact one above.5. These are different3D selectors with separate hair/contact/contour-safety gates, not automatic improvements or optical-alpha fits. V1 capture preparation is preserved but superseded because it did not retain density/interval/RGB arrays.

In that example alpha<.01 requires `p < -log(.99)/(sigma*delta) = 3.35e-9`, far below.001. A semantic class belief is not automatically a physical density fraction. Adding an empty-ray optical-alpha target would not resolve this safely: hidden actors behind a known nonactor could then be penalized throughout the ray. Visibility-weighted class fitting and physical export selection must remain separate hypotheses.

## Controlled GPU plan

The launch-ready first phase is **read-only capture with zero optimizer creation**, on dev3's historical Python/Torch runtime. The existing controller acquires its global device lock, rejects a busy device, sets the exact physical UUID before Torch import, and records process/GPU/OOM evidence every15seconds. The worker verifies its actual compute-PID→UUID mapping and rejects foreign compute processes. Guards require8GiB available RAM and20GiB free disk initially, abort below2GiB available RAM, above6GiB worker RSS or12GiB GPU reservation, and limit this read-only technical capture to1800seconds.

Capture strictly restores all38 source model tensors, restores the checkpoint RNG, checks model/checkpoint/query hashes and unchanged RNG afterward, and records actual packed samples/weights. Native renderer versus custom composition must agree within2e-5 per chunk. Nine saved190k **model-render** samples independently verify full-scene and accepted-box RGB parity; these are never fit-cache rows and use no heldout source RGB/masks. Runtime transformed rays must match separately generated floor-camera rays after applying the transform exactly once.

An actual dev3 write preflight found that its ubuntu UID1000 could not create output beneath the new package owned by olive's UID1001. Only the new package-root ACL was changed to grant UID1000 access; no code/source bytes changed. A subsequent [remote preflight receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_capture_plan_v2/dev3_write_and_input_preflight.json) proves mkdir/write/read/remove under ubuntu and all245 input hashes. This was a preparation permission issue, not a checkpoint or training failure.

The capture reports frozen native alpha and maximum accepted-box actor contribution for every supervised TRAIN camera. Its initial fitting stub refused any positive cap/A<.95; that conservative preparation gate was superseded after root reviewed the source evidence. The separately reviewed technical worker retains the original positive labels, reports the irreducible conditional likelihood bound and does not filter or normalize them to insidecap. No clipping of targets or invented opacity labels is automatic.

### Completed native read-only GPU gate

Root authorized captureV2 only. On dev3's leased L40S the worker captured all69,888 rays and returned0, with no optimizer construction or updates. All38 model tensors, source checkpoint, TRAIN queries and restored native RNG remained unchanged; model digest `059a8f2760c4b59b42870a7096e5c55c102315fffcbe9585f5bef678da06b9ee`. Saved190k-model parity across nine pixels had maximum full-scene RGB error1.79e-7 and AABB-inside error1.19e-7. All runtime coordinate checks passed. Final supervisor evidence shows the worker exited, GPU memory returned to3MiB, no compute processes remain and no OOM occurred. [Capture receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_capture_plan_v2/capture/capture.json), [baseline parity](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_capture_plan_v2/capture/baseline_parity.json), [supervision](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_capture_plan_v2/manual_supervision.jsonl).

There are136/30,720 positive rays with accepted-box cap/A<.95. Root's subsequent source review found genuine hair, cloth and shoe material plus some partial hair gaps; these are not136 certified perfect semantic labels, and the complete set remains unchanged for the controlled head test. Actual captured maximum density is65,659,972 globally and2,526,100.75 inside the accepted box. The latter maximum has native numerical weight0 at the inspected ray. Even hypothetical uniform p=.001 inside the box yields recomputed alpha>.1 on3,895/39,168 negative queries and alpha>.99 on15; this is a mathematical counterfactual, not fitted semantics or alphaGT. [Full capacity/density audit](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_capacity_CPU_v1/capacity_and_density.json).

### TRAIN bag-positive qualification before fitting

Root's separate native review identified91 distal-bag/floor pixels mislabeled as actor in IA/LA/LB/MC. None coincides exactly with the fixed sampled69,888 queries, but23 IA and2 LA reviewed pixels would pass the high-coverage/eroded-interior eligibility rule. Thus this rule does not certify true actor membership. Their nominal connected components have63,200/61,948/68,737 pixels in IA/LA/LB, so automatically deleting a whole component would remove actor material.

The [sampled-query source ROIs](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_bag_positive_audit_CPU_v1/query_source_ROIs.jpg) and [nearest positive native source crops](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_bag_positive_audit_CPU_v1/nearest_positive_native_sources.jpg) expose nearby sampled positives for manual qualification. IA[1019,567] and IA[1013,562] appear on bag folds in native source; they need explicit future UNKNOWN qualification before fitting. Other nearby shoe/contact positives require care. No query, frozen mask or capture-manifest label was changed; no new negative or whole-ray emptiness claim was introduced. [Exact91 overlap receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_bag_positive_audit_CPU_v1/reviewed91_overlap.json).

Root confirmed exactly those two IA bag-positive queries: camera-ray IDs302/150, global query IDs40622/40470. Their confidence becomes0 (UNKNOWN) only in new head fitting; targets remain1, and all original archives/masks stay untouched. Both have cap/A=1, illustrating that high native capacity does not certify semantic membership. IA's other nearby positives and LA/LB/MC shoe/ankle positives remain. The [qualified-label receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_head_technical_plan_v1/label_qualification.json) is root-reviewed, and root personally inspected all seven136-case overview sheets; [capacity source review](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_capacity_source_review_CPU_v1/root_manual_review.json).

### Completed head-only technical gate

Root authorized only32 updates, exact restoration plus32 more, and an independent64-update run. All three returned0 with native checkpoint/caches unchanged and no native model/optimizer or source RGB targets loaded. Resume-start head, Adam, global RNG and sampler states match exactly; the64 camera histories match. Subsequent resumed-versus-independent64 trajectories have maximum logit difference9.06e-6, logit RMSE1.18e-7, maximum probability difference2.38e-7 and probability RMSE4.56e-9. These measured CUDA grid-sample backward atomic differences are not bitwise equivalence. [Comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_head_technical_plan_v1/CPU_comparison/comparison.json), [independent root technical review](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_head_technical_plan_v1/root_technical_review.json).

The independent64 anchor is SHA256 `4c79ce1dd111d584d12ff0b09d959c4f44870a081aee84cdeee38b4041a456c3`. Its probability range is.95347–.99732;92.976% of grid nodes remain exactly at initial logit4.59512 (p=.99). This is an explicit preservation prior, not semantic certainty. Native-weight0 samples have no direct class gradient; shared vertices can learn from other samples, and unchanged values are not a perfect unobserved-node certificate. No hidden support gate or additional prior was introduced. Hard/exact-zero selectors still retain untouched p=.99 regions. The64-step result is a restoration test, not learned cleanup.

The separately reviewed first-physical plan resumes64→2064 (2000 additional head updates) and stops at a mandatory physical visual gate: all3 full-HD views, box-only nohead control, soft/hard/exact-zero selectors with recomputed transmittance, and a separately labeled original-weight contribution partition. Traverse all originally occupied native intervals with sampler sigma_fn=None and no opacity early termination; preserve original core/EMA/bounds and state hashes. Native-grid-unqueried geometry remains unidentifiable. Report fixedV1 PSNR/SSIM/LPIPS plus hair/contact/shoe/floor crops and newly visible original-weight0 sample mass. EVAL source RGB/masks are post-render evaluation only. No automatic continuation follows the visual gate;256³ remains unlaunched.

The [sealed first-physical controller](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_first_physical_plan_v1/controller_plan.json) contains four jobs:2000 head updates, TRAIN-only class distributions, physical rendering, and post-render metrics/source guards. TRAIN distributions exclude the two qualified UNKNOWN queries and separate positive, negative, bag, near/far-clear and136-capacity groups; native-visibility class posteriors and unweighted/visibility-weighted sample probabilities remain distinct. Thresholds.1/.5 and smoothstep[.1,.5] stay fixed, without EVAL-only threshold selection or automatic promotion. Eight semantic and four physical-selector CPU tests plus four worker dry-runs passed under the actual dev3 runtime. [Actual remote write/input preflight](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_first_physical_plan_v1/dev3_actual_preflight.json) verified115 bound files and four specs. The subsequent authorized model/head/RNG/traversal results are recorded below.

### Completed2000-update head and TRAIN audit

Root authorized the four bounded phases. Head64→2064 and the TRAIN-only audit returned0, with exact64-start restoration, unchanged native checkpoint/caches and no native updates. Head2064 SHA256 is `d0bc8f42e53d9e38e1502da91c203ead0d93ab2468fce4ae577bdca3a9b4a228`. TRAIN positive native-visibility conditional posterior has median.999940 and p01=.97605. The768 qualified bag negatives have median.01625/p99=.0630;3/768 exceed.1 and none exceed.5. Across all39,168 negatives,4.12% exceed.1 and3 exceed.5. These are classifier diagnostics under native visibility, not physical alpha or clean-asset evidence.

All136 original capacity cases remain positive: their inside-native-mass conditional posterior has median.999934/minimum.92274, while the full-native posterior remains capped (median.8072). The inside-cap ratio is explicitly diagnostic and never changes fitting targets. The negative inside-native-mass tail still reaches≈.99, consistent with weak/hidden preservation priors. [TRAIN distributions](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_first_physical_plan_v1/TRAIN_distributions/distributions.json).

The first physical renderer stopped before producing images because an overly strict guard required literal background policy `black`. The historical native config uses `random`, whose exact bound `RGBRenderer.combine_rgb` returns unblended compRGB (black equivalent) during rendering. A CPU proof under actual dev3 Torch2.4.1 confirms exact random/black equality, unchanged RNG, global background overrideNone and no CUDA initialization. Head2064/TRAIN results are unaffected; original failed logs and terminal controller state are preserved. The [narrow two-job render/metrics recovery](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_render_recovery_v1/controller_plan.json) changes only that guard and adds provenance; no head rerun, threshold/prior/support/sampler change or weakened identity gate is introduced. [Source-qualified CPU proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_render_recovery_v1/background_policy_CPU_proof.json), [guard-only diff](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_render_recovery_v1/renderer_guard_only.diff).

### Completed physical visual gate: p=.99 prior fails isolation

Root authorized the recovery; all three full-HD physical renders and fixedV1 evaluation returned0. Native38 tensors, head, original occupancy and restored RNG remain unchanged. Sampler sigma_fn=None/no opacity early termination was verified. The box-only control matches selected190k metrics to8.35e-8dB PSNR,9.93e-9 SSIM and4.97e-9 LPIPS. Final supervision shows no live worker/compute process,3MiB GPU memory and no OOM. [Physical receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_render_recovery_v1/renders/complete.json), [metrics](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_render_recovery_v1/visual_gate/metrics.json).

| Render, same fixedV1 actor targets | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Nohead accepted-box control |30.568816|.839111|.047971|
| Physical soft sigma*p |30.213129|.837091|.051092|
| Physical hard p≥.1 |30.026098|.836480|.051803|
| Physical hard p≥.5 |29.864843|.835102|.053313|
| Physical exact-zero smoothstep[.1,.5] |29.962618|.835860|.052595|

All three [body comparisons](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_render_recovery_v1/visual_gate/all_actors.jpg) and all nine [hair/contact/feet comparisons](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_render_recovery_v1/visual_gate/all_details.jpg) were inspected. Body/hair appearance is broadly retained, but every physical selector retains or exposes extensive fragmented floor around the feet. None passes the clean-asset goal. The original-weight semantic contribution partition looks cleaner because it retains old occlusion; it is explicitly a nonphysical diagnostic and must not be promoted as actor isolation.

At2064,91.75% of grid nodes remain exactly at the preservation prior. Native numerical-weight0 samples inside the accepted box number49.66M/DC,40.49M/HC and29.48M/KA; their mean actor probability remains.9768/.9776/.9709. Hard.5 exposes native-weight0 mass>.01 at23,227/72,222/6,171 pixels respectively. These are visibility/prior diagnostics, not physical emptiness or alphaGT. Good visible-class scores therefore do not establish clean3D export.

Root's next requested CPU-only controlled proposal changes only the initial actor probability to.01: fresh seed42/head/Adam/sampler for2064 updates matching the original64+2000 camera history, unchanged caches/two UNKNOWN overrides/136 cases/objective/LR/grid/selectors. It is an explicit removal prior for unobserved regions, with hair/feet risks and no free-space claim. No further GPU work is authorized by the completed gate itself.

### Completed p=.01 counterexperiment (preparation provenance)

[CPU readiness](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_prior001_candidate_v1/CPU_readiness.json) and the [controlled-factor proof](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_prior001_candidate_v1/controlled_factor_proof.json) bind the2064 fresh seed42 camera draws to the actual original2064 history exactly. The conditional-class objective helper remains byte-identical and selector mathematics is AST-identical. Initial probability is explicit in the spec, initializer, checkpoint, completion, TRAIN audit and renderer's initial-node count; hardcoded.99 diagnostics are removed. Nine semantic/five physical CPU tests and all five worker dry-runs pass on actual dev3, including a real write/remove probe and126 bound input hashes. These were prelaunch checks; root subsequently released the exact bounded five-job DAG, which completed below.

The [five-job proposal](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_prior001_candidate_v1/controller_plan.json) stops after fresh2064, TRAIN distributions, three physical views, fixed evaluation and paired CPU body/hair/contact/feet/hidden-mass review. Root-template authorizations were false at preparation time; the later actual root release authorized only the bounded five jobs. Only the explicit unknown-region prior changes; no support gate, new label, density/RGB fitting or semantic certainty/empty-space claim is added.

### Completed p=.01 physical gate and matched prior comparison

The released fresh2064 head, TRAIN audit, three physical full-HD renders, fixed evaluation and paired CPU review all returned0. The head checkpoint SHA256 is `1d493003ded6d56fb652515fea8d79d905eafe59cb3a7dddd2f5577997d9b90a`. All2064 actual camera packets match the original p=.99 trajectory; native geometry/RGB, objective, Adam/LR, grid, labels and physical selectors remain fixed. Source/model/head/RNG identity gates pass; box-control PSNR agrees with selected190k within1.34e-7dB. [Paired comparison](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_prior001_candidate_v1/paired_review/comparison.json), [physical receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_prior001_candidate_v1/renders/complete.json).

| p=.01 render, same fixedV1 targets | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Nohead accepted-box control |30.568816|.839111|.047971|
| Physical soft sigma*p |29.573848|.832209|.056482|
| Physical hard p≥.1 |29.087799|.829026|.058507|
| Physical hard p≥.5 |29.006416|.828417|.058906|
| Physical smoothstep[.1,.5] |29.059815|.828818|.058693|

For hard.5, exterior alpha>.1 outside the fixed actor-mask3px dilation falls from1,277,493 box-control pixels to7,872 across three views (99.384% reduction). This is a fixed diagnostic region, not certified actor alphaGT; distant residual components still remain. Newly visible native-weight0 mass>.01 occurs at279/DC,42/HC and0/KA pixels versus23,227/72,222/6,171 under p=.99. Hidden-sample mean p becomes.02326/.02413/.03051. This shows strong sensitivity to the explicit unobserved-region prior, without proving those regions physically empty. TRAIN positive median conditional probability is.98402 (p01=.86127), bag median5.32e-5; all136 capacity cases remain, inside-cap posterior median.9822/minimum.7785. [TRAIN distributions](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_prior001_candidate_v1/TRAIN_distributions/distributions.json).

All three body and nine hair/contact/feet guards are saved in [paired body](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_prior001_candidate_v1/paired_review/all_actors_prior_pair.jpg) and [paired details](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_prior001_candidate_v1/paired_review/all_details_prior_pair.jpg). Physical hard selectors preserve broad actor appearance while removing much floor/bag density. Soft sigma*p still leaks, as predicted. Small HC rig and DC bag/shoe-contact remnants plus distant islands prevent claiming all artifacts removed. No new thresholds, labels, support gates or automatic continuation were introduced.

### Fixed interior versus boundary CPU diagnosis

The existing fixed region helper defines core as exact Euclidean distance>8px and boundary as original mask minus core. Original sealed crops, AlexNet LPIPS without resizing and neutral exterior=.5 remain fixed. This diagnostic never redefines primary V1 metrics or fit labels. [All72 measurements](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_fixed_core_boundary_CPU_v1/metrics.json), [all3 body guards](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_fixed_core_boundary_CPU_v1/all_actors_fixed_guard.jpg), [all9 details](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_fixed_core_boundary_CPU_v1/all_details_fixed_guard.jpg).

| Render | Core PSNR / SSIM / LPIPS | Boundary PSNR / SSIM / LPIPS |
|---|---|---|
| Nohead box |32.710591 / .856139 / .02405054 |27.856511 / .919930 / .01287970 |
| p=.01 hard.1 |32.411254 / .854870 / .02470459 |25.439596 / .893226 / .01954493 |
| p=.01 hard.5 |32.351933 / .854656 / .02483967 |25.348731 / .891937 / .01979383 |
| p=.99 hard.5 |32.656404 / .855776 / .02442106 |26.509038 / .909820 / .01667116 |

Five of six individual actor cores change by at most.00750dB under p=.01 hard.5. KA gray changes by−2.13762dB, dominating the aggregate. Its largest changed interior component has231pixels around[953,471]..[971,498] on a bag/floor region; a smaller75pixel component around[1070,263]..[1080,278] lies under the clasped hands/wrist. Root personally reviewed calibrated native source crops: the large change is visibly bag/floor removal, while the wrist/sleeve-seam gap remains ambiguous. Thus core metrics contain annotation confounding but do not establish that every fidelity change is annotation-only.

### Calibrated RAW context and potential TRAIN-node support

[Four native-source contexts](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_residual_source_CPU_v1/all_native_source_contexts.jpg) bind actual rig calibration, source hashes, geometric pixel footprints and original RAW-display contract; these are source displays, not optical alphaGT. Root's [manual review receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_residual_source_CPU_v1/root_manual_source_review.json) records HC[875,602] as a visible floor-mounted rig/clamp, DC[1097,572] as bag/shoe contact and the KA wrist gap as ambiguous. No masks or labels change, and an occluding source object cannot make the full ray empty.

A separate CPU classifier-only ray audit identifies nodes touching sampled p≥.5 intervals and accumulates frozen TRAIN native-weight times trilinear basis on those nodes. HC touches36nodes supported by10positive and23negative query rays; leading positive exposures are LA[1020,367] global54223 and MA[1221,435] global57826, visibly face regions in TRAIN source. DC touches20nodes,85positive/336negative queries, led by MA[1039,574] global57771, LA[871,558] global54594 and LA[881,557] global54586 on shoe/contact sources. [Exact potential exposures and source hashes](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_residual_exposure_CPU_v1/potential_exposure.json), [HC TRAIN source patches](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_residual_exposure_CPU_v1/HC_isolated_fleck_potential_TRAIN_support.png), [DC patches](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_native_semantic_residual_exposure_CPU_v1/DC_gray_shoe_bag_edge_potential_TRAIN_support.png).

This potential shared-node exposure is neither historical training gradient nor an actual EVAL density-contributor trace: EVAL sigma/depth sample arrays were not saved. These face-source queries may support different locations along the whole HC classifier ray; without selected physical EVAL weights/depth, this establishes neither residual contribution nor face-to-floor geometry aliasing and cannot justify source relabeling. Native-weight≤1e-8 samples are omitted only from this exposure diagnostic, with no training/filter change. An actual residual contributor trace would require a separately reviewed read-only model query. All this follow-up is CPU-only; no new GPU work or automatic promotion occurs.

## Insights

A full-scene RGB model avoids forcing observed background color into a bounded actor field, but frozen native visibility still leaves hidden class membership underconstrained. Small class probabilities do not imply small physical opacity, and retaining an unseen-region foreground prior can expose large unwanted layers after genuine background removal. The completed controlled prior test greatly reduces hidden-density artifacts, while showing residual geometry/class ambiguities and boundary/source-label limitations. Direct-LC semantics must fit its own captured native visibility with identity floor coordinates; transferring this NGP head is outside the proposed scope.

Code: `scripts/marshawn_frozen_semantic_head.py`, `scripts/prepare_marshawn_frozen_semantic_cpu.py`, `scripts/run_marshawn_frozen_semantic_probe.py`; tests: `tests/test_marshawn_frozen_semantic_head.py`. This is an opt-in prototype and does not modify the existing LookCloser implementation or active campaigns. Architecture paragraph, if promoted: “An independent frozen-field semantic grid may learn view-independent actor membership from native full-ray termination weights. Native RGB/density remain immutable; physical actor export recomputes visibility through a separately reviewed3D membership selector. UNKNOWN source regions, optical-coverage uncertainty and dense-background residual opacity remain explicit.”
