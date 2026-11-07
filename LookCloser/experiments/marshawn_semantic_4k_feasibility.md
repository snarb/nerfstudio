# Semantic LookCloser4K: isolated adapter feasibility

## Scope and current source

Read-only inspection of the frozen R0 source at
`/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_candidate_v2_20261106/code`.
No worker/model, dataset, release or queue was changed. Bind the eventual
completed R0/R1/W32 winner later; do not assume today's R0 is final ancestry.
The existing4K source generation has63 TRAIN3840×2024 and three byte-identical
FHD EVAL1920×1012. Supplementary EVAL4K diagnostics stay separate.

## What already supports mixed TRAIN/EVAL resolution

The dataparser uses per-frame intrinsics/dimensions and separate TRAIN/EVAL
caches. TRAIN63 is homogeneous4K; EVAL3 is homogeneousFHD. The sampler requires
a homogeneous TRAIN cache, not equal TRAIN/EVAL dimensions. Its raster
validation uses actual image dimensions, and V1 evaluation compares the
rendered camera raster with its unchanged FHD target. `downscale_factor=1`,
no pose auto-scaling and the floor-frame AABB remain exact.

The4K loss/confidence rasters already derive from original8K actor/hair/veto
archives, not HD upscaling. W32/W64 widths remain native RAW pixels; reviewed
physical morphology uses33/65/129 kernels rather than17/33/65. Preserve those
existing source receipts, original positive/unknown/veto semantics and MC's
known inherited bag inclusion. These rasters govern loss/sampling; they must
not silently rebuild or redefine the3D semantic admission support.

## Blocking source contract and safe adapter

Frozen `marshawn_semantic_admission.load_support` explicitly requires its
HD `transforms_sha256` and `source_ready_sha256` to match the supplied data.
The worker currently sets `semantic_data=spec['data']`, so simply pointing it
at4K correctly fails. Do not edit old receipts, replace hashes, or skip checks.

Use a new isolated generation with a two-source derivation adapter:

1. TRAIN data is the new source-bound4K generation with its own complete map
   inventory and reviewed READY. Existing HD data remains the geometry source.
2. Invoke the unchanged strict semantic loader against the original HD data,
   original manifest and original grid. The adapter independently verifies a
   hash-bound HD→4K derivation receipt: exact63 camera order/splits,
   original poses/world-to-box/AABB,2×TRAIN intrinsics/size, unchanged EVAL9
   files/rows/crops, native confidence lineage, and frozen photometry.
3. Require the same source grid bytes and final mapped128³ boolean support,
   currently mapped SHA
   `d60baa9186a63f3769204b93cb46dcfd2ea16e3d41a72ca788934fe19bd2659f`.
   Original manifest SHA is
   `fce7fc24735cd62d5242ce719964c5d6d61291feef1cc1c536c12f2ce194b15e`;
   grid archive SHA is
   `e804d736dd766e8adf11df07e346a6317af87e0b106a94dfc524ab32f1690d71`.
   Preserve the whole-cell conservative cover, W64 pad1 semantics,103 witnesses
   and two qualified shoe anchors. A W32 loss variant does not imply narrowing
   this W64 admission grid.
4. Retain density-selector, EMA/core update, native dilation policy and actual
   occupancy-capture code unchanged. R0/R1/W32 choices come only from the chosen
   reviewed parent. Freeze a new minimal config/derivation adapter diff and
   explicit release; never reuse the current generation's19-artifact release.

No ordinary resume check is relaxed. A cold4K comparison is a new generation;
resuming HD optimizer state into4K would be a distinct continuation experiment,
not the default resolution comparison.

## Frequency contract and causality limitations

All63 maps must be present, finite scalar resolutions with original metadata
schedule16levels/min16/max8192/hash23,patch8/stride8 and correct source hashes.
The4K map shape is253×480, exactly covering2024×3840. ExistingHD126×240 maps
cover1008 of1012rows, with the last4rows handled by existing clamp policy.
No HD map resize, partial map set, missing-map fallback or patch16 is acceptable.
Patch16 would introduce an8-row uncovered4K tail and is separately deferred.

The proposed LC experiment is explicitly a **4K RGB/confidence +4K4000-step
frequency-map bundle**. HD maps used1000 updates/level, so it is not pure
resolution causality. Keep all censor masks/counts and non-promotion labels.
Lower censoring is not a proxy for better3D reconstruction. Native4K remains
the cleaner resolution-only control because it does not change this learned
frequency preprocessing bundle.

Another measured-code limitation must be exposed: current R0 uses `legacy`
projection, which multiplies scalar2D frequency by pixel focal/depth. Doubling
pixel focal at4K doubles that product for unchanged2D scalar values, although
the2D regressor's scalar resolution is per unitUV. `normalized_jacobian` has
explicit normalized handling but changes other mode semantics too. Do not
silently switch projection inside the4K adapter. Preserve the selected parent
mode, record resulting grid-level/clamp distributions, and treat projection
as a separately reviewed factor if needed. Existing1024-update receipts must
show actual finite updates and eligibility rather than config-only checks.

## Memory and parity plan

Actual stored shapes/dtypes imply these63-camera CPU allocations:

| Component | GiB |
|---|---:|
| float32 TRAIN RGB cache |5.472|
| sampler five bool rasters + two float32 weights + actor union |6.384|
| int64 valid-pixel index pool upper bound |3.648|
| cached calibration-valid boolean images |0.456|

These total15.96GiB before core/band/FAS index pools, Python/Torch overhead,
image-stack construction peaks and evaluation cache. Actual current HD semantic
producer PID718446 measured RSS7,822,964KiB and peak RSS8,231,144KiB; its large
virtual reservation is not resident RAM. Budget approximately24–36GiB resident
per4K run initially, with a measured CPU-cache admission gate and conservative
headroom; this estimate is not a measured4K training peak. Keep frozen
`images_on_gpu=False`, `masks_on_gpu=False`, `cache_train_rays=False` and
`cpu_fas_prefetch=False`. Full ray caching alone would add at least10.94GiB.
Do not launch eight4K runs based only on available VRAM. At this read-only
check, silver reported MemTotal1,486,024,808KiB and MemAvailable1,417,876,104KiB
(approximately1.32TiB available), so host RAM currently has ample aggregate
headroom; still measure each new4K cache peak and recheck at admission.

After root reviews the selected ancestry and all63 maps, the concrete gates are:

- CPU parser/cache dry run:63 homogeneous4K TRAIN,3 exactFHD EVAL, unique complete
  camera IDs, full aligned RGB/valid/confidence raster indexing and measured RSS.
  Preserve historical cache/FAS ordering semantics; do not fix unrelated sampler
  behavior in this comparison.
- Source adapter negative tests: reject altered pose/AABB/camera membership,
  changed EVAL pixels, wrong scaled intrinsics, any unbound mask/map or HD grid,
  and missing source hashes. Compare all mapped support bytes and boundary/witness
  queries with the original geometry source.
- Separately released short actual-Adam/AMP smoke and real1024 frequency-grid
  update, actual semantic/core/active snapshots, finite confidence-normalized
  actor objective and strict resume. Check GPU/CPU peaks and all3 unchanged V1
  cameras before production. No GPU run is authorized by this document.
- A completed3D comparison uses identical primary V1 scores and per-view/actor
  visual/geometry checks. Optional4K model views are supplementary matched-raster
  diagnostics under separate release. Keep temporal stability and hair/feet
  boundaries explicit; do not declare the map/support ideal from metrics alone.

## CPU adapter implementation and actual checks

Root authorized an isolated CPU implementation. Prepared generation:
`/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_4k_adapter_proposal`.
All1495 parent source files remain byte-identical; only standalone
`marshawn_semantic_4k_derivation.py` is added. It validates the dual-source
receipt and returns a proposed configuration without constructing a model or
optimizer. The unchanged strict semantic loader still receives the originalHD
geometry source. Ordinary training/resume and the semantic field/EMA/core/
occupancy code are untouched; there is no production entrypoint release.

The real CPU check verifies654 source hashes, all66 camera rows by identity,
exact ordering within each split, all9 EVAL bytes/crops, native-mask lineage,
and the original mapped128³ support SHA. The exporter changes only overall
TRAIN/EVAL interleaving, so a naive positional zip of all66frames would be
incorrect. The actual frozen parser confirms63TRAIN3840×2024 and3EVAL1920×1012,
with identical poses/AABB and exactly scaledTRAIN intrinsics.

The normalized real config differs in exactly two fields:
`pipeline.datamanager.data` and `pipeline.datamanager.dataparser.data`.
The entire model config, including `semantic_data` pointing toHD, is identical.
Final CPU CLI/config checks hide CUDA devices and confirm CUDA was not
initialized; no model or optimizer was instantiated. Ten tests reject changed
pose/intrinsics/EVAL/splits/order/duplicate cameras, incomplete/unbound maps,
source tampering, and attempts to write READY or overwrite either dataset.

Adapter SHA `f7bdb5fe24e2caaa64836c3f27a2af69be570e05550046427f0f199ff874f380`;
derivation receipt SHA
`8e4286677c6f9ea188fa089dce3add4c8b492a361ce15414b91caa205a0eaa46`;
config receipt SHA
`454aca57fb0feb05069b6d15956c2210fe9c3ee524a67e113f7d5dc04c7a309b`.
`source.diff`, `source_inventory.json`, `preparation.json` and CPU logs are
retained. The full cache gate, all63-map qualification, READY construction,
selected final R0/R1/W32 ancestry and GPU release remain pending.

## Actual CPU cache and sampling gate, 2026-11-06

The experimental63-map aggregate passed: all654 source files and63 map/metadata/censor sets were verified. Overall unresolved patch fraction is51.65%; this remains a censored experimental candidate, not certified recoverable detail. No SSIM threshold, RGB, confidence or geometry change was made. Source generation and all63 original map outputs remain immutable.

Isolated `C/semantic_4k_cpu_cache_v1` now contains the native-derived4K source plus SHA-verified maps. Actual frozen datamanager construction completed in22.18s: TRAIN tensor63×2024×3840×3, EVAL3×1012×1920×3, RGB5.472GiB. The shuffled cache contains all63 unique absolute camera IDs. Actual pipeline metadata/map loader loaded63 tensors253×480, patch8/stride8, with complete coverage and no tail.

Actual original W64/body80 initialization found275,519 eligible FAS cells. At steps0/4096/8192,4096-ray batches covered all63 cameras; quotas were2731/0/546/819,2731/0/546/819 and1366/1365/546/819. RGB and confidence gathers matched exact source camera/pixel positions, and every drawn pixel was valid. Proposal means .68548/.68548/.68945 are analytic, unchanged by batch observations. Full CPU check took63.42s, peakRSS17.512GiB. Only cache-loading thread count was bounded to2; ordered future consumption and sampling implementations are unchanged. CUDA was hidden before imports; no radiance model or optimizer was instantiated. Three negative cache/source tests plus10 earlier derivation tests passed. A first pre-cache harness attempt referenced the projection flag on model instead of pipeline; it failed before loading and is preserved in `precache_attribute_failure/`.

A conservative projection diagnostic found that **both HD and4K** actor-FAS map cells exceed8192 under the lower bound `f2d*focal/max_camera_to_AABB_corner_distance` (all63 cameras). Separately, all checked4K AABB-hit sampled rays exceed8192 even at the far ray/AABB intersection. These are geometry-only bounds, not observed rendered-depth grid updates. The existing legacy formula is preserved. This does not make2D maps irrelevant: body80 FAS still uses their levels. Actual step1024 pre/post histograms and touched/admitted max-level counts are mandatory in the separate GPU technical gate.

Technical proposal: cold0..1024 with the native frequency update, save complete state and actual schema2 occupancy; then resume1025..1056 and verify32 attempted iterations, nonempty actual Adam, finite combined gradients on successful updates, strict full-state/RNG/support/cache restoration, and all three unchanged V1 FHD renders. Separate technical-only entrypoint consumes an explicit dual-source contract and654+189 inventories; the frozen production worker still requires its original READY contract. No READY, production release, hidden Jacobian switch, or claim of pure-resolution causality: this remains **4K RGB/confidence +4000-step maps versusHD+1000-step maps**. CPU source/config checks do not imply GPU technical success or quality promotion.

Runnable technical preparation is isolated at `C/semantic_4k_technical_v1` and staged under the matching silver `/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/` directory. All1495 original source files are unchanged;1499 total files include the CPU derivation adapter and three technical-only helpers. Remote CPU verification passed all1499 code +654 source +189 map/metadata/censor hashes and unchanged63/3 split, with CUDA uninitialized. Normalized config diff is exactly the two datamanager/parser paths. The production worker and strict ordinary resume are untouched.

Policy SHA `23ed49ca6377dd9c382b2de4b49a0c2e994db86f32b946e1214e0ad0854e86ff`, contract SHA `654db3a9bac693abf42f8719a80d17404c4af8bbeb9a8cdd9fb08a768e382054`. Concrete proposed admission is fresh-idle silverGPU2 under its shared exclusive lock,10GiB allocator/12GiB measured cap,900s total and5s resource checks. It terminates only its own child process group on failure. No GPU technical run is authorized by the template. Twenty-two CPU tests passed, including strict sources, policy/production rejection, state/RNG digest behavior and actual checkpoint inventory/pruning.

Prerelease review caught an incorrectly duplicated instrumentation call in checkpoint inventory; it was removed before any GPU execution. A separate CLI-only module alias could shadow the original factory; that was corrected to the new entrypoint basename, with a regression test. Earlier source/diff/contract/policy receipts remain in `prerelease_review_corrections/`. Existing completion receipts are also required to match the exact policy before an idempotent return. These corrections do not alter any frozen scientific model implementation.

## Released technical result and separate production preparation

Root release `f437be448ebadc86abafc3fa68f75c07ac41b18e7565329e617960746d4ec91c` launched exclusiveGPU2 supervisor1240294. **PASS in167.56s**, peak7868MiB, no OOM. Cold stage reached1024; actual Adam1023→1055 across the32 resumed attempts, all32 successful and combined gradients finite. All seven serialized-state hashes, RNG/cache order, originalHD mapped support and actual schema2 learned-core/active checks passed. Three FHD V1 views were rendered at1056 and transfer hashes verified. Technical1056 checkpointSHA `8f31d80e869cde4566acb98f50a6ff3cd4f3187ff37f72bf2c564b278734bc9c` is retained remotely. Original pruning removed1024 after successful resume/1056save; its SHA, full-state proof and both actual-grid sidecars remain, but its weights are not claimed retained.

Actual native frequency update1024:2048 finite rendered-depth samples,283 positions insideAABB,268 admitted. It touched336 unique frequency cells, all set tolevel15;266 unique touched cells came from admitted positions and all werelevel15. Histogram changed from2,097,152 zero-level cells to2,096,816 level0 +336 level15. Raw projected frequencies exceeded8192 for every sample. The unchanged legacy path also clamps outside positions to boundary frequency cells; this is not an occupancy/free-space certificate. Observer did not advance RNG or alter occupancy.

Manual inspection of all three views/two actors found correct view/pose/raster correspondence and no gross channel-format mismatch. Early-cold blur, fuzzy halos and floor/background clumps remain; no quality promotion. Contacts and signed source hashes: `C/semantic_4k_technical_v1/visual_review/`.

The separately staged **unlaunched** `C/semantic_4k_production_v1` retains all1499 technical snapshot files (including all1495 original semantic files);1502 total with a new production adapter/worker and unchanged evaluation archiver. The production train loop, evaluator, checkpoint selector and pruning bodies are AST-identical to the originalHD worker. Normalized config differs only in two datamanager/parser data paths; semantic/model/EMA/core implementation and originalHD geometry source remain unchanged. Seven production-specific CPU tests cover release/source/phase guards, exact loop parity, immutable source hashes, unlisted mask-override rejection and durable-launch plan binding.

A new experimental data-ready receipt exists **only in this new generation**, after actual63-map, CPUcache, GPUresume and technical visual checks. It binds654 source files +189 map/metadata/censor files, preserves original source-generation metadata, explicitly labels the4K+4000-step-map bundle and is not execution authorization or default promotion. Actual dataset layout is781 files (654 originals +126 frequency files +READY); extra actor-mask override files are rejected. Remote CPU validation passed all1502 code +654 source +189 map hashes.

Prepared production: originalR0/W64/body80/sourcepad1/coarse.00625/lambda.01, seed42; cold full106316 ancestry then unchanged plateau. First15188 V1 metrics/renders/schema2 plus independent103TRAIN/12DC audit. Physical TRAIN audit coordinates must map continuously as `2*(HD+.5)-.5` and reproduce original ray origins/directions; DC remains FHD. Separate durable controller uses any idle silverGPU under shared locks; released launcher starts the unchanged remote checkpoint archiver/local fetch watcher and idempotent dispatcher. No controller or watcher has been launched by preparation.

Prepared contractSHA `180978cd020989f1da1be6e7c8aed4ca7c1d6a3de75cb8ded06ef061e7711fd7`; controller planSHA `950f74969291baf7aa1efc80843ef7de555c029420d58a81ba094cff6e1121b6`; launch planSHA `e3df8c9e432a89ac6818384cbd4c9d3b3d9db0fc21428ca0f624cd95b49fbf83`; READY SHA `9eb1e43d470fdc51396fc28ac7f97b28f6f1e94c03f8c2da4ac9c02f00eab912`. Execution still requires the exact root production release; the existing template is false.

## Released cold run and first15188 gate, 2026-11-07

Root independently verified14 bindings +1502 code +654 source +189 map artifacts and released exact production root receipt SHA `547fc55d7cefdc290a2809d6f9cb3833602057f35093538e49798f790ecbae25`. This supersedes the preparation-only status above. Idempotent launcher started dispatcher2102689, archive/fetch watcher2102686 and silverGPU1 trainer1301297. Separate read-only health watcher2103046 checks process identity, progress, GPU memory and OOM evidence every60s; durable receipts are in `C/semantic_4k_production_health_v1`. It neither changes nor stops training.

| Same V1 FHD evaluation at15188 | PSNR | SSIM | Masked LPIPS |
|---|---:|---:|---:|
| Original HD R0/W64 |31.414393|.84105578|.0311726214|
| Experimental4K+4000maps R0/W64 |31.293130|.83676706|.0300620813|

Black LPIPS .0319978644→.0301002562; grey .0303473783→.0300239064. DC .0364644583→.0344707016; HC .0289360508→.0276536494; KA .0281173550→.0280618928. Early average LPIPS improves3.56%, while PSNR/SSIM decline. This is not completed-model selection or pure-resolution causality.

Root and fleet agents viewed all three actual GT/HD/4K contacts at `C/semantic_4k_review_15188/{DC,HC,KA}_GT_HD_4K.png`; image hashes and observations are bound in `visual_review.json`. Gross pose/hair/shoes remain; some clothing/hair details improve. Wide exterior haze and interleg/floor clumps persist in all three. No geometry or quality promotion. First15188 checkpoint, config, actual schema2 grid and renders are protected by the archive watcher at `independent_occupancy/semantic_4k_original_r0/step-000015188`; independent physical103TRAIN+12DC ray audit remains a separate gate.
