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
   signed/hash-bound HD→4K derivation receipt: exact63 camera order/splits,
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
Do not launch eight4K runs based only on available VRAM.

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
