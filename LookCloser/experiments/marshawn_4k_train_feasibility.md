# Marshawn 3840×2024 TRAIN / frozen FHD EVAL feasibility

## What was inspected

Read-only design for a later resolution factor. No dataset, masks, frequency
maps, optimizer runs or GPU probes were created. Evidence inventory:
`/fsx/tmp/marshawn_001660_hd/tuning_20261106/resolution_4k_design/feasibility.json`.

The existing AA diagnostic in `resolution_feasibility_AA_v2` reconstructs the
current FHD PNG byte-for-byte before deriving the higher-resolution source. Its
4K→FHD area-downsample differs by mean1.729 and p99=11 display codes: spatial
filtering and remapping do not commute. Therefore an upscaled FHD image or
downsampled 4K image cannot replace the frozen evaluation target.

## Source-preserving derivation

1. Use the existing63 TRAIN cameras only, from their original8192×4320 TIFFs.
   All63 files exist, totaling12.46GiB. This inspection checked existence/size,
   not fresh hashes of all TIFF payloads; export must verify and bind those hashes.
   Keep filename/camera ordering and the existing floor-frame poses unchanged.
   Copy the accepted box byte-for-byte; do not apply `world_to_box` a second time.
2. Reuse the complete source photometry contract, not global gain4 alone:
   vendor Log3G10 decode, positive RWG, RWG→Rec709 matrix, positive Rec709,
   native-linear AREA prefilter to target raster, one calibrated lens remap,
   then the existing float32 gain4 and inherited TRAIN-fit radial k4/center plus
   per-camera scalar EV, followed by unchanged Reinhard/sRGB8 rounding. The fit
   SHA is `272478de1e9d563961d0553c8009c2d5972fdf628d407f40d90485d6e9af9355`.
   Preserve arithmetic ordering from `probe_marshawn_source_resolution.py`.
3. For TRAIN only, double `w,h,fl_x,fl_y,cx,cy`. Keep poses and zero residual
   distortion unchanged. The current transform file already uses per-frame
   intrinsics and has no global width/focal fields to override them.
4. Derive masks from `confidence_native_hair_v2/raw/<camera>.npz`. All66 archive
   hashes were verified against the campaign-bound manifest; they contain native
   `black`, `grey`, reviewed add-only-hair `nominal`, `source_ineligible` and crop
   placement. This reuses the reviewed native source, not upscaled HD semantics.
   Recompute `native_confidence` at native widths32/64, embed in the8192×4320
   raster, and call the existing AREA-before-remap coverage helper with the4K
   target intrinsics. W64 continues to mean64 **native** pixels; do not change it
   to128. Veto coverage>1e-6 overrides every weight; nominal/actor coverage uses.5.
5. Reconstruct validity using the original helper's native map-center rule
   `0≤mx≤8191, 0≤my≤4319`; all current TRAIN valid rasters contain every pixel.
   Do not silently substitute a stricter kernel-footprint validity policy.
   Assign reviewed added hair to the nearest projected actor exactly as the
   current builder does; remove actor overlap. Recompute body80 core/band at
   twice the target-pixel radii: elliptical kernel17→33,33→65,65→129, preserving
   the existing upper75% wider-band policy. Otherwise resolution would also alter
   the physical sampling margin. Preserve the underlying native confidence curve.
6. Symlink/copy DC,HC,KA PNGs, validity PNGs and semantic NPZs unchanged. Preserve
   their per-frame transforms/intrinsics, sealed actor crops and every byte hash.
   The inventory records all nine evaluation-file hashes and full camera rows.

## Loader and memory feasibility

The parser supports per-frame dimensions. All TRAIN images remain homogeneous
at3840×2024, and all EVAL images remain homogeneous at1920×1012 in a separate
cache. Consequently the body80 adapter's rigid NHWC/full-cache requirements and
single train width remain valid. Mixing sizes **within** TRAIN would fail that
adapter and is outside this proposal. Frequency lookup uses each train camera's
actual dimensions; validation rendering uses the independent eval camera.

The preserved recipe uses CPU float32 image caching, `images_on_gpu=False` and
`cache_train_rays=False`. Estimated memory per4K run:

| Component | GiB |
|---|---:|
|63 RGB images, float32 |5.47|
|Adapter masks and two float32 confidence arrays |6.38|
|Dataset validity cache |0.46|
|Valid-pixel int64 pools |3.65|
|Core/band int64 pools, estimated4× current counts |0.23|
|Additional RGB collation peak |5.47|

Allow20–35GiB CPU RAM/run including allocator/model/Python FAS cells and decode
temporaries; this is an estimate, not a measured4K peak. Silver currently has
about1.3TiB available RAM, while representative FHD trainer RSS is7.2–7.4GiB.
Rendering batch/ray budget and model architecture stay unchanged. Avoid enabling
a full4K ray cache merely for this experiment. Decode native RGB/masks with a
bounded CPU worker count and stage derived data to silver's `/cache` filesystem.

## Frequency preparation feasibility

Refit all63 maps from the new RGB bytes. Never resize, reuse or reinterpret the
FHD maps. Preserve the current recipe: SSIM.95, patch/stride8,16 levels,
min16/max8192, hash23,1000 updates/level, training batch16384, eval batch1024.

The grid grows from126×240 to253×480:4.016× as many patches. FHD's last four
raster rows were uncovered;2024 is divisible by8, so4K covers all rows. Raw map
storage across63 cameras is only29.19MiB; one4K float32 RGB tensor is88.95MiB.
The2D model size and total1,008,000 optimizer updates remain unchanged. Worst-case
SSIM work is1,943,040 patches or1904 batches per image. The fixed sample budget
falls from134.9 to33.7 random samples/source pixel, so late-level censoring/noise
must be audited rather than calling every maximum-level patch real detail.

No timing benchmark was run. Training work stays roughly fixed while patch
evaluation grows approximately4×; actual elapsed time requires one approved
TRAIN-only pilot. Frequency thresholds remain unchanged here; tuning them comes
after structural/resolution selection.

## Proposed receipts and release gates

- Before export, bind raw TIFF, vendor decoder, rig camera/image files, inherited
  fit/contract, native semantic archives/hair-review, all helper sources, original
  transforms, accepted box and evaluation-file hashes.
- During each TRAIN export, require an exact FHD RGB replay and a FHD array-level
  semantic/weight replay from the native archives. Write separate4K image and
  semantic hashes plus coverage, veto, overlap and physical-margin statistics.
  NPZ container bytes need not replay identically; individual arrays must.
- Preview native hair/feet/occlusion contours on several diverse TRAIN cameras,
  including the AA source diagnostic, before approving all63 outputs.
- Before map fitting, verify split isolation, dimensions and scaled-intrinsic
  rays. Fit an approved TRAIN pilot to measure time, memory and unresolved-patch
  behavior, then use a locked resumable63-camera queue with source-bound receipts.
- Seal a new ready/inventory generation only when all63 new maps exist and every
  evaluation hash/crop/camera and the box remain unchanged. A separate cold
  training arm must inherit the then-reviewed best structural recipe, not an
  obsolete original-DAG winner. No training release is granted by this design.
