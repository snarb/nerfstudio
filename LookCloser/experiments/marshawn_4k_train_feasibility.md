# Marshawn 3840×2024 TRAIN / frozen FHD EVAL feasibility

## What was inspected

The initial inspection was read-only: no dataset, masks, frequency maps,
optimizer runs or GPU probes were created at that stage. Evidence inventory:
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

## Subsequent authorized four-camera CPU pilot

`scripts/build_marshawn_4k_pilots.py` implements only AA,NE,JA,MC with at most two
CPU workers. Five unit tests passed. The isolated helper snapshot and CPU logs
are in `resolution_4k_pilots_builder`; detached builder PID2058684 finished in
about31seconds. Per-camera elapsed time was14–16seconds. No GPU, frequency fitting
or training was used, and the other59 TRAIN cameras were not exported.

Each pilot passed exact FHD RGB replay, all eight semantic-array comparisons,
native W32/W64 replay, source veto and calibrated validity replay. An independent
post-check matched all32 semantic-array byte hashes against the frozen dataset.
All nine EVAL files, camera rows/crops and the accepted box remain unchanged.
The four new TRAIN images are3840×2024; `pilot_transforms.json` explicitly names
only these four TRAIN views and three unchanged EVAL views. There is no training
`ready.json` or map directory.

Actual-pixel hair, feet and contact review crops are under
`resolution_4k_pilots/review/<camera>/`; their three panels show source RGB,
contours and W64 confidence. Contours are cyan for the black actor, yellow for
the grey actor, green for the filled sampling core and magenta for source veto.
Hair scopes come from the reviewed native RAW tiles through the inverse lens
mapping. Overview images are display thumbnails; anatomical crops retain scale1.

All four overviews and hair scopes plus selected feet crops were visually
inspected. No systematic contour remapping offset was seen. In MC's grey-foot
scope, confidence/core includes a neighboring dark object or fabric: its
anatomical/occlusion identity still needs root review. Exact FHD replay establishes
that this is inherited source semantics, not a newly introduced4K mapping error.
No source correction or automatic manual-review approval was made.

Receipts: `resolution_4k_pilots/complete.json`, `inspection.json` and per-camera
`replay/*.json`. The6.82MB `four_camera_review.zip` contains review crops and
provenance. Full63 export and any frequency pilot remain unauthorized until the
root reviewer accepts these results.

## Reviewed full63 source export and prepared AA frequency pilot

Root's pilot `root_review.json` approved the CPU-only scope extension. The wrapper
`scripts/build_marshawn_4k_train.py` verifies that review and every original helper
hash, then delegates to the byte-identical pilot builder with the63 TRAIN camera
names. Exactly two CPU workers were used; four-pilot artifacts remain unchanged.

`resolution_4k_train63/complete.json` records63 successful exact FHD RGB/semantic/
confidence/validity replays. An independent post-check verified504 semantic-array
byte hashes. The four prior4K pilot images and semantic-array hashes also match
the new generation. All EVAL files/crops/camera rows and the accepted box are
unchanged. The inherited MC bag inclusion remains explicitly labeled a v1 mask
error, with no geometric-GT claim or silent correction.

The new `transforms.json` describes63 TRAIN4K and3 EVAL FHD images;
`source_metadata.json` binds source hashes and per-camera crops. There is **no
ready.json and no frequency-map directory**. Source completion does not authorize
NeRF training or certify all source masks geometrically.

A separate single-camera AA frequency.95 pilot is prepared in
`resolution_4k_AA_frequency`, with controller root
`resolution_4k_AA_frequency_controller`. Its exact AA image, source completion,
metadata and receipt have been staged and hash-verified on silver. The worker
preserves the16-level/1000-step/.95/patch8/hash23 recipe and records read-only
per-level SSIM counts, exact final unresolved-patch mask, map histogram, time,
memory and actor frequency overlays. This distinguishes censored maximum-level
patches from patches that actually meet the final-level threshold.

Nine CPU tests passed, including review gates, EVAL split exclusion, release
refusal before CUDA imports and progressive censoring accounting. The frozen
heatmap API was separately checked on CPU. No GPU fit has started, no GPU lock is
held, and the release file remains absent. GPU0 belongs first to the separate
exponential-geometry technical gates; root must explicitly release it afterward.

After that release, root must create the staged worker's `root_release.json`
from the unauthorized template with `frequency_fit_authorized:true`,
`reviewer:"root"`, the exact spec hash and source-completion hash. Then:

```bash
C=/fsx/tmp/marshawn_001660_hd/tuning_20261106
"$C/python" "$C/transport_v2_code/LookCloser/scripts/marshawn_tuning_controller.py" launch --root "$C/resolution_4k_AA_frequency_controller"
```

The dispatcher uses the shared per-GPU flock and verifies the GPU is idle before
starting. The pilot map alone cannot qualify a63-map dataset or create READY.

### AA frequency pilot: actual outcome and longer-fit control

The authorized 1000-update pilot has now completed (the earlier preparation
status above is historical). Its actual SSIM counters, source-bound map and actor
overlays are in `resolution_4k_AA_frequency/result`. Runtime was84.18seconds;
peak allocated GPU memory was1.391GiB. Of121440 patches,96160 (79.183%)
remained below SSIM.95 after the final level. Maximum-level assignments cover
85.638%; the difference is patches that actually passed at level15.

The original HD AA log at
`frequency_radial_k4/next4_AA_EA_GA_NA/worker0.log` records final resolved=71.5%,
therefore unresolved is approximately28.5%, with only0.1 percentage-point log
precision. This is an observed fitting counter, not inferred from maximum-level
map occupancy. Exact original per-patch censoring was not found.

|4K cohort|Patches|Exact unresolved|Maximum level|
|---|---:|---:|---:|
|Actor interior (coverage≥90%)|4014|80.84%|85.40%|
|Black actor interior|2766|87.96%|90.42%|
|Grey actor interior|1248|65.06%|74.28%|
|Actor boundary|494|47.57%|57.09%|
|Outside actors|116932|79.26%|85.77%|

Actual source/heatmap overlays were inspected. Large dark-clothing and floor
regions are nearly uniformly maximum-level, as well as hair; this is not
selective evidence of tiny detail. The source shows folds, lettering and hair,
but sensor noise and insufficient fitting remain confounded. No claim is made
that these masks are geometric truth. Full cohort statistics and source map
hashes are in `result/cohort_analysis.json`.

The next prepared control keeps source pixels, seed42, SSIM.95, architecture,
patch8 and optimizer math fixed and changes only updates per level1000→4000.
This restores approximately the samples-per-pixel budget lost when raster area
quadrupled. `scripts/prepare_marshawn_4k_AA_long_fit.py` creates an independent
frozen worker, exact three-site source diff, strict recipe, hash receipts and
unauthorized release template. Six CPU tests passed, and eight staged remote
hashes were verified. The new controller is prepared at
`resolution_4k_AA_frequency_4000_controller`; it is **not launched** and its
source-bound root release is absent. Expected fitting time is roughly5–6minutes,
to be measured rather than assumed. Review should compare exact censoring,
actor regions and visible detail before authorizing all63 maps. No threshold
relaxation, READY creation or NeRF training is included.

### AA4000 completed; source staging and next diagnostic

Root reviewed and released the4000-update control. It completed in303.79seconds
with unchanged1.391GiB peak allocation and no OOM; its GPU lock is released.
Exact unresolved fell79.183%→45.278%, while maximum-level assignment fell
85.638%→56.378%. Actor interior censoring fell80.84%→56.33%, black87.96%→69.41%,
grey65.06%→27.32%, boundary47.57%→19.03%, outside79.26%→45.01%.
There were43895 failed→passed patches and2721 passed→failed patches: changing
the update count changes the optimization trajectory, so this is not a strictly
monotonic per-patch comparison. Actual same-source censor overlays show more
resolved folds, grey clothing and floor, but dark clothing remains heavily
censored. Residual censoring cannot yet be uniquely assigned to sensor noise,
true texture, fitting budget or capacity.

`scripts/compare_marshawn_4k_frequency_fits.py` reproduces cohort statistics,
verifies source/map/censor hashes and creates actual source/1000/4000 comparison
panels. Artifacts are in `resolution_4k_AA_frequency_4000/result`. No NeRF quality
or geometry approval follows from these fitting statistics.

The complete4K source generation was independently transferred to silver:
654/654 file hashes match, including unchanged FHD EVAL and floor-frame poses.
`resolution_4k_train63_transfer/verification.json` binds that check; READY is
absent. Two independent63-camera fleet queues are prepared at
`resolution_4k_frequency_queue_{1000,4000}_fleet`, with exact per-camera workers,
source hashes and unauthorized release templates. They can use any idle
silverGPU0..7 under shared locks. Only one reviewed recipe may be launched;
admission waits for existing training queues to claim their initial slots.
Active jobs are never preempted. Serial preparations remain preserved and
unlaunched. No full63 frequency queue has been authorized.

A single AA16000 diagnostic is separately prepared for localGPU0, pending root
review/release. Its only fitting change is4000→16000 updates; the three-site
diff and exact input hashes are preserved. Local/silver Python, Torch, CUDA,
TCNN Python/native binary, Torch extension and frozen preprocessor hashes match,
and both GPUs are RTXPRO6000Blackwell sm120. Twelve CPU tests passed. Runtime is
estimated around20minutes, with actual measurement required. Its purpose is to
test whether the remaining high dark-clothing censoring is still budget-limited
before relaxing SSIM or authorizing all-camera fitting.

The proposed16000 run now has a separate `_16000_residual` generation, preserving
the plain preparation. A return-value wrapper retains the actual fitted2D model
without changing training or random debug sampling. After fitting restores the
original SSIM function, a bound helper renders five fixed source-declared ROIs
(black shirt, black pants, hair, grey folds, shoe) at level15. It writes GT,
prediction, signed float residuals and patchwise SSIM7 on the original8×8 grid;
PNG residuals use a labeled8× visualization only. Before/after model-state and
Python/NumPy/Torch CPU/CUDA RNG hashes must agree, and model train/eval flags are
restored. CPU tests verify preservation and deliberate RNG-mutation rejection.
This generation remains pending explicit root release; the actual residuals
have not yet been measured.

### AA16000 residual diagnostic: completed, recipe rejected

Root authorized the residual generation. It completed1189.93seconds of fitting
with1.391GiB peak allocation; actual model and Python/NumPy/Torch CPU/CUDA RNG
hashes agreed before and after all five post-fit renders. Ten raw evidence
artifact hashes were verified. The source-only fixed ROI montage was manually
inspected in `resolution_4k_AA_frequency_16000_residual/result/residual_evidence/`.

Increasing4000→16000 updates made exact unresolved **worse**,45.278%→81.727%.
Actor interior censoring worsened56.33%→77.85%; black69.41%→88.25%; grey27.32%→
54.81%. There were3180 failed→passed patches and47443 passed→failed. Consequently
the larger budget is not a qualified frequency recipe and does not authorize
the63-camera fit.

|Fixed ROI|Mean SSIM8|Fraction passing.95|Mean absolute RGB residual|
|---|---:|---:|---:|
|Black shirt|0.89000|0.78%|0.01289|
|Black pants|0.86393|0.39%|0.01463|
|Hair|0.91540|35.16%|0.01392|
|Grey folds|0.94884|54.30%|0.00941|
|Shoe|0.92111|29.17%|0.01256|

Predictions retain folds, hair and highlights but show discrepant colored fine
texture on dark cloth. Residuals occur within smooth regions as well as on
details. This evidence does not isolate sensor noise as the cause. The frozen
2D fitter uses constant AdamLR0.01 without decay; an optimization/LR comparison
is a plausible next diagnostic, not a proven explanation. Relaxing the SSIM
threshold would conceal the unresolved optimization question. All original
sources, ordinary training jobs and full63-map release status remain unchanged.
